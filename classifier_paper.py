import torch
import itertools
import json
import hashlib
import os
import umap
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch_geometric
import torch_scatter
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, random_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR, CosineAnnealingLR
import wandb
import numpy as np
from scipy.ndimage import convolve1d
from collections import Counter
from pymatgen.core import Structure
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch.utils.data import Dataset as DS  # collision
from torch_geometric.utils.convert import (
    to_scipy_sparse_matrix,
)  # for representation purposes
from torch_geometric.utils import (
    to_dense_adj,
    to_dense_batch,
)  # for representation purposes

# For LDS
from collections import Counter
from scipy.ndimage import convolve1d
from scipy.stats import gaussian_kde
from utils import get_lds_kernel_window
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning.loggers import WandbLogger
from ocpmodels.common.registry import registry
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from utils import get_lds_kernel_window
from loss import weighted_mse_loss
from torchmetrics import F1Score
from torchmetrics.classification import BinaryF1Score
from data_model_utils import MyDataModule, MyOwnDataset, collate_fn, ocp_model

# wandb_logger = WandbLogger(
#    project="magmom", name="classifier_separate_file_boundary_0.5_ternary"
# )


class MagmomClassifier(LightningModule):
    def __init__(self, backbone, boundary):
        super(MagmomClassifier, self).__init__()
        self.backbone = backbone
        self.boundary = boundary

    def forward(self, x, training=True):
        if not training:
            self.backbone.eval()
        print("Module is in training?", self.backbone.training)
        magmoms = self.backbone(x)
        return magmoms

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer, mode="max", patience=50, factor=0.5
                ),
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_acc",
            },
        }

    def training_step(self, batch, batch_idx):
        yhat = self(batch)[0].squeeze()
        # y = torch.LongTensor(
        #    [
        #        0 if (-self.boundary < magmom < self.boundary) else 1
        #        for magmom in batch.y
        #    ]
        # ).to(
        #    self.device
        # )  # FIXME: How to pick m in | x | > m
        y = torch.LongTensor(
            [
                0
                if (magmom <= -self.boundary)
                else 1
                if (-self.boundary < magmom < self.boundary)
                else 2
                for magmom in batch.y
            ]
        ).to(
            self.device
        )  # Three classes #TERNARY
        # y = torch.LongTensor(
        #    [0 if abs(magmom) < self.boundary else 1 for magmom in batch.y]
        # ).to(self.device)
        # weights = torch.tensor([0.55967772, 4.68916834]).to(self.device)  # binary
        weights = torch.tensor([6.27546332, 0.37310474, 6.2329898]).to(
            self.device
        )  # Three classes
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
        loss = criterion(yhat, y)
        # Let's derive the weights per batch based on convolution with Gaussian 1D kernel
        # if not validation:
        #    with torch.no_grad():
        #        labels = batch.y.detach().cpu().numpy()
        #        bins = np.linspace(min(labels), max(labels), 100)
        #        bin_index_per_label = np.digitize(labels, bins)
        #        Nb = max(bin_index_per_label) + 1
        #        num_samples_of_bins = dict(Counter(bin_index_per_label))
        #        emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(Nb)]
        #        lds_kernel_window = get_lds_kernel_window(kernel="gaussian", ks=5, sigma=2)
        #        eff_label_dist = convolve1d(np.array(emp_label_dist), weights=lds_kernel_window, mode="constant")
        #        eff_num_per_label = [
        #            eff_label_dist[bin_idx] for bin_idx in bin_index_per_label
        #        ]
        #        weights = [np.float32(1 / x) for x in eff_num_per_label]
        #        scaling = len(weights) / np.sum(weights)
        #        weights = torch.tensor([scaling * x for x in weights]).to(self.device)
        # loss = nn.MSELoss()(yhat.type(y.dtype), y)
        # signs = torch.tensor([-1 if label < 0 else 1 for label in y]).to(self.device)
        # zeros = torch.tensor(0).expand_as(signs).to(self.device)
        # ones = torch.tensor(1).expand_as(signs).to(self.device)
        # loss = (yhat - y) ** 2 #+ torch.max(zeros, ones - signs * yhat)
        # if not validation:
        #    loss *= weights
        # loss = torch.mean(loss)
        self.log("train_loss", loss.item(), on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        yhat = self(batch, training=False)[0].squeeze()
        preds = torch.argmax(yhat, axis=1)
        # y = torch.LongTensor(
        #    [0 if abs(magmom) < self.boundary else 1 for magmom in batch.y]
        # ).to(self.device)
        y = torch.LongTensor(
            [
                0
                if (magmom <= -self.boundary)
                else 1
                if (-self.boundary < magmom < self.boundary)
                else 2
                for magmom in batch.y
            ]
        ).to(self.device)
        # y = torch.LongTensor(
        #    [
        #        0 if (-self.boundary < magmom < self.boundary) else 1
        #        for magmom in batch.y
        #    ]
        # ).to(self.device)
        criterion = torch.nn.CrossEntropyLoss()
        num_correct = (preds == y).sum().item()
        num_samples = y.shape[0]
        val_loss = criterion(yhat, y)

        # val_loss = nn.MSELoss()(yhat, y)
        # val_mae = nn.L1Loss()(yhat, y)
        # total_val_loss = self.training_step(batch, batch_idx, validation=True)
        self.log("val_loss", val_loss.item(), on_epoch=True, batch_size=16)

        # self.log('val_mae', val_mae.item(), on_epoch=True)
        # self.log('total_val_loss', total_val_loss.item(), on_epoch=True)
        return {
            "val_loss": val_loss,
            "num_correct": num_correct,
            "num_samples": num_samples,
            "yhat": preds,
            "y": y,
        }

    def validation_epoch_end(self, outputs):
        if self.global_rank == 0:
            gathered = self.all_gather(outputs)
            #            val_acc = sum(output["num_correct"] for output in gathered) / sum(
            #                output["num_samples"] for output in gathered
            #            )
            yhats = (
                torch.hstack([output["yhat"] for output in gathered])
                .squeeze()
                .to(self.device)
            )
            ys = (
                torch.hstack([output["y"] for output in gathered])
                .squeeze()
                .to(self.device)
            )
            # f1 = BinaryF1Score().to(self.device)
            f1 = F1Score(task="multiclass", num_classes=3).to(self.device)
            val_acc = f1(yhats, ys).item() * 100.0
            self.log("val_acc", val_acc, rank_zero_only=True)

        return val_acc


if __name__ == "__main__":
    pl.seed_everything(1, workers=True)
    classifier_model = MagmomClassifier(ocp_model, boundary=0.1)
    dataset_object = MyOwnDataset(structures=None, root="disk_data")
    len_train_o = int(0 * len(dataset_object))
    len_val_o = int(0 * len(dataset_object))
    len_test_o = len(dataset_object) - len_train_o - len_val_o
    train_set_o, val_set_o, test_set_o = torch.utils.data.random_split(
        dataset_object, [len_train_o, len_val_o, len_test_o]
    )
    dmo = MyDataModule(
        batch_size=1024,
        train_set=train_set_o,
        val_set=val_set_o,
        test_set=test_set_o,
        train_collate_fn=collate_fn,  # To take abs and >= 1
        val_collate_fn=collate_fn,
        test_collate_fn=collate_fn,
    )
    testloader = dmo.test_dataloader()
    breakpoint()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_all_embs = torch.empty(0, 256)
    classifier_model = classifier_model.to(device)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",  # need to implement recall score (tp / (tp + fn))
        mode="max",
        dirpath="magmom_checkpoints",
        filename="classifier_paper-{epoch}-{step}-{val_acc:.4f}",
    )
    learning_rate_callback = LearningRateMonitor(logging_interval="epoch")
    trainer = pl.Trainer(
        gpus=1,
        # deterministic=True,
        max_epochs=10000,
        # precision=16,
        # gradient_clip_val=0.5,
        # gradient_clip_algorithm="value",
        # strategy="ddp",
        # logger=wandb_logger,
        callbacks=[learning_rate_callback, checkpoint_callback],
    )
    # trainer.fit(
    #    classifier_model,
    #    dmo.train_dataloader(),
    #    dmo.val_dataloader(),
    # )
    bmp = checkpoint_callback.best_model_path
    bmp = "checkpoints/classifier_paper-epoch=12-step=2392-val_acc=93.6278.ckpt"
    classifier_model.load_state_dict(torch.load(f"{bmp}")["state_dict"])
    pred_masks = []
    label_masks = []
    with torch.no_grad():
        classifier_model.eval()  # Very important to be in inference mode
        for test_batch in testloader:
            test_batch = test_batch.to(device)
            pred_mask, _, embedding = classifier_model(test_batch)
            pred_masks.extend(torch.argmax(pred_mask, dim=1).cpu().numpy().tolist())
            # label_mask = [
            #    0 if abs(magmom) < classifier_model.boundary else 1
            #    for magmom in test_batch.y
            # ]  # binary
            # label_mask = [
            #    0
            #    if magmom <= -classifier_model.boundary
            #    else 2
            #    if magmom >= classifier_model.boundary
            #    else 1
            #    for magmom in test_batch.y
            # ]  # ternary

            # label_masks.extend(label_mask)
            test_all_embs = torch.vstack((test_all_embs, embedding.cpu()))

    # Visualize the embedding partition within the test set
    figure = plt.figure(figsize=(20.0, 20.0))
    ax1 = figure.add_subplot(111)
    reducer = umap.UMAP(random_state=42)
    reduced_emb = reducer.fit_transform(test_all_embs.numpy())

    colors = np.array(["blue", "red", "green"])
    ax1.scatter(
        reduced_emb[:, 0],
        reduced_emb[:, 1],
        c=colors[np.array(pred_masks)],
        s=500,
        alpha=0.2,
    )
    # Print out the F1 Score on the test set
    # f1_test = F1Score(task="multiclass", num_classes=3).to(device)
    # f1_score_test = (
    #    f1_test(
    #        torch.tensor(pred_masks).to(device), torch.tensor(label_masks).to(device)
    #    )
    #    * 100
    # )
    # print(f1_score_test)
    figure.savefig("classifier_paper_test_set_ternary_RuPtMoOx_case_study.pdf")
