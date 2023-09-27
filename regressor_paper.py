from ocpmodels.common.utils import setup_imports
import torch
import itertools
import json
import hashlib
import os
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
from pytorch_lightning import LightningModule, LightningDataModule
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import setup_imports
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from data_model_utils import EmbeddingData, FFBlock, EmbeddingRegressor, MyDataModule
from pytorch_lightning.loggers import WandbLogger
from matplotlib import pyplot as plt

if __name__ == '__main__':

    pl.seed_everything(42, workers=True)
    # wandb_logger = WandbLogger(
    #    project="magmom", name="embedding_regressor_paper_no_zeros_no_sign_0.2cutoff"
    # )
    setup_imports()

    dataset = EmbeddingData("embeddings_for_ref_paper.pt")
    len_train = int(0.8 * len(dataset))
    len_val = int(0.1 * len(dataset))
    len_test = len(dataset) - len_train - len_val
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [len_train, len_val, len_test]
    )
    dm = MyDataModule(
        batch_size=256,
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        # train_collate_fn=train_collate_fn,  # To take abs and >= 1
        # val_collate_fn=train_collate_fn,
        # test_collate_fn=test_collate_fn,
    )


    model = EmbeddingRegressor()

    checkpoint_callback = ModelCheckpoint(
        monitor="embedding_val_loss",  # need to implement recall score (tp / (tp + fn))
        mode="min",
        dirpath="magmom_checkpoints",
        filename="regressor_paper_no_zeros_no_sign_0.1cutoff-{epoch}-{step}-{embedding_val_loss:.4f}",
    )
    learning_rate_callback = LearningRateMonitor(logging_interval="epoch")
    # trainer = pl.Trainer(
    #    gpus=1,
    #    # deterministic=True,
    #    max_epochs=10000,
    #    # precision=16,
    #    # gradient_clip_val=0.5,
    #    # gradient_clip_algorithm="value",
    #    # strategy="ddp",
    #    #    logger=wandb_logger,
    #    callbacks=[learning_rate_callback, checkpoint_callback],
    # )
    # trainer.fit(
    #    model,
    #    dm.train_dataloader(),
    #    dm.val_dataloader(),
    # )
    bmp = checkpoint_callback.best_model_path
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bmp = "checkpoints/regressor_paper_no_zeros_no_sign-epoch=1386-step=59641-embedding_val_loss=0.0902.ckpt"
    model.load_state_dict(torch.load(f"{bmp}", map_location=device)["state_dict"])
    model = model.to(device)
    test_data = dm.test_dataloader()
    test_labels = []
    test_preds = []
    with torch.no_grad():
        for x in test_data:
            model.eval()  # This is really important since we are using batchnorm and Dropout
            breakpoint()
            test_labels.extend(torch.abs(x[1]).cpu().numpy().tolist())
            x = x[0].to(device)
            pred_magmoms = model(x).squeeze().cpu().numpy().tolist()
            test_preds.extend(pred_magmoms)
    fig = plt.figure(figsize=(10.0, 10.0))

    import seaborn as sns

    sns.set_theme(style="ticks")

    sns.jointplot(x=test_labels, y=test_preds, kind="hex", color="#4CB391")
    plt.savefig("seaborn_parity_test_set_0.1cutoff_EVAL.pdf")

