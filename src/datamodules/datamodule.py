import os
import sys

import lightning as L
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
print(f"parent_dir: {parent_dir}")
sys.path.insert(0, parent_dir)

from src.datasets.dataset import Power_PV_Dataset

dataset_dict = dict(
    power_uci=Power_PV_Dataset,
    pv_dacon=Power_PV_Dataset,
)


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        print(f"[dm] data_name: {cfg.dataset_name}")
        dataset_maker = dataset_dict[cfg.dataset_name]
        self.dataset = dataset_maker(cfg)
        # self.train_start, self.train_end = dataset.get_train_idx()
        # self.dataloaders = self.data_provider(cfg, dataset)

        self.train_dataset = None
        self.val_dataset = None
        self.pred_dataset = None
        self.test_dataset = None

        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers
        # self.pred_idx = [i - self.cfg.seq_len for i in self.cfg.test_indices]

    def setup(self, stage):
        self.stage = stage
        if stage == "fit":
            self.train_idx, __test_idx = train_test_split(
                list(range(len(self.dataset))), test_size=0.2, shuffle=False
            )
            self.val_idx, self.test_idx = train_test_split(
                __test_idx, test_size=0.5, shuffle=False
            )
            self.train_dataset = Subset(self.dataset, self.train_idx)
            self.val_dataset = Subset(self.dataset, self.val_idx)

        if stage == "test":
            self.test_dataset = Subset(self.dataset, self.test_idx)

        if stage == "predict":
            self.predict_dataset = Subset(self.dataset, self.test_idx)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,  # type: ignore
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,  # type: ignore
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,  # type: ignore
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):  # -> DataLoader[Any]:
        return DataLoader(
            self.predict_dataset,  # type: ignore
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )


if __name__ == "__main__":

    import os

    import yaml
    from easydict import EasyDict as edict

    datamodule_name = "pv_dacon"
    print(f">>>>>>>>>> [{datamodule_name}] <<<<<<<<<<<<<<<")
    cwd = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    print(f"parent_dir: {parent_dir}")
    sys.path.append(os.path.join(cwd, "src"))
    cfg_data_path = f"{cwd}/configs/datamodule/{datamodule_name}.yaml"
    with open(cfg_data_path) as f:
        cfg = edict(yaml.safe_load(f))
        cfg = cfg.cfg  # type: ignore

    data_dir = f"{cwd}/data"
    data_dir = f"{data_dir}/{cfg.datafolder_name}"
    cfg.data_path = f"{data_dir}/{cfg.file_name}.csv"
    cfg.scaler_path = f"{data_dir}/{cfg.file_name}_scaler.pkl"
    cfg.test_run = False

    datamodule = DataModule(cfg)

    datamodule.setup("fit")

    # sys.exit()
    datamodule.setup("predict")
    test_dataloader = datamodule.predict_dataloader()

    for i, batch in enumerate(test_dataloader):
        print(
            f'{i:<5d}: {batch["seq_x"].shape}, {batch["seq_y"].shape}, {batch["seq_xt"].shape}, {batch["seq_yt"].shape}'
        )
