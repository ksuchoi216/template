import sys

import lightning as L
import pandas as pd
from torch.utils.data import DataLoader, Dataset

# from src.datasets.sensor_dataset import SensorDataset

dataset_dict = {
    "sensor": SensorDataset,
}


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        self.save_hyperparameters()
        print(f"[datamodule] data_name: {cfg.dataset_name}")
        self.dataset = dataset_dict[cfg.dataset_name]

        self.train_dataset = None
        self.val_dataset = None
        self.pred_dataset = None
        self.test_dataset = None
        self.cfg = cfg

    def setup(self, stage):
        self.stage = stage

    def data_provider(self, cfg, stage):

        if stage in ["val", "test", "pred"]:
            shuffle_flag = False
            drop_last = True
            batch_size = cfg.batch_size
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = cfg.batch_size  # bsz for train and valid

        dataset = self.dataset(cfg, stage)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            drop_last=drop_last,
            num_workers=cfg.num_workers,
        )
        return dataloader

    def train_dataloader(self):
        return self.data_provider(self.cfg, "train")

    def val_dataloader(self):
        return self.data_provider(self.cfg, "val")

    def test_dataloader(self):
        return self.data_provider(self.cfg, "test")

    def predict_dataloader(self):
        return self.data_provider(self.cfg, "pred")
