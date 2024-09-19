import glob
import os
import re
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# warning ignore
warnings.filterwarnings("ignore")


class Power_PV_Dataset(Dataset):
    def __init__(self, cfg):
        self.seq_len = cfg.seq_len
        self.label_len = cfg.label_len
        self.pred_len = cfg.pred_len

        self.features = cfg.features
        self.target = cfg.target
        self.scale = cfg.scale
        self.timeenc = cfg.timeenc
        self.freq = cfg.freq

        self.test_run = cfg.test_run
        self.__read_data__(cfg)

    def get_train_idx(self):
        return self.train_start, self.train_end

    def __read_data__(self, cfg):
        # * load data
        print(f"file was loaded from {cfg.data_path}")
        df_raw = pd.read_csv(cfg.data_path)
        if self.test_run:
            df_raw = df_raw[: cfg.test_run_len]

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove("dt")
        df_raw = df_raw[["dt"] + cols + [self.target]]
        df_raw["dt"] = pd.to_datetime(df_raw["dt"])

        # # * split
        self.train_start = 0
        self.train_end = int(len(df_raw) * 0.7)

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        # * scaling
        self.scaler = StandardScaler()
        train_data = df_data[self.train_start : self.train_end]
        self.scaler.fit(train_data.values)
        print(f"saving scaler to {cfg.scaler_path}")
        joblib.dump(
            self.scaler,
            cfg.scaler_path,
        )

        self.data = self.scaler.transform(df_data.values)

        # * feature engineering
        df_stamp = df_raw[["dt"]]
        df_stamp["dt"] = pd.to_datetime(df_stamp.dt)
        df_stamp["month"] = df_stamp.dt.apply(lambda row: row.month, 1)  # type: ignore
        df_stamp["day"] = df_stamp.dt.apply(lambda row: row.day, 1)  # type: ignore
        df_stamp["weekday"] = df_stamp.dt.apply(lambda row: row.weekday(), 1)  # type: ignore
        df_stamp["hour"] = df_stamp.dt.apply(lambda row: row.hour, 1)  # type: ignore
        self.data_stamp = df_stamp.drop(columns=["dt"]).values

        print(f"[ds] data_stamp shape: {self.data_stamp.shape}")
        print(f"[ds] data shape: {self.data.shape}")
        print(f"[ds] test_run: {self.test_run}")

    def __len__(self):
        return len(self.data) - self.pred_len - self.seq_len  # type: ignore

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data[s_begin:s_end]  # type: ignore
        seq_y = self.data[r_begin:r_end]  # type: ignore
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        batch = dict(
            index=torch.tensor([s_begin, s_end, r_begin, r_end], dtype=torch.int64),
            seq_x=torch.tensor(seq_x, dtype=torch.float32),
            seq_y=torch.tensor(seq_y, dtype=torch.float32),
            seq_xt=torch.tensor(seq_x_mark, dtype=torch.float32),
            seq_yt=torch.tensor(seq_y_mark, dtype=torch.float32),
        )
        return batch


if __name__ == "__main__":

    import os

    import yaml
    from easydict import EasyDict as edict

    datamodule_name = "pv_dacon"
    print(f">>>>>>>>>> [{datamodule_name}] <<<<<<<<<<<<<<<")
    cwd = os.getcwd()
    cfg_data_path = f"{cwd}/configs/datamodule/{datamodule_name}.yaml"
    with open(cfg_data_path) as f:
        cfg = edict(yaml.safe_load(f))
        cfg = cfg.cfg  # type: ignore

    data_dir = f"{cwd}/data"
    data_dir = f"{data_dir}/{cfg.datafolder_name}"
    cfg.data_path = f"{data_dir}/{cfg.file_name}.csv"
    cfg.scaler_path = f"{data_dir}/{cfg.file_name}_scaler.pkl"
    cfg.test_run = False

    dataset = Power_PV_Dataset(cfg)
    print(f"dataset length: {len(dataset)}")
    # sys.exit()
    from torch.utils.data import Subset

    # dataset = Subset(dataset, cfg.pred_indices)
    # check dataset
    print(f"dataset length: {len(dataset)}")
    dataset0 = dataset[0]
    print(f"batch keys: {dataset0.keys()}")
    print(f"seq_x shape: {dataset0['seq_x'].shape}")
    print(f"seq_y shape: {dataset0['seq_y'].shape}")
    print(f"seq_xt shape: {dataset0['seq_xt'].shape}")
    print(f"seq_yt shape: {dataset0['seq_yt'].shape}")

    seq_len = cfg.seq_len
    pred_len = cfg.pred_len
    label_len = cfg.label_len

    shuffle_flag = True
    drop_last = True
    batch_size = cfg.batch_size

    from torch.utils.data import DataLoader, Subset

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last,
    )

    for i, batch in enumerate(dataloader):
        print(
            f'{i:<5d}: {batch["seq_x"].shape}, {batch["seq_y"].shape}, {batch["seq_xt"].shape}, {batch["seq_yt"].shape}'
        )
        break

    dataset = Power_PV_Dataset(cfg)

    print(f"dataset length: {len(dataset)}")
