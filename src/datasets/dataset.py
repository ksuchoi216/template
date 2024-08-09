import glob
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sktime.datasets import load_from_tsfile_to_dataframe
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")


class Dataset_Others(Dataset):
    def __init__(self, cfg, stage):
        if cfg.seq_len == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = cfg.seq_len
            self.label_len = cfg.label_len
            self.pred_len = cfg.pred_len
        # init
        assert stage in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2, "pred": 3}
        self.set_type = type_map[stage]

        self.features = cfg.features
        self.target = cfg.target
        self.scale = cfg.scale
        self.timeenc = cfg.timeenc
        self.freq = cfg.freq
        self.__read_data__(cfg)

    def __read_data__(self, cfg):
        self.scaler = StandardScaler()

        data_path = f"{cfg.data_dir}/{cfg.datafolder_name}/{cfg.file_name}.csv"
        print(f"file was loaded from {data_path}")
        df_raw = pd.read_csv(data_path)

        cols = list(df_raw.columns)
        # print(f"[ds] cols: {cols}")
        cols.remove(self.target)
        cols.remove("dt")
        df_raw = df_raw[["dt"] + cols + [self.target]]
        df_raw["dt"] = pd.to_datetime(df_raw["dt"])

        # print(df_raw)
        val_start = int(len(df_raw) * 0.7)
        test_start = int(len(df_raw) * (0.7 + 0.2))

        start_indices = [0, val_start, test_start]
        end_indices = [val_start, test_start, len(df_raw)]
        start_idx = start_indices[self.set_type]
        end_idx = end_indices[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        print(f"[ds] df_data: {df_data.shape}\n{df_data}")
        if self.scale:
            train_data = df_data[start_indices[0] : end_indices[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["dt"]][start_idx:end_idx]
        df_stamp["dt"] = pd.to_datetime(df_stamp.dt)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.dt.apply(lambda row: row.month, 1)  # type: ignore
            df_stamp["day"] = df_stamp.dt.apply(lambda row: row.day, 1)  # type: ignore
            df_stamp["weekday"] = df_stamp.dt.apply(lambda row: row.weekday(), 1)  # type: ignore
            df_stamp["hour"] = df_stamp.dt.apply(lambda row: row.hour, 1)  # type: ignore
            data_stamp = df_stamp.drop(columns=["dt"]).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["dt"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[start_idx:end_idx]  # type: ignore
        self.data_y = data[start_idx:end_idx]  # type: ignore
        self.data_stamp = data_stamp

        print(f"data_x shape: {self.data_x.shape}")
        print(f"data_y shape: {self.data_y.shape}")
        print(f"data_stamp shape: {self.data_stamp.shape}")

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        batch = dict(
            seq_x=torch.tensor(seq_x, dtype=torch.float32),
            seq_y=torch.tensor(seq_y, dtype=torch.float32),
            seq_xt=torch.tensor(seq_x_mark, dtype=torch.float32),
            seq_yt=torch.tensor(seq_y_mark, dtype=torch.float32),
        )

        return batch
