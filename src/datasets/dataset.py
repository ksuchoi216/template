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


class Custom_Dataset(Dataset):
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
        self.stage = stage
        type_map = {"train": 0, "val": 1, "test": 2, "pred": 3}
        self.set_type = type_map[stage]

        self.features = cfg.features
        self.target = cfg.target
        self.scale = cfg.scale
        self.timeenc = cfg.timeenc
        self.freq = cfg.freq

        self.test_run = cfg.test_run
        self.__read_data__(cfg)

    def __read_data__(self, cfg):
        # * load data
        data_dir = f"{cfg.data_dir}/{cfg.datafolder_name}"
        data_path = f"{data_dir}/{cfg.file_name}.csv"
        print(f"file was loaded from {data_path}")
        df_raw = pd.read_csv(data_path)
        if self.test_run:
            df_raw = df_raw[: cfg.run_len]

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove("dt")
        df_raw = df_raw[["dt"] + cols + [self.target]]
        df_raw["dt"] = pd.to_datetime(df_raw["dt"])

        # * split
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

        # * scaling
        self.scaler = StandardScaler()
        train_data = df_data[start_indices[0] : end_indices[0]]
        self.scaler.fit(train_data.values)
        data = self.scaler.transform(df_data.values)

        # * feature engineering
        df_stamp = df_raw[["dt"]][start_idx:end_idx]
        df_stamp["dt"] = pd.to_datetime(df_stamp.dt)
        df_stamp["month"] = df_stamp.dt.apply(lambda row: row.month, 1)  # type: ignore
        df_stamp["day"] = df_stamp.dt.apply(lambda row: row.day, 1)  # type: ignore
        df_stamp["weekday"] = df_stamp.dt.apply(lambda row: row.weekday(), 1)  # type: ignore
        df_stamp["hour"] = df_stamp.dt.apply(lambda row: row.hour, 1)  # type: ignore
        data_stamp = df_stamp.drop(columns=["dt"]).values

        # * save scaler
        if self.stage == "train" and not self.test_run:
            joblib.dump(
                self.scaler,
                f"{cfg.data_dir}/{cfg.datafolder_name}/scaler_{cfg.file_name}.pkl",
            )

        # * slice data
        self.data_stamp = data_stamp  # type: ignore
        self.data_x = data[start_idx:end_idx]  # type: ignore
        self.data_y = data[start_idx:end_idx]  # type: ignore
        print(f"data_stamp shape: {self.data_stamp.shape}")
        print(f"data_x shape: {self.data_x.shape}")
        print(f"data_y shape: {self.data_y.shape}")

        if self.test_run:
            self.past_indices = [3, 6]
        else:
            self.past_indices = cfg.past_indices

        self.past_len = len(self.past_indices)
        self.oldest_past = np.max(self.past_indices)

        self.length = len(self.data_x) - self.seq_len - self.pred_len - self.oldest_past
        print(f"length: {self.length}")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        s_begin = index + self.oldest_past
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        past_x = np.vstack([self.data_x[s_begin - i, :] for i in self.past_indices])
        past_xt = np.vstack(
            [self.data_stamp[s_begin - i, :] for i in self.past_indices]
        )

        batch = dict(
            seq_x=torch.tensor(seq_x, dtype=torch.float32),
            seq_y=torch.tensor(seq_y, dtype=torch.float32),
            seq_xt=torch.tensor(seq_x_mark, dtype=torch.float32),
            seq_yt=torch.tensor(seq_y_mark, dtype=torch.float32),
            past_x=torch.tensor(past_x, dtype=torch.float32),
            past_xt=torch.tensor(past_xt, dtype=torch.float32),
        )

        return batch


if __name__ == "__main__":
    print("dataset test")

    data_x = np.random.rand(30, 321)
    data_y = np.random.rand(30, 321)
    data_stamp = np.random.rand(30, 4)

    past_indices = [1, 3]
    past_len = len(past_indices)
    oldest_past = np.max(past_indices)
    print(f"oldest_past: {oldest_past}")

    seq_len = 5
    pred_len = 2
    label_len = 2

    length = len(data_x) - seq_len - pred_len - oldest_past
    for i in range(0, length):
        s_begin = i + oldest_past
        s_end = s_begin + seq_len
        r_begin = s_end - label_len
        r_end = r_begin + label_len + pred_len
        past = s_begin - oldest_past

        print(
            f"s_begin: {s_begin}, s_end: {s_end}, r_begin: {r_begin}, r_end: {r_end} past: {past}"
        )
