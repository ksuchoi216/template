import sys
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch

from src.datasets.feat_time import TimeCovariates


class TSDataset(Dataset):
    def __init__(
        self,
        split_index,
        dt,
        df,
        mode,
        seq_len,
        label_len,
        pred_len,
        target_col,
        past_len,
        past_interval,
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        timecovariates = TimeCovariates(dt)
        dft = timecovariates.get_covariates()[
            split_index[mode][0] : split_index[mode][1]
        ]
        df = df[split_index[mode][0] : split_index[mode][1]]

        self.past_len = past_len * past_interval
        self.interval_index = np.array(
            [i * past_interval for i in range(1, past_len + 1)]
        )

        if target_col is not None:
            self.target_idx = df.columns.get_loc(target_col)
        else:
            self.target_idx = None

        print(f"[{mode}] target idx: {self.target_idx}")

        self.data = df.values
        self.data_time = dft.values

        print(
            f"[{mode}] data shape: {self.data.shape}, data_time shape: {self.data_time.shape}"
        )

    def __len__(self):
        return len(self.data) - self.past_len - self.seq_len - self.pred_len

    def __getitem__(self, idx):
        x_start = idx + self.past_len
        x_end = x_start + self.seq_len
        y_start = x_end - self.label_len
        y_end = x_end + self.pred_len

        # print(f'x_start: {x_start}, x_end: {x_end}, y_start: {y_start}, y_end: {y_end}')
        seq_x = self.data[x_start:x_end, :]
        seq_y = self.data[y_start:y_end, :]
        seq_xt = self.data_time[x_start:x_end, :]
        seq_yt = self.data_time[y_start:y_end, :]
        past_x = np.vstack([self.data[x_start - i, :] for i in self.interval_index])
        past_xt = np.vstack(
            [self.data_time[x_start - i, :] for i in self.interval_index]
        )

        seq_x = torch.tensor(seq_x, dtype=torch.float32)
        seq_y = torch.tensor(seq_y, dtype=torch.float32)
        seq_xt = torch.tensor(seq_xt, dtype=torch.float32)
        seq_yt = torch.tensor(seq_yt, dtype=torch.float32)
        past_x = torch.tensor(past_x, dtype=torch.float32)
        past_xt = torch.tensor(past_xt, dtype=torch.float32)

        batch_dic = dict(
            seq_x=seq_x,
            seq_y=seq_y,
            seq_xt=seq_xt,
            seq_yt=seq_yt,
            past_x=past_x,
            past_xt=past_xt,
            target_idx=self.target_idx,
        )

        return batch_dic
