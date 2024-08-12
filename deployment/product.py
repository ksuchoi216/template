import os
import warnings
from pprint import pprint

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from easydict import EasyDict as edict
from einops import rearrange
from joblib.memory import re

warnings.filterwarnings("ignore")


def load_config(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = edict(yaml.safe_load(f))
    return cfg


def add_time_feat(df, time_col_num=0, ff=60):
    x = df.values
    ts = x[:, time_col_num]
    cos_time = np.cos(2 * np.pi * ff * ts)
    sin_time = np.sin(2 * np.pi * ff * ts)
    feat_x = np.column_stack([x, cos_time, sin_time])

    df = pd.DataFrame(feat_x, columns=df.columns.tolist() + ["cos_time", "sin_time"])
    return df


class DataManager:
    def __init__(self, cfg):
        self.data_foldername = cfg.data_foldername
        self.base_dir = cfg.base_dir
        self.seq_len = cfg.seq_len
        self.seq_len_resampled = cfg.seq_len_resampled
        self.fundamental_freq = cfg.fundamental_freq

    def load_data(self, data_num) -> pd.DataFrame:
        path = f"{self.base_dir}/{self.data_foldername}/{data_num}.csv"

        df = pd.read_csv(path)
        # print(f"df shape: {df.shape}, {len(df)}")
        assert len(df) != 0, f"Failed to load data from {path}"

        return df

    def resample_df(self, df, resample_freq="500us"):
        df = df.copy()
        org_length = len(df)
        today = pd.to_datetime("today").date()
        df["dt"] = pd.to_datetime(df["t"], origin=today, unit="s")
        df = df.set_index("dt")
        resampled_df = df.resample(resample_freq).mean().interpolate()
        resampled_df = resampled_df.reset_index()
        ratio = org_length / len(resampled_df)
        if len(resampled_df) > self.seq_len_resampled:
            resampled_df = resampled_df.iloc[: self.seq_len_resampled]

        return resampled_df, ratio

    def forward(self, filename, start_idx=1000):
        def slice_df(df, start_idx):
            return df.iloc[start_idx : start_idx + self.seq_len]

        df = self.load_data(filename)
        df = slice_df(df, start_idx)
        df = add_time_feat(df, time_col_num=0, ff=self.fundamental_freq)
        resampled_df, ratio = self.resample_df(df, resample_freq="500us")
        resampled_df = resampled_df.drop(columns=["t", "dt"], inplace=False)
        df = df.drop(columns=["t"], inplace=False)

        x, resampled_x = df.values, resampled_df.values

        x = torch.tensor(x, dtype=torch.float64)
        resampled_x = torch.tensor(resampled_x, dtype=torch.float64)

        return x, resampled_x


class Model:
    def __init__(self, cfg):
        self.model_paths, self.scaler_paths = self.get_paths(cfg)
        self.__load_model__()

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def get_paths(self, cfg):
        base_dir = cfg.base_dir
        model_paths = {}
        scaler_paths = {}
        for i in range(1, 4):
            model_name = f"model{i}"
            model_path = f"{base_dir}/models/{model_name}.pt"
            scaler_path = f"{base_dir}/scalers/{model_name}_scaler.pkl"
            model_paths[model_name] = model_path
            scaler_paths[model_name] = scaler_path

        return model_paths, scaler_paths

    def __load_model__(self):
        def load_model_scaler(model_path, scaler_path):
            model = torch.jit.load(model_path)
            model.eval()
            scaler = joblib.load(scaler_path)
            return model, scaler

        for model_name, model_path in self.model_paths.items():
            scaler_path = self.scaler_paths[model_name]
            model, scaler = load_model_scaler(model_path, scaler_path)
            setattr(self, model_name, model)
            setattr(self, f"{model_name}_scaler", scaler)

    def apply_scaler(self, x, scaler):
        # convert Torch tensor to numpy array
        x = x.numpy()
        seq_len, dim = x.shape
        x = rearrange(x, "s d -> 1 (s d)")
        x = scaler.transform(x)
        x = rearrange(x, "1 (s d) -> 1 s d", s=seq_len, d=dim)
        x = torch.tensor(x, dtype=torch.float32)
        return x

    def predict(self, x, resampled_x):

        x0 = self.apply_scaler(x, self.model1_scaler)  # type: ignore
        x1 = self.apply_scaler(resampled_x, self.model2_scaler)  # type: ignore
        x2 = self.apply_scaler(resampled_x, self.model3_scaler)  # type: ignore

        out0 = self.softmax(self.model1(x0))  # type: ignore
        y1 = int(torch.argmax(out0, dim=1))
        out1 = self.sigmoid(self.model2(x1))  # type: ignore
        y2 = 1 if out1 >= 0.5 else 0
        out2 = self.sigmoid(self.model3(x2))  # type: ignore
        y3 = 1 if out2 >= 0.5 else 0

        return [y1, y2, y3]


def compute_mode(arr):
    y1 = arr[0]
    y2 = arr[1]
    y3 = arr[2]

    if y3 == 1:
        result_num = 4
    elif y2 == 1:
        result_num = 4
    else:
        result_num = y1 + 1

    return result_num


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(description="prediction by model")
    parser.add_argument(
        "--config_dir",
        type=str,
        default="./product/configs",
        help="base directory",
    )
    parser.add_argument(
        "--filename",
        type=int,
        default=0,
        help="filename",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=1000,
        help="data start point for slicing data",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    config_dir = args.config_dir
    # filename = args.filename
    # start_idx = args.start_idx

    cfg = load_config(f"{config_dir}/config.yaml")
    datamanager = DataManager(cfg)
    model = Model(cfg)

    start_coords = np.arange(32000, 42000, 2000)
    selected_files = [0, 1, 2, 4, 5, 7, 9]

    total_result_arr = []
    for i, filename in enumerate(selected_files):
        for j, start_idx in enumerate(start_coords):
            x, resampled_x = datamanager.forward(filename, start_idx)
            result_arr = model.predict(x, resampled_x)

            total_result_arr.append(result_arr)

    total_result_arr = np.array(total_result_arr)

    y_data = np.load("./test/y_data.npy")
    # print(total_result_arr)
    # print(y_data)

    y_preds = []
    for result_arr in total_result_arr:
        result_num = compute_mode(result_arr)
        y_preds.append(result_num)

    print(f"mode: {result_num}")

    # y_trues = []
    # for result_arr in y_data:
    #     result_num = compute_mode(result_arr)
    #     y_trues.append(result_num)

    # final_result = []
    # for y_pred, y_true in zip(y_preds, y_trues):
    #     if y_pred == y_true:
    #         result = True
    #     else:
    #         result = False
    #     # print(result)
    #     final_result.append([result, y_pred, y_true])

    # final_result = np.array(final_result)
    # print(f"[result, y_pred, y_true]")
    # print(final_result)
