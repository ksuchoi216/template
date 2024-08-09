import sys

import torch
import torch.nn as nn
from einops import rearrange


class TimeDistributed(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        x = rearrange(x, "b s d -> b d s")
        x = self.module(x)
        x = rearrange(x, "b d s -> b s d")
        return x


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.seq_len = cfg.seq_len
        self.pred_len = cfg.pred_len
        self.label_len = cfg.label_len
        self.printout = cfg.printout
        self.features = cfg.features
        print(f"seq_len: {self.seq_len}, pred_len: {self.pred_len}")
        print(f"label_len: {self.label_len}, features: {self.features}")

        # * lstm
        self.lstm_num_layers = cfg.lstm_num_layers
        self.lstm_hidden_size = cfg.lstm_hidden_size
        self.num_direction = 2 if cfg.bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=cfg.input_dim,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            batch_first=True,
            bidirectional=cfg.bidirectional,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.flatten = nn.Flatten()
        fc_input_size = self.seq_len * self.lstm_hidden_size * self.num_direction
        print(f"fc_input_size: {fc_input_size}") if self.printout else None
        self.fc = nn.Linear(fc_input_size, cfg.fc_hidden_size)

        self.relu = nn.ReLU()
        self.out_layer = nn.Linear(cfg.fc_hidden_size, self.pred_len)

    def forward(self, batch):
        x = batch["seq_x"]
        print(f"x shape execute: {x.shape}") if self.printout else None

        h0 = torch.zeros(
            self.lstm_num_layers * self.num_direction, x.size(0), self.lstm_hidden_size
        ).to(x.device)
        c0 = torch.zeros(
            self.lstm_num_layers * self.num_direction, x.size(0), self.lstm_hidden_size
        ).to(x.device)
        x, (h_out, _) = self.lstm(x, (h0, c0))
        print(f"x shape after lstm: {x.shape}") if self.printout else None
        x = self.flatten(self.dropout(x))
        print(f"x shape after flatten: {x.shape}") if self.printout else None
        x = self.relu(self.fc(x))
        print(f"x shape after fc: {x.shape}") if self.printout else None
        x = self.out_layer(x)
        print(f"x shape after out_layer: {x.shape}") if self.printout else None
        return x


if __name__ == "__main__":
    import os

    import yaml
    from easydict import EasyDict as edict

    datamodule_name = "base_electricity"
    model_name = "lstm"
    print(f">>>>>>>>>> [{model_name}][{datamodule_name}] <<<<<<<<<<<<<<<")

    cwd = os.getcwd()
    cfg_model_path = f"{cwd}/configs/runner/{model_name}.yaml"
    with open(cfg_model_path) as f:
        cfgm = edict(yaml.safe_load(f))
        model_target = cfgm.model._target_  # type: ignore
        cfgm = cfgm.model.cfg  # type: ignore

    cfg_data_path = f"{cwd}/configs/datamodule/{datamodule_name}.yaml"
    with open(cfg_data_path) as f:
        cfgdm = edict(yaml.safe_load(f))
        cfgdm = cfgdm.cfg  # type: ignore

    print(f"model_target: {model_target}")
    assert cfgm.model_name == model_name, f"{cfgm.model_name} != {model_name}"

    cfgm.input_dim = cfgdm.input_dim
    cfgm.seq_len = cfgdm.seq_len
    cfgm.pred_len = cfgdm.pred_len
    cfgm.label_len = cfgdm.label_len
    cfgm.features = cfgdm.features
    cfgm.batch_size = cfgdm.batch_size

    batch = dict(
        seq_x=torch.randn(cfgm.batch_size, cfgm.seq_len, cfgm.input_dim),
        seq_y=torch.randn(cfgm.batch_size, cfgm.pred_len, cfgm.input_dim),
        seq_xt=torch.randn(cfgm.batch_size, cfgm.seq_len, 4),
        seq_yt=torch.randn(cfgm.batch_size, cfgm.pred_len, 4),
    )

    for k, v in batch.items():
        print(f"{k}: {v.shape}")

    cfgm.printout = True
    model = Model(cfgm)
    y_pred = model(batch)

    print(f'seq_y: {batch["seq_y"].shape}')
    print(f"out: {y_pred.shape}")
    t_dim = -1 if model.features == "MS" else 0
    print(f"t_dim: {t_dim}")

    seq_y = batch["seq_y"]
    print(f"seq_y: {seq_y.shape}")

    # seq_y[..., t_dim] = y_pred

    # b, s, d = seq_y.shape
    # seq_y = rearrange(seq_y, "b s d -> (b s) d")
    # print(f"seq_y: {seq_y.shape}")

    # seq_y = rearrange(seq_y, "(b s) d -> b s d", b=b, s=s)
