import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
print(f"parent_dir: {parent_dir}")
sys.path.insert(0, parent_dir)


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        # print(f"x shape in moving_avg: {x.shape}")
        x = self.avg(rearrange(x, "b s d -> b d s"))
        # print(f"x shape after avg: {x.shape}")
        x = rearrange(x, "b d s -> b s d")
        return x


class time_decomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        # print(f"moving_mean: {moving_mean.shape}")

        res = x - moving_mean
        return res, moving_mean


class freq_decomp(nn.Module):
    def __init__(self):
        super().__init__()
        # self.slice_size = slice_size

    def forward(self, x):
        freq = torch.fft.rfft(x, dim=1)
        freq = torch.abs(freq)
        # freq = freq[:, : self.slice_size, :]

        return freq


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
        # ! ===============================================================
        self.printout = cfg.printout
        print(f"printout: {self.printout}")

        self.seq_len = cfg.seq_len
        self.pred_len = cfg.pred_len
        self.label_len = cfg.label_len
        self.features = cfg.features
        self.input_dim = cfg.input_dim
        self.time_dim = cfg.time_dim
        # self.past_len = cfg.past_len
        print(f"seq_len: {self.seq_len}")
        print(f"label_len: {self.label_len}, features: {self.features}")

        self.add_timefeat = cfg.add_timefeat
        if self.add_timefeat:
            self.input_dim += self.time_dim

        print(f"input_dim: {self.input_dim}, time_dim: {self.time_dim}")
        # ! ===============================================================

        self.lstm_num_layers = cfg.lstm_num_layers
        self.lstm_hidden_size = cfg.lstm_hidden_size
        self.num_direction = 2 if cfg.bidirectional else 1

        self.has_cnn = cfg.has_cnn
        self.has_time_decomp = cfg.has_time_decomp
        self.has_freq_decomp = cfg.has_freq_decomp
        print(f"[m] has_time_decomp: {self.has_time_decomp}") if self.printout else None
        print(f"[m] has_freq_decomp: {self.has_freq_decomp}") if self.printout else None

        # sys.exit()
        if self.has_time_decomp:
            self.time_decomp = time_decomp(cfg.kernel_size)
            self.trend_layers = nn.ModuleList()
            self.seasonal_layers = nn.ModuleList()
            self.time_hidden_size = cfg.time_hidden_size

            for i in range(self.input_dim):
                self.trend_layers.append(nn.Linear(self.seq_len, self.time_hidden_size))
                self.seasonal_layers.append(
                    nn.Linear(self.seq_len, self.time_hidden_size)
                )

            trend_out_in = self.input_dim * self.time_hidden_size
            self.trend_out_layer = nn.Linear(trend_out_in, self.pred_len)
            seasonal_out_in = self.input_dim * self.time_hidden_size
            self.seasonal_out_layer = nn.Linear(seasonal_out_in, self.pred_len)

        if self.has_freq_decomp:
            self.freq_decomp = freq_decomp()
            self.freq_layers = nn.ModuleList()

            freq_input_size = (
                self.seq_len // 2 + 1 if self.seq_len % 2 == 0 else self.seq_len // 2
            )
            print(f"freq_input_size: {freq_input_size}")

            self.freq_hidden_size = cfg.freq_hidden_size
            for i in range(self.input_dim):
                self.freq_layers.append(
                    nn.Linear(freq_input_size, self.freq_hidden_size)
                )

            freq_out_in = self.input_dim * self.freq_hidden_size
            print(f"freq_out_in: {freq_out_in}")

            self.freq_out_layer = nn.Linear(freq_out_in, self.pred_len)

        print(f"[m] has_cnn: {self.has_cnn}") if self.printout else None
        if self.has_cnn:
            self.init_conv1d = TimeDistributed(
                nn.Conv1d(
                    cfg.input_dim,
                    cfg.cnn_out_channel,
                    kernel_size=cfg.cnn_kernel_size,
                    padding="same",
                )
            )

            layers = []
            for _ in range(cfg.cnn_n_blocks):
                layers.append(
                    TimeDistributed(
                        nn.Conv1d(
                            cfg.cnn_out_channel,
                            cfg.cnn_out_channel,
                            kernel_size=cfg.cnn_kernel_size,
                            padding="same",
                        ),
                    )
                )
                layers.append(nn.ReLU())
                layers.append(TimeDistributed(nn.Dropout(cfg.dropout)))
                layers.append(
                    TimeDistributed(nn.MaxPool1d(kernel_size=cfg.maxpool_size))
                )
            # print(f"[model] layers: {layers}")
            self.conv1d = nn.ModuleList(layers)
            self.input_dim = cfg.cnn_out_channel

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.lstm_hidden_size,
            num_layers=cfg.lstm_num_layers,
            batch_first=True,
            bidirectional=cfg.bidirectional,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.flatten = nn.Flatten()

        if self.has_cnn:
            __seq_len = self.seq_len // (cfg.maxpool_size**cfg.cnn_n_blocks)
        else:
            __seq_len = self.seq_len

        fc_input_size = __seq_len * self.lstm_hidden_size * self.num_direction
        print(f"[model] fc_input_size: {fc_input_size}") if self.printout else None

        self.fc = nn.Linear(fc_input_size, cfg.fc_hidden_size)
        self.relu = nn.ReLU()
        self.out_layer = nn.Linear(cfg.fc_hidden_size, self.pred_len)

        if self.has_freq_decomp and self.has_time_decomp:
            multiple = 4
        elif self.has_time_decomp:
            multiple = 3
        elif self.has_freq_decomp:
            multiple = 2

        if self.has_time_decomp or self.has_freq_decomp:
            assemble_in = multiple * self.pred_len
            self.assemble = nn.Linear(assemble_in, self.pred_len, bias=False)

        # sys.exit()

    # def forward(self, seq_x, seq_xt, past_x, past_xt, dec_input, seq_yt):
    def forward(self, batch_x):
        # ! ===============================================================
        seq_x = batch_x["seq_x"]
        seq_xt = batch_x["seq_xt"]
        # past_x = batch_x["past_x"]
        # x = x[:, :, -1].reshape(-1, self.seq_len, self.input_dim)
        # x_enc = batch["seq_x"]
        # x_mark_enc = batch["seq_xt"]
        # x_dec = batch["dec_input"]
        # x_mark_dec = batch["seq_yt"]
        # ! ===============================================================

        x = seq_x
        if self.add_timefeat:
            x = torch.cat([seq_x, seq_xt], dim=-1)

        if self.has_time_decomp:
            seasonal, trend = self.time_decomp(x)
            print(f"ss shape: {seasonal.shape}") if self.printout else None
            print(f"tr shape: {trend.shape}") if self.printout else None

            seasonal = rearrange(seasonal, "b s d -> b d s")
            trend = rearrange(trend, "b s d -> b d s")

            seasonal_out = torch.zeros(
                seasonal.shape[0],
                seasonal.shape[1],
                self.time_hidden_size,
                dtype=seasonal.dtype,
            ).to(seasonal.device)

            trend_out = torch.zeros(
                trend.shape[0], trend.shape[1], self.time_hidden_size, dtype=trend.dtype
            ).to(trend.device)

            for i in range(self.input_dim):
                seasonal_out[:, i, :] = self.seasonal_layers[i](seasonal[:, i, :])
                trend_out[:, i, :] = self.trend_layers[i](trend[:, i, :])

            seasonal_out = self.flatten(self.dropout(seasonal_out))
            trend_out = self.flatten(self.dropout(trend_out))

            seasonal_out = self.seasonal_out_layer(seasonal_out)
            trend_out = self.trend_out_layer(trend_out)
            print(f"ss_out shape: {seasonal_out.shape}") if self.printout else None
            print(f"tr_out shape: {trend_out.shape}") if self.printout else None

        if self.has_freq_decomp:
            freq = self.freq_decomp(x)
            freq = rearrange(freq, "b s d -> b d s")
            print(f"freq org shape: {freq.shape}") if self.printout else None
            freq_out = torch.zeros(
                [freq.shape[0], freq.shape[1], self.freq_hidden_size], dtype=freq.dtype
            ).to(freq.device)
            print(f"freq_out shape: {freq_out.shape}") if self.printout else None
            for i in range(self.input_dim):
                print(f"in: {freq[:, i, :].shape}") if self.printout else None
                freq_out[:, i, :] = self.freq_layers[i](freq[:, i, :])
                print(f"out: {freq_out[:, i, :].shape}") if self.printout else None
            print(f"freq_out shape: {freq_out.shape}") if self.printout else None

            freq_out = self.flatten(self.dropout(freq_out))
            freq_out = self.freq_out_layer(freq_out)
            print(f"freq_out shape: {freq_out.shape}") if self.printout else None

        if self.has_cnn:
            x = self.init_conv1d(x)
            print(f"x shape after init_conv1d: {x.shape}") if self.printout else None
            for layer in self.conv1d:
                x = layer(x)
                print(f"x in cnn: {x.shape}") if self.printout else None

        print(f"x shape before lstm: {x.shape}") if self.printout else None
        h0 = torch.zeros(
            self.lstm_num_layers * self.num_direction, x.size(0), self.lstm_hidden_size
        ).to(x.device)
        c0 = torch.zeros(
            self.lstm_num_layers * self.num_direction, x.size(0), self.lstm_hidden_size
        ).to(x.device)
        x, (h_out, _) = self.lstm(x, (h0, c0))

        # x = x + seq_x
        print(f"x shape after lstm: {x.shape}") if self.printout else None
        x = self.flatten(self.dropout(x))
        print(f"x shape after flatten: {x.shape}") if self.printout else None
        x = self.relu(self.fc(x))
        print(f"x shape after fc: {x.shape}") if self.printout else None
        x = self.out_layer(x)
        print(f"x shape after out_layer: {x.shape}") if self.printout else None

        if self.has_time_decomp and self.has_freq_decomp:
            x = torch.cat([x, seasonal_out, trend_out, freq_out], dim=1)
            x = self.assemble(x)
            print(f"both x: {x.shape}") if self.printout else None
        elif self.has_time_decomp:
            x = torch.cat([x, seasonal_out, trend_out], dim=1)
            x = self.assemble(x)
            print(f"time x: {x.shape}") if self.printout else None
        elif self.has_freq_decomp:
            x = torch.cat([x, freq_out], dim=1)
            print(f"x shape: {x.shape}") if self.printout else None
            x = self.assemble(x)
            print(f"freq x: {x.shape}") if self.printout else None

        # if self.add_past:

        print(f"x out: {x.shape}") if self.printout else None

        return x


if __name__ == "__main__":
    import os

    import yaml
    from easydict import EasyDict as edict

    datamodule_name = "pv_dacon"
    runner_name = "lstm"
    print(f">>>>>>>>>> [{runner_name}][{datamodule_name}] <<<<<<<<<<<<<<<")

    cwd = os.getcwd()
    cfg_model_path = f"{cwd}/configs/runner/{runner_name}.yaml"
    with open(cfg_model_path) as f:
        cfgm = edict(yaml.safe_load(f))
        model_target = cfgm.model._target_  # type: ignore
        cfgm = cfgm.model.cfg  # type: ignore

    cfg_data_path = f"{cwd}/configs/datamodule/{datamodule_name}.yaml"
    with open(cfg_data_path) as f:
        cfgdm = edict(yaml.safe_load(f))
        cfgdm = cfgdm.cfg  # type: ignore

    print(f"model_target: {model_target}")

    cfgm.input_dim = cfgdm.input_dim
    cfgm.seq_len = cfgdm.seq_len
    cfgm.pred_len = cfgdm.pred_len
    cfgm.label_len = cfgdm.label_len
    # cfgm.past_len = cfgdm.past_len
    cfgm.features = cfgdm.features
    cfgm.batch_size = cfgdm.batch_size
    cfgm.time_dim = cfgdm.time_dim

    print(f"time_dim: {cfgm.time_dim}")
    seq_y = torch.randn(cfgm.batch_size, cfgm.pred_len, cfgm.input_dim)
    dec_input = torch.zeros_like(seq_y[:, -cfgm.label_len :, :]).float()
    dec_input = torch.cat([seq_y[:, : cfgm.label_len, :], dec_input], dim=1).float()

    batch = dict(
        seq_x=torch.randn(cfgm.batch_size, cfgm.seq_len, cfgm.input_dim),
        seq_xt=torch.randn(cfgm.batch_size, cfgm.seq_len, cfgm.time_dim),
        seq_yt=torch.randn(
            cfgm.batch_size, cfgm.pred_len + cfgm.label_len, cfgm.time_dim
        ),
        # past_x=torch.randn(cfgm.batch_size, past_len, cfgm.past_len),
        # past_xt=torch.randn(cfgm.batch_size, past_len, cfgm.time_dim),
        dec_input=dec_input,
    )

    print(f"seq_y: {seq_y.shape}")
    for k, v in batch.items():
        print(f"{k}: {v.shape}")

    cfgm.printout = True
    print(f"{cfgm.features} Model ============================")
    model = Model(cfgm)
    y_pred = model(batch)
    print(f"y_pred: {y_pred.shape}")

    if ismlflow := False:
        print(f'{"="*20} Model ============================')
        import mlflow.pytorch

        # Set the tracking URI
        mlflow.set_tracking_uri("http://192.168.1.100:25001")

        # Initialize the MLflow client
        client = mlflow.tracking.MlflowClient()

        # Download the model artifact
        model_uri = "runs:/b38f91b8ca0e494493f729055c3ea37a/model"

        # Load the model using PyTorch
        loaded_model = mlflow.pytorch.load_model(model_uri)

        # Example: Use the loaded model for inference
        output = loaded_model(batch)
        print(output)
