import torch.nn as nn
import torch
from src.models.layers.Embed import DataEmbedding_wo_pos
from einops import rearrange
import sys


class Model(nn.Module):
    def __init__(
        self,
        input_size,
        input_size_2nd,
        input_size_3rd,
        seq_len,
        pred_len,
        label_len,
        past_len,
        hidden_size,
        n_layer_lstm,
        bidirectional,
        embed="timeF",
        freq="t",
        dropout=0.05,
        cnn_out_channel=128,
        kernel_size=3,
        n_layer_cnn=2,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.past_len = past_len
        print("=" * 25)
        print(f"|| seq_len: {seq_len:10} ||")
        print(f"|| pred_len: {pred_len:9} ||")
        print(f"|| label_len: {label_len:8} ||")
        print("=" * 25)

        self.input_size = input_size
        self.input_size_2nd = input_size_2nd
        self.input_size_3rd = input_size_3rd
        self.hidden_size = hidden_size
        self.n_layer_lstm = n_layer_lstm
        self.embed = embed
        self.freq = freq
        self.dropout = dropout
        self.n_layer_cnn = n_layer_cnn

        if bidirectional:
            self.num_direction = 2
        else:
            self.num_direction = 1

        self.short_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=n_layer_lstm,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.short_conv1d = nn.Conv1d(
            in_channels=input_size_2nd,
            out_channels=cnn_out_channel,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
        )

        self.short_convlist = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        cnn_out_channel,
                        cnn_out_channel,
                        kernel_size,
                        stride=1,
                        padding=kernel_size // 2,
                    ),
                )
                for _ in range(n_layer_cnn)
            ]
        )

        self.short_normlist = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(cnn_out_channel),
                    nn.ReLU(),
                )
                for _ in range(n_layer_cnn)
            ]
        )
        self.short_mapping = nn.Linear(self.seq_len * 2, self.pred_len)
        self.long_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=n_layer_lstm,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.long_conv1d = nn.Conv1d(
            in_channels=input_size_2nd,
            out_channels=cnn_out_channel,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
        )

        self.long_convlist = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        cnn_out_channel,
                        cnn_out_channel,
                        kernel_size,
                        stride=1,
                        padding=kernel_size // 2,
                    ),
                )
                for _ in range(n_layer_cnn)
            ]
        )

        self.long_normlist = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(cnn_out_channel),
                    nn.ReLU(),
                )
                for _ in range(n_layer_cnn)
            ]
        )
        self.long_mapping = nn.Linear(self.past_len * 2, self.pred_len)

        self.fc = nn.Linear(self.pred_len, self.pred_len * 2)
        self.fc2 = nn.Linear(self.pred_len * 2, self.pred_len)

    def forward(self, seq_x, seq_xt, past_x, past_xt, dec_input, seq_yt):
        # * power lstm + weather cnn
        short_power_x = seq_x[:, :, :8]
        # print(f"short_power_x: {short_power_x.shape}, seq_xt: {seq_xt.shape}")
        short_power_x = torch.cat([short_power_x, seq_xt], dim=-1)
        # print(f"short_power_x: {short_power_x.shape}")
        short_weather_x = seq_x[:, :, 8:]

        # print(
        # f"short_power_x: {short_power_x.shape}, weather_x: {short_weather_x.shape}"
        # )
        h0 = torch.zeros(
            self.n_layer_lstm * self.num_direction,
            short_power_x.size(0),
            self.hidden_size,
        ).to(short_power_x.device)
        c0 = torch.zeros(
            self.n_layer_lstm * self.num_direction,
            short_power_x.size(0),
            self.hidden_size,
        ).to(short_power_x.device)
        # print(f"x shape: {x.shape}, h0 shape: {h0.shape}, c0 shape: {c0.shape}")
        short_power_x, (h_out, _) = self.short_lstm(short_power_x, (h0, c0))
        short_power_x = torch.squeeze(short_power_x)
        # print(f"short_power_x shape after lstm: {short_power_x.shape}")

        short_weather_x = rearrange(short_weather_x, "b s d -> b d s")
        short_weather_x = self.short_conv1d(short_weather_x)
        for i in range(self.n_layer_cnn):
            # print(f"short_weather_x shape before cnn: {i} -> {short_weather_x.shape}")
            short_weather_x = self.short_convlist[i](short_weather_x)
            short_weather_x = rearrange(short_weather_x, "b d s -> b s d")
            short_weather_x = self.short_normlist[i](short_weather_x)
            short_weather_x = rearrange(short_weather_x, "b s d -> b d s")
            # print(f"short_weather_x shape after cnn: {i} -> {short_weather_x.shape}")
        short_weather_x = rearrange(short_weather_x, "b d s -> b s d")
        short_weather_x = torch.squeeze(short_weather_x)
        # print(f"short_weather_x shape after cnn: {short_weather_x.shape}")

        short_x = torch.cat([short_power_x, short_weather_x], dim=-1)
        short_x = self.short_mapping(short_x)
        # print(f"short_x shape: {short_x.shape}")

        # * long power lstm + weather cnn
        long_power_x = past_x[:, :, :8]
        long_power_x = torch.cat([long_power_x, past_xt], dim=-1)
        long_weather_x = past_x[:, :, 8:]

        # print(f"long_power_x: {long_power_x.shape}, weather_x: {long_weather_x.shape}")
        h0 = torch.zeros(
            self.n_layer_lstm * self.num_direction,
            long_power_x.size(0),
            self.hidden_size,
        ).to(long_power_x.device)
        c0 = torch.zeros(
            self.n_layer_lstm * self.num_direction,
            long_power_x.size(0),
            self.hidden_size,
        ).to(long_power_x.device)
        # print(f"x shape: {x.shape}, h0 shape: {h0.shape}, c0 shape: {c0.shape}")
        long_power_x, (h_out, _) = self.long_lstm(long_power_x, (h0, c0))
        long_power_x = torch.squeeze(long_power_x)
        # print(f"long_power_x shape after lstm: {long_power_x.shape}")

        long_weather_x = rearrange(long_weather_x, "b s d -> b d s")
        long_weather_x = self.long_conv1d(long_weather_x)
        for i in range(self.n_layer_cnn):
            # print(f"long_weather_x shape before cnn: {i} -> {long_weather_x.shape}")
            long_weather_x = self.long_convlist[i](long_weather_x)
            long_weather_x = rearrange(long_weather_x, "b d s -> b s d")
            long_weather_x = self.long_normlist[i](long_weather_x)
            long_weather_x = rearrange(long_weather_x, "b s d -> b d s")
            # print(f"long_weather_x shape after cnn: {i} -> {long_weather_x.shape}")
        long_weather_x = rearrange(long_weather_x, "b d s -> b s d")
        long_weather_x = torch.squeeze(long_weather_x)
        # print(f"long_weather_x shape after cnn: {long_weather_x.shape}")
        long_x = torch.cat([long_power_x, long_weather_x], dim=-1)
        long_x = self.long_mapping(long_x)
        # print(f"long_x shape: {long_x.shape}")

        x = short_x + long_x
        # print(f"x shape after sum: {x.shape}")

        xt = seq_xt[:, -self.pred_len :, :]
        # print(f"xt shape: {xt.shape}")
        x = self.fc(x)
        x = self.fc2(x)
        # print(f"final x shape: {x.shape}")
        # x = torch.cat((x, time_feat), dim=-1)
        # print(f"x shape after cat: {x.shape}")

        return x
