import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from src.models.layers.Embed import DataEmbedding_wo_pos


class Model(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        label_len,
        past_len,
        input_size,
        hidden_size,
        kernel_size,
        n_layer_sign,
        n_layer_offset,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.past_len = past_len

        self.input_mapping = nn.Linear(input_size, hidden_size)

        self.sign_layers = nn.ModuleList()
        for i in range(n_layer_sign):
            self.sign_layers.append(
                nn.Conv1d(hidden_size, hidden_size, kernel_size, padding="same")
            )
            self.sign_layers.append(nn.BatchNorm1d(hidden_size))
            self.sign_layers.append(nn.LeakyReLU(negative_slope=0.1))

        self.offset_layers = nn.ModuleList()
        for i in range(n_layer_offset):
            self.offset_layers.append(
                nn.Conv1d(hidden_size, hidden_size, kernel_size=1, padding="same")
            )
            self.offset_layers.append(nn.BatchNorm1d(hidden_size))
            self.offset_layers.append(nn.LeakyReLU(negative_slope=0.1))

        self.sign_locally_connected = nn.Conv1d(
            in_channels=hidden_size,  # assuming main has the shape (batch_size, channels, sequence_length)
            out_channels=1,
            kernel_size=1,
            padding=0,  # 'valid' padding in Keras is equivalent to no padding in PyTorch
        )

        self.main_final_mapping = nn.Linear(self.seq_len, self.pred_len)

        self.sub_reduce_mapping = nn.Linear(hidden_size, 1)
        self.sub_final_mapping = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, seq_x, seq_xt, past_x, past_xt, dec_input, seq_yt):
        x = seq_x
        # print(f"x shape: {x.shape}")
        x_in = self.input_mapping(x)
        # print(f"x shape after input_mapping: {x.shape}")

        x = rearrange(x_in, "b s d -> b d s")
        # print(f"x shape after rearrange: {x.shape}")

        for layer in self.sign_layers:
            x = layer(x)
        x_sign = rearrange(x, "b d s -> b s d")
        # print(f"x_sign shape after rearrange: {x_sign.shape}")

        x = rearrange(x_in, "b s d -> b d s")
        for layer in self.offset_layers:
            x = layer(x)
        x_offset = rearrange(x, "b d s -> b s d")
        # print(f"x_offset shape after rearrange: {x_offset.shape}")

        significance = F.softmax(x_sign, dim=2)
        # print(f"x_sign shape after softmax: {x_sign.shape}")
        x_off = x_in + x_offset
        x = x_off * significance
        # print(f"x shape after multiply: {x.shape}")

        x = rearrange(x, "b s d -> b d s")
        x = self.sign_locally_connected(x)
        x = torch.squeeze(x)
        # print(f"x shape after locally_connected: {x.shape}")
        y = self.main_final_mapping(x)
        # print(f"x shape after main_final_mapping: {x.shape}")

        y_sub = self.sub_reduce_mapping(x_off)
        y_sub = torch.squeeze(y_sub)
        y_sub = self.sub_final_mapping(y_sub)
        # print(f"y_sub shape: {y_sub.shape}")

        return y, y_sub
