import torch.nn as nn
import torch
from einops import rearrange


class _MixingBlock(nn.Module):
    def __init__(self, seq_len, n_feature, hidden_size, dropout):
        super(_MixingBlock, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=1)
        self.lin_time = nn.Linear(in_features=seq_len, out_features=seq_len)
        self.lin_feat1 = nn.Linear(in_features=n_feature, out_features=hidden_size)
        self.lin_feat2 = nn.Linear(in_features=hidden_size, out_features=n_feature)

        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.ReLU()

    def forward(self, x):
        # Time mixing
        y = x.unsqueeze(1)  # b, 1, s, d
        y = self.bn(y)
        y = y.squeeze(1)  # b, s, d
        y = rearrange(y, "b d s -> b s d")
        y = self.act(self.lin_time(y))
        y = rearrange(y, "b s d -> b d s")
        y = self.dropout(y)
        x = x + y  # residual

        # Feature mixing
        y = x.unsqueeze(1)  # b, 1, s, d
        y = self.bn(y)
        y = y.squeeze(1)  # b, s, d
        y = self.act(self.lin_feat1(y))
        y = self.dropout(y)
        y = self.lin_feat2(y)
        y = self.dropout(y)
        x = x + y  # residual

        return x


class Model(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        label_len,
        input_size,
        hidden_size,
        dropout,
        n_block,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len

        mixing_layers = []
        for _ in range(n_block):
            mixing_layers.append(
                _MixingBlock(seq_len, input_size, hidden_size, dropout)
            )

        self.mixing_layers = nn.ModuleList(mixing_layers)
        self.temp_proj_layer = nn.Linear(in_features=seq_len, out_features=pred_len)

    def forward(self, seq_x, seq_xt, dec_input, seq_yt):
        x = torch.cat([seq_x, seq_xt], dim=-1)

        for mixing_layer in self.mixing_layers:
            x = mixing_layer(x)

        x = rearrange(x, "b d s -> b s d")
        x = self.temp_proj_layer(x)
        x = rearrange(x, "b s d -> b d s")

        return x
