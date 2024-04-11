import torch.nn as nn
import torch
from src.models.layers.Embed import DataEmbedding_wo_pos
from einops import rearrange


class Model(nn.Module):
    def __init__(self, input_size, seq_len, pred_len, label_len, d_model, individual):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len

        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.d_model = d_model
        self.individual = individual
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.d_model):
                self.Linear.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, seq_x, seq_xt, dec_input, seq_yt):
        x = torch.cat([seq_x, seq_xt], dim=-1)

        if self.individual:
            output = torch.zeros(
                [x.size(0), self.pred_len, x.size(2)], dtype=x.dtype
            ).to(x)
            for i in range(self.d_model):
                output[:, :, i] = self.Linear[i](x[:, :, i])
            x = output
            # print(f"x.shape: {x.shape}")
        else:
            x = rearrange(x, "b s d -> b d s")
            x = self.Linear(x)
            x = rearrange(x, "b d s -> b s d")
        return x
