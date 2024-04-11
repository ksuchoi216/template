import torch.nn as nn
import torch
from src.models.layers.Embed import DataEmbedding_wo_pos
from einops import rearrange


class Model(nn.Module):
    def __init__(
        self,
        input_size,
        seq_len,
        pred_len,
        label_len,
    ):
        super().__init__()

    def forward(self, seq_x, seq_xt, dec_input, seq_yt):

        return 0
