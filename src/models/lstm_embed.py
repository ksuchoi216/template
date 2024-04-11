import torch.nn as nn
import torch
from src.models.layers.Embed import DataEmbedding_wo_pos


class Model(nn.Module):
    def __init__(
        self,
        input_size,
        seq_len,
        pred_len,
        label_len,
        num_layers,
        bidirectional,
        d_model=512,
        embed="timeF",
        freq="t",
        dropout=0.05,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = 48
        self.pred_len = pred_len
        self.label_len = label_len
        self.num_layers = num_layers

        self.d_model = d_model
        self.embed = embed
        self.freq = freq
        self.dropout = dropout
        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(
            self.input_size, self.d_model, self.embed, self.freq, self.dropout
        )

        if bidirectional:
            self.num_direction = 2
        else:
            self.num_direction = 1

        self.lstm = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.fc = nn.Linear(
            self.hidden_size * self.num_direction, self.hidden_size // 2
        )
        self.fc2 = nn.Linear(self.hidden_size // 2, 1)

    def forward(self, seq_x, seq_xt, dec_input, seq_yt):
        # print(x.shape)
        # change x as type torch.float32
        x = self.enc_embedding(seq_x, seq_xt)

        h0 = torch.zeros(
            self.num_layers * self.num_direction, x.size(0), self.hidden_size
        ).to(x.device)
        c0 = torch.zeros(
            self.num_layers * self.num_direction, x.size(0), self.hidden_size
        ).to(x.device)
        # print(f"h0, c0 {type(h0)} {type(c0)}")
        # print(x.shape)
        x, (h_out, _) = self.lstm(x, (h0, c0))
        # print(x.shape)

        x = self.fc(x)
        # print(x.shape)
        x = self.fc2(x)
        x = torch.squeeze(x)
        # print(x.shape)
        return x
