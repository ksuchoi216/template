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
        self.input_size = input_size
        self.hidden_size = 48
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.n_layer_lstm = n_layer_lstm

        self.input_size = input_size
        self.embed = embed
        self.freq = freq
        self.dropout = dropout
        # Embedding
        # self.enc_embedding = DataEmbedding_wo_pos(
        #     self.input_size, self.input_size, self.embed, self.freq, self.dropout
        # )
        self.n_layer_cnn = n_layer_cnn

        if bidirectional:
            self.num_direction = 2
        else:
            self.num_direction = 1

        self.conv1d = nn.Conv1d(
            in_channels=input_size,
            out_channels=cnn_out_channel,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
        )

        self.convlist = nn.ModuleList(
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

        self.normlist = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(cnn_out_channel),
                    nn.ReLU(),
                )
                for _ in range(n_layer_cnn)
            ]
        )

        self.lstm = nn.LSTM(
            input_size=cnn_out_channel,
            hidden_size=self.hidden_size,
            num_layers=n_layer_lstm,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.fc = nn.Linear(
            self.hidden_size * self.num_direction + 7, self.hidden_size // 2
        )
        self.fc2 = nn.Linear(self.hidden_size // 2, 1)

        self.fc3 = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, seq_x, seq_xt, past_x, dec_input, seq_yt):
        # x = self.enc_embedding(seq_x, seq_xt)
        # print(f"seq_x: {seq_x.shape}, seq_xt: {seq_xt.shape}")
        # print(f"dec_input: {dec_input.shape}, seq_yt: {seq_yt.shape}")
        # print(f"past_x: {past_x.shape}")

        x = torch.cat([seq_x, seq_xt], dim=-1)
        # x = seq_x
        x = rearrange(x, "b s d -> b d s")
        # print(f"x shape: {x.shape}")
        x = self.conv1d(x)

        for i in range(self.n_layer_cnn):
            # print(f"x shape before cnn: {i} -> {x.shape}")
            x = self.convlist[i](x)
            x = rearrange(x, "b d s -> b s d")
            x = self.normlist[i](x)
            x = rearrange(x, "b s d -> b d s")
            # print(f"x shape after cnn: {i} -> {x.shape}")
        x = rearrange(x, "b d s -> b s d")

        # print(f"x shape before lstm: {x.shape}")

        h0 = torch.zeros(
            self.n_layer_lstm * self.num_direction, x.size(0), self.hidden_size
        ).to(x.device)
        c0 = torch.zeros(
            self.n_layer_lstm * self.num_direction, x.size(0), self.hidden_size
        ).to(x.device)
        # print(f"x shape: {x.shape}, h0 shape: {h0.shape}, c0 shape: {c0.shape}")
        x, (h_out, _) = self.lstm(x, (h0, c0))
        # print(f"x shape after lstm: {x.shape}")

        look_yt = seq_yt[:, self.label_len :, :]
        look_xt = seq_xt[:, -self.pred_len :, :]

        time_feat = torch.cat([look_yt, look_xt], dim=1)
        # print(f"time feat shape: {time_feat.shape}")

        x = torch.cat((x, time_feat), dim=-1)
        # print(f"x shape after cat: {x.shape}")

        x = self.fc(x)
        x = self.fc2(x)
        x = torch.squeeze(x)
        x = self.fc3(x)
        return x
