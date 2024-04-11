import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        label_len,
        past_len,
        input_size,
        num_layers,
        bidirectional,
        dropout,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = 48
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.num_layers = num_layers

        if bidirectional:
            self.num_direction = 2
        else:
            self.num_direction = 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.fc = nn.Linear(
            self.hidden_size * self.num_direction, self.hidden_size // 2
        )
        self.fc2 = nn.Linear(self.hidden_size // 2, 1)

        self.fc3 = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, seq_x, seq_xt, past_x, past_xt, dec_input, seq_yt):
        x = seq_x
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
        # print(f"before lstm x: {x.shape}")

        x = self.fc(x)
        # print(f"f1 x: {x.shape}")

        x = self.fc2(x)
        # print(f"f2 x: {x.shape}")
        x = torch.squeeze(x)

        x = self.fc3(x)
        # print(f"x shape: {x.shape}")
        return x
