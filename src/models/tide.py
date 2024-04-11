import torch.nn as nn
import torch
from einops import rearrange


class _ResidualBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_size,
        dropout,
        use_layer_norm,
    ):
        super().__init__()

        # dense layer with ReLU activation with dropout
        self.dense = nn.Sequential(
            nn.Linear(input_dim, hidden_size),  # d -> h
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),  # h -> d'
            nn.Dropout(dropout),
        )
        # linear skip connection from input to output of self.dense
        self.skip = nn.Linear(input_dim, output_dim)

        # layer normalization as output
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None

    def forward(self, x):
        # print("x:", x.shape)
        y1 = self.dense(x)
        # print("y1:", y1.shape)
        y2 = self.skip(x)
        # print("y2:", y2.shape)

        x = y1 + y2

        # layer normalization
        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return x


class Model(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        label_len,
        input_size,
        past_cov_dim,
        temporal_width_past,
        future_cov_dim,
        temporal_width_future,
        hidden_size,
        output_dim,
        use_layer_norm,
        dropout,
        num_encoder_layers,
        num_decoder_layers,
        decoder_output_dim,
        nr_params,
        temporal_decoder_hidden,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len

        self.past_cov_dim = past_cov_dim
        self.decoder_output_dim = decoder_output_dim
        self.nr_params = nr_params

        self.past_cov_projection = _ResidualBlock(
            input_dim=self.past_cov_dim,
            output_dim=temporal_width_past,
            hidden_size=hidden_size,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        )
        self.future_cov_projection = _ResidualBlock(
            input_dim=future_cov_dim,
            output_dim=temporal_width_future,
            hidden_size=hidden_size,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        )
        encoder_dim = (
            self.seq_len * (input_size + temporal_width_past)
            + self.pred_len * temporal_width_future
        )

        self.encoders = nn.Sequential(
            _ResidualBlock(
                input_dim=encoder_dim,
                output_dim=hidden_size,
                hidden_size=hidden_size,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
            ),
            *[
                _ResidualBlock(
                    input_dim=hidden_size,
                    output_dim=hidden_size,
                    hidden_size=hidden_size,
                    use_layer_norm=use_layer_norm,
                    dropout=dropout,
                )
                for _ in range(num_encoder_layers - 1)
            ],
        )
        self.decoders = nn.Sequential(
            *[
                _ResidualBlock(
                    input_dim=hidden_size,
                    output_dim=hidden_size,
                    hidden_size=hidden_size,
                    use_layer_norm=use_layer_norm,
                    dropout=dropout,
                )
                for _ in range(num_decoder_layers - 1)
            ],
            # add decoder output layer
            _ResidualBlock(
                input_dim=hidden_size,
                output_dim=decoder_output_dim * pred_len * nr_params,
                hidden_size=hidden_size,
                use_layer_norm=use_layer_norm,
                dropout=dropout,
            ),
        )
        decoder_input_dim = (decoder_output_dim + temporal_width_future) * nr_params
        print(f"decoder_input_dim: {decoder_input_dim}")

        self.temporal_decoder = _ResidualBlock(
            input_dim=decoder_input_dim,
            output_dim=output_dim * self.nr_params,
            hidden_size=temporal_decoder_hidden,
            use_layer_norm=use_layer_norm,
            dropout=dropout,
        )

        self.lookback_skip = nn.Linear(self.seq_len, self.pred_len * self.nr_params)

    def forward(self, seq_x, seq_xt, dec_input, seq_yt):
        future_time_feat = seq_yt[:, self.label_len :, :]
        past_time_feat = seq_xt

        # x_past = torch.cat([seq_x, past_time_feat], dim=2)
        # x_future = torch.cat([seq_x, future_time_feat], dim=2)
        # print(f"x_past: {x_past.shape}, x_future: {x_future.shape}")

        past_time_feat = self.past_cov_projection(past_time_feat)
        future_time_feat = self.future_cov_projection(future_time_feat)
        # print(
        # f"past_time_feat.shape: {past_time_feat.shape}, future_time_feat.shape: {future_time_feat.shape}"
        # )

        # print(f"seq_x: {seq_x.shape}")
        encoded = [seq_x, past_time_feat, future_time_feat]
        encoded = [t.flatten(start_dim=1) for t in encoded if t is not None]
        encoded = torch.cat(encoded, dim=1)
        # print(f"encoded.shape: {encoded.shape}")
        encoded = self.encoders(encoded)
        # print(f"encoded.shape: {encoded.shape}")
        decoded = self.decoders(encoded)
        # print(f"decoded.shape: {decoded.shape}")

        decoded = rearrange(
            decoded,
            "b (p do) -> b p do",
            p=self.pred_len,
        )

        # print(f"decoded.shape: {decoded.shape}")
        # print(f"future_time_feat.shape: {future_time_feat.shape}")
        concated = [decoded, future_time_feat]
        # concated = [t.flatten(start_dim=1) for t in concated if t is not None]
        temporal_decoder_input = torch.cat(concated, dim=2)

        # print(f"temporal_decoder_input.shape: {temporal_decoder_input.shape}")

        temporal_decoded = self.temporal_decoder(temporal_decoder_input)
        # print(f"temporal_decoded.shape: {temporal_decoded.shape}")

        skip = self.lookback_skip(rearrange(seq_x, "b s d -> b d s"))
        skip = rearrange(skip, "b d s -> b s d")
        # print(f"temporal_decoded.shape: {temporal_decoded.shape}, skip: {skip.shape}")
        y = temporal_decoded + skip
        # print(f"y.shape: {y.shape}")

        return y


import sys

"""
L=seq_len, H=pred_len - label_len
lookback 1:L
prediction L+1:L+H

y-power: 1:L
x-time: 1:L+H
a-other: constant value

"""
