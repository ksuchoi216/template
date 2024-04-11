import re
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.autograd import Variable
from torch.nn import functional as tf


def init_hidden(x, hidden_size, num_direction=1, xavier=True):
    if xavier:
        return nn.init.xavier_normal_(
            torch.zeros(num_direction, x.size(0), hidden_size)
        )
    return Variable(torch.zeros(num_direction, x.size(0), hidden_size))


class InputAttentionEncoder(nn.Module):
    def __init__(self, N, M, T, stateful=False):
        """
        :param: N: int
            number of time serieses
        :param: M:
            number of LSTM units
        :param: T:
            number of timesteps
        :param: stateful:
            decides whether to initialize cell state of new time window with values of the last cell state
            of previous time window or to initialize it with zeros
        """
        super().__init__()
        self.N = N
        self.M = M
        self.T = T

        self.encoder_lstm = nn.LSTMCell(input_size=self.N, hidden_size=self.M)

        # equation 8 matrices
        self.W_e = nn.Linear(2 * self.M, self.T)
        self.U_e = nn.Linear(self.T, self.T, bias=False)
        self.v_e = nn.Linear(self.T, 1, bias=False)

    def forward(self, inputs):
        encoded_inputs = torch.zeros((inputs.size(0), self.T, self.M)).to(inputs.device)

        # initiale hidden states
        h_tm1 = torch.zeros((inputs.size(0), self.M)).to(inputs.device)
        s_tm1 = torch.zeros((inputs.size(0), self.M)).to(inputs.device)

        for t in range(self.T):
            # concatenate hidden states
            h_c_concat = torch.cat((h_tm1, s_tm1), dim=1)

            # attention weights for each k in N (equation 8)
            x = self.W_e(h_c_concat).unsqueeze_(1).repeat(1, self.N, 1)
            y = self.U_e(inputs.permute(0, 2, 1))
            z = torch.tanh(x + y)
            e_k_t = torch.squeeze(self.v_e(z))

            # normalize attention weights (equation 9)
            alpha_k_t = F.softmax(e_k_t, dim=1)

            # weight inputs (equation 10)
            weighted_inputs = alpha_k_t * inputs[:, t, :]

            # calculate next hidden states (equation 11)
            h_tm1, s_tm1 = self.encoder_lstm(weighted_inputs, (h_tm1, s_tm1))

            encoded_inputs[:, t, :] = h_tm1
        return encoded_inputs


class TemporalAttentionDecoder(nn.Module):
    def __init__(self, M, P, T, O, stateful=False):
        """
        :param: M: int
            number of encoder LSTM units
        :param: P:
            number of deocder LSTM units
        :param: T:
            number of timesteps
        :param: stateful:
            decides whether to initialize cell state of new time window with values of the last cell state
            of previous time window or to initialize it with zeros
        """
        super().__init__()
        self.M = M
        self.P = P
        self.T = T
        self.stateful = stateful

        self.decoder_lstm = nn.LSTMCell(input_size=1, hidden_size=self.P)

        # equation 12 matrices
        self.W_d = nn.Linear(2 * self.P, self.M)
        self.U_d = nn.Linear(self.M, self.M, bias=False)
        self.v_d = nn.Linear(self.M, 1, bias=False)

        # equation 15 matrix
        self.w_tilda = nn.Linear(self.M + 1, 1)

        # equation 22 matrices
        self.W_y = nn.Linear(self.P + self.M, self.P)
        self.v_y = nn.Linear(self.P, O)

    def forward(self, encoded_inputs, y):

        # initializing hidden states
        d_tm1 = torch.zeros((encoded_inputs.size(0), self.P)).to(encoded_inputs.device)
        s_prime_tm1 = torch.zeros((encoded_inputs.size(0), self.P)).to(
            encoded_inputs.device
        )
        # print(d_tm1.device)
        # print(s_prime_tm1.device)
        for t in range(self.T):
            # concatenate hidden states
            d_s_prime_concat = torch.cat((d_tm1, s_prime_tm1), dim=1)
            # print(d_s_prime_concat)
            # temporal attention weights (equation 12)
            x1 = (
                self.W_d(d_s_prime_concat)
                .unsqueeze_(1)
                .repeat(1, encoded_inputs.shape[1], 1)
            )
            y1 = self.U_d(encoded_inputs)
            z1 = torch.tanh(x1 + y1)
            l_i_t = self.v_d(z1)
            # print(f"l_i_t shape: {l_i_t.shape}")

            # normalized attention weights (equation 13)
            beta_i_t = F.softmax(l_i_t, dim=1)
            # print(f"beta_i_t shape: {beta_i_t.shape}")

            # create context vector (equation_14)
            c_t = torch.sum(beta_i_t * encoded_inputs, dim=1)
            # print(f"c_t shape: {c_t.shape}")

            # concatenate c_t and y_t
            y_c_concat = torch.cat((c_t, y[:, t, :]), dim=1)
            # print(f"y_c_concat shape: {y_c_concat.shape}")
            # create y_tilda
            y_tilda_t = self.w_tilda(y_c_concat)

            # calculate next hidden states (equation 16)
            d_tm1, s_prime_tm1 = self.decoder_lstm(y_tilda_t, (d_tm1, s_prime_tm1))

        # concatenate context vector at step T and hidden state at step T
        d_c_concat = torch.cat((d_tm1, c_t), dim=1)
        # print(f"d_tm1 shape: {d_tm1.shape}")
        # print(f"c_t shape: {c_t.shape}")
        # print(f"d_c_concat shape: {d_c_concat.shape}")

        # calculate output
        y_Tp1 = self.v_y(self.W_y(d_c_concat))
        # print(f"y_Tp1 shape: {y_Tp1.shape}")
        return y_Tp1


class Model(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        label_len,
        past_len,
        target_idx,
        feat_dim,
        encoder_hidden_size,
        decoder_hidden_size,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.label_len = label_len
        self.target_idx = target_idx
        self.encoder_input_size = feat_dim

        self.encoder = InputAttentionEncoder(
            self.encoder_input_size, encoder_hidden_size, seq_len
        )

        self.decoder = TemporalAttentionDecoder(
            encoder_hidden_size, decoder_hidden_size, seq_len, pred_len
        )

    def forward(self, seq_x, seq_xt, past_x, past_xt, dec_input, seq_yt):
        x = seq_x[:, :, : self.encoder_input_size]
        y = seq_x[:, :, self.target_idx].unsqueeze(2)
        # print(f"x shape: {x.shape}")
        # print(f"y shape: {y.shape}")

        x_encoded = self.encoder(x)
        # print(f"x_encoded shape: {x_encoded.shape}")

        out = self.decoder(x_encoded, y)
        # print(f"out shape: {out.shape}")
        return out
