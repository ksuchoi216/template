import torch
import torch.nn as nn
import torch.nn.functional as F


class SOCNNLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred, y_sub, alpha=0.1):
        main_loss = F.mse_loss(y_true, y_pred)
        aux_loss = F.mse_loss(y_true, y_sub)
        loss = main_loss + alpha * aux_loss
        return loss
