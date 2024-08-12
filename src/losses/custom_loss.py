import torch
import torch.nn as nn


class NMAELoss(nn.Module):
    def __init__(self):
        super(NMAELoss, self).__init__()

    def forward(self, y_true, y_pred):
        mean_absolute_error_true = torch.mean(torch.abs(y_true - torch.mean(y_true)))
        nmae = torch.mean(torch.abs((y_true - y_pred) / mean_absolute_error_true))
        return nmae


class NMSELoss(nn.Module):
    def __init__(self):
        super(NMSELoss, self).__init__()

    def forward(self, true_values, predicted_values):
        variance_true = torch.var(true_values)
        nmse = torch.mean(((true_values - predicted_values) / variance_true) ** 2)
        return nmse


class SOCNNLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_true, y_pred, y_sub, alpha=0.1):
        main_loss = F.mse_loss(y_true, y_pred)
        aux_loss = F.mse_loss(y_true, y_sub)
        loss = main_loss + alpha * aux_loss
        return loss
