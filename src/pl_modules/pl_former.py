import lightning as L
from torchmetrics import MeanMetric

import torch

from src.pl_modules.pl_baseline import TSForecastTask


class TSForecastFormer(TSForecastTask):
    def __init__(
        self, optimizer, scheduler, model, metrics, criterion, monitor, **kwargs
    ):
        super().__init__(
            optimizer, scheduler, model, metrics, criterion, monitor, **kwargs
        )

    def prepare_pred(self, y_pred):
        y_pred = y_pred[:, :, self.target_idx]
        return y_pred
