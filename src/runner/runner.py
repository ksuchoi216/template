import os
import sys
import time

import joblib
import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from einops import rearrange
from lightning.pytorch.loggers import WandbLogger
from torchmetrics import MeanMetric

# from src.utils import plot_pred_dic


class Runner(L.LightningModule):
    def __init__(
        self,
        info,
        optimizer,
        scheduler,
        model,
        metric_selection,
        metrics,
        criterion,
        setup,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["criterion", "model"])
        self.info = info
        self.model = model
        self.criterion = criterion[criterion.selection]
        self.monitor = setup.monitor

        self.train_metric_selection = metric_selection["train"]
        self.test_metric_selection = metric_selection["test"]
        self._set_metrics()
        print(f"[run] self.train_metric_selection: {self.train_metric_selection}")
        print(f"[run] self.test_metric_selection: {self.test_metric_selection}")

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.printout = self.model.printout
        self.pred_len = self.model.pred_len

    def _set_metrics(self):
        for k in self.train_metric_selection:
            setattr(self, f"train_{k}", self.hparams.metrics[k])  # type: ignore
        for k in self.test_metric_selection:
            setattr(self, f"val_{k}", self.hparams.metrics[k])  # type: ignore
        for k in self.test_metric_selection:
            setattr(self, f"test_{k}", self.hparams.metrics[k])  # type: ignore

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())  # type: ignore
        if self.hparams.scheduler is not None:  # type: ignore
            scheduler = self.hparams.scheduler(optimizer=optimizer)  # type: ignore
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.monitor,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def prepare_batch(self, batch):

        seq_y = batch["seq_y"]
        dec_input = torch.zeros_like(seq_y[:, -self.model.label_len :, :]).float()
        dec_input = torch.cat(
            [seq_y[:, : self.model.label_len, :], dec_input], dim=1
        ).float()

        batch_x = dict(
            seq_x=batch["seq_x"],
            seq_xt=batch["seq_xt"],
            seq_yt=batch["seq_yt"],
            # past_x=batch["past_x"],
            # past_xt=batch["past_xt"],
            dec_input=dec_input,
        )
        seq_y = batch["seq_y"]

        self.t_dim = -1 if self.model.features == "MS" else 0
        y_true = seq_y[:, -self.model.pred_len :, self.t_dim :].squeeze()

        for k, v in batch_x.items():
            if isinstance(v, torch.Tensor):
                print(f"[run] {k}: {v.shape}") if self.printout else None

        return batch_x, y_true

    def prepare_loss(self, y_pred, y_true):
        return y_pred.reshape(-1, 1), y_true.reshape(-1, 1)

    def step_forward(self, batch, batch_idx, stage, stage_loss, stage_metric_selection):
        batch_x, y_true = self.prepare_batch(batch)
        y_pred = self.model(batch_x)
        print(f"[run] y_pred: {y_pred.shape}") if self.printout else None
        print(f"[run] y_true: {y_true.shape}") if self.printout else None

        y_pred, y_true = self.prepare_loss(y_pred, y_true)

        loss = self.criterion(y_pred, y_true)
        stage_loss(loss)
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        for key in stage_metric_selection:
            metric = getattr(self, f"{stage}_{key}")
            res = metric(y_pred, y_true)
            self.log(
                f"{stage}/{key}",
                res,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        return loss

    def training_step(self, batch, batch_idx):
        return self.step_forward(
            batch, batch_idx, "train", self.train_loss, self.train_metric_selection
        )

    def validation_step(self, batch, batch_idx):
        return self.step_forward(
            batch, batch_idx, "val", self.val_loss, self.test_metric_selection
        )

    def test_step(self, batch, batch_idx):
        return self.step_forward(
            batch, batch_idx, "test", self.test_loss, self.test_metric_selection
        )

    def predict_step(self, batch, batch_id):
        def inverse_data(base, y, pred_len=True):
            import copy

            base = copy.deepcopy(base)
            y = copy.deepcopy(y)
            # print(f"[1] base:\n{base[:, :, -1]}") if self.printout else None
            if pred_len:
                base = base[:, -self.pred_len :, :]
            base[:, :, -1] = y
            # print(f"[2] base:\n{base[:, :, -1]}") if self.printout else None
            base = base.cpu().numpy()
            b, s, d = base.shape
            base = base.reshape(b * s, d)
            base = self.scaler.inverse_transform(base)
            base = base.reshape(b, s, d)
            # print(f"[3] base:\n{base[:, :, -1]}") if self.printout else None
            print(f"base shape: {base.shape}")

            inversed_y = base[:, :, -1].astype(int)

            return inversed_y

        if batch_id == 0:
            self.scaler = joblib.load(self.info.scaler_path)
            torch.set_grad_enabled(False)
            self.model.eval()

        batch_x, y_true = self.prepare_batch(batch)
        # for k, v in batch_x.items():
        #     if isinstance(v, torch.Tensor):
        #         print(f"[pred] {k}: {v.shape}") if self.printout else None

        y_pred = self.model(batch_x)
        seq_y = batch["seq_y"]
        # print(f"y_pred start==========================================")
        inversed_y_pred = inverse_data(seq_y, y_pred)
        # print(f"y_true start==========================================")
        inversed_y_true = inverse_data(seq_y, y_true)

        seq_x = batch["seq_x"]
        x_input = seq_x[:, :, -1]
        inversed_input_x = inverse_data(seq_x, x_input, pred_len=False)

        pred_dics = dict(
            x_input=inversed_input_x,
            y_pred=inversed_y_pred,
            y_true=inversed_y_true,
        )

        # for k, v in pred_dics.items():
        #     print(f"[pred] {k}: {v.shape}")

        return pred_dics

    def forward(self, batch):
        return self.model(batch)
