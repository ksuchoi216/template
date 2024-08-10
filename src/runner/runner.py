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
        for k, v in batch.items():
            print(f"[run] {k}: {v.shape}")

        batch_x = dict(
            seq_x=batch["seq_x"],
            seq_xt=batch["seq_xt"],
            seq_yt=batch["seq_yt"],
            past_x=batch["past_x"],
            past_xt=batch["past_xt"],
        )
        seq_y = batch["seq_y"]

        self.t_dim = -1 if self.model.features == "MS" else 0
        y_true = seq_y[:, -self.model.pred_len :, self.t_dim :].squeeze()

        return batch_x, y_true

    def prepare_loss(self, y_pred, y_true):
        return y_pred.reshape(-1, 1), y_true.reshape(-1, 1)

    def training_step(self, batch, batch_idx):
        batch_x, y_true = self.prepare_batch(batch)
        y_pred = self.model(batch_x)
        y_pred, y_true = self.prepare_loss(y_pred, y_true)

        print(f"[run] y_pred: {y_pred.shape}, type(y_pred): {type(y_pred)}")
        print(f"[run] y_true: {y_true.shape}, type(y_true): {type(y_true)}")
        print(f"self.criterion: {self.criterion}")

        loss = self.criterion(y_pred, y_true)
        print(f"[run] loss: {loss}")
        self.train_loss(loss)
        self.log(
            "train/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        for key in self.train_metric_selection:
            metric = getattr(self, f"train_{key}")
            res = metric(y_pred, y_true)
            self.log(
                f"train/{key}",
                res,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        return loss

    def validation_step(self, batch, batch_idx):
        print(f"[run] validation_step: {batch_idx}")
        batch_x, y_true = self.prepare_batch(batch)
        y_pred = self.model(batch_x)
        y_pred, y_true = self.prepare_loss(y_pred, y_true)
        loss = self.criterion(y_pred, y_true)

        self.val_loss(loss)
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        for key in self.test_metric_selection:
            metric = getattr(self, f"val_{key}")
            res = metric(y_pred, y_true)
            self.log(
                f"val/{key}",
                res,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        return loss

    def test_step(self, batch, batch_idx):
        batch_x, y_true = self.prepare_batch(batch)
        y_pred = self.model(batch_x)
        y_pred, y_true = self.prepare_loss(y_pred, y_true)
        loss = self.criterion(y_pred, y_true)

        self.test_loss(loss)
        self.log(
            "test/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        for key in self.test_metric_selection:
            metric = getattr(self, f"test_{key}")
            res = metric(y_pred, y_true)
            self.log(
                f"test/{key}",
                res,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        return loss

    # def predict_step(self, batch, batch_id):
    #     def transform_data(seq_y, y_pred, t_dim):
    #         b, s, d = seq_y.shape
    #         seq_y[..., t_dim] = y_pred

    #         seq_y = rearrange(seq_y, "b s d -> (b s) d")

    #         seq_y = scaler.inverse_transform(seq_y)
    #         seq_y = rearrange(seq_y, "(b s) d -> b s d", b=b, s=s)
    #         y_pred = seq_y[..., t_dim].squeeze()
    #         return y_pred

    #     scaler = joblib.load(f"{self.info.data_dir}/scaler.pkl")
    #     batch_x, y_true = self.prepare_batch(batch)
    #     y_pred = self.model(batch_x)

    #     seq_y = batch_x["seq_y"]

    #     y_true = transform_data(seq_y, y_true, self.t_dim)
    #     y_pred = transform_data(seq_y, y_pred, self.t_dim)
    #     res = np.concatenate([y_true, y_pred], axis=1)
    #     df = pd.DataFrame(res, columns=["y_true", "y_pred"])
    #     ts = int(time.time())
    #     res_dir = f"{self.info.output_dir}/res/{ts}_{self.info.run_name}"
    #     if not os.path.exists(self.info.output_dir):
    #         os.makedirs(self.info.output_dir)

    #     if not os.path.exists(res_dir):
    #         os.makedirs(res_dir)

    #     df.to_csv(
    #         f"{res_dir}/res.csv",
    #         index=False,
    #     )

    # def on_validation_epoch_end(self):
    #     val_y_trues = torch.cat(self.val_y_trues)
    #     val_y_preds = torch.cat(self.val_y_preds)
    #     val_y_prods = torch.cat(self.val_y_prods)
    #     print(f"[exp] val_y_trues: {val_y_trues.shape}") if self.printout else None
    #     print(f"[exp] val_y_preds: {val_y_preds.shape}") if self.printout else None
    #     print(f"[exp] val_y_prods: {val_y_prods.shape}") if self.printout else None

    #     for key in self.save_metric_selection:
    #         metric = getattr(self, f"save_{key}")
    #         if key == "cm":
    #             y_out = val_y_preds
    #         else:
    #             y_out = val_y_prods
    #         metric.update(y_out, val_y_trues)
    #         fig, ax = metric.plot()
    #         fig.savefig(
    #             f"{self.info.output_dir}/val_{self.info.dataset_name}_{self.info.x_data_name}_{key}.png"
    #         )
    #         # self.logger.experiment.log({key: [wandb.Image(fig)]})

    #     self.val_y_trues = []
    #     self.val_y_preds = []
    #     self.val_y_prods = []

    # def on_test_epoch_end(self):
    #     test_y_trues = torch.cat(self.test_y_trues)
    #     test_y_preds = torch.cat(self.test_y_preds)
    #     test_y_prods = torch.cat(self.test_y_prods)
    #     print(f"[exp] test_y_trues: {test_y_trues.shape}") if self.printout else None
    #     print(f"[exp] test_y_preds: {test_y_preds.shape}") if self.printout else None
    #     print(f"[exp] test_y_prods: {test_y_prods.shape}") if self.printout else None

    #     for key in self.save_metric_selection:
    #         metric = getattr(self, f"save_{key}")
    #         if key == "cm":
    #             y_out = test_y_preds
    #         else:
    #             y_out = test_y_prods
    #         metric.update(y_out, test_y_trues)
    #         fig, ax = metric.plot()
    #         fig.savefig(
    #             f"{self.info.output_dir}/test_{self.info.dataset_name}_{self.info.x_data_name}_{key}.png"
    #         )
    #         # self.logger.experiment.log({key: [wandb.Image(fig)]})

    #     self.test_y_trues = []
    #     self.test_y_preds = []
    #     self.test_y_prods = []

    # def predict_step(self, batch, batch_id):
    #     self.dropout.train()

    #     batch_x, batch_y = self.prepare_batch(batch)

    #     y_preds = []
    #     for _ in range(self.mc_iteration):
    #         y_out = self.dropout(self.model(batch_x))
    #         y_pred, y_true = self.prepare_eval(y_out, batch_y)
    #         y_preds.append(y_pred)

    #     y_preds = torch.tensor(y_preds)
    #     return {"y_pred": y_preds, "y_true": y_true}
