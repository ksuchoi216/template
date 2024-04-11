import lightning as L
import torch
from torchmetrics import MeanMetric


class TSForecastTask(L.LightningModule):
    def __init__(
        self,
        optimizer,
        scheduler,
        model,
        metrics,
        criterion,
        monitor,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["criterion"])

        self.model = model
        self.criterion = criterion
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self._set_metrics()

    def _set_metrics(self):
        for k in self.hparams.metrics.train:
            setattr(self, f"train_{k}", self.hparams.metrics.train[k])
        for k in self.hparams.metrics.val:
            setattr(self, f"val_{k}", self.hparams.metrics.val[k])

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.hparams.monitor,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def prepare_batch(self, batch):
        seq_x = batch["seq_x"]
        seq_y = batch["seq_y"]
        seq_xt = batch["seq_xt"]
        seq_yt = batch["seq_yt"]
        self.target_idx = batch["target_idx"][0]
        past_x = batch["past_x"]
        past_xt = batch["past_xt"]
        input_x = seq_x[:, :, self.target_idx]

        # * decoder input
        dec_input = torch.zeros_like(seq_y[:, -self.model.label_len :, :]).float()
        dec_input = torch.cat(
            [seq_y[:, : self.model.label_len, :], dec_input], dim=1
        ).float()

        seq_y = seq_y[:, -self.model.pred_len :, self.target_idx]
        return seq_x, seq_y, seq_xt, seq_yt, past_x, past_xt, input_x, dec_input

    def prepare_pred(self, y_pred):
        # y_pred = y_pred[:, :, self.target_idx]
        return y_pred

    def training_step(self, batch, batch_idx):
        seq_x, seq_y, seq_xt, seq_yt, past_x, past_xt, input_x, dec_input = (
            self.prepare_batch(batch)
        )
        y_true = seq_y
        y_pred = self.model(seq_x, seq_xt, past_x, past_xt, dec_input, seq_yt)

        y_pred = self.prepare_pred(y_pred)
        y_true, y_pred = y_true.reshape(-1), y_pred.reshape(-1)

        loss = self.criterion(y_pred, y_true)
        self.train_loss(loss)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        for key in self.hparams.metrics.train:
            metric = getattr(self, f"train_{key}")
            metric(y_pred, y_true)
            self.log(
                f"train/{key}",
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        return loss

    def validation_step(self, batch, batch_idx):
        seq_x, seq_y, seq_xt, seq_yt, past_x, past_xt, input_x, dec_input = (
            self.prepare_batch(batch)
        )
        y_true = seq_y
        y_pred = self.model(seq_x, seq_xt, past_x, past_xt, dec_input, seq_yt)

        y_pred = self.prepare_pred(y_pred)
        y_true, y_pred = y_true.reshape(-1), y_pred.reshape(-1)

        loss = self.criterion(y_pred, y_true)
        self.val_loss(loss)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        for key in self.hparams.metrics.val:
            metric = getattr(self, f"val_{key}")
            metric(y_pred, y_true)
            self.log(
                f"val/{key}",
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        return {"grountruth": y_true, "prediction": y_pred}

    def predict_step(self, batch, batch_idx=0):
        seq_x, seq_y, seq_xt, seq_yt, past_x, past_xt, input_x, dec_input = (
            self.prepare_batch(batch)
        )
        y_true = seq_y
        y_pred = self.model(seq_x, seq_xt, past_x, past_xt, dec_input, seq_yt)
        y_pred = self.prepare_pred(y_pred)

        return {"y_pred": y_pred, "y_true": y_true, "input_x": input_x}
