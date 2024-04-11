from src.pl_modules.pl_baseline import TSForecastTask


class TSForecastSOCNN(TSForecastTask):
    def __init__(
        self, optimizer, scheduler, model, metrics, criterion, monitor, **kwargs
    ):
        super().__init__(
            optimizer, scheduler, model, metrics, criterion, monitor, **kwargs
        )

    def training_step(self, batch, batch_idx):
        seq_x, seq_y, seq_xt, seq_yt, past_x, past_xt, input_x, dec_input = (
            self.prepare_batch(batch)
        )
        y_true = seq_y
        y_pred, y_sub = self.model(seq_x, seq_xt, past_x, past_xt, dec_input, seq_yt)

        y_pred = self.prepare_pred(y_pred)
        y_true, y_pred = y_true.reshape(-1), y_pred.reshape(-1)
        y_sub = y_sub.reshape(-1)

        loss = self.criterion(y_pred, y_true, y_sub)
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
        y_pred, y_sub = self.model(seq_x, seq_xt, past_x, past_xt, dec_input, seq_yt)

        y_pred = self.prepare_pred(y_pred)
        y_true, y_pred = y_true.reshape(-1), y_pred.reshape(-1)
        y_sub = y_sub.reshape(-1)

        loss = self.criterion(y_pred, y_true, y_sub)
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
        y_pred, y_sub = self.model(seq_x, seq_xt, past_x, past_xt, dec_input, seq_yt)
        y_pred = self.prepare_pred(y_pred)

        return {"y_pred": y_pred, "y_true": y_true, "input_x": input_x}
