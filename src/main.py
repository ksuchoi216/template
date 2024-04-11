import os
import sys
import lightning as L
import mlflow
import hydra
import pickle
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.cli import LightningCLI

from src import utils

log = utils.get_pylogger(__name__)


def execute(cfg: DictConfig):
    if cfg.get("seed"):
        L.seed_everything(cfg.seed)

    log.info(f"datamodel {cfg.datamodule._target_}")
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"pl_module {cfg.pl_module._target_}")
    pl_module = hydra.utils.instantiate(cfg.pl_module)

    log.info(f"callbacks loading...")
    callbacks = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info(f"trainer {cfg.trainer._target_}")
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks)

    train_metrics, test_metrics = {}, {}

    if cfg.trainer.get("limit_val_batches"):
        datamodule.setup("fit")
        train_dataloader = datamodule.train_dataloader()
        batch = next(iter(train_dataloader))
        pl_module.training_step(batch, 1)

        # sys.exit()
        trainer.fit(model=pl_module, datamodule=datamodule)
        train_metrics = trainer.callback_metrics
        print(f"train_metrics: {train_metrics}")
        res = trainer.predict(model=pl_module, datamodule=datamodule)
        draw_path = utils.draw_subplot_res(cfg, "test", res, row_num=5, col_num=3)
        print(train_metrics)
        run_name = utils.get_run_name(cfg)
        return train_metrics

    if not cfg.trainer.get("limit_val_batches"):
        # csvlogger = CSVLogger(save_dir=f"cfg.save.log_dir")
        mlflow.set_tracking_uri(f"{cfg.mlflow.url}")
        mlflow.set_experiment(
            f"{cfg.datamodule.task_name}_{cfg.mlflow.experiment_name}"
        )
        # cli = LightningCLI()
        # if cli.trainer.global_rank == 0:
        if trainer.global_rank == 0:
            mlflow.pytorch.autolog(log_datasets=False)
            run_name = utils.get_run_name(cfg)
            mlflow.start_run(run_name=run_name, description="None")
        # description = OmegaConf.to_yaml(cfg)

        # with mlflow.start_run(run_name=run_name, description="None") as run:

        log.info("Starting training!")
        if cfg.get("train"):
            ckpt_path = cfg.paths.ckpt_dir
            if not os.path.exists(ckpt_path) or cfg.resume == False:
                ckpt_path = None
            trainer.fit(model=pl_module, datamodule=datamodule, ckpt_path=ckpt_path)
            train_metrics = trainer.callback_metrics
            res = trainer.predict(model=pl_module, datamodule=datamodule)
            draw_path = utils.draw_subplot_res(cfg, run_name, res, row_num=5, col_num=3)
            if trainer.global_rank == 0:
                mlflow.log_artifact(draw_path, artifact_path="result_image")
            # log.info(f"Best model ckpt at {trainer.checkpoint_callback.best_model_path}")

        if cfg.get("test"):
            ckpt_path = cfg.paths.ckpt_dir
            ckpt_path = trainer.checkpoint_callback.best_model_path
            if ckpt_path == "":
                ckpt_path = None
            trainer.test(model=pl_module, datamodule=datamodule, ckpt_path=ckpt_path)
            test_metrics = trainer.callback_metrics

        if trainer.global_rank == 0:
            mlflow.end_run()
        metric_dict = {**train_metrics, **test_metrics}
        return metric_dict


@hydra.main(version_base="1.2", config_path="./configs", config_name="train.yaml")
def main(cfg: DictConfig):
    # seq_len = cfg.datamodule.dataset.seq_len
    OmegaConf.update(cfg, "pl_module.model.seq_len", cfg.datamodule.dataset.seq_len)

    metric_dict = execute(cfg)
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )
    return metric_value


if __name__ == "__main__":
    main()
