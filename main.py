import os
import pickle
import sys
import warnings
from datetime import datetime
from pprint import pprint
from urllib import parse

import hydra
import lightning as L
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from src import utils

# from torch import inference_mode


warnings.filterwarnings("ignore", category=FutureWarning)

log = utils.get_pylogger(__name__)


def execute(cfg: DictConfig):
    if cfg.get("seed"):
        L.seed_everything(cfg.seed)

    log.info(f"datamodel {cfg.datamodule._target_}")
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"runner {cfg.runner._target_}")
    runner = hydra.utils.instantiate(cfg.runner)

    log.info(f"callbacks loading...")
    callbacks = utils.instantiate_callbacks(cfg.get("callbacks"))
    print(f"callbacks: {callbacks}")

    log.info(f"trainer {cfg.trainer._target_}")
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks)
    train_metrics, test_metrics = {}, {}

    if cfg.run_mode.test_run:
        log.info(f"[main] initial testing ...")
        datamodule.setup("fit")
        train_dataloader = datamodule.train_dataloader()
        batch = next(iter(train_dataloader))
        runner.training_step(batch, 1)

        trainer.fit(model=runner, datamodule=datamodule)
        train_metrics = trainer.callback_metrics
        print(f"train_metrics: {train_metrics}")
        trainer.test(model=runner, datamodule=datamodule)
        trainer.predict(model=runner, datamodule=datamodule)

        return train_metrics

    if not cfg.run_mode.test_run:
        log.info("[main] training ...")

        if cfg.run_mode.get("train"):
            ckpt_path = cfg.paths.ckpt_dir
            if not os.path.exists(ckpt_path) or cfg.run_mode.resume == False:
                ckpt_path = None
            trainer.fit(model=runner, datamodule=datamodule, ckpt_path=ckpt_path)
            train_metrics = trainer.callback_metrics

            # result = trainer.predict(model=runner, datamodule=datamodule)

        if cfg.run_mode.get("test"):
            ckpt_path = cfg.paths.ckpt_dir
            ckpt_path = trainer.checkpoint_callback.best_model_path
            if ckpt_path == "":
                ckpt_path = None
            trainer.test(model=runner, datamodule=datamodule)
            test_metrics = trainer.callback_metrics
            # runner.to_onnx("./model.onnx", datamodule, export_params=True)

        if cfg.run_mode.get("pred"):
            trainer.predict(model=runner, datamodule=datamodule)

        print(f"train_metrics: {train_metrics}")
        print(f"test_metrics: {test_metrics}")

        metric_dict = {**train_metrics, **test_metrics}
        return metric_dict


@hydra.main(version_base="1.2", config_path="./configs", config_name="train.yaml")
def main(cfg: DictConfig):
    wandb.require("core")

    cfg = utils.update_cfg(cfg)
    metric_dict = execute(cfg)
    if cfg.run_mode.hp_search:
        metric_value = utils.get_metric_value(
            metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
        )
        return metric_value


if __name__ == "__main__":
    main()
