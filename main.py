import os
import pickle
import sys
import warnings
from datetime import datetime
from pprint import pprint
from urllib import parse

import hydra
import lightning as L
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from src import utils

# from torch import inference_mode


warnings.filterwarnings("ignore")
log = utils.get_pylogger(__name__)
# from src.utils import organize_pred_dics, plot_pred_dic


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
    # sys.exit()
    mlflow.set_tracking_uri(f"{cfg.mlflow.tracking_uri}")
    experiment_name = cfg.mlflow.experiment_name

    client = mlflow.tracking.MlflowClient()
    # Get the experiment details
    experiment = client.get_experiment_by_name(experiment_name)
    print(f'Experiment "{experiment_name}" details: {experiment}')
    # Check if the experiment exists and is deleted
    if experiment is not None and experiment.lifecycle_stage == "deleted":
        # Permanently delete the experiment
        client.restore_experiment(experiment.experiment_id)
        # client.delete_experiment(experiment.experiment_id)
        # print(f"Experiment '{experiment_name}' permanently deleted.")

    mlflow.set_experiment(experiment_name)
    mlflow.pytorch.autolog(
        log_datasets=cfg.mlflow.log_datasets,
        log_models=cfg.mlflow.log_models,
        checkpoint_monitor=cfg.runner.setup.monitor,
        checkpoint_mode=cfg.runner.setup.mode,
        checkpoint_save_best_only=True,
    )

    with mlflow.start_run(
        run_name=cfg.mlflow.run_name,
        description=cfg.mlflow.run_description,
    ) as run:
        if cfg.run_mode.test_run:
            log.info(f"[main] initial testing ...")

            datamodule.setup("fit")
            train_dataloader = datamodule.train_dataloader()
            batch = next(iter(train_dataloader))
            runner.training_step(batch, 1)

            trainer.fit(model=runner, datamodule=datamodule)
            train_metrics = trainer.callback_metrics
            print(f"train_metrics: {train_metrics}")

            if cfg.run_mode.get("test"):
                trainer.test(model=runner, datamodule=datamodule)

            if cfg.run_mode.get("pred"):
                pred_dics = trainer.predict(model=runner, datamodule=datamodule)
                organized_pred_dics = organize_pred_dics(pred_dics)
                fig_path = plot_pred_dic(cfg, organized_pred_dics)
                mlflow.log_table(
                    data=organized_pred_dics,
                    artifact_file="predictions/predictions.json",
                )
                mlflow.log_artifact(fig_path, artifact_path="predictions")

            mlflow.end_run()
            # mlflow.delete_run(run.info.run_id)

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
                pred_dics = trainer.predict(model=runner, datamodule=datamodule)
                # organized_pred_dics = organize_pred_dics(pred_dics)
                # fig_path = plot_pred_dic(cfg, organized_pred_dics)
                # mlflow.log_table(
                #     data=organized_pred_dics,
                #     artifact_file="predictions/predictions.json",
                # )
                # mlflow.log_artifact(fig_path, artifact_path="predictions")

            # save config
            output_config_path = f"{cfg.paths.output_dir}/configs.yaml"
            with open(output_config_path, "w") as f:
                OmegaConf.save(cfg, f)
            mlflow.log_artifact(output_config_path, artifact_path="configs")

            # save dataset
            if cfg.mlflow.log_datasets:
                dataset_path = f"{cfg.paths.output_dir}/dataset"
                if not os.path.exists(dataset_path):
                    os.makedirs(dataset_path)
                # mlflow.log_artifact(dataset_path, artifact_path="dataset")

            mlflow.end_run()
            print(f"train_metrics: {train_metrics}")
            print(f"test_metrics: {test_metrics}")
            metric_dict = {**train_metrics, **test_metrics}
            return metric_dict


@hydra.main(version_base="1.2", config_path="./configs", config_name="train.yaml")
def main(cfg: DictConfig):
    # wandb.require("core")

    cfg = utils.update_cfg(cfg)
    metric_dict = execute(cfg)
    if cfg.run_mode.hp_search:
        metric_value = utils.get_metric_value(
            metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")  # type: ignore
        )
        return metric_value


if __name__ == "__main__":
    main()  # type: ignore
