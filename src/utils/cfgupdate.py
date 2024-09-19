import os
import sys

from omegaconf import OmegaConf


def get_run_name(cfg):
    cfgdm = cfg.datamodule.cfg
    cfgm = cfg.runner.model.cfg
    cfgr = cfg.runner

    run_name = (
        f"[{cfg.memory.runner}]"
        + f"[{cfg.memory.datamodule}]"
        + f"_lr{cfgr.optimizer.lr}"
        + f"_{cfgr.criterion.selection}"
        + f"_dim{cfgm.input_dim}"
    )

    if cfg.run_mode.test_run:
        run_name += "_test"
    else:
        run_name_extra = cfg.memory.run_name_extra
        run_name_extra = run_name_extra.split("/")
        for extra in run_name_extra:
            print(f"extra: {extra}")
            if extra in cfgm:
                run_name += f"_{extra}{cfgm[extra]}"

    return run_name


def get_data_path(cfg):
    cfgdm = cfg.datamodule.cfg
    data_dir = f"{cfgdm.data_dir}/{cfgdm.datafolder_name}"
    data_path = f"{data_dir}/{cfgdm.file_name}.csv"
    scaler_path = f"{data_dir}/scaler_{cfg.memory.datamodule}.pkl"

    return data_path, scaler_path


def setup_mlflow_cfg(cfg):
    experiment_name = f"{cfg.memory.experiment_name}-{cfg.memory.experiment_sub_name}"
    run_name = get_run_name(cfg)
    print(
        f"cfg.mlflow.log_datasets: {cfg.mlflow.log_datasets}, type: {type(cfg.mlflow.log_datasets)}"
    )
    print(f"cfg.mlflow.log_models: {cfg.mlflow.log_models}")
    OmegaConf.update(cfg, "mlflow.run_name", run_name)
    OmegaConf.update(cfg, "mlflow.experiment_name", experiment_name)
    OmegaConf.update(cfg, "mlflow.run_name", run_name)
    OmegaConf.update(cfg, "runner.info.run_name", run_name)
    print(f"[cfgupdater] experiment_name: {experiment_name}")
    print(f"[cfgupdater] run_name: {cfg.memory.run_name}")

    return cfg


def update_cfg(cfg):
    # * run & experiment name

    cfg = setup_mlflow_cfg(cfg)

    data_path, scaler_path = get_data_path(cfg)
    OmegaConf.update(cfg, "datamodule.cfg.data_path", data_path)
    OmegaConf.update(cfg, "datamodule.cfg.scaler_path", scaler_path)
    OmegaConf.update(cfg, "runner.info.scaler_path", scaler_path)
    print(f"[cfgupdater] data_path: {data_path}")
    print(f"[cfgupdater] scaler_path: {scaler_path}")

    # * test_run
    if cfg.run_mode.test_run:
        monitor = cfg.runner.setup.monitor.replace("val", "train")
        OmegaConf.update(cfg, "runner.setup.monitor", monitor)
        OmegaConf.update(cfg, "callbacks.model_checkpoint.monitor", monitor)
        OmegaConf.update(cfg, "callbacks.early_stopping.monitor", monitor)

    # * log model config to wandb (only datamodule config can be uploaded automatically)
    OmegaConf.update(cfg, "datamodule.cfg.cfg_model", cfg.runner.model.cfg)

    # sys.exit()
    return cfg
