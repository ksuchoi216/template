import os
import sys

from omegaconf import OmegaConf


def get_run_name(cfg):
    cfgm = cfg.runner.model.cfg
    cfgdm = cfg.datamodule.cfg
    cfgr = cfg.runner

    run_name = (
        f"[{cfg.names.runner}]"
        + f"[{cfg.names.datamodule}]"
        + f"[{cfgr.criterion.selection}]"
        + f"dim{cfgm.input_dim}"
    )
    return run_name


def update_cfg(cfg):
    project_name = f"{cfg.names.project}-{cfg.names.exp}"
    run_name = get_run_name(cfg)
    print(f"[cfgupdater] project_name: {project_name}")
    print(f"[cfgupdater] run_name: {run_name}")
    # * Wandb logger
    if cfg.run_mode.test_run:
        monitor = cfg.runner.setup.monitor.replace("val", "train")
        OmegaConf.update(cfg, "runner.setup.monitor", monitor)
        OmegaConf.update(cfg, "callbacks.model_checkpoint.monitor", monitor)
        OmegaConf.update(cfg, "callbacks.early_stopping.monitor", monitor)
    else:
        OmegaConf.update(cfg, "trainer.logger.project", project_name)
        OmegaConf.update(cfg, "trainer.logger.name", run_name)
        OmegaConf.update(cfg, "runner.info.run_name", run_name)

    # sys.exit()
    return cfg
