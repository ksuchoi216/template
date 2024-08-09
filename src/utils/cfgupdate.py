import os
import sys

from omegaconf import OmegaConf


def get_run_name(cfg):
    cfgm = cfg.runner.model.cfg
    cfgdm = cfg.datamodule.cfg
    cfgr = cfg.runner
    criterion_name = cfg.runner.criterion.selection

    filenum = cfgdm.filenum
    if filenum is not None:
        filenum = "".join([str(i) for i in filenum])
    else:
        filenum = "all"

    snum = cfgdm.snum
    if snum is not None:
        snum = "".join([str(i) for i in cfgdm.snum])
    else:
        snum = "all"

    run_name = cfg.names.run_name
    if run_name is None:
        run_name = ""
    else:
        run_name = f"[{run_name}]"

    datamodule_name = cfg.names.datamodule
    runner_name = cfg.names.runner
    run_name = (
        f"[{runner_name}]"
        + f"[{datamodule_name}]"
        + f"[{cfgr.criterion.selection}]"
        + f"f{filenum}_s{snum}_l{cfgm.num_layers}_dim{cfgm.input_dim}"
    )
    return run_name


def update_cfg(cfg):
    # * Callbacks
    OmegaConf.update(
        cfg, "callbacks.model_checkpoint.monitor", cfg.runner.setup.monitor
    )
    OmegaConf.update(cfg, "callbacks.model_checkpoint.mode", cfg.runner.setup.mode)
    OmegaConf.update(cfg, "callbacks.early_stopping.monitor", cfg.runner.setup.monitor)
    OmegaConf.update(cfg, "callbacks.early_stopping.mode", cfg.runner.setup.mode)

    print(f"[cfgupdater] cfg.run_mode.hp_search: {cfg.run_mode.hp_search}")
    # sys.exit()
    # * Wandb logger
    if not cfg.run_mode.test_run and not cfg.run_mode.hp_search:
        project_name = f"{cfg.names.project}-{cfg.names.exp}"
        run_name = get_run_name(cfg)
        OmegaConf.update(cfg, "trainer.logger.project", project_name)
        OmegaConf.update(cfg, "trainer.logger.name", run_name)
        OmegaConf.update(cfg, "runner.info.run_name", run_name)
        print(f"[cfgupdater] run_name: {run_name}")

    return cfg
