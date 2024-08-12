import os
import pickle
import shutil
import sys
import test
from datetime import datetime
from urllib import parse

import hydra
import joblib
import lightning as L
import numpy as np

# import onnx
import torch
import wandb
from easydict import EasyDict as edict
from einops import rearrange
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from torch import inference_mode

from src import utils

log = utils.get_pylogger(__name__)


class ModelDeployment:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model_test = cfg.deployment.model_test

    def get_paths(self, model_num):
        cfgd = edict(self.cfg.deployment)
        deployment_dir = cfgd.deployment_dir  # type: ignore
        model_name = f"model{model_num}"
        model_key = cfgd.model_key  # type: ignore
        model_dic = cfgd[model_name]
        org_scaler_path = model_dic.scaler
        wandb_path = model_dic[model_key]
        test_data_dir = cfgd.test_data_dir  # type: ignore

        new_model_name = f"{model_name}"

        path_dic = edict(
            save_ckpt=deployment_dir,
            model_ckpt=f"{deployment_dir}/model.ckpt",
            org_scaler=org_scaler_path,
            model_pt=f"{deployment_dir}/models/{new_model_name}.pt",
            new_scaler=f"{deployment_dir}/scalers/{new_model_name}_scaler.pkl",
            wandb=wandb_path,
            test_data=test_data_dir,
        )

        for k, v in path_dic.items():
            print(f"{k}: {v}")

        return path_dic

    def load_model_from_ckpt(
        self,
        path_dic,
        delete_ckpt_file=True,
        save_model=True,
        save_torchscript=True,
        copy_scaler=True,
        return_value=False,
    ):
        wandb_path = path_dic.wandb
        model_ckpt_path = path_dic.model_ckpt
        save_ckpt_dir = path_dic.save_ckpt

        run = wandb.init()
        print(f"wandb_path: {wandb_path}")
        artifact = run.use_artifact(wandb_path, type="model")  # type: ignore
        save_ckpt_dir = artifact.download(save_ckpt_dir)
        print(f"downloaded artifact to: {save_ckpt_dir}")
        run.finish()  # type: ignore

        model = hydra.utils.instantiate(self.cfg.exp.model)
        checkpoint = torch.load(model_ckpt_path)
        ckpt = checkpoint["state_dict"]

        model_state_dict_keys = list(model.state_dict().keys())
        # print(f"model_state_dict_keys: {model_state_dict_keys}")
        __ckpt = {}

        for k, v in ckpt.items():
            k = k.replace("model.", "")
            __ckpt[k] = v

        # print(f"__ckpt: {__ckpt.keys()}")
        model.load_state_dict(__ckpt)
        model.eval()

        if delete_ckpt_file:
            os.remove(model_ckpt_path)
            print(f"deleted ckpt file: {model_ckpt_path}")

        if save_model:
            model_pt_path = path_dic.model_pt
            if save_torchscript:
                model = torch.jit.script(model)  # TorchScript 형식으로 내보내기
                torch.jit.save(model, model_pt_path)
                # model.save("model.pt")  # 저장하기
            else:
                torch.save(model, model_pt_path)
            print(f"model saved as pt: {model_pt_path}")

        if copy_scaler:
            org_scaler_path = path_dic.org_scaler
            new_scaler_path = path_dic.new_scaler
            shutil.copy(org_scaler_path, new_scaler_path)
            print(f"copied scaler to: {new_scaler_path}")

            scaler = joblib.load(new_scaler_path)

        model = torch.jit.load(model_pt_path)
        model.eval()

        if return_value:
            return model, scaler
        else:
            return 0

    def forward(self, model_num):
        path_dic = self.get_paths(model_num)

        log.info(f"loading model from ckpt...")
        self.load_model_from_ckpt(path_dic)


@hydra.main(version_base="1.2", config_path="./configs", config_name="train.yaml")
def main(cfg: DictConfig):
    model_num = cfg.deployment.model_num
    wandb.require("core")

    log.info(f"updating cfg...")
    cfg = utils.update_cfg(cfg)

    modeldeployment = ModelDeployment(cfg)
    modeldeployment.forward(model_num)


if __name__ == "__main__":
    main()

# def test_model(self, model_pt_path, torchscript=False):
#     x = torch.randn(1, self.cfg.datamodule.cfg.seq_len, 8)

#     if torchscript:
#         model = torch.jit.load(model_pt_path)
#         model.eval()
#         print(model)
#     else:
#         model = torch.load(model_pt_path)
#         model.eval()
#     y = model(x)
#     print(y)
