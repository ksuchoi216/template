#!/bin/bash
export HYDRA_FULL_ERROR=1

# * names
project_name="forecasting"
exp_name="test"

# * option
max_epochs=10

# ! run ========================================
datamodule_name="base_electricity"
runner_name="lstm"
echo ">>>>>>> single gpu[$runner_name][$datamodule_name] <<<<<<<<<<"
python main.py \
    datamodule=$datamodule_name \
    runner=$runner_name \
    runner.model.cfg.printout="False" \
    trainer.max_epochs=$max_epochs \
    names.project=$project_name \
    names.datamodule=$datamodule_name \
    names.runner=$runner_name \
    names.exp=$exp_name
# ! =============================================
