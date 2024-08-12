#!/bin/bash
export HYDRA_FULL_ERROR=1

# * names
project_name="forecasting"
exp_name="test"

# * selection
datamodule_name="base_electricity"
runner_name="lstm"

# * option
max_epochs=10

# * condition
istest=1
istest2=0
ismultigpustest=0
ismultigpus=0
ishp=0

if [ $istest -eq 1 ]; then
    echo ">>>>>> test[$runner_name][$datamodule_name] <<<<<<<<"
    python main.py \
        datamodule=$datamodule_name \
        runner=$runner_name \
        runner.model.cfg.printout="True" \
        trainer=test \
        run_mode.test_run="True" \
        names.project=$project_name \
        names.datamodule=$datamodule_name \
        names.runner=$runner_name \
        names.exp=$exp_name
elif [ $istest2 -eq 1 ]; then
    echo ">>>>>> test2[$runner_name][$datamodule_name] <<<<<<<<"
    python main.py \
        datamodule=$datamodule_name \
        runner=$runner_name \
        runner.model.cfg.printout="True" \
        trainer=test2 \
        run_mode.test_run="True"
elif [ $ismultigpustest -eq 1 ]; then
    echo ">>>>>> multi gpus test[$runner_name][$datamodule_name] <<<<<<<<"
    python main.py \
        datamodule=$datamodule_name \
        runner=$runner_name \
        runner.model.cfg.printout="True" \
        trainer=test_multigpus \
        run_mode.test_run="True"
elif [ $ishp -eq 1 ]; then
    echo ">>>>>>> hp search[$runner_name][$datamodule_name] <<<<<<<"
    python main.py --multirun \
        datamodule=$datamodule_name \
        runner=$runner_name \
        runner.model.cfg.printout="False" \
        run_mode.hp_search="True" \
        run_mode.pred="False" \
        trainer=hp_search \
        trainer.max_epochs=$max_epochs \
        hp_search=$runner_name
elif [ $ismultigpus -eq 1 ]; then
    echo ">>>>>>> multi gpus[$runner_name][$datamodule_name] <<<<<<<"
    python main.py \
        datamodule=$datamodule_name \
        runner=$runner_name \
        runner.model.cfg.printout="False" \
        trainer=multigpus \
        trainer.max_epochs=$max_epochs \
        names.project=$project_name \
        names.datamodule=$datamodule_name \
        names.runner=$runner_name \
        names.exp=$exp_name
else
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
fi
