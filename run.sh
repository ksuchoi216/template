#!/bin/bash
export HYDRA_FULL_ERROR=1
# * experiment_name
experiment_name="pv_dacon"
experiment_sub_name="modeling"
# * options
datamodule="pv_dacon"
runner="lstm"
# run_name_extra="lstm_num_layers/add_timefeat/add_past/bidirectional/has_cnn/cnn_n_blocks/has_time_decomp/has_freq_decomp"
run_name_extra="lstm_num_layers/add_timefeat/add_past/bidirectional/has_cnn/cnn_n_blocks/has_time_decomp/has_freq_decomp"
log_datasets="True"
log_models="True"

# * runner option
max_epochs=30

# * condition
istest=0
ismultigpustest=0
ismultigpus=0
ishp=0

if [ $istest -eq 1 ]; then
    echo ">>>>>> test[$runner][$datamodule] <<<<<<<<"
    python main.py \
        memory.experiment_name=$experiment_name \
        memory.datamodule=$datamodule \
        memory.runner=$runner \
        memory.experiment_sub_name=$experiment_sub_name \
        memory.printout="True" \
        memory.run_name_extra=$run_name_extra \
        mlflow.log_datasets=$log_datasets \
        mlflow.log_models=$log_models \
        datamodule=$datamodule \
        trainer=test \
        runner=$runner \
        run_mode.test_run="True"
elif [ $ishp -eq 1 ]; then
    echo ">>>>>>> hp search[$runner][$datamodule] <<<<<<<"
    python main.py --multirun \
        memory.experiment_name=$experiment_name \
        memory.datamodule=$datamodule \
        memory.runner=$runner \
        memory.experiment_sub_name=$experiment_sub_name \
        memory.printout="False" \
        memory.run_name_extra=$run_name_extra \
        datamodule=$datamodule \
        runner=$runner \
        run_mode.hp_search="True" \
        run_mode.pred="False" \
        trainer=hp_search \
        trainer.max_epochs=$max_epochs \
        hp_search=$runner
elif [ $ismultigpustest -eq 1 ]; then
    echo ">>>>>> multi gpus test[$runner][$datamodule] <<<<<<<<"
    python main.py \
        memory.experiment_name=$experiment_name \
        memory.datamodule=$datamodule \
        memory.runner=$runner \
        memory.experiment_sub_name=$experiment_sub_name \
        memory.printout="True" \
        memory.run_name_extra=$run_name_extra \
        mlflow.log_datasets=$log_datasets \
        mlflow.log_models=$log_models \
        datamodule=$datamodule \
        runner=$runner \
        trainer=test_multigpus \
        run_mode.test_run="True"
elif [ $ismultigpus -eq 1 ]; then
    echo ">>>>>>> multi gpus[$runner][$datamodule] <<<<<<<"
    python main.py \
        memory.experiment_name=$experiment_name \
        memory.datamodule=$datamodule \
        memory.runner=$runner \
        memory.experiment_sub_name=$experiment_sub_name \
        memory.printout="False" \
        memory.run_name_extra=$run_name_extra \
        mlflow.log_datasets=$log_datasets \
        mlflow.log_models=$log_models \
        datamodule=$datamodule \
        runner=$runner \
        trainer=multigpus \
        trainer.max_epochs=$max_epochs
else
    echo ">>>>>>> single gpu[$runner][$datamodule] <<<<<<<<<<"
    python main.py \
        memory.experiment_name=$experiment_name \
        memory.datamodule=$datamodule \
        memory.runner=$runner \
        memory.experiment_sub_name=$experiment_sub_name \
        memory.printout="False" \
        memory.run_name_extra=$run_name_extra \
        mlflow.log_datasets=$log_datasets \
        mlflow.log_models=$log_models \
        datamodule=$datamodule \
        runner=$runner \
        trainer.max_epochs=$max_epochs
fi
