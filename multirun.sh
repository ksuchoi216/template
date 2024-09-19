#!/bin/bash
export HYDRA_FULL_ERROR=1
# * project
project_name="pv_dacon"
exp_name="seq_length"
# * options
datamodule="pv_dacon"
runner="lstm"
extra_list="lstm_num_layers/add_timefeat/add_past/bidirectional/has_cnn/has_time_decomp/has_freq_decomp"

# * runner option
max_epochs=30

# * condition
istest=0
ismultigpustest=0
ismultigpus=0
ishp=0

# for loop for seq_len
for seq_len in 24 48 72 96 120 144 168 192 216 240; do
    echo ">>>>>>> single gpu[$runner][$datamodule][$seq_len] <<<<<<<<<<"
    python main.py \
        memory.project=$project_name \
        memory.datamodule=$datamodule \
        memory.runner=$runner \
        memory.exp=$exp_name \
        memory.printout="False" \
        memory.extra_list=$extra_list \
        datamodule=$datamodule \
        runner=$runner \
        trainer.max_epochs=$max_epochs \
        datamodule.cfg.seq_len=$seq_len
done
