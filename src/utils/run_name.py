def get_run_name(cfg):
    run_name = ""
    added_name = cfg.mlflow.added_name
    if added_name is None:
        added_name = ""

    cfgpl = cfg.pl_module
    cfgm = cfg.pl_module.model
    task_name = cfg.datamodule.task_name
    if cfgpl.name == "LSTM":
        run_name = f"{task_name}_{cfgpl.name}{added_name}_lr{cfgpl.optimizer.lr}_ip{cfgm.input_size}_sl{cfgm.seq_len}_pl{cfgm.pred_len}_bi{cfgm.bidirectional}"
    elif cfgpl.name == "LSTM-CNN" or cfgpl.name == "LSTM-CNN-Hybrid":
        run_name = f"{task_name}_{cfgpl.name}{added_name}_lr{cfgpl.optimizer.lr}_ip{cfgm.input_size}_sl{cfgm.seq_len}_pl{cfgm.pred_len}_bi{cfgm.bidirectional}_in{cfgm.input_size}_co{cfgm.cnn_out_channel}_ks{cfgm.kernel_size}_cl{cfgm.n_layer_cnn}_ll{cfgm.n_layer_lstm}"
    elif cfgpl.name == "Linear" or cfgpl.name == "DLinear" or cfgpl.name == "NLinear":
        run_name = f"{task_name}_{cfgpl.name}{added_name}_lr{cfgpl.optimizer.lr}_ip{cfgm.input_size}_sl{cfgm.seq_len}_pl{cfgm.pred_len}_in{cfgm.input_size}_i{cfgm.individual}"
    elif cfgpl.name == "TSMixer":
        run_name = f"{task_name}_{cfgpl.name}{added_name}_lr{cfgpl.optimizer.lr}_sl{cfgm.seq_len}_pl{cfgm.pred_len}_d{cfgm.input_size}_h{cfgm.hidden_size}_n{cfgm.n_block}"
    elif cfgpl.name == "TiDE":
        run_name = f"{task_name}_{cfgpl.name}{added_name}_lr{cfgpl.optimizer.lr}_sl{cfgm.seq_len}_pl{cfgm.pred_len}_d{cfgm.input_size}"
    elif cfgpl.name == "Autoformer":
        run_name = f"{task_name}_{cfgpl.name}{added_name}_lr{cfgpl.optimizer.lr}_sl{cfgm.seq_len}_pl{cfgm.pred_len}_d{cfgm.d_model}_el{cfgm.e_layers}_dl{cfgm.d_layers}_ma{cfgm.moving_avg}"
    elif cfgpl.name == "SOCNN":
        run_name = f"{task_name}_{cfgpl.name}{added_name}_lr{cfgpl.optimizer.lr}_sl{cfgm.seq_len}_pl{cfgm.pred_len}_d{cfgm.input_size}_k{cfgm.kernel_size}_ls{cfgm.n_layer_sign}_lo{cfgm.n_layer_offset}"
    elif cfgpl.name == "DA-RNN":
        run_name = f"{task_name}_{cfgpl.name}{added_name}_lr{cfgpl.optimizer.lr}_sl{cfgm.seq_len}_pl{cfgm.pred_len}_el{cfgm.encoder_hidden_size}_dl{cfgm.decoder_hidden_size}"
    print("=" * 96)
    print(f"|| run_name: {run_name:80} ||")
    print("=" * 96)
    return run_name
