import sys
import lightning as L
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def draw_subplot_res(cfg, run_name, res, row_num=4, col_num=4, **kwargs):
    draw_dir = cfg.paths.draw_dir
    seed = cfg.seed
    batch_size = cfg.datamodule.batch_size
    seq_len = cfg.datamodule.dataset.seq_len
    pred_len = cfg.datamodule.dataset.pred_len

    L.seed_everything(seed)

    res_len = len(res)
    # random sample index from a range (0, res_len)
    data_idx = np.random.choice(res_len, size=row_num, replace=False)
    batch_idx = np.random.choice(batch_size // 5, size=col_num, replace=False)
    # print(f"batch_idx: {batch_idx}")

    data_len = len(data_idx) * len(batch_idx)

    subplot_col_num = col_num
    if data_len % col_num == 0:
        subplot_row_num = data_len // subplot_col_num
    else:
        subplot_row_num = data_len // subplot_col_num + 1
    # print(f'data_len: {data_len}, subplot_row_num: {subplot_row_num}, subplot_col_num: {subplot_col_num}')
    fig, axs = plt.subplots(
        subplot_row_num, subplot_col_num, figsize=(20, 5 * subplot_row_num)
    )
    axs = axs.flatten()
    t = np.arange(seq_len + pred_len)
    cnt = 0
    for i, _data_idx in enumerate(data_idx):
        _res = res[_data_idx]
        for j, _batch_idx in enumerate(batch_idx):
            pred = _res["y_pred"][_batch_idx]
            true = _res["y_true"][_batch_idx]
            input = _res["input_x"][_batch_idx]

            axs[cnt].plot(t[seq_len:], pred, label="pred", color="red")
            axs[cnt].plot(t[seq_len:], true, label="true", color="deepskyblue")
            axs[cnt].plot(t[:seq_len], input, label="input", color="dodgerblue")
            axs[cnt].set_title(f"data_idx: {data_idx}, batch_idx:{_batch_idx}")
            axs[cnt].legend()

            if "xlabel" in kwargs:
                x_col = kwargs["xlabel"]
                axs[i].set_xlabel(x_col)
            if "ylabel" in kwargs:
                y_col = kwargs["ylabel"]
                axs[i].set_ylabel(y_col)
            if "xlim" in kwargs:
                xlim = kwargs["xlim"]
                axs[i].set_xlim(xlim[0], xlim[1])
            if "ylim" in kwargs:
                ylim = kwargs["ylim"]
                axs[i].set_ylim(ylim[0], ylim[1])
            cnt += 1

    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()
    # convert today to string
    today = datetime.now().strftime("%Y%m%d%H%M%S")
    # save matplotlib figure to local files

    draw_path = f"{draw_dir}/{today}_resplot_{run_name}.png"
    try:
        fig.savefig(draw_path)
        print(f"save plot to {draw_path} successfully")
    except Exception as e:
        print(f"Error: {e}")

    return draw_path
