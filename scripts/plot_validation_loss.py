import datetime
import json
import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.axes import Axes

from . import naming
from .utils import main_logger

logger = main_logger.getChild(__name__)


#########################
## Shared Plot Options ##
#########################


alpha: float = 0.8
fill: bool = True
capsize: float = 10.0
rotate_ytick: bool = False
raw_label: bool = True
fontsize: float = 26
num_bins: int = 100
nospace: bool = True
legend_fontsize: float = 16
legend_out: bool = False

markersize: float = 12.0

# manual_adjust: bool = False
manual_adjust: bool = False
adjust_left: float = 0.125
adjust_right: float = 0.9
adjust_top: float = 0.9
adjust_bottom: float = 0.1
adjust_wspace: float = 0.1
adjust_hspace: float = 0.1

# tight_layout option
tightlayout_pad: float = 0.3
tightlayout_w_pad: float = 0.3
tightlayout_h_pad: float = 0.3

# Set colors and markers to use.
colors: list[str] = [
    "gold",
    "magenta",
    "tab:brown",
    "tab:green",
    "tab:pink",
    "tab:purple",
    "tab:blue",
    "tab:orange",
    "tab:gray",
    "tab:red",
    "tab:gray",
]

markers: list[str] = ["o", "v", "^", "s", "d"]

# Set fontsize and adjust subplots.
plt.rcParams["font.size"] = fontsize

if nospace:
    plt.rcParams["savefig.pad_inches"] = 0


# Ad hoc
def format_yticks(x: float, pos: float) -> str:
    if x <= 1:
        return f"{x:.2f}"
    else:
        return f"{x:3g}"


########################
## Parameters to Vary ##
########################


@dataclass
class ExecParams:
    ## Parameters to vary.

    # Setting name (e.g., model name, beam search options...)
    setting_key: str

    # Model
    train_setting_key: str

    # Datasets
    dataset_key_l: list[str]

    # Measure
    measure_key: str
    measure_name: str

    figsize: tuple[float, float]


# Datasets
# dataset_key -> dataset_name
dataset_key_name_dict: dict[str, str] = {
    "ptb": "English",
    "ctb": "Chinese",
    "spmrl_french": "French",
    "spmrl_german": "German",
    "spmrl_korean": "Korean",
    "spmrl_basque": "Basque",
    "spmrl_hebrew": "Hebrew",
    "spmrl_hungarian": "Hungarian",
    "spmrl_polish": "Polish",
    "spmrl_polish_remove_lowest": "Polish-additional",
    "spmrl_swedish": "Swedish",
}

measure_key_name_dict: dict[str, str] = {
    "gold_joint_neg_ll": r"Validation Loss ($\downarrow$)",
}

measure_key_l: list[str] = [
    "gold_joint_neg_ll",
]


random_seeds: list[int] = [1111, 2222, 3333]  # wisteria


beam_options: list[tuple[int, int, int]] = [
    # (beam_size, min_shift_size, inference_batch_size)
    # We set min_shift_size 1/50 of beam size.
    (50, 1, 320),
    (200, 4, 80),
    (800, 16, 20),
]

# Train/Eval setting keys.

train_other_params: dict[str, dict[str, Any]] = {"default_setting": {}}

eval_other_params: dict[str, dict[str, Any]] = {
    f"beam_{beam_size}-shift_{min_shift_size}": {
        "beam_size": beam_size,
        "min_shift_size": min_shift_size,
        "inference_batch_size": batch_size,
    }
    for beam_size, min_shift_size, batch_size in beam_options
}


# Strategy.

# Add left-n-corner.
n_corner_l: list[int] = [1, 2, 3]

left_n_corner_key_l: list[str] = ["top-down"] + [f"left-{n}-corner" for n in n_corner_l] + ["bottom-up"]
left_n_corner_name_l: list[str] = [str(0)] + [str(n) for n in n_corner_l] + ["inf"]


# Add uniform speculation.
real_pos_l: list[float] = [0.26, 0.35, 0.65, 0.74]

uniform_spec_key_l = ["top-down"] + [f"uniform-spec-{real_pos}" for real_pos in real_pos_l] + ["bottom-up"]
# uniform_spec_name_l = ["0.0"] + [str(real_pos) for real_pos in real_pos_l] + ["0.99"]
uniform_spec_name_l = [".0"] + [str(real_pos)[1:] for real_pos in real_pos_l] + [".99"]  # Remove the first 0


# Add local-first.
# Note that the order is descending.
height_l: list[int] = [3, 2, 1]

local_first_key_l = ["top-down"] + [f"local-first-{height}" for height in height_l] + ["bottom-up"]
local_first_name_l = ["inf"] + [str(height) for height in height_l] + ["0"]


# Add global-first.
depth_l: list[int] = [3, 2, 1]

global_first_key_l = ["top-down"] + [f"global-first-{depth}" for depth in depth_l] + ["bottom-up"]
global_first_name_l = ["inf"] + [str(depth) for depth in depth_l] + ["-1"]

# Strategy groups to plot.
strategy_group_key_l: list[str] = ["left-n-corner", "uniform-spec", "local-first", "global-first"]
strategy_group_name_l: list[str] = strategy_group_key_l

# strategy_group_key -> xlabel
strategy_xlabel: dict[str, str] = {
    "left-n-corner": r"$n$",
    "uniform-spec": r"$\theta$",
    "local-first": r"$h$",
    "global-first": r"$d$",
}

# Other settings (e.g., model hyperparameters, beam search options...)
# This is useful for iterating over complex combination of settings (which cannot be done by simple for loops).
# setting_key -> specific setting dict.
plot_other_params: dict[str, dict[str, Any]] = {
    "all_dataset_plot": {
        "dataset_key_l": [
            "ptb",
            "ctb",
            "spmrl_basque",
            "spmrl_french",
            "spmrl_german",
            "spmrl_hebrew",
            "spmrl_hungarian",
            "spmrl_korean",
            "spmrl_polish",
            # "spmrl_polish_remove_lowest",  # Use the same source as normal polish.
            "spmrl_swedish",
        ],
        "figsize": (4 * 4, 4.5 * 10),
    },
    "sub_dataset_plot": {"dataset_key_l": ["ptb", "ctb", "spmrl_german", "spmrl_korean"], "figsize": (4 * 4, 4.5 * 4)},
    "polish_remove_lowest_plot": {
        "dataset_key_l": ["spmrl_polish_remove_lowest"],
        "figsize": (4 * 4, 4.5 * 1.2),
    },
}

#######################################
## Paths, Device and Script Settings ##
#######################################

base_dir: Path = Path("../tmp")

# Device setting
num_parallel: int = 10
use_gpu: bool = False
gpu_ids: list[int] = []

debug: bool = True
# debug: bool = False

# Used for log file name for each subprocess.
log_time: str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

# Set debug params.
if debug:
    base_dir = Path("../debug")
    # dataset_key_l = ["debug_data"]

    dataset_key_name_dict: dict[str, str] = {
        "debug_data": "Debug",
    }

    measure_key_l: list[str] = [
        "gold_joint_neg_ll",
    ]

    random_seeds: list[int] = [9999999, 888888]

    beam_options: list[tuple[int, int, int]] = [
        # (beam_size, min_shift_size, inference_batch_size)
        # We set min_shift_size 1/50 of beam size.
        (2, 1, 5),
        (4, 1, 5),
    ]

    train_other_params: dict[str, dict[str, Any]] = {
        "debug_default": {
            # Model hyperparameters
            "stack_size": 80,  # Beamsearching top-down with subwords requries larger stack_size and max_comp_nodes for the debug data. (when a model is trained well, stack_size and max_comp_nodes do not need to be as large as this.)
            # "stack_size": 20,
            "larger_stack_size": 150,  # Beamsearching top-down with subwords requries larger stack_size and max_comp_nodes for the debug data. (when a model is trained well, stack_size and max_comp_nodes do not need to be as large as this.)
            "w_dim": 7,
            "h_dim": 9,
            "num_layers": 2,
            "regularizer": "dropout",
            # Supervised train hyperparameters
            "num_epochs": 2,
            "num_steps": 10,
            "print_every": 1,
            "valid_every": 10,
            "val_batch_size": 3,
        }
    }
    eval_other_params: dict[str, dict[str, Any]] = {
        f"beam_{beam_size}-shift_{min_shift_size}": {
            "beam_size": beam_size,
            "min_shift_size": min_shift_size,
            "inference_batch_size": batch_size,
            "word_sync_step_limit": 3,
            "inference_block_size": 2,
            "eval_filter_by_parsability": False,  # This optioned should be False for debug data, since word_sync_step_limit may be too tight.
        }
        for beam_size, min_shift_size, batch_size in beam_options
    }

    # Set strategies.
    # strategy_params: dict[str, dict[str, Any]]: strategy_key -> {func_key:, params:}
    strategy_params: dict[str, dict[str, Any]] = {
        "top-down": {"func_key": "top-down", "params": {}},
        "bottom-up": {"func_key": "bottom-up", "params": {}},
        "left-1-corner": {"func_key": "left-n-corner", "params": {"n": 1}},
        "uniform-spec-0.26": {"func_key": "uniform-speculation", "params": {"real_pos": 0.26}},
        "local-first-1": {"func_key": "local-first", "params": {"height": 1}},
        "global-first-1": {"func_key": "global-first", "params": {"depth": 1}},
    }

    # Add left-n-corner.
    n_corner_l: list[int] = [1, 1, 1]

    left_n_corner_key_l: list[str] = ["top-down"] + [f"left-{n}-corner" for n in n_corner_l] + ["bottom-up"]
    left_n_corner_name_l: list[str] = [str(0)] + [str(n) for n in n_corner_l] + ["inf"]

    # Add uniform speculation.
    real_pos_l: list[float] = [0.26, 0.26, 0.26, 0.26]

    uniform_spec_key_l = ["top-down"] + [f"uniform-spec-{real_pos}" for real_pos in real_pos_l] + ["bottom-up"]
    uniform_spec_name_l = ["0.0"] + [str(real_pos) for real_pos in real_pos_l] + ["0.99"]

    # Add local-first.
    # Note that the order is descending.
    height_l: list[int] = [1, 1, 1]

    local_first_key_l = ["top-down"] + [f"local-first-{height}" for height in height_l] + ["bottom-up"]
    local_first_name_l = ["inf"] + [str(height) for height in height_l] + ["0"]

    # Add global-first.
    depth_l: list[int] = [1, 1, 1]

    global_first_key_l = ["top-down"] + [f"global-first-{depth}" for depth in depth_l] + ["bottom-up"]
    global_first_name_l = ["inf"] + [str(depth) for depth in depth_l] + ["-1"]

    # Strategy groups to plot.
    strategy_group_key_l: list[str] = ["left-n-corner", "uniform-spec", "local-first", "global-first"]
    strategy_group_name_l: list[str] = strategy_group_key_l

    # Script setting.

    # num_parallel: int = 1
    num_parallel: int = 3

    # use_gpu: bool = True
    use_gpu: bool = False
    gpu_ids: list[int] = []

    plot_other_params: dict[str, dict[str, Any]] = {"debug_plot": {"dataset_key_l": ["debug_data", "debug_data"]}}

# Arrange data.

# figsize: tuple[float, float] = (figsize_x, figsize_y)

measure_name_l: list[str] = [measure_key_name_dict[key] for key in measure_key_l]

train_setting_key_l: list[str] = list(train_other_params.keys())
eval_setting_key_l: list[str] = list(eval_other_params.keys())

beam_opt_key_l: list[str] = [f"beam-{bs}" for bs, _, _ in beam_options]
beam_opt_name_l: list[str] = beam_opt_key_l

assert len(eval_setting_key_l) == len(beam_opt_key_l)

strategy_keys: list[list[str]] = []
strategy_names: list[list[str]] = []
for strategy_group_key in strategy_group_key_l:
    if strategy_group_key == "left-n-corner":
        strategy_keys.append(left_n_corner_key_l)
        strategy_names.append(left_n_corner_name_l)

    elif strategy_group_key == "uniform-spec":
        strategy_keys.append(uniform_spec_key_l)
        strategy_names.append(uniform_spec_name_l)

    elif strategy_group_key == "local-first":
        strategy_keys.append(local_first_key_l)
        strategy_names.append(local_first_name_l)

    elif strategy_group_key == "global-first":
        strategy_keys.append(global_first_key_l)
        strategy_names.append(global_first_name_l)
    else:
        raise Exception(f"No such strategy {strategy_group_key}")


###############
## Functions ##
###############


def gen_execparams() -> Generator[ExecParams, None, None]:
    """Generate parameters for exec_single by varying parameters."""

    for setting_key in plot_other_params:
        for measure_key, measure_name in zip(measure_key_l, measure_name_l):
            for train_setting_key in train_setting_key_l:
                yield ExecParams(
                    setting_key=setting_key,
                    train_setting_key=train_setting_key,
                    measure_key=measure_key,
                    measure_name=measure_name,
                    **plot_other_params[setting_key],
                )


def exec_single(params: ExecParams) -> None:
    """Function to be executed in parallel."""

    pid = multiprocessing.current_process().pid

    print(f"{pid=}: Start plotting for {params=}")

    all_res: dict[tuple[str, str, int], float] = dict()

    # First, load data.
    for dataset_key in params.dataset_key_l:
        for strategy_key_l in strategy_keys:
            for strategy_key in strategy_key_l:
                for eval_setting_key, beam_opt_key in zip(eval_setting_key_l, beam_opt_key_l):
                    for seed in random_seeds:
                        gold_plot_key = (dataset_key, strategy_key, seed)

                        eval_output_file: Path = naming.get_eval_file(
                            base_dir=base_dir,
                            treebank_key=dataset_key,
                            strategy=strategy_key,
                            eval_setting_key=eval_setting_key,
                            train_setting_key=params.train_setting_key,
                            train_seed=seed,
                        )

                        with eval_output_file.open(mode="r") as f:
                            tmp_res_dict = json.load(f)

                            # Get the measure to plot.
                            val = tmp_res_dict[params.measure_key]

                            # Since different beam options have the same gold performance (because gold performance is not relavant to beam search), just check if the results are really the same.
                            if gold_plot_key in all_res:
                                assert all_res[gold_plot_key] == val
                            else:
                                all_res[gold_plot_key] = val

    # Plot
    dataset_name_l = [dataset_key_name_dict[dataset_key] for dataset_key in params.dataset_key_l]

    save_filepath = naming.get_plot_val_loss_file(
        base_dir=base_dir, setting_key=params.setting_key, measure_key=params.measure_key
    )
    save_filepath.parent.mkdir(parents=True, exist_ok=True)

    # Set row/colmn size.
    nrows = len(params.dataset_key_l)
    ncols = len(strategy_keys)

    # Create fig and axes.
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=params.figsize,
        sharex=False,
        sharey="row",
        squeeze=False,
        constrained_layout=True,
    )

    # Adjust.
    if manual_adjust:
        fig.subplots_adjust(
            left=adjust_left,
            right=adjust_right,
            top=adjust_top,
            bottom=adjust_bottom,
            wspace=adjust_wspace,
            hspace=adjust_hspace,
        )

    # Set common x/y-labels
    fig.suptitle(f"{params.measure_name}", y=1.0, verticalalignment="bottom")

    # Plot.
    for row_i in range(nrows):
        dataset_key = params.dataset_key_l[row_i]
        dataset_name = dataset_name_l[row_i]

        for col_i in range(ncols):
            strategy_group_key = strategy_group_key_l[col_i]
            strategy_group_name = strategy_group_name_l[col_i]
            strategy_key_l: list[str] = strategy_keys[col_i]

            ax = axes[row_i][col_i]
            assert isinstance(ax, Axes)

            # Set yaxis formatter.
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_yticks))

            # Set ax title..
            if row_i == 0:
                ax.set_title(strategy_group_name)

            # Set xlabel.
            if row_i == nrows - 1:
                ax.set_xlabel(strategy_xlabel[strategy_group_key_l[col_i]])

            # Set xticks.
            xticks = list(range(len(strategy_key_l)))
            ax.set_xticks(xticks)
            ax.set_xticklabels(strategy_names[col_i])

            # Set ylabel.
            if col_i == 0:
                ax.set_ylabel(f"{dataset_name_l[row_i]}")

            # Get the data
            data: list[list[float]] = []
            for strategy_key in strategy_key_l:
                seed_res: list[float] = [all_res[(dataset_key, strategy_key, seed)] for seed in random_seeds]

                data.append(seed_res)

            # Calcualte mean and standard error.
            y_mean = np.mean(data, axis=1)
            y_std = np.std(data, axis=1)
            y_stderr = y_std / np.sqrt(len(random_seeds))

            ax.errorbar(
                x=xticks,
                y=y_mean,
                yerr=y_stderr,
                marker=markers[0],
                color=colors[0],
                label=None,
                markersize=markersize,
                alpha=alpha,
                capsize=capsize,
            )

    # Save the plot.
    if not manual_adjust:
        # fig.tight_layout(pad=tightlayout_pad, w_pad=tightlayout_w_pad, h_pad=tightlayout_h_pad)
        pass

    fig.savefig(str(save_filepath), bbox_inches="tight")

    print(f"{pid=}: Finish plotting for {params=}")


def main():
    logger.info("Start plot!!!")

    with multiprocessing.Pool(processes=num_parallel) as executer:
        # Pass the seed and hparams.
        executer.map(
            exec_single,
            gen_execparams(),
        )

    logger.info("Finish plot!!!")


if __name__ == "__main__":
    main()
