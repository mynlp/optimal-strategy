import datetime
import json
import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator

import numpy as np

from . import naming
from .utils import main_logger

logger = main_logger.getChild(__name__)


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
    dataset_key_l_l: list[list[str]]  # multi-row plot.

    # Measure
    measure_key: str
    measure_name: str


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
    "spmrl_swedish": "Swedish",
}

measure_key_name_dict: dict[str, str] = {
    # Parsing performance
    "labeled_corpus_f1": r"Labeled Parsing F1 Score ($\uparrow$)",
    #
    # Language modeling performance
    "corpus_nll": r"Marginalized NLL ($\downarrow$)",  # Negative log likelihood.
    #
    # PPL marginalized over beams.
    "corpus_token_ppl": r"Marginalized Token PPL ($\downarrow$)",
    #
    # Structure-conditioned token probability of the best action and gold action.
    # Corpus level ppl.
    "structure_cond_token_ppl": r"Structure-conditioned Token PPL ($\downarrow$)",
    "joint_neg_ll": r"joint_neg_ll ($\downarrow$)",
    "action_only_neg_ll": r"action_only_neg_ll ($\downarrow$)",
    "token_only_neg_ll": r"token_only_neg_ll ($\downarrow$)",
}

# True if the measure is better when larger.
measure_key_larger_better_dict: dict[str, bool] = {
    # Parsing performance
    "labeled_corpus_f1": True,
    # Language modeling performance
    # PPL marginalized over beams.
    "corpus_token_ppl": False,
}
measure_key_l: list[str] = [
    # Parsing performance
    "labeled_corpus_f1",
    # Language modeling performance
    "corpus_token_ppl",
    #
]


def parsing_score_str(x: float) -> str:
    return f"{x:.1f}"


def ppl_str(x: float) -> str:
    return f"{x}"


measure_key_value_str_func: dict[str, Any] = {
    # Parsing performance
    "labeled_corpus_f1": parsing_score_str,
    # Language modeling performance
    # PPL marginalized over beams.
    "corpus_token_ppl": ppl_str,
}

# Scale some values for clarity.
measure_key_scale: dict[str, float] = {
    # Parsing performance
    "labeled_corpus_f1": 100.0,
    # Language modeling performance
    # PPL marginalized over beams.
    "corpus_token_ppl": 1.0,
}

plot_gold_action_value_measures: set[str] = set(
    [
        "structure_cond_token_ppl",
        "joint_neg_ll",
        "action_only_neg_ll",
        "token_only_neg_ll",
    ]
)

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

# strategy_params: dict[str, dict[str, Any]]: strategy_key -> {func_key:, params:}
strategy_params: dict[str, dict[str, Any]] = dict()
strategy_names: dict[str, str] = dict()

# Add top-down.
strategy_params["top-down"] = {"func_key": "top-down", "params": {}}
# strategy_names["top-down"] = "top-down"
strategy_names["top-down"] = "TD"

# Add bottom-up.
strategy_params["bottom-up"] = {"func_key": "bottom-up", "params": {}}
# strategy_names["bottom-up"] = "bottom-up"
strategy_names["bottom-up"] = "BU"

# Add left-n-corner.
n_corner_l: list[int] = [1, 2, 3]
for n in n_corner_l:
    strategy_params[f"left-{n}-corner"] = {"func_key": "left-n-corner", "params": {"n": n}}
    # strategy_names[f"left-{n}-corner"] = f"left-n-corner-{n}"
    strategy_names[f"left-{n}-corner"] = f"LC-{n}"

# Add uniform speculation.
real_pos_l: list[float] = [0.26, 0.35, 0.65, 0.74]
for real_pos in real_pos_l:
    strategy_params[f"uniform-spec-{real_pos}"] = {"func_key": "uniform-speculation", "params": {"real_pos": real_pos}}
    # strategy_names[f"uniform-spec-{real_pos}"] = f"uniform-spec-{real_pos}"
    strategy_names[f"uniform-spec-{real_pos}"] = f"US-{real_pos}"

# Add local-first.
height_l: list[int] = [1, 2, 3]
for height in height_l:
    strategy_params[f"local-first-{height}"] = {"func_key": "local-first", "params": {"height": height}}
    # strategy_names[f"local-first-{height}"] = f"local-first-{height}"
    strategy_names[f"local-first-{height}"] = f"LF-{height}"

# Add global-first.
depth_l: list[int] = [1, 2, 3]
for depth in depth_l:
    strategy_params[f"global-first-{depth}"] = {"func_key": "global-first", "params": {"depth": depth}}
    # strategy_names[f"global-first-{depth}"] = f"global-first-{depth}"
    strategy_names[f"global-first-{depth}"] = f"GF-{depth}"

# Get list of strategy keys.
strategies: list[str] = list(strategy_params.keys())

# Other settings (e.g., model hyperparameters, beam search options...)
# This is useful for iterating over complex combination of settings (which cannot be done by simple for loops).
# setting_key -> specific setting dict.
plot_other_params: dict[str, dict[str, Any]] = {
    "all_dataset_plot": {
        "dataset_key_l_l": [
            ["ptb", "ctb", "spmrl_french", "spmrl_german", "spmrl_korean"],
            ["spmrl_basque", "spmrl_hebrew", "spmrl_hungarian", "spmrl_polish", "spmrl_swedish"],
        ]
    }
}

topn_plot: int = 2

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

    dataset_key_name_dict: dict[str, str] = {
        "debug_data": "Debug",
    }

    measure_key_l: list[str] = [
        # Parsing performance
        "labeled_corpus_f1",
        # Language modeling performance
        # PPL marginalized over beams.
        "corpus_token_ppl",
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
    strategies: list[str] = list(strategy_params.keys())

    # Script setting.

    # num_parallel: int = 1
    num_parallel: int = 3

    # use_gpu: bool = True
    use_gpu: bool = False
    gpu_ids: list[int] = []

    plot_other_params: dict[str, dict[str, Any]] = {
        "debug_plot": {"dataset_key_l_l": [["debug_data", "debug_data"], ["debug_data", "debug_data"]]}
    }

# Arrange data.

measure_name_l: list[str] = [measure_key_name_dict[key] for key in measure_key_l]

train_setting_key_l: list[str] = list(train_other_params.keys())
eval_setting_key_l: list[str] = list(eval_other_params.keys())

beam_opt_key_l: list[str] = [f"beam-{bs}" for bs, _, _ in beam_options]
beam_opt_name_l: list[str] = [f"{bs}" for bs, _, _ in beam_options]

assert len(eval_setting_key_l) == len(beam_opt_key_l)

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

    print(f"{pid=}: Start making table for {params=}")

    res: dict[tuple[str, str], tuple[list[str], list[float], list[float]]] = dict()

    # First, load data. and retrieve the best strategies for each dataset and beam size.
    for dataset_key_l in params.dataset_key_l_l:
        for dataset_key in dataset_key_l:
            for eval_setting_key, beam_opt_key in zip(eval_setting_key_l, beam_opt_key_l):
                table_key = (dataset_key, beam_opt_key)
                values: list[float] = []
                stderrs: list[float] = []

                for strategy_key in strategies:
                    seed_res: list[float] = []

                    for seed in random_seeds:
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
                            val = tmp_res_dict[params.measure_key] * measure_key_scale[params.measure_key]

                            seed_res.append(val)

                    mean_value = np.mean(seed_res)
                    std = np.std(seed_res)
                    stderr = std / np.sqrt(len(random_seeds))

                    values.append(mean_value)
                    stderrs.append(stderr)

                # Calculate the best strategy.
                ascending_index: list[int] = np.argsort(values).tolist()
                if measure_key_larger_better_dict[params.measure_key]:
                    # Convert to descending order.
                    sorted_index = ascending_index[::-1]
                else:
                    sorted_index = ascending_index

                sorted_strategies = [strategies[ind] for ind in sorted_index]
                sorted_values = [values[ind] for ind in sorted_index]
                sorted_stderrs = [stderrs[ind] for ind in sorted_index]
                res[table_key] = (sorted_strategies, sorted_values, sorted_stderrs)

    nrows = 1 + max([len(l) for l in params.dataset_key_l_l])

    # Plot
    table_string_l: list[str] = []
    table_string_l.append(r"\begin{table*}[t]")
    table_string_l.append(r"\centering")
    table_string_l.append(r"\begin{tabular}{" + "c" * nrows + r"}")

    for dataset_key_l in params.dataset_key_l_l:
        header: str = (
            "Beam & " + " & ".join([f"{dataset_key_name_dict[dataset_key]}" for dataset_key in dataset_key_l]) + r" \\"
        )

        table_string_l.append(header)
        table_string_l.append(r"\hline\hline")

        for beam_opt_key, beam_opt_name in zip(beam_opt_key_l, beam_opt_name_l):
            for i in range(topn_plot):
                table_line_str: str = f"{beam_opt_name}" if i == 0 else ""

                for dataset_key in dataset_key_l:
                    tmp_strategies, tmp_values, tmp_stderrs = res[(dataset_key, beam_opt_key)]

                    tmp_strategy_name = strategy_names[tmp_strategies[i]]

                    if i == 0:
                        tmp_strategy_name = r"\textbf{" + tmp_strategy_name + r"}"

                    tmp_strategy_value = measure_key_value_str_func[params.measure_key](tmp_values[i])
                    tmp_strategy_stderr = tmp_stderrs[i]

                    table_line_str += (
                        f" & {tmp_strategy_name}"
                        + r" {\small "
                        + f"({tmp_strategy_value}"
                        + r"$\pm$"
                        + f"{tmp_strategy_stderr:.1f})"
                        + r"}"
                    )

                table_line_str += r" \\"
                table_string_l.append(table_line_str)

            table_string_l.append(r"\hline")

        table_string_l.append(r"\\")

    table_string_l.append(r"\end{tabular}")
    table_string_l.append(r"\caption{Caption}")
    table_string_l.append(r"\label{tab:strategy_parsing}")
    table_string_l.append(r"\end{table*}")

    save_filepath = naming.get_best_strategy_table_file(
        base_dir=base_dir, setting_key=params.setting_key, measure_key=params.measure_key
    )
    save_filepath.parent.mkdir(parents=True, exist_ok=True)

    with save_filepath.open(mode="w") as f:
        f.write("\n".join(table_string_l))

    print(f"{pid=}: Finish making table for {params=}")


def main():
    logger.info("Start making table!!!")

    with multiprocessing.Pool(processes=num_parallel) as executer:
        # Pass the seed and hparams.
        executer.map(
            exec_single,
            gen_execparams(),
        )

    logger.info("Finish making table!!!")


if __name__ == "__main__":
    main()
