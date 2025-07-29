import datetime
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator

from . import naming
from .utils import get_thread_id, main_logger, thread_log, thread_run_subprocess

logger = main_logger.getChild(__name__)

########################
## Default Parameters ##
########################


@dataclass
class TrainExecParams:
    ## Parameters to vary.

    # Dataset
    dataset_key: str

    random_seed: int

    # Model training
    strategy: str

    # Setting name (e.g., model name, beam search options...)
    setting_key: str

    ## Parameters to use the default values.

    # Model hyperparameters

    stack_size: int = 150  # Seems CTB and SPMRL_FRENCH requires larger stack size for subwords.

    larger_stack_size: int = 300  # Stack size used when the default stack size is not enough.

    w_dim: int = 256
    h_dim: int = 256
    num_layers: int = 2
    regularizer: str = "dropout"
    dropout: float = 0.3  # Seems to be better than 0.1
    # dropout: float = 0.1  # as in the batched rnng paper

    # Supervised training options
    batch_size: int = -1  # Dummy value, we set batch_size for each dataset.

    num_epochs: int = 80
    num_steps: int = 8000

    lr: float = 0.001

    print_every: int = 10
    valid_every: int = 200  # -1 means validation is performed every 100 steps(batches).
    val_batch_size: int = 4096


@dataclass
class EvalExecParams:
    ## Parameters to vary.

    train_params: TrainExecParams

    # Setting name (e.g., model name, beam search options...)
    setting_key: str

    ## Parameters to use the default values.

    # Beam search options
    beam_size: int  # Note that both of open_beam and step_compelete_beam have the same beam_size, so the beam size in total is beam_size*2.
    min_shift_size: int  # beam_size / 50 seems to work well while being much faster.

    inference_batch_size: int

    inference_block_size: int = 100000000000  # No limit

    word_sync_step_limit: int = 20  # For safety, this limit should be larger than 3. (e.g., in case SHIFT is not possible, probably NT and REDUCE is needed to reduce the stack size.)

    # Evaluation options
    # Note that these parameters shouldn't affect the results.
    eval_batch_size: int = 8192
    eval_block_size: int = 8192

    # Only evaluate parsable data for given nt_insert_pos_limit.
    eval_filter_by_parsability: bool = True


########################
## Parameters to Vary ##
########################

# Datasets.
dataset_key_l: list[str] = [
    "ptb",
    "ctb",
    "spmrl_french",
    "spmrl_german",
    "spmrl_korean",
    "spmrl_basque",
    "spmrl_hebrew",
    "spmrl_hungarian",
    "spmrl_polish",
    "spmrl_polish_remove_lowest",
    "spmrl_swedish",
]

# We use smaller batch_size for smaller datasets (i.e., datasets that have less than 10K data.)
# Note that actual batches may contain less than the values here depending on the max length diff constraint.
dataset_train_batch_sizes: dict[str, int] = {
    "ptb": 512,
    "ctb": 512,
    "spmrl_french": 512,
    "spmrl_german": 512,
    "spmrl_korean": 512,
    "spmrl_basque": 128,
    "spmrl_hebrew": 128,
    "spmrl_hungarian": 128,
    "spmrl_polish": 128,
    "spmrl_polish_remove_lowest": 128,
    "spmrl_swedish": 128,
}

split_for_eval: str = "dev"
assert split_for_eval in {"train", "dev", "test"}

# Strategy.

# strategy_params: dict[str, dict[str, Any]]: strategy_key -> {func_key:, params:}
strategy_params: dict[str, dict[str, Any]] = dict()

# Add top-down.
strategy_params["top-down"] = {"func_key": "top-down", "params": {}}

# Add bottom-up.
strategy_params["bottom-up"] = {"func_key": "bottom-up", "params": {}}

# Add left-n-corner.
n_corner_l: list[int] = [1, 2, 3]
for n in n_corner_l:
    strategy_params[f"left-{n}-corner"] = {"func_key": "left-n-corner", "params": {"n": n}}

# Add uniform speculation.
real_pos_l: list[float] = [0.26, 0.35, 0.65, 0.74]
for real_pos in real_pos_l:
    strategy_params[f"uniform-spec-{real_pos}"] = {"func_key": "uniform-speculation", "params": {"real_pos": real_pos}}

# Add local-first.
height_l: list[int] = [1, 2, 3]
for height in height_l:
    strategy_params[f"local-first-{height}"] = {"func_key": "local-first", "params": {"height": height}}

# Add global-first.
depth_l: list[int] = [1, 2, 3]
for depth in depth_l:
    strategy_params[f"global-first-{depth}"] = {"func_key": "global-first", "params": {"depth": depth}}

# Get list of strategy keys.
strategies: list[str] = list(strategy_params.keys())


random_seeds: list[int] = [1111, 2222, 3333]  # wisteria


beam_options: list[tuple[int, int, int]] = [
    # (beam_size, min_shift_size, inference_batch_size)
    # We set min_shift_size 1/50 of beam size.
    (50, 1, 320),
    (200, 4, 80),
    (800, 16, 20),
]


# Other settings (e.g., model hyperparameters, beam search options...)
# This is useful for iterating over complex combination of settings (which cannot be done by simple for loops).
# setting_key -> specific setting dict.
train_other_params: dict[str, dict[str, Any]] = {"default_setting": {}}  # just give dummy center

eval_other_params: dict[str, dict[str, Any]] = {
    f"beam_{beam_size}-shift_{min_shift_size}": {
        "beam_size": beam_size,
        "min_shift_size": min_shift_size,
        "inference_batch_size": batch_size,
    }
    for beam_size, min_shift_size, batch_size in beam_options
}

#######################################
## Paths, Device and Script Settings ##
#######################################

base_dir: Path = Path("../tmp")

# Device setting

# num_parallel: int = 1
num_parallel: int = 8
use_gpu: bool = True
# gpu_ids: list[int] = [0]
gpu_ids: list[int] = [0, 1, 2, 3, 4, 5, 6, 7]

train_force_update: bool = False
# train_force_update: bool = True
beam_force_update: bool = False
# beam_force_update: bool = True
eval_force_update: bool = False
# eval_force_update: bool = True

skip_beam: bool = False
skip_eval: bool = False
# skip_beam: bool = True
# skip_eval: bool = True


debug: bool = True
# debug: bool = False

# Used for log file name for each subprocess.
log_time: str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

# Set debug params.
if debug:
    base_dir = Path("../debug")
    dataset_key_l = ["debug_data"]

    dataset_train_batch_sizes: dict[str, int] = {"debug_data": 4}

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

    beam_options: list[tuple[int, int, int]] = [
        # (beam_size, min_shift_size, inference_batch_size)
        # We set min_shift_size 1/50 of beam size.
        (2, 1, 5),
        (4, 1, 5),
    ]

    # random_seeds: list[int] = [888888]
    random_seeds: list[int] = [9999999, 888888]

    # num_parallel: int = 1
    num_parallel: int = 3

    # use_gpu: bool = True
    use_gpu: bool = False
    gpu_ids: list[int] = [0]

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


###############
## Functions ##
###############


def train_gen_execparams() -> Generator[TrainExecParams, None, None]:
    """Generate parameters for exec_single by varying parameters."""

    for dataset_key in dataset_key_l:
        for strategy in strategies:
            for train_setting_key in train_other_params:
                for random_seed in random_seeds:
                    yield TrainExecParams(
                        dataset_key=dataset_key,
                        random_seed=random_seed,
                        strategy=strategy,
                        setting_key=train_setting_key,
                        batch_size=dataset_train_batch_sizes[dataset_key],
                        **train_other_params[train_setting_key],
                    )


def eval_gen_execparams() -> Generator[EvalExecParams, None, None]:
    """Generate parameters for exec_single by varying parameters."""

    for train_params in train_gen_execparams():
        for eval_setting_key in eval_other_params:
            yield EvalExecParams(
                train_params=train_params,
                setting_key=eval_setting_key,
                **eval_other_params[eval_setting_key],
            )


def train_exec_single(params: TrainExecParams) -> None:
    """Function to be executed in parallel."""

    thread_log(f"Execution for {params=}")

    # List of commands to execute
    commands: list[str] = []
    steps: list[str] = []  # Just for logging.

    # Set device; gpu_id is None if the device to be used is not gpu.
    gpu_id = None
    thread_id: int = get_thread_id()
    if use_gpu and (thread_id < len(gpu_ids)):
        gpu_id = gpu_ids[thread_id]

    gpu_option: list[str] = (
        ["CUDA_VISIBLE_DEVICES={}".format(gpu_id)] if gpu_id is not None else ["CUDA_VISIBLE_DEVICES=-1"]
    )

    # We only use one gpu, and the gpu count always start from 0 (regardless of actual gpu id).
    device_flag: list[str] = ["--device cuda", "--gpu 0"] if gpu_id is not None else ["--device cpu"]

    #########################
    ## Supervised Training ##
    #########################

    # First, check if the result file already exists or not.
    # If the result file alredy exists, the execution is skipped unless train_force_update is True.

    train_output_file: Path = naming.get_train_model_file(
        base_dir=base_dir,
        treebank_key=params.dataset_key,
        strategy=params.strategy,
        setting_key=params.setting_key,
        train_seed=params.random_seed,
    )

    if train_force_update or (not train_output_file.exists()):
        # Make the directories just in case.
        train_output_file.parent.mkdir(parents=True, exist_ok=True)

        _, train_file, val_file, _ = naming.get_processed_treebank_names(
            base_dir=base_dir, treebank_key=params.dataset_key
        )

        train_val_res_file = naming.get_train_val_file(
            base_dir=base_dir,
            treebank_key=params.dataset_key,
            strategy=params.strategy,
            setting_key=params.setting_key,
            train_seed=params.random_seed,
        )

        sp_model_path: Path = naming.get_sp_model_path(base_dir=base_dir, treebank_key=params.dataset_key)

        train_command_str: str = " ".join(
            gpu_option
            + [
                "python train.py",
                # Files.
                f"--train_file {train_file.resolve()}",
                f"--val_file {val_file.resolve()}",
                f"--save_path {train_output_file.resolve()}",
                f"--val_res_path {train_val_res_file.resolve()}",
                f"--strategy {params.strategy}",
                # Model hyperparameters.
                f"--stack_size {params.stack_size}",
                f"--larger_stack_size {params.larger_stack_size}",
                f"--w_dim {params.w_dim}",
                f"--h_dim {params.h_dim}",
                f"--num_layers {params.num_layers}",
                f"--regularizer {params.regularizer}",
                f"--dropout {params.dropout}",
                f"--lr {params.lr}",
                # Supervised train hyperparameters.
                f"--batch_size {params.batch_size}",
                f"--val_batch_size {params.val_batch_size}",
                f"--num_epochs {params.num_epochs}",
                f"--num_steps {params.num_steps}",
                f"--print_every {params.print_every}",
                f"--valid_every {params.valid_every}",
                f"--seed {params.random_seed}",
                f"--sp_model {sp_model_path.resolve()}",
            ]
            + device_flag
        )

        commands.append(train_command_str)
        steps.append("train")
    else:
        thread_log(f"Skip supervised training for {params}")

    # Execute the commands.
    if len(commands) == 0:
        thread_log(f"Skip execution for {params}")
    else:
        log_file: Path = naming.get_train_log_file(
            base_dir=base_dir,
            treebank_key=params.dataset_key,
            strategy=params.strategy,
            setting_key=params.setting_key,
            train_seed=params.random_seed,
            log_time=log_time,
        )
        log_file.parent.mkdir(parents=True, exist_ok=True)

        step_str: str = ", ".join(steps)
        command_str: str = "; ".join(["set -e"] + commands)

        thread_log(f"GPU id: {gpu_id}; Start {step_str} for {params}")

        thread_run_subprocess(command_str=command_str, log_file=log_file)

        thread_log(f"GPU id: {gpu_id}; End {step_str} for {params}")


def eval_exec_single(params: EvalExecParams) -> None:
    """Function to be executed in parallel."""

    thread_log(f"Execution for {params=}")

    # List of commands to execute
    commands: list[str] = []
    steps: list[str] = []  # Just for logging.

    # Set device; gpu_id is None if the device to be used is not gpu.
    gpu_id = None
    thread_id: int = get_thread_id()
    if use_gpu and (thread_id < len(gpu_ids)):
        gpu_id = gpu_ids[thread_id]

    gpu_option: list[str] = (
        ["CUDA_VISIBLE_DEVICES={}".format(gpu_id)] if gpu_id is not None else ["CUDA_VISIBLE_DEVICES=-1"]
    )

    # We only use one gpu, and the gpu count always start from 0 (regardless of actual gpu id).
    device_flag: list[str] = ["--device cuda", "--gpu 0"] if gpu_id is not None else ["--device cpu"]

    # Ad hoc setting.
    if params.train_params.dataset_key == "ptb" and split_for_eval == "dev":
        eval_split = "dev_22"
    else:
        eval_split = split_for_eval

    ###############
    ## Inference ##
    ###############

    # First, check if the result file already exists or not.
    # If the result file alredy exists, the execution is skipped unless beam_force_update is True.

    inference_output_file: Path = naming.get_inference_file(
        base_dir=base_dir,
        treebank_key=params.train_params.dataset_key,
        strategy=params.train_params.strategy,
        eval_setting_key=params.setting_key,
        train_setting_key=params.train_params.setting_key,
        train_seed=params.train_params.random_seed,
    )

    train_output_file: Path = naming.get_train_model_file(
        base_dir=base_dir,
        treebank_key=params.train_params.dataset_key,
        strategy=params.train_params.strategy,
        setting_key=params.train_params.setting_key,
        train_seed=params.train_params.random_seed,
    )

    if (not skip_beam) and (beam_force_update or train_force_update or (not inference_output_file.exists())):
        # Make the directories just in case.
        inference_output_file.parent.mkdir(parents=True, exist_ok=True)

        eval_text_file: Path = naming.get_treebank_text_filepath(
            base_dir=base_dir, treebank_key=params.train_params.dataset_key, split=eval_split
        )

        beam_command_str: str = " ".join(
            gpu_option
            + [
                "python inference.py",
                f"--test_file {eval_text_file.resolve()}",
                f"--model_file {train_output_file.resolve()}",
                f"--output_file {inference_output_file.resolve()}",
                f"--stack_size {params.train_params.stack_size}",
                f"--larger_stack_size {params.train_params.larger_stack_size}",
                # Beam search hyperparameters.
                f"--beam_size {params.beam_size}",
                f"--min_shift_size {params.min_shift_size}",
                f"--word_sync_step_limit {params.word_sync_step_limit}",
                f"--batch_size {params.inference_batch_size}",
                f"--block_size {params.inference_block_size}",
                f"--seed {params.train_params.random_seed}",
            ]
            + device_flag
        )

        commands.append(beam_command_str)
        steps.append("inference")
    else:
        thread_log(f"Skip inference for {params}")

    ##############
    ## Evaluate ##
    ##############

    # First, check if the result file already exists or not.
    # If the result file alredy exists, the execution is skipped unless eval_force_update is True.

    eval_output_file: Path = naming.get_eval_file(
        base_dir=base_dir,
        treebank_key=params.train_params.dataset_key,
        strategy=params.train_params.strategy,
        eval_setting_key=params.setting_key,
        train_setting_key=params.train_params.setting_key,
        train_seed=params.train_params.random_seed,
    )

    # Skip eval if beam is skipped (which means there may be no beam search results needed for evaluation).
    if (not skip_beam and not skip_eval) and (
        eval_force_update or beam_force_update or train_force_update or (not eval_output_file.exists())
    ):
        # Make the directories just in case.
        eval_output_file.parent.mkdir(parents=True, exist_ok=True)

        eval_text_file: Path = naming.get_treebank_text_filepath(
            base_dir=base_dir, treebank_key=params.train_params.dataset_key, split=eval_split
        )
        eval_gold_tree_file: Path = naming.get_treebank_filepath(
            base_dir=base_dir, treebank_key=params.train_params.dataset_key, split=eval_split
        )
        strategy_params_file: Path = naming.get_strategy_params_filepath(base_dir=base_dir)

        filter_by_parsability_option: str = "--filter_by_parsability" if params.eval_filter_by_parsability else ""

        eval_command_str: str = " ".join(
            gpu_option
            + [
                "python evaluate.py",
                f"--inference_file {inference_output_file.resolve()}",
                f"--gold_tree_file {eval_gold_tree_file.resolve()}",
                f"--text_file {eval_text_file.resolve()}",
                f"--model_file {train_output_file.resolve()}",
                f"--output_file {eval_output_file.resolve()}",
                f"--strategy_params_file {strategy_params_file.resolve()}",
                f"--strategy {params.train_params.strategy}",
                f"--stack_size {params.train_params.stack_size}",
                f"--larger_stack_size {params.train_params.larger_stack_size}",
                f"--word_sync_step_limit {params.word_sync_step_limit}",
                f"--seed {params.train_params.random_seed}",
                filter_by_parsability_option,
            ]
            + device_flag
        )

        commands.append(eval_command_str)
        steps.append("evaluation")
    else:
        thread_log(f"Skip evaluation for {params}")

    # Execute the commands.
    if len(commands) == 0:
        thread_log(f"Skip execution for {params}")
    else:
        log_file: Path = naming.get_eval_log_file(
            base_dir=base_dir,
            treebank_key=params.train_params.dataset_key,
            strategy=params.train_params.strategy,
            eval_setting_key=params.setting_key,
            train_setting_key=params.train_params.setting_key,
            train_seed=params.train_params.random_seed,
            log_time=log_time,
        )
        log_file.parent.mkdir(parents=True, exist_ok=True)

        step_str: str = ", ".join(steps)
        command_str: str = "; ".join(["set -e"] + commands)

        thread_log(f"GPU id: {gpu_id}; Start {step_str} for {params}")

        thread_run_subprocess(command_str=command_str, log_file=log_file)

        thread_log(f"GPU id: {gpu_id}; End {step_str} for {params}")


def main():
    logger.info("Start train and eval!!!")

    # with ThreadPoolExecutor(max_workers=num_parallel) as executer:
    with ThreadPoolExecutor(max_workers=num_parallel, thread_name_prefix="Thread") as executer:
        # First train
        executer.map(
            train_exec_single,
            train_gen_execparams(),
        )

    # with ThreadPoolExecutor(max_workers=num_parallel) as executer:
    with ThreadPoolExecutor(max_workers=num_parallel, thread_name_prefix="Thread") as executer:
        # Next evaluate.
        executer.map(
            eval_exec_single,
            eval_gen_execparams(),
        )

    logger.info("Finish train and eval!!!")


if __name__ == "__main__":
    main()
