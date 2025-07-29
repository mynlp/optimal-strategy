import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from . import naming

#########################
## Paths and Variables ##
#########################

# Set the base directory where all datasets are stored.
data_base_dir = "/home/$USER"  # local

ptb_dir = data_base_dir + "/resources/PTB3/treebank_3/parsed/mrg/wsj/"
ctb_dir = data_base_dir + "/resources/ctb5.1_507K/data/bracketed/"
spmrl_basque_dir = data_base_dir + "/resources/SPMRL/SPMRL_SHARED_2014_NO_ARABIC/BASQUE_SPMRL/gold/ptb/"
spmrl_french_dir = data_base_dir + "/resources/SPMRL/SPMRL_SHARED_2014_NO_ARABIC/FRENCH_SPMRL/gold/ptb/"
spmrl_german_dir = data_base_dir + "/resources/SPMRL/SPMRL_SHARED_2014_NO_ARABIC/GERMAN_SPMRL/gold/ptb/"
spmrl_hebrew_dir = data_base_dir + "/resources/SPMRL/SPMRL_SHARED_2014_NO_ARABIC/HEBREW_SPMRL/gold/ptb/"
spmrl_hungarian_dir = data_base_dir + "/resources/SPMRL/SPMRL_SHARED_2014_NO_ARABIC/HUNGARIAN_SPMRL/gold/ptb/"
spmrl_korean_dir = data_base_dir + "/resources/SPMRL/SPMRL_SHARED_2014_NO_ARABIC/KOREAN_SPMRL/gold/ptb/"
spmrl_polish_dir = data_base_dir + "/resources/SPMRL/SPMRL_SHARED_2014_NO_ARABIC/POLISH_SPMRL/gold/ptb/"
spmrl_swedish_dir = data_base_dir + "/resources/SPMRL/SPMRL_SHARED_2014_NO_ARABIC/SWEDISH_SPMRL/gold/ptb/"

dataset_key_dir_dict: dict[str, str] = {
    "ptb": ptb_dir,
    "ctb": ctb_dir,
    "spmrl_basque": spmrl_basque_dir,
    "spmrl_french": spmrl_french_dir,
    "spmrl_german": spmrl_german_dir,
    "spmrl_hebrew": spmrl_hebrew_dir,
    "spmrl_hungarian": spmrl_hungarian_dir,
    "spmrl_korean": spmrl_korean_dir,
    "spmrl_polish": spmrl_polish_dir,
    "spmrl_polish_remove_lowest": spmrl_polish_dir,  # Use the same source as normal polish.
    "spmrl_swedish": spmrl_swedish_dir,
}


base_dir: Path = Path("../tmp")

split_for_eval: str = "dev"
assert split_for_eval in {"train", "dev", "test"}

max_sentence_length: int = 10000  # This means length is not limited.

nt_insert_pos_limit: int = 10

# Set subword vocabulary size.

vocabsize_large: int = (
    5000  # Vocab size for larger datasets: ptb, ctb, spmrl_french, spmrl_german, spmrl_hungarian, spmrl_korean.
)
vocabsize_small: int = 1500  # Vocab size for smaller datasets: spmrl_basque, spmrl_hebrew, spmrl_polish, spmrl_swedish. These datasets have roughly 5K~8K different words (with vocab_min_freq==2).

# Set strategy params.
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


debug: bool = True
# debug: bool = False

# Set debug params.
if debug:
    base_dir = Path("../debug")
    vocabsize_debug: int = 500
    nt_insert_pos_limit: int = 10

    strategy_params: dict[str, dict[str, Any]] = {
        "top-down": {"func_key": "top-down", "params": {}},
        "bottom-up": {"func_key": "bottom-up", "params": {}},
        "left-1-corner": {"func_key": "left-n-corner", "params": {"n": 1}},
        "uniform-spec-0.26": {"func_key": "uniform-speculation", "params": {"real_pos": 0.26}},
        "local-first-1": {"func_key": "local-first", "params": {"height": 1}},
        "global-first-1": {"func_key": "global-first", "params": {"depth": 1}},
    }

# Get file to save strategy_params.
strategy_params_file: Path = naming.get_strategy_params_filepath(base_dir=base_dir)
strategy_params_file.parent.mkdir(parents=True, exist_ok=True)


def get_treebanks():
    options: list[str] = []

    for dataset_key, dataset_dir in dataset_key_dir_dict.items():
        # Set options for ptb.
        if dataset_key == "ptb":
            options += [
                f"--ptb_dir {ptb_dir}",
                f"--ptb_train_filepath {naming.get_treebank_filepath(base_dir=base_dir, treebank_key='ptb', split='train').resolve()}",
                f"--ptb_dev_22_filepath {naming.get_treebank_filepath(base_dir=base_dir, treebank_key='ptb', split='dev_22').resolve()}",
                f"--ptb_dev_24_filepath {naming.get_treebank_filepath(base_dir=base_dir, treebank_key='ptb', split='dev_24').resolve()}",
                f"--ptb_test_filepath {naming.get_treebank_filepath(base_dir=base_dir, treebank_key='ptb', split='test').resolve()}",
            ]
        # Set options for other datasets.
        else:
            options += [
                f"--{dataset_key}_dir {dataset_dir}",
                f"--{dataset_key}_train_filepath {naming.get_treebank_filepath(base_dir=base_dir, treebank_key=dataset_key, split='train').resolve()}",
                f"--{dataset_key}_dev_filepath {naming.get_treebank_filepath(base_dir=base_dir, treebank_key=dataset_key, split='dev').resolve()}",
                f"--{dataset_key}_test_filepath {naming.get_treebank_filepath(base_dir=base_dir, treebank_key=dataset_key, split='test').resolve()}",
            ]

    command: str = " ".join(["python ./get_treebank.py"] + options)

    # Execute command.
    p = subprocess.run(command, shell=True)
    p.check_returncode()


def preprocess_treebanks():
    for dataset_key in dataset_key_dir_dict.keys():
        if dataset_key == "ptb":
            # We use section 22 for dev.
            dev_file = naming.get_treebank_filepath(base_dir=base_dir, treebank_key="ptb", split="dev_22")
        else:
            dev_file = naming.get_treebank_filepath(base_dir=base_dir, treebank_key=dataset_key, split="dev")

        processed_treebank_base, _, _, _ = naming.get_processed_treebank_names(
            base_dir=base_dir, treebank_key=dataset_key
        )
        processed_treebank_base.parent.mkdir(parents=True, exist_ok=True)

        # Set vocabsize.
        match dataset_key:
            case "ptb" | "ctb" | "spmrl_french" | "spmrl_german" | "spmrl_hungarian" | "spmrl_korean":
                vocabsize = vocabsize_large
            case _:
                vocabsize = vocabsize_small

        preprocess_command_str: str = " ".join(
            [
                "python ./preprocess.py",
                f"--trainfile {naming.get_treebank_filepath(base_dir=base_dir, treebank_key=dataset_key, split='train')}",
                f"--valfile {dev_file}",
                f"--strategy_params_file {strategy_params_file}",
                f"--nt_insert_pos_limit {nt_insert_pos_limit}",
                f"--vocabsize {vocabsize}",  # vocabsize is not used when vocabminfreq is >0.
                f"--seqlength {max_sentence_length}",
                f"--outputfile {processed_treebank_base}",
            ]
        )

        if dataset_key == "ptb" and split_for_eval == "dev":
            eval_split = "dev_22"
        else:
            eval_split = split_for_eval

        eval_text_file: Path = naming.get_treebank_text_filepath(
            base_dir=base_dir, treebank_key=dataset_key, split=eval_split
        )
        eval_gold_tree_file: Path = naming.get_treebank_filepath(
            base_dir=base_dir, treebank_key=dataset_key, split=eval_split
        )

        get_text_command: str = " ".join(
            [
                "python get_texts_from_trees.py",
                f"--input_tree_file {eval_gold_tree_file.resolve()}",
                f"--output_file {eval_text_file.resolve()}",
            ]
        )

        command_str: str = f"set -e; {preprocess_command_str}; {get_text_command}"

        # Execute commands.
        print()
        print(f"Start processing {dataset_key}")
        print()
        try:
            p = subprocess.run(command_str, shell=True)
            p.check_returncode()
        except Exception:
            exit(1)

        print()
        print(f"Finish processing {dataset_key}")
        print()


def prepare_debug_data():
    """Debug data (train, dev, test splits) are taken from PTB train split."""

    # First copy debug data files to debug directory.
    dataset_dir: Path = base_dir.joinpath("dataset")
    dataset_dir.mkdir(parents=True, exist_ok=True)

    commands: list[str] = ["set -e", f"cp -r ./debug_data {dataset_dir.resolve()}"]

    processed_treebank_base, _, _, _ = naming.get_processed_treebank_names(base_dir=base_dir, treebank_key="debug_data")
    processed_treebank_base.parent.mkdir(parents=True, exist_ok=True)

    preprocess_command_str: str = " ".join(
        [
            "python ./preprocess.py",
            f"--trainfile {naming.get_treebank_filepath(base_dir=base_dir, treebank_key='debug_data', split='train')}",
            f"--valfile {naming.get_treebank_filepath(base_dir=base_dir, treebank_key='debug_data', split='dev')}",
            f"--strategy_params_file {strategy_params_file}",
            f"--nt_insert_pos_limit {nt_insert_pos_limit}",
            f"--vocabsize {vocabsize_debug}",  # vocabsize is not used when vocabminfreq is >0.
            f"--seqlength {max_sentence_length}",
            f"--outputfile {processed_treebank_base}",
        ]
    )
    commands.append(preprocess_command_str)

    eval_text_file: Path = naming.get_treebank_text_filepath(
        base_dir=base_dir, treebank_key="debug_data", split=split_for_eval
    )
    eval_gold_tree_file: Path = naming.get_treebank_filepath(
        base_dir=base_dir, treebank_key="debug_data", split=split_for_eval
    )

    get_text_command: str = " ".join(
        [
            "python get_texts_from_trees.py",
            f"--input_tree_file {eval_gold_tree_file.resolve()}",
            f"--output_file {eval_text_file.resolve()}",
        ]
    )
    commands.append(get_text_command)

    command_str = "; ".join(commands)

    # Execute commands.
    print()
    print("Start processing debug_data")
    print()
    try:
        p = subprocess.run(command_str, shell=True)
        p.check_returncode()
    except Exception:
        exit(1)

    print()
    print("Finish processing debug_data")
    print()


def main():
    # First, generate strategy parameter file.
    with strategy_params_file.open(mode="w") as f:
        f.write(json.dumps(strategy_params))

    if not debug:
        get_treebanks()
        preprocess_treebanks()

    else:
        prepare_debug_data()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(name)s:%(levelname)s: %(message)s",
    )

    main()
