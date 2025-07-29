from pathlib import Path

##############
## Strategy ##
##############


def get_strategy_params_filepath(base_dir: Path) -> Path:
    return Path.joinpath(base_dir, "strategy", "strategy_params.json")


##############
## Datasets ##
##############


def get_treebank_filepath(base_dir: Path, treebank_key: str, split: str) -> Path:
    return Path.joinpath(base_dir, "dataset", treebank_key, f"{treebank_key}_{split}.tree")


def get_treebank_text_filepath(base_dir: Path, treebank_key: str, split: str) -> Path:
    return get_treebank_filepath(base_dir=base_dir, treebank_key=treebank_key, split=split).with_suffix(".tokens")


def get_processed_treebank_names(base_dir: Path, treebank_key: str) -> tuple[Path, Path, Path, Path]:
    base: Path = Path.joinpath(base_dir, "processed_dataset", treebank_key, treebank_key)
    train: Path = Path.joinpath(base_dir, "processed_dataset", treebank_key, f"{treebank_key}-train.json")
    val: Path = Path.joinpath(base_dir, "processed_dataset", treebank_key, f"{treebank_key}-val.json")
    test: Path = Path.joinpath(base_dir, "processed_dataset", treebank_key, f"{treebank_key}-test.json")

    return base, train, val, test


def get_sp_model_path(base_dir: Path, treebank_key: str) -> Path:
    return Path.joinpath(base_dir, "processed_dataset", treebank_key, f"{treebank_key}-spm.model")


#########################
## Supervised Training ##
#########################


def get_train_model_file(base_dir: Path, treebank_key: str, strategy: str, setting_key: str, train_seed: int) -> Path:
    return Path.joinpath(
        base_dir,
        "model",
        "train",
        treebank_key,
        strategy,
        setting_key,
        f"seed_{train_seed}",
        "rnng.pt",
    )


def get_train_val_file(base_dir: Path, treebank_key: str, strategy: str, setting_key: str, train_seed: int) -> Path:
    return Path.joinpath(
        base_dir,
        "model",
        "train",
        treebank_key,
        strategy,
        setting_key,
        f"seed_{train_seed}",
        "val_res.jsonl",
    )


###############
## Inference ##
###############


def get_inference_file(
    base_dir: Path,
    treebank_key: str,
    strategy: str,
    eval_setting_key: str,
    train_setting_key: str,
    train_seed: int,
) -> Path:
    return Path.joinpath(
        base_dir,
        "model",
        "inference",
        treebank_key,
        strategy,
        train_setting_key,
        eval_setting_key,
        f"seed_{train_seed}",
        "inference_output.jsonl",
    )


################
## Evaluation ##
################


def get_eval_file(
    base_dir: Path,
    treebank_key: str,
    strategy: str,
    eval_setting_key: str,
    train_setting_key: str,
    train_seed: int,
) -> Path:
    return Path.joinpath(
        base_dir,
        "model",
        "evaluate",
        treebank_key,
        strategy,
        train_setting_key,
        eval_setting_key,
        f"seed_{train_seed}",
        "eval_results.json",
    )


##########
## Plot ##
##########


def get_best_strategy_table_file(base_dir: Path, setting_key: str, measure_key: str) -> Path:
    return Path.joinpath(base_dir, "analysis", "best_strategy_table", setting_key, f"{measure_key}.txt")


def get_plot_val_loss_file(base_dir: Path, setting_key: str, measure_key: str) -> Path:
    return Path.joinpath(base_dir, "analysis", "plot_val_loss", setting_key, f"{measure_key}.png")


def get_plot_inference_results_file(base_dir: Path, setting_key: str, measure_key: str) -> Path:
    return Path.joinpath(base_dir, "analysis", "plot_inference_results", setting_key, f"{measure_key}.png")


#########
## Log ##
#########


def get_train_log_file(
    base_dir: Path,
    treebank_key: str,
    strategy: str,
    setting_key: str,
    train_seed: int,
    log_time: str,
) -> Path:
    return Path.joinpath(
        base_dir,
        "log",
        "train",
        treebank_key,
        strategy,
        setting_key,
        f"seed_{train_seed}",
        f"{log_time}.log",
    )


def get_eval_log_file(
    base_dir: Path,
    treebank_key: str,
    strategy: str,
    eval_setting_key: str,
    train_setting_key: str,
    train_seed: int,
    log_time: str,
) -> Path:
    return Path.joinpath(
        base_dir,
        "log",
        "eval",
        treebank_key,
        strategy,
        train_setting_key,
        eval_setting_key,
        f"seed_{train_seed}",
        f"{log_time}.log",
    )
