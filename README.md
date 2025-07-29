# Is Incremental Structure Prediction Process Universal across Languages?: Revisiting Parsing Strategy through Speculation

This repository provides the experimental codes for the CONLL2025 paper [Is Incremental Structure Prediction Process Universal across Languages?: Revisiting Parsing Strategy through Speculation](https://aclanthology.org/2025.conll-1.29/).

# Environment

Simply running `uv run python ...` automatically recovers the environment.

# Experiments

## Preprocess

`script/preprocess.py` is used. Hyperparameters for preprocess are directly written in the script.

Run

```
uv run python -m scripts.preprocess
```

For full experiments, you need to modify the script and set the following variables:

```
data_base_dir: str = "PathToDirectory" # Set the directory where all datasets are placed.
debug: bool = False # Set False, the default is True.
```

## Training and Evaluation

`script/train_and_eval.py` is used. Hyperparameters are directly written in the script.

Run

```
uv run python -m scripts.train_and_eval
```

For full experiments, you need to modify the script and set the following variables:

```
base_dir: Path = Path("../tmp") # Set the directory where all results will be placed.
debug: bool = False # Set False, the default is True

num_parallel: int # Number of parallel process.
use_gpu: bool # Wether to use gpu or not.
gpu_ids: list[int] # List of gpus ids for parallel processing.
```

## Analysis

`script/bst_strategy_table.py`, `scripts/plot_inference_results.py`, and `scripts/plot_validation_loss.py` are used. Hyperparameters are directly written in the script.

Run

```
uv run python -m scripts.best_strategy_table
uv run python -m scripts.plot_inference_results
uv run python -m scripts.plot_validation_loss
```

For full experiments, you need to modify the script and set the following variables:

```
base_dir: Path = Path("../tmp") # Set the directory where all results will be placed.
debug: bool = False # Set False, the default is True

num_parallel: int # Number of parallel process.
use_gpu: bool # Wether to use gpu or not.
gpu_ids: list[int] # List of gpus ids for parallel processing.
```

# License

This code is distributed under the MIT License.