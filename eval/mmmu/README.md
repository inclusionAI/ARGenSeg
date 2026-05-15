# README for Evaluation

## 🌟 Overview

This script provides an evaluation pipeline for `MMMU`.

## 🗂️ Data Preparation

Before starting to download the data, please create the `data` folder.

### MMMU

The evaluation script will automatically download the MMMU dataset from HuggingFace, and the cached path is `data/MMMU`.

## 🏃 Evaluation Execution

To run the evaluation, execute the following command on an 8-GPU setup:

```shell
torchrun --nproc_per_node=8 eval/mmmu/evaluate_mmmu.py --checkpoint ${CHECKPOINT}
```

Alternatively, you can run the following simplified command:

```shell
GPUS=8 sh evaluate.sh ${CHECKPOINT} mmmu-val
```

### Arguments

The following arguments can be configured for the evaluation script:

| Argument         | Type   | Default             | Description                                                                                                       |
| ---------------- | ------ | ------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `--checkpoint`   | `str`  | `''`                | Path to the model checkpoint.                                                                                     |
| `--datasets`     | `str`  | `'MMMU_validation'` | Comma-separated list of datasets to evaluate.                                                                     |
| `--dynamic`      | `flag` | `False`             | Enables dynamic high resolution preprocessing.                                                                    |
| `--max-num`      | `int`  | `6`                 | Maximum tile number for dynamic high resolution.                                                                  |
| `--load-in-8bit` | `flag` | `False`             | Loads the model weights in 8-bit precision.                                                                       |
| `--auto`         | `flag` | `False`             | Automatically splits a large model across 8 GPUs when needed, useful for models too large to fit on a single GPU. |
