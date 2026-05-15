# README for Evaluation

## 🌟 Overview

This script provides an evaluation pipeline for `RefCOCO`.

## 🗂️ Data Preparation

Before starting to download the data, please create the `data` folder.

### RefCOCO/RefCOCO+/RefCOCO-g

Follow the instructions below to prepare the data:

```shell
# Step 1: Create the data directory
mkdir -p data/refcoco && cd data/refcoco

# Step 2: Download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/refcoco/refcoco_val.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/refcoco/refcoco_testA.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/refcoco/refcoco_testB.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/refcoco%2B/refcoco%2B_val.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/refcoco%2B/refcoco%2B_testA.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/refcoco%2B/refcoco%2B_testB.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/refcocog/refcocog_val.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/refcocog/refcocog_test.jsonl

cd ../..
```

After preparation is complete, the directory structure is:

```shell
data/refcoco
├── refcocog_test.jsonl
├── refcocog_val.jsonl
├── refcoco_testA.jsonl
├── refcoco+_testA.jsonl
├── refcoco_testB.jsonl
├── refcoco+_testB.jsonl
├── refcoco_val.jsonl
└── refcoco+_val.jsonl
```

## 🏃 Evaluation Execution

To run the evaluation, execute the following command on an 8-GPU setup:

```shell
torchrun --nproc_per_node=8 eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT}
```

Alternatively, you can run the following simplified command:

```shell
GPUS=8 sh evaluate.sh ${CHECKPOINT} refcoco
```

### Arguments

The following arguments can be configured for the evaluation script:

| Argument         | Type   | Default                                                                                                                               | Description                                                                                                       |
| ---------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `--checkpoint`   | `str`  | `''`                                                                                                                                  | Path to the model checkpoint.                                                                                     |
| `--datasets`     | `str`  | `refcoco_val`, `refcoco_testA`, `refcoco_testB` , `refcoco+_val`, `refcoco+_testA`, `refcoco+_testB`, `refcocog_val`, `refcocog_test` | Comma-separated list of datasets to evaluate.                                                                     |
| `--dynamic`      | `flag` | `False`                                                                                                                               | Enables dynamic high resolution preprocessing.                                                                    |
| `--max-num`      | `int`  | `6`                                                                                                                                   | Maximum tile number for dynamic high resolution.                                                                  |
| `--load-in-8bit` | `flag` | `False`                                                                                                                               | Loads the model weights in 8-bit precision.                                                                       |
| `--auto`         | `flag` | `False`                                                                                                                               | Automatically splits a large model across 8 GPUs when needed, useful for models too large to fit on a single GPU. |
