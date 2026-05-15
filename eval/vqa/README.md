# README for Evaluation

## 🌟 Overview

This script provides an evaluation pipeline for visual question answering across 9 datasets: `VQAv2`, `TextVQA`, `AI2D`.

## 🗂️ Data Preparation

Before starting to download the data, please create the `data` folder.

### VQAv2

Follow the instructions below to prepare the data:

```shell
# Step 1: Create the data directory
mkdir -p data/vqav2 && cd data/vqav2

# Step 2: Make sure you have downloaded COCO images
ln -s ../coco/train2014 ./
ln -s ../coco/val2014 ./
ln -s ../coco/test2015 ./

# Step 3: Download questions and annotations
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip && unzip v2_Annotations_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip && unzip v2_Questions_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip && unzip v2_Annotations_Val_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip && unzip v2_Questions_Val_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip && unzip v2_Questions_Test_mscoco.zip

# Step 4: Download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vqav2/vqav2_train.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vqav2/vqav2_val.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/vqav2/vqav2_testdev.jsonl

cd ../..
```

After preparation is complete, the directory structure is:

```shell
data/vqav2
├── train2014 -> ../coco/train2014
├── val2014 -> ../coco/val2014
├── test2015 -> ../coco/test2015
├── v2_mscoco_train2014_annotations.json
├── v2_mscoco_train2014_complementary_pairs.json
├── v2_mscoco_val2014_annotations.json
├── v2_OpenEnded_mscoco_test2015_questions.json
├── v2_OpenEnded_mscoco_test-dev2015_questions.json
├── v2_OpenEnded_mscoco_train2014_questions.json
├── v2_OpenEnded_mscoco_val2014_questions.json
├── vqav2_testdev.jsonl
├── vqav2_train.jsonl
└── vqav2_val.jsonl
```

### TextVQA

Follow the instructions below to prepare the data:

```shell
# Step 1: Create the data directory
mkdir -p data/textvqa && cd data/textvqa

# Step 2: Download images
wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip && unzip train_val_images.zip

# Step 3: Download converted files
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_train_annotations.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_train_questions.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_train.jsonl
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_val_annotations.json
wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/textvqa/textvqa_val_questions.json
wget https://huggingface.co/OpenGVLab/InternVL/raw/main/textvqa_val.jsonl
wget https://huggingface.co/OpenGVLab/InternVL/raw/main/textvqa_val_llava.jsonl

cd ../..
```

After preparation is complete, the directory structure is:

```shell
data/textvqa
├── TextVQA_Rosetta_OCR_v0.2_test.json
├── TextVQA_Rosetta_OCR_v0.2_train.json
├── TextVQA_Rosetta_OCR_v0.2_val.json
├── textvqa_train_annotations.json
├── textvqa_train.jsonl
├── textvqa_train_questions.json
├── textvqa_val_annotations.json
├── textvqa_val.jsonl
├── textvqa_val_llava.jsonl
├── textvqa_val_questions.json
└── train_images
```

### AI2D

Follow the instructions below to prepare the data：

```bash
# Step 1: Create the data directory
mkdir -p data/ai2diagram && cd data/ai2diagram

# Step 2: Download converted files
wget https://huggingface.co/OpenGVLab/InternVL/raw/main/ai2d_test_vlmevalkit.jsonl -O test_vlmevalkit.jsonl
wget https://huggingface.co/OpenGVLab/InternVL/resolve/main/AI2D_TEST.zip && unzip AI2D_TEST.zip

# Step 3: Download images from Google Drive (optional, provided by InternLM-XComposer)
# https://drive.google.com/file/d/1dqqa3MnrxMXaU_K9JA6C83je32ibwdOY/view?usp=sharing
# images should be placed in `data/ai2diagram/ai2d/abc_images` and `data/ai2diagram/ai2d/images`
cd ../..
```

After preparation is complete, the directory structure is:

```
data/ai2diagram
 ├── test_vlmevalkit.jsonl
 ├── ai2d # (optional)
 │    ├── abc_images
 │    └── images
 └── AI2D_TEST
```

## 🏃 Evaluation Execution

To run the evaluation, execute the following command on an 8-GPU setup:

```shell
torchrun --nproc_per_node=8 eval/caption/evaluate_caption.py --checkpoint ${CHECKPOINT} --datasets ${DATASETS}
```

Alternatively, you can run the following simplified command:

```shell
# Test VQAv2 val
GPUS=8 sh evaluate.sh ${CHECKPOINT} vqa-vqav2-val
# Test VQAv2 testdev
GPUS=8 sh evaluate.sh ${CHECKPOINT} vqa-vqav2-testdev
# Test AI2D test
GPUS=8 sh evaluate.sh ${CHECKPOINT} vqa-ai2d-test
# Test TextVQA val
GPUS=8 sh evaluate.sh ${CHECKPOINT} vqa-textvqa-val
```

### Arguments

The following arguments can be configured for the evaluation script:

| Argument         | Type   | Default     | Description                                                                                                       |
| ---------------- | ------ | ----------- | ----------------------------------------------------------------------------------------------------------------- |
| `--checkpoint`   | `str`  | `''`        | Path to the model checkpoint.                                                                                     |
| `--datasets`     | `str`  | `okvqa_val` | Comma-separated list of datasets to evaluate.                                                                     |
| `--dynamic`      | `flag` | `False`     | Enables dynamic high resolution preprocessing.                                                                    |
| `--max-num`      | `int`  | `6`         | Maximum tile number for dynamic high resolution.                                                                  |
| `--load-in-8bit` | `flag` | `False`     | Loads the model weights in 8-bit precision.                                                                       |
| `--auto`         | `flag` | `False`     | Automatically splits a large model across 8 GPUs when needed, useful for models too large to fit on a single GPU. |
