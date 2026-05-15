# Evaluation Guide

This document provides instructions for evaluating ARGenSeg on various benchmarks.

## Data Preparation

> **Note:** For detailed data download instructions for specific benchmarks, please refer to their respective README files linked below.

### Common Data: COCO Images

COCO images are required for multiple evaluation tasks:

```bash
# Create the data directory
mkdir -p data/coco && cd data/coco

# Download and unzip image files
wget http://images.cocodataset.org/zips/train2014.zip && unzip train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip && unzip val2014.zip

cd ../..
```

Directory structure:
```
data/coco
├── train2014
└── val2014
```

### Benchmark-Specific Data

| Benchmark | Data Guide |
|-----------|------------|
| RefCOCO Series (Segmentation) | [eval/refcoco_seg/README.md](./refcoco_seg/README.md) |
| RefCOCO Series (Comprehension) | [eval/refcoco/README.md](./refcoco/README.md) |
| POPE | [eval/pope/README.md](./pope/README.md) |
| VQA (TextVQA) | [eval/vqa/README.md](./vqa/README.md) |
| MMMU | [eval/mmmu/README.md](./mmmu/README.md) |

## Quick Start

### Referring Expression Segmentation

```bash
# Evaluate on all RefCOCO splits
bash eval/refcoco_seg/multi_eval_refcoco.sh 1,2,3,4,5,6,7,8

# Evaluate on gRefCOCO
bash eval/refcoco_seg/multi_eval_refcoco.sh 9,10,11
```

See [eval/refcoco_seg/README.md](./refcoco_seg/README.md) for detailed instructions.

### Visual Grounding (Comprehension)

```bash
torchrun --nproc_per_node=8 eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} \
  --datasets 'refcoco_testA'
```
See [eval/refcoco/README.md](./refcoco/README.md) for detailed instructions.

### Other Benchmarks

| Benchmark | Command |
|-----------|---------|
| [POPE](./pope/README.md) | `GPUS=8 sh evaluate.sh ${CHECKPOINT} pope` |
| [MMMU-val](./mmmu/README.md) | `GPUS=8 sh evaluate.sh ${CHECKPOINT} mmmu-val` |
| [TextVQA](./vqa/README.md) | `GPUS=8 sh evaluate.sh ${CHECKPOINT} vqa-textvqa-val` |