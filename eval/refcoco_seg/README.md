# Referring Expression Segmentation Evaluation

This directory contains evaluation scripts for referring expression segmentation on RefCOCO series benchmarks.

## Quick Start

### Step 1: Prepare Data

```bash
# Create data directory
mkdir -p data/refcoco && cd data/refcoco
ln -s ../coco/train2014 ./

# Download PSALM format annotations
# Google Drive: https://drive.google.com/file/d/1EcC1tl1OQRgIqqy7KFG7JZz2KHujAQB3/view
# Baidu Cloud: https://pan.baidu.com/s/1NRGJGkJDUGn8CU-sU5ScOg (code: hust)
```

Expected directory structure:
```
data/refcoco
├── refcoco/
│   ├── val_psalm.json
│   ├── testA_psalm.json
│   └── testB_psalm.json
├── refcoco+/
│   ├── val_psalm.json
│   ├── testA_psalm.json
│   └── testB_psalm.json
├── refcocog/
│   ├── val_psalm.json
│   └── test_psalm.json
├── grefcoco/
│   ├── grefcoco_val.json
│   ├── grefcoco_testA.json
│   └── grefcoco_testB.json
└── train2014/  # COCO train2014 images
```

### Step 2: Run Evaluation

```bash
# Evaluate on all RefCOCO splits
sh eval/refcoco_seg/multi_eval_refcoco.sh 1,2,3,4,5,6,7,8

# Evaluate on gRefCOCO
sh eval/refcoco_seg/multi_eval_refcoco.sh 9,10,11

# Evaluate on specific split (e.g., RefCOCO val)
sh eval/refcoco_seg/multi_eval_refcoco.sh 1
```

### Step 3: Check Results

Results are saved to:
```
eval/refcoco_seg/res/YYYYMMDD-HHMMSS-{model_name}/
├── 1.log (RefCOCO val)
├── 2.log (RefCOCO+ val)
└── ...
```

Each log contains **cIoU** and **gIoU** metrics.