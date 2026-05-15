set -x

declare -A TEST_SETS=(
    ["1"]="refcoco/val_psalm.json"
    ["2"]="refcoco+/val_psalm.json"
    ["3"]="refcocog/val_psalm.json"
    ["4"]="refcoco/testA_psalm.json"
    ["5"]="refcoco/testB_psalm.json"
    ["6"]="refcoco+/testA_psalm.json"
    ["7"]="refcoco+/testB_psalm.json"
    ["8"]="refcocog/test_psalm.json"
    ["9"]="grefcoco/grefcoco_val.json"
    ["10"]="grefcoco/grefcoco_testA.json"
    ["11"]="grefcoco/grefcoco_testB.json"
)

CHOICES=$1

if [ -z "$CHOICES" ]; then
    echo "Usage: $0 {1,2,3,...}"
    exit 1
fi

IFS=',' read -r -a CHOICE_ARRAY <<< "$CHOICES"

timestamp=$(date +"%Y%m%d-%H%M%S")

checkpoint="pretrained/InternVL2_5-ARGenSeg-8B"


for CHOICE in "${CHOICE_ARRAY[@]}"; do
    if [ -z "${TEST_SETS[$CHOICE]}" ]; then
        echo "Invalid choice: $CHOICE. Skipping..."
        continue
    fi

    output_file="eval/refcoco_seg/res/${timestamp}/$CHOICE.log"
    OUTPUT_DIR="eval/refcoco_seg/res/${timestamp}"
    if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    fi

    echo "Evaluation Data is: ${TEST_SETS[$CHOICE]}"
    echo "Log path: $output_file"

    # echo
    # nohup \
    # CUDA_VISIBLE_DEVICES=0 \
    torchrun --nproc_per_node=8 \
        eval/refcoco_seg/referring_segmentation.py \
            --model_path $checkpoint \
            --image_folder data/refcoco/train2014 \
            --json_path data/refcoco/${TEST_SETS[$CHOICE]} \
            --dynamic_image_size 1 \
            --gen_img_reso 256 \
            --pad2square False \
            2>&1 | tee -a "$output_file"
        # > "$output_file" 2>&1 &
done