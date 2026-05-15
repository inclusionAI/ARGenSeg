CHECKPOINT="pretrained/InternVL2_5-ARGenSeg-8B"


GPUS=8 sh evaluate.sh ${CHECKPOINT} pope \
# --dynamic