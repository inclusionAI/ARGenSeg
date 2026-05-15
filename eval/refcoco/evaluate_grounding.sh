CHECKPOINT="pretrained/InternVL2_5-ARGenSeg-8B"

torchrun --nproc_per_node=8 eval/refcoco/evaluate_grounding.py --checkpoint ${CHECKPOINT} \
  --datasets 'refcoco_testA'
  # --dynamic
  # --datasets 'refcoco_testA' \