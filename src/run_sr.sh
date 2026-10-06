#!/bin/bash
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() {  # dataset seed epochs tasks...
  ds=$1; seed=$2; ep=$3; shift 3
  for fold in 0 1 2 3 4; do for task in "$@"; do for m in yolo26n yolo26s rtdetr-l; do
    b=8; [ $m = rtdetr-l ] && b=4
    .venv/bin/python src/sr_eval.py --dataset $ds --task $task --model $m --fold $fold --seed $seed --epochs $ep --batch $b >> logs/sr.log 2>&1
  done; done; done
}
run indoor 0 100 person gender allclass
run outdoor 0 30 person gender posture allclass
run indoor 1 100 person gender allclass
run indoor 2 100 person gender allclass
echo ALLDONE >> logs/sr.log
