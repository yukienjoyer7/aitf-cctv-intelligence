#!/bin/bash
# Run the full CCTV-SR benchmark: every (dataset, fold, task, model, seed) job through src/sr_eval.py.
# Each job writes results/sr/<tag>.json and is skipped if that file exists, so this script can be stopped and
# restarted at any time (this is how the benchmark was run). Output goes to logs/sr.log.
#
# Budget notes:
#   * indoor  (~90 train frames/fold): 100 epochs.
#   * outdoor (~730 train frames/fold, about 8x more): 30 epochs, which is still more optimisation steps.
#   * RT-DETR-L uses batch 4 because batch 8 ran out of memory on the 6 GB GPU; the YOLO models use batch 8.
# NOTE: in practice only indoor seed 0 and outdoor seed 0 (folds 0-3 for most tasks) were completed before the
# run was stopped; the later calls below are the intended full plan.
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # reduces GPU memory fragmentation

# usage: run <dataset> <seed> <epochs> <task>...
run() {
  local dataset=$1 seed=$2 epochs=$3
  shift 3
  for fold in 0 1 2 3 4; do
    for task in "$@"; do
      for model in yolo26n yolo26s rtdetr-l; do
        batch=8
        [ "$model" = rtdetr-l ] && batch=4
        .venv/bin/python src/sr_eval.py --dataset "$dataset" --task "$task" --model "$model" \
          --fold "$fold" --seed "$seed" --epochs "$epochs" --batch "$batch" >> logs/sr.log 2>&1
      done
    done
  done
}

run indoor 0 100 person gender allclass
run outdoor 0 30 person gender posture allclass
run indoor 1 100 person gender allclass
run indoor 2 100 person gender allclass
echo ALLDONE >> logs/sr.log
