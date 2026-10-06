#!/bin/bash
cd "$(dirname "$0")/.."
for seed in 0 1 2; do for fold in 0 1 2 3 4; do for task in person gender; do for m in yolo26n yolo26s rtdetr-l; do
  .venv/bin/python src/train_eval.py --model $m --task $task --fold $fold --seed $seed --epochs 100 >> logs/train.log 2>&1
done; done; done; done
echo ALLDONE >> logs/train.log
