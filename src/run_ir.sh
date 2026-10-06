#!/bin/bash
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for seed in 0 1 2; do for fold in 0 1 2 3 4; do for task in person gender; do
  .venv/bin/python src/ir_eval.py --task $task --fold $fold --seed $seed >> logs/ir.log 2>&1
done; done; done
echo ALLDONE >> logs/ir.log
