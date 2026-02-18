#!/bin/bash
set -e
export PYTHONPATH=.

COMMAND=$1

case "$COMMAND" in
    "sanity")
        echo "Running sanity check..."
        uv run scripts/sanity_check.py
        ;;
    "train")
        echo "Training models with configs at configs/train/* ..."
        uv run ./run_sequence.py
        ;;
    "train_single")
        echo "Training models with unified classes..."
        uv run scripts/train.py ./run_sequence.py --pattern single*.yaml
        ;;
    "train_multy")
      echo "Training models with different classes..."
      uv run scripts/train.py ./run_sequence.py --pattern multy*.yamll
      ;;
    "eval")
        echo "Evaluating model(s)..."
        uv run scripts/eval.py --weights-dir runs/detect/VehicleCountingModel --pattern */weights/best.pt
        ;;
    "compare")
        echo "Comparing models..."
        uv run scripts/compare.py
        ;;
    *)
        echo "Unknown command: $1. Available commands: sanity, train, train_single, train_multy, eval, compare"
        exit 1
        ;;
esac
