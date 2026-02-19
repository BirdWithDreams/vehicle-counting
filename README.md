# Vehicle Counting Model

## Project Structure

```text
VehicleCountingModel/
├── artifacts/          # Generated artifacts (leaderboards, evaluations)
├── configs/            # Configuration files (hydra/omegaconf style)
├── data/               # Dataset directory
├── runs/               # Training results (Ultralytics & CometML)
├── scripts/            # Core python scripts
├── run_sequence.py     # Automation script for running multiple experiments
...
```

## Scripts & Workflow

- **`scripts/train.py`**: Trains the model based on provided configs.
  - **Input**: Config files from `configs/train/`.
  - **Output**: Trained model in `runs/detect/<model_name>` and summary in `train_result_summary.json`.
- **`scripts/eval.py`**: Evaluates trained models on the test set.
  - **Input**: Path to trained weights or directory.
  - **Output**: Evaluation metrics (JSON) and per-image counts in `artifacts/experiments/evals`.
- **`scripts/compare.py`**: Aggregates evaluation results into a leaderboard.
  - **Input**: Metrics from `artifacts/experiments/evals`.
  - **Output**: `artifacts/leaderboards/leaderboard.md` and `.csv`.
- **`run_sequence.py`**: Utility to run multiple training configurations sequentially.
  - Supports `--include` to explicitly select configs and `--skip` to exclude them.
  - Logic: `selected = set(include) - set(skip)`.
  - Example: `python run_sequence.py --include single_baseline_y26n --skip single_aug_y26n`

### Run Scripts (`run.ps1` / `run.sh`)

Convenience wrappers to execute common tasks.
Usage: `./run.ps1 <command>` (PowerShell) or `./run.sh <command>` (Bash)

Available commands:
- **`sanity`**: Runs `scripts/sanity_check.py` to verify environment and configs.
- **`train`**: Runs all training configs found in `configs/train/vehicles`.
- **`train_single`**: Runs only single-class configs (`single*.yaml`).
- **`train_multy`**: Runs only multi-class configs (`multy*.yaml`).
- **`eval`**: Evaluates all models in `runs/detect/VehicleCountingModel`.
- **`compare`**: Generates the leaderboard from evaluation results.


## Configs Explanation (`configs/train/vehicles`)

- **`*_baseline`**: Standard training setup (640sz, SGD, 120ep). Serves as a starting point for comparisons.
- **`single_aug`**: Applies heavy augmentation (Mosaic 1.0, Mixup 0.15, HSV, Flip, etc.) to improve generalization on the single vehicle class.
- **`*_freeze`**: Freezes the first 10 layers of the backbone. Useful for preventing overfitting or speeding up training on smaller datasets.
- **`multy_imbalance`**: Uses Focal Loss (`fl_gamma: 1.5`) and class weighting to handle class imbalance in multi-class detection scenarios.
- **`single_hires`**: Uses high-resolution input (960px). Beneficial for detecting small objects but requires more computational resources.
- **`multy_stable` / `multy_prod`**: The "Production" configuration. Uses AdamW optimizer, longer training (220ep), cosine LR schedule, and patience to maximize performance.

## Leaderboard & Analysis

| model_id             | task     |    map50 |   map50_95 |   counting_mae |   counting_rmse |   counting_mape | weights                                                               |
|:---------------------|:---------|---------:|-----------:|---------------:|----------------:|----------------:|:----------------------------------------------------------------------|
| multy_prod_y26n      | Multiple | 0.764472 |   0.57275  |        1.53968 |         1.93136 |         1.53968 | runs\detect\VehicleCountingModel\multy_prod_y26n\weights\best.pt      |
| single_stable_y26n   | Single   | 0.775779 |   0.55261  |        1.68254 |         2.10064 |         1.68254 | runs\detect\VehicleCountingModel\single_stable_y26n\weights\best.pt   |
| multy_baseline_y26n  | Multiple | 0.777155 |   0.578014 |        1.71429 |         2.31626 |         1.71429 | runs\detect\VehicleCountingModel\multy_baseline_y26n\weights\best.pt  |
| single_freeze_y26n   | Single   | 0.706583 |   0.552867 |        1.73016 |         2.29907 |         1.73016 | runs\detect\VehicleCountingModel\single_freeze_y26n\weights\best.pt   |
| single_hires_y26n    | Single   | 0.640864 |   0.449799 |        1.73016 |         2.32652 |         1.73016 | runs\detect\VehicleCountingModel\single_hires_y26n\weights\best.pt    |
| single_aug_y26n      | Single   | 0.774641 |   0.57017  |        1.87302 |         2.34352 |         1.87302 | runs\detect\VehicleCountingModel\single_aug_y26n\weights\best.pt      |
| multy_freeze_y26n    | Multiple | 0.763698 |   0.569126 |        1.93651 |         2.31626 |         1.93651 | runs\detect\VehicleCountingModel\multy_freeze_y26n\weights\best.pt    |
| single_baseline_y26n | Single   | 0.72443  |   0.536141 |        2.03175 |         2.55107 |         2.03175 | runs\detect\VehicleCountingModel\single_baseline_y26n\weights\best.pt |
| multy_imbalance_y26n | Multiple | 0.696639 |   0.491176 |        2.25397 |         3.10657 |         2.25397 | runs\detect\VehicleCountingModel\multy_imbalance_y26n\weights\best.pt |

**Analysis**:
- **`multy_prod_y26n` (Multi-class Stable)** achieves the best performance (~0.76 mAP50, ~1.54 MAE), balancing detection accuracy with counting precision.
- **`single_aug_y26n`** demonstrates that heavy augmentation significantly improves performance over the baseline (~0.77 vs 0.72 mAP), suggesting better generalization.
- **`single_hires_y26n`** performed worse than expected, possibly due to the smaller batch size required by higher resolution or optimization difficulties.
- **`single_freeze_y26n`** offers stable performance, slightly lower than full training, but establishes a solid baseline with less computational cost.

## Running Inference

Run YOLO with the best model weights using the following command template:

```
yolo TASK MODE ARGS
```

Example (using `multy_prod_y26n`):

```bash
yolo detect predict \
    model=runs/detect/VehicleCountingModel/multy_prod_y26n/weights/best.pt \
    source=path/to/image.jpg \
    conf=0.20 \
    iou=0.60 \
    imgsz=640
```

## Metrics Explanation

- **mAP50 (Mean Average Precision @ IoU 0.5)**: A standard object detection metric measuring the average precision of checking if a predicted box overlaps with a ground truth box by at least 50%.
- **mAP50-95**: The average of mAP calculated at IoU thresholds from 0.5 to 0.95 (step 0.05). This is a stricter measure of localization accuracy.
- **Counting MAE (Mean Absolute Error)**: The average absolute difference between the predicted count of vehicles and the actual count per image. Lower values indicate better counting performance.
- **Counting RMSE (Root Mean Squared Error)**: Similar to MAE but penalizes larger errors more heavily. Useful for identifying if the model has occasional large counting misses.
- **Counting MAPE (Mean Absolute Percentage Error)**: The average percentage error in counting. This metric is useful for understanding the relative error magnitude regardless of the total number of vehicles.
