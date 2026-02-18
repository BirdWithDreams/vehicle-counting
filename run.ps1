param (
    [Parameter(Mandatory=$true)]
    [string]$command
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "$PWD"

switch ($command) {
    "sanity" {
        Write-Host "Running sanity check..."
        uv run scripts/sanity_check.py
    }
    "train" {
        Write-Host "Training models with configs at configs/train/* ..."
        uv run ./run_sequence.py
    }
    "train_single" {
        Write-Host "Training models with unified classes..."
        uv run ./run_sequence.py --pattern single*.yaml
    }
    "train_multy" {
        Write-Host "Training models with different classes..."
        uv run ./run_sequence.py --pattern multy*.yaml
    }
    "eval" {
        Write-Host "Evaluating model(s)..."
        uv run scripts/eval.py --weights-dir runs/detect/VehicleCountingModel --pattern */weights/best.pt
    }
    "compare" {
        Write-Host "Comparing models..."
        uv run scripts/compare.py
    }
    Default {
        Write-Error "Unknown command: $command. Available commands: sanity, train, train_single, train_multy, eval, compare"
        exit 1
    }
}
