from pathlib import Path

def get_experiment_dir(project_cfg: dict, train_cfg: dict) -> Path:
    root = Path(project_cfg["paths"]["artifacts_dir"]) / "experiments"
    tag = train_cfg.get("name", "exp")
    mode = ["multy", "single"][train_cfg["single_cls"]]
    return root / f"{mode}__{tag}"
