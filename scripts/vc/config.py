from pathlib import Path
import yaml

def load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())

def merge_inherits(cfg: dict, base_dir: str = "configs") -> dict:
    """Supports simple 'inherits: train/base.yaml' pattern."""
    inherits = cfg.get("inherits")
    if not inherits:
        return cfg
    base = load_config(str(Path(base_dir) / inherits))
    base.update({k: v for k, v in cfg.items() if k != "inherits"})
    return base
