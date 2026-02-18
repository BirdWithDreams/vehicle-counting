import json
from pathlib import Path
import random
import numpy as np
import os
from datetime import datetime

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def dump_json(path: Path, obj) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2))

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")
