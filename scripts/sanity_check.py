from dotenv import load_dotenv
load_dotenv()

import os
from pathlib import Path
import yaml

def main():
    assert os.getenv("COMET_API_KEY"), "COMET_API_KEY missing (set it in .env)"
    for p in ["configs/project.yaml", "configs/train/base.yaml", "configs/eval/base.yaml"]:
        assert Path(p).exists(), f"Missing: {p}"

    cfg = yaml.safe_load(Path("configs/project.yaml").read_text())
    print("Project:", cfg["project_name"])
    print("Artifacts dir:", cfg["paths"]["artifacts_dir"])
    print("Runs dir:", cfg["paths"]["runs_dir"])
    print("OK")

if __name__ == "__main__":
    main()
