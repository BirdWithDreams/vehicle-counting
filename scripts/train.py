import os
from pathlib import Path
import click
from dotenv import load_dotenv

load_dotenv()

import comet_ml
comet_ml.login()

from ultralytics import YOLO

from vc.config import load_config, merge_inherits
from vc.utils import set_seed, ensure_dir, dump_json
from vc.paths import get_experiment_dir

@click.command()
@click.option("--project", default="configs/project.yaml", help="Path to project config", show_default=True)
@click.option("--train", required=True, help="Path to train config (e.g. configs/train/yolo11s_768_adamw.yaml)")
def main(project, train):
    project_cfg = load_config(project)
    train_cfg = merge_inherits(load_config(train), base_dir="configs")

    set_seed(project_cfg["seed"])

    exp_dir = get_experiment_dir(project_cfg, train_cfg)
    ensure_dir(exp_dir)

    # Save resolved configs for reproducibility
    dump_json(exp_dir / "resolved_project.json", project_cfg)
    dump_json(exp_dir / "resolved_train.json", train_cfg)

    run_name = train_cfg.get("name")
    if not run_name:
        run_name = Path(train).stem

    # Set Comet ML environment variables explicitly
    os.environ["COMET_PROJECT_NAME"] = project_cfg["project_name"]
    os.environ["COMET_EXPERIMENT_NAME"] = run_name

    model = YOLO(train_cfg["model"])

    results = model.train(
        data=train_cfg["data"],
        epochs=train_cfg["epochs"],
        imgsz=train_cfg["imgsz"],
        batch=train_cfg["batch"],
        patience=train_cfg["patience"],
        optimizer=train_cfg["optimizer"],
        lr0=train_cfg["lr0"],
        freeze=train_cfg.get("freeze"),
        lrf=train_cfg["lrf"],
        weight_decay=train_cfg["weight_decay"],
        warmup_epochs=train_cfg["warmup_epochs"],
        cos_lr=train_cfg["cos_lr"],
        mosaic=train_cfg["mosaic"],
        mixup=train_cfg["mixup"],
        close_mosaic=train_cfg["close_mosaic"],
        scale=train_cfg["scale"],
        translate=train_cfg["translate"],
        degrees=train_cfg["degrees"],
        shear=train_cfg["shear"],
        perspective=train_cfg["perspective"],
        hsv_h=train_cfg["hsv_h"],
        hsv_s=train_cfg["hsv_s"],
        hsv_v=train_cfg["hsv_v"],
        fliplr=train_cfg["fliplr"],
        flipud=train_cfg["flipud"],
        amp=train_cfg["amp"],
        deterministic=train_cfg["deterministic"],
        device=project_cfg["device"],
        workers=project_cfg["workers"],
        cache=project_cfg["cache"],
        project=project_cfg["project_name"],
        name=run_name,
        save=train_cfg["save"],
        save_period=train_cfg["save_period"],
        val=train_cfg["val"],
    )

    # Ultralytics already writes run artifacts; we also store a pointer
    dump_json(exp_dir / "train_result_summary.json", {"run_dir": str(results.save_dir)})

if __name__ == "__main__":
    main()
