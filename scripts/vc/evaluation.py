from pathlib import Path
import yaml
import numpy as np
from ultralytics.data.utils import check_det_dataset

def _load_dataset(data_yaml: str):
    data = check_det_dataset(data_yaml)
    return data  # includes paths to labels/images per split

def _count_labels(label_path: Path) -> int:
    if not label_path.exists():
        return 0
    # YOLO labels: cls x y w h
    n = 0
    for line in label_path.read_text().strip().splitlines():
        if line.strip():
            n += 1
    return n

def evaluate_counting(
    model,
    data_yaml: str,
    imgsz: int,
    conf: float,
    iou: float,
    split: str = "val",
    merge_to_vehicle: bool = True,
    match_iou: float = 0.5,
):
    data = _load_dataset(data_yaml)
    img_dir = Path(data[split])
    # Ultralytics dataset dict also provides label paths via data['labels']
    # but for simplicity we use convention: labels mirror images path with /labels and .txt
    # If your structure differs, adjust here.

    # Predict over split folder
    preds = model.predict(
        source=str(img_dir),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        verbose=False,
        save=False,
        stream=False,
        max_det=300,
    )

    per_image = []
    for r in preds:
        img_path = Path(r.path)
        # Common Roboflow structure: .../images/xxx.jpg and labels in .../labels/xxx.txt
        label_path = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")

        gt = _count_labels(label_path)
        pred = 0 if r.boxes is None else len(r.boxes)

        per_image.append({"image": str(img_path), "gt": gt, "pred": pred, "err": pred - gt})

    errs = np.array([x["err"] for x in per_image], dtype=float)
    gt_counts = np.array([x["gt"] for x in per_image], dtype=float)

    mae = float(np.mean(np.abs(errs)))
    rmse = float(np.sqrt(np.mean(errs**2)))
    mape = float(np.mean(np.abs(errs) / np.maximum(gt_counts, 1.0)))  # avoid div by zero

    return {"n": len(per_image), "mae": mae, "rmse": rmse, "mape": mape, "per_image": per_image}
