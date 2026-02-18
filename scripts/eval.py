import json
from pathlib import Path
from typing import List, Optional, Tuple

import click
from dotenv import load_dotenv
from ultralytics import YOLO

from vc.config import load_config, merge_inherits
from vc.evaluation import evaluate_counting
from vc.utils import dump_json, ensure_dir, now_iso


def _looks_like_glob(s: str) -> bool:
    return any(ch in s for ch in ["*", "?", "[", "]"])


def _resolve_weights(
        weights: Tuple[str, ...],
        weights_dir: Optional[str],
        pattern: str,
) -> List[Path]:
    """
    Resolve:
      - explicit files
      - globs
      - directory recursive search
    into a de-duplicated, sorted list of existing weight files.
    """
    found: List[Path] = []

    # explicit weights / globs
    for w in weights:
        w = str(w).strip()
        if not w:
            continue
        if _looks_like_glob(w):
            found.extend(Path().glob(w))
        else:
            p = Path(w)
            if p.is_dir():
                found.extend(p.rglob(pattern))
            else:
                found.append(p)

    # weights_dir recursive
    if weights_dir:
        d = Path(weights_dir)
        if d.exists() and d.is_dir():
            found.extend(d.rglob(pattern))
        else:
            raise click.ClickException(f"--weights-dir does not exist or is not a directory: {weights_dir}")

    # keep only existing files
    files = [p for p in found if p.exists() and p.is_file()]

    # de-duplicate by resolved path
    uniq = {}
    for p in files:
        try:
            uniq[str(p.resolve())] = p
        except Exception:
            uniq[str(p)] = p

    # stable order
    out = sorted(uniq.values(), key=lambda x: str(x))
    return out


def _write_jsonl(path: Path, rows: List[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: List[dict]) -> None:
    # Minimal CSV writer without pandas dependency
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    # union of keys for header (stable-ish)
    keys = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)

    def esc(v) -> str:
        s = "" if v is None else str(v)
        if any(c in s for c in [",", '"', "\n", "\r"]):
            s = '"' + s.replace('"', '""') + '"'
        return s

    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(esc(r.get(k)) for k in keys) + "\n")


@click.command()
@click.option("--project", default="configs/project.yaml", help="Path to project config", show_default=True)
@click.option("--data", default="./data/data.yaml", required=True, help="Path to data config")
@click.option(
    "--eval", "eval_config", default="configs/eval/counting.yaml", help="Path to eval config", show_default=True
    )
@click.option(
    "--weights",
    multiple=True,
    required=False,
    help="Path(s) to model weights OR glob(s) OR directory(ies). Can be provided multiple times.",
)
@click.option(
    "--weights-dir",
    required=False,
    default=None,
    help="Directory to recursively search for weights (e.g. runs/detect/VehicleCountingModel).",
)
@click.option(
    "--pattern",
    default="**/weights/best.pt",
    show_default=True,
    help="Recursive pattern used when searching directories.",
)
@click.option("--out", default="artifacts/experiments/evals", help="Output directory", show_default=True)
@click.option(
    "--fail-fast/--no-fail-fast",
    default=False,
    show_default=True,
    help="Stop on first failure (default: continue and record in errors.jsonl).",
)
def main(project, data, eval_config, weights, weights_dir, pattern, out, fail_fast):
    load_dotenv()

    project_cfg = load_config(project)
    eval_cfg = merge_inherits(load_config(eval_config), base_dir="configs")

    weight_files = _resolve_weights(weights=weights, weights_dir=weights_dir, pattern=pattern)
    if not weight_files:
        raise click.ClickException(
            "No weight files found. Provide --weights (file/glob/dir) and/or --weights-dir."
        )

    # One run root folder for this evaluation batch
    run_root = Path(out) / f"eval_{now_iso().replace(':', '-')}"
    ensure_dir(run_root)

    summary_rows: List[dict] = []
    error_rows: List[dict] = []

    click.echo(f"Found {len(weight_files)} weight file(s). Saving run to: {run_root}")

    for wf in weight_files:
        wf = Path(wf)

        args = load_config(wf.parent.parent / "args.yaml")
        model_id = args["name"]
        model_out = run_root / model_id
        ensure_dir(model_out)

        click.echo(f"\n=== Evaluating: {wf} ===")
        click.echo(f"Model ID: {model_id}")

        try:
            model = YOLO(str(wf))

            # Ultralytics validation (mAP)
            val_res = model.val(
                data=data,
                imgsz=args["imgsz"],
                conf=eval_cfg["conf"],
                iou=eval_cfg["iou"],
                max_det=eval_cfg["max_det"],
                device=project_cfg["device"],
                split=eval_cfg["split"],
                project=project_cfg["project_name"] + "Eval",
                name=model_id,
                save_json=True,
                single_cls=args["single_cls"],
            )

            # Counting metrics (image-level count error)
            counting_res = evaluate_counting(
                model=model,
                data_yaml=data,
                imgsz=args["imgsz"],
                conf=eval_cfg["conf"],
                iou=eval_cfg["iou"],
                split=eval_cfg["split"],
                merge_to_vehicle=eval_cfg.get("counting", {}).get("merge_to_vehicle", True),
                match_iou=eval_cfg.get("counting", {}).get("match_iou", 0.5),
            )

            metrics = {
                "model_id": model_id,
                "weights": str(wf),
                "split": eval_cfg["split"],
                "map50": float(val_res.box.map50),
                "map50_95": float(val_res.box.map),
                "counting_mae": float(counting_res["mae"]),
                "counting_rmse": float(counting_res["rmse"]),
                "counting_mape": float(counting_res["mape"]),
                "n_images": int(counting_res["n"]),
            }

            dump_json(model_out / "metrics.json", metrics)
            dump_json(model_out / "counting_per_image.json", counting_res["per_image"])

            summary_rows.append(metrics)
            click.echo(f"Saved: {model_out}")

        except Exception as e:
            err = {
                "model_id": model_id,
                "weights": str(wf),
                "error": repr(e),
            }
            error_rows.append(err)
            click.echo(f"[ERROR] {model_id}: {e}")

            if fail_fast:
                raise

    # Aggregates
    _write_jsonl(run_root / "summary.jsonl", summary_rows)
    _write_csv(run_root / "summary.csv", summary_rows)
    if error_rows:
        _write_jsonl(run_root / "errors.jsonl", error_rows)

    click.echo("\n=== Done ===")
    click.echo(f"Summary: {run_root / 'summary.jsonl'}")
    click.echo(f"CSV:     {run_root / 'summary.csv'}")
    if error_rows:
        click.echo(f"Errors:  {run_root / 'errors.jsonl'}")


if __name__ == "__main__":
    main()
