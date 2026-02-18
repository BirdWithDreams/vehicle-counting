import click
from pathlib import Path
import pandas as pd
import yaml

from vc.utils import ensure_dir

@click.command()
@click.option("--compare", default="configs/compare/leaderboard.yaml", help="Path to leaderboard config", show_default=True)
@click.option("--metrics_root", default="artifacts/experiments/evals", help="Path to metrics root directory", show_default=True)
@click.option("--out", default="artifacts/leaderboards", help="Output directory", show_default=True)
def main(compare, metrics_root, out):
    cfg = yaml.safe_load(Path(compare).read_text())

    metrics_files = list(Path(metrics_root).rglob("metrics.json"))
    if not metrics_files:
        raise SystemExit(f"No metrics.json found under {metrics_root}")

    rows = []
    for f in metrics_files:
        rows.append(pd.read_json(f, typ="series"))
    df = pd.DataFrame(rows)
    df["task"] = ""

    cols = ["model_id", "task"] + cfg["include"] + ["weights"]
    df = df[cols].sort_values(cfg["sort_by"], ascending=True)

    out_dir = Path(out)
    ensure_dir(out_dir)
    df['task'] = df['model_id'].apply(lambda x: ["Multiple", "Single"]["single" in x])
    df.to_csv(out_dir / "leaderboard.csv", index=False)
    df.to_markdown(out_dir / "leaderboard.md", index=False)
    print("Wrote:", out_dir)

if __name__ == "__main__":
    main()
