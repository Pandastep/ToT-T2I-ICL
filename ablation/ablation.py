#ablation.py
import argparse
import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import csv


@dataclass
class VariantResult:
    name: str
    results_dir: str
    eval_out_dir: str
    summary_path: str
    n_scored: Optional[int]
    clip_mean: Optional[float]
    clip_std: Optional[float]
    clip_min: Optional[float]
    clip_max: Optional[float]


def run_cmd(cmd: str, cwd: Optional[str] = None) -> None:
    # Windows-friendly: allow string commands; use shell=True for simplicity with quoted paths.
    print(f"\n[CMD] {cmd}")
    completed = subprocess.run(cmd, cwd=cwd, shell=True)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with code {completed.returncode}: {cmd}")


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_summary_json(eval_out_dir: str) -> str:
    """
    Your evaluation script seems to output summary.json in out_dir.
    If your filename differs, adjust here.
    """
    cand = os.path.join(eval_out_dir, "summary.json")
    if os.path.isfile(cand):
        return cand
    # fallback: search
    for root, _, files in os.walk(eval_out_dir):
        if "summary.json" in files:
            return os.path.join(root, "summary.json")
    raise FileNotFoundError(f"summary.json not found under: {eval_out_dir}")


def parse_clip_metrics(summary: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """
    Expects format like:
    summary['aggregates']['by_method'][0] contains clip_mean/std/min/max, n
    """
    by_method = summary.get("aggregates", {}).get("by_method", [])
    if not by_method:
        return dict(n=None, mean=None, std=None, min=None, max=None)

    row = by_method[0]
    return dict(
        n=row.get("n"),
        mean=row.get("clip_mean"),
        std=row.get("clip_std"),
        min=row.get("clip_min"),
        max=row.get("clip_max"),
    )


def to_latex_table(rows: List[VariantResult]) -> str:
    # Compact IEEE-friendly table (small)
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{CLIP similarity ablation for Tree-of-Thought variants.}")
    lines.append("\\label{tab:ablation_tot}")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\hline")
    lines.append("Variant & Mean & Std & Min & Max \\\\")
    lines.append("\\hline")
    for r in rows:
        if r.clip_mean is None:
            lines.append(f"{r.name} & - & - & - & - \\\\")
        else:
            lines.append(
                f"{r.name} & {r.clip_mean:.4f} & {r.clip_std:.4f} & {r.clip_min:.4f} & {r.clip_max:.4f} \\\\"
            )
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="ablation_config.json", help="Path to ablation config JSON.")
    ap.add_argument("--cwd", type=str, default=".", help="Project root (where main3.py/evaluation.py live).")
    ap.add_argument("--skip_inference", action="store_true", help="Do not run inference; only run evaluation+collect.")
    args = ap.parse_args()

    cfg = load_json(args.config)

    project_cwd = os.path.abspath(args.cwd)
    run_inference = bool(cfg.get("run_inference", True)) and (not args.skip_inference)

    variants = cfg["variants"]
    eval_tpl = cfg["eval_cmd_template"]

    collected: List[VariantResult] = []

    for v in variants:
        name = v["name"]
        results_dir = v["results_dir"]
        eval_out_dir = v["eval_out_dir"]
        infer_cmd = v.get("infer_cmd")

        # 1) Inference (optional)
        if run_inference:
            if not infer_cmd:
                raise ValueError(f"Variant {name} missing infer_cmd in config.")
            run_cmd(infer_cmd, cwd=project_cwd)

        # 2) Evaluation (CLIP)
        # eval_method: if you store images under results_dir, evaluation.py likely needs method flag.
        # Use the variant name or 'tot'. If your evaluation expects fixed method label, set it here.
        eval_method = "tot"  # change if needed
        eval_cmd = eval_tpl.format(results_dir=results_dir, eval_out_dir=eval_out_dir, eval_method=eval_method)
        run_cmd(eval_cmd, cwd=project_cwd)

        # 3) Collect summary
        summary_path = find_summary_json(os.path.join(project_cwd, eval_out_dir))
        summary = load_json(summary_path)
        m = parse_clip_metrics(summary)

        collected.append(
            VariantResult(
                name=name,
                results_dir=results_dir,
                eval_out_dir=eval_out_dir,
                summary_path=summary_path,
                n_scored=m["n"],
                clip_mean=m["mean"],
                clip_std=m["std"],
                clip_min=m["min"],
                clip_max=m["max"],
            )
        )

    # 4) Save CSV/JSON
    out_csv = os.path.join(project_cwd, "ablation_summary.csv")
    out_json = os.path.join(project_cwd, "ablation_summary.json")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["variant", "n_scored", "clip_mean", "clip_std", "clip_min", "clip_max", "summary_path"])
        for r in collected:
            w.writerow([r.name, r.n_scored, r.clip_mean, r.clip_std, r.clip_min, r.clip_max, r.summary_path])

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "variant": r.name,
                    "n_scored": r.n_scored,
                    "clip_mean": r.clip_mean,
                    "clip_std": r.clip_std,
                    "clip_min": r.clip_min,
                    "clip_max": r.clip_max,
                    "summary_path": r.summary_path,
                }
                for r in collected
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n[OK] Saved:")
    print(" -", out_csv)
    print(" -", out_json)

    # 5) Print LaTeX table
    print("\n[LaTeX]")
    print(to_latex_table(collected))


if __name__ == "__main__":
    main()
