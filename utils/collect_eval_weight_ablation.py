import json
import csv
import re
from pathlib import Path
from statistics import mean, stdev

EVAL_ROOT = Path("eval_weight_ablation_clip_csr")

OUT_CSV_TASK = EVAL_ROOT / "weight_ablation_clip_csr_by_task.csv"
OUT_CSV_MODE = EVAL_ROOT / "weight_ablation_clip_csr_by_mode.csv"
OUT_JSON_MODE = EVAL_ROOT / "weight_ablation_clip_csr_by_mode.json"

summary_paths = sorted(EVAL_ROOT.glob("task*_weight_*/summary.json"))

if not summary_paths:
    raise FileNotFoundError(f"No summary.json files found under {EVAL_ROOT}")

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_name(folder_name):
    # task1_weight_no_constraints -> task_id=1, weight_mode=no_constraints
    m = re.match(r"task(\d+)_weight_(.+)", folder_name)
    if not m:
        raise ValueError(f"Unexpected folder name: {folder_name}")
    return int(m.group(1)), m.group(2)

def first_row(summary, key):
    rows = summary.get("aggregates", {}).get(key, [])
    return rows[0] if rows else {}

task_rows = []

for path in summary_paths:
    task_id, weight_mode = parse_name(path.parent.name)
    s = read_json(path)

    clip = first_row(s, "clip_by_method")
    csr = first_row(s, "csr_by_method")

    task_rows.append({
        "task_id": task_id,
        "weight_mode": weight_mode,

        "n_clip": clip.get("n"),
        "clip_mean": clip.get("clip_mean"),
        "clip_std": clip.get("clip_std"),
        "clip_min": clip.get("clip_min"),
        "clip_max": clip.get("clip_max"),

        "n_csr": csr.get("n"),
        "csr_mean": csr.get("csr_mean"),
        "csr_std": csr.get("csr_std"),
        "csr_min": csr.get("csr_min"),
        "csr_max": csr.get("csr_max"),

        "summary_path": str(path),
    })

# Save per-task table
with open(OUT_CSV_TASK, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "task_id",
            "weight_mode",
            "n_clip",
            "clip_mean",
            "clip_std",
            "clip_min",
            "clip_max",
            "n_csr",
            "csr_mean",
            "csr_std",
            "csr_min",
            "csr_max",
            "summary_path",
        ],
    )
    writer.writeheader()
    writer.writerows(task_rows)

# Aggregate by weight mode using per-sample CSVs, not repeated task means
mode_data = {}

for task_row in task_rows:
    mode = task_row["weight_mode"]
    mode_data.setdefault(mode, {
        "clip_values": [],
        "csr_values": [],
    })

    metrics_path = Path(task_row["summary_path"]).parent / "metrics_per_sample.csv"

    with open(metrics_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("clip") not in ("", None):
                mode_data[mode]["clip_values"].append(float(row["clip"]))

            if row.get("csr") not in ("", None):
                mode_data[mode]["csr_values"].append(float(row["csr"]))

mode_rows = []

for mode, data in sorted(mode_data.items()):
    clip_values = data["clip_values"]
    csr_values = data["csr_values"]

    mode_rows.append({
        "weight_mode": mode,

        "n_clip": len(clip_values),
        "clip_mean": mean(clip_values) if clip_values else None,
        "clip_std": stdev(clip_values) if len(clip_values) > 1 else 0.0,
        "clip_min": min(clip_values) if clip_values else None,
        "clip_max": max(clip_values) if clip_values else None,

        "n_csr": len(csr_values),
        "csr_mean": mean(csr_values) if csr_values else None,
        "csr_std": stdev(csr_values) if len(csr_values) > 1 else 0.0,
        "csr_min": min(csr_values) if csr_values else None,
        "csr_max": max(csr_values) if csr_values else None,
    })

with open(OUT_CSV_MODE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "weight_mode",
            "n_clip",
            "clip_mean",
            "clip_std",
            "clip_min",
            "clip_max",
            "n_csr",
            "csr_mean",
            "csr_std",
            "csr_min",
            "csr_max",
        ],
    )
    writer.writeheader()
    writer.writerows(mode_rows)

with open(OUT_JSON_MODE, "w", encoding="utf-8") as f:
    json.dump(mode_rows, f, indent=2, ensure_ascii=False)

print("\n=== Weight ablation: CLIP + CSR ===")
for r in mode_rows:
    print(
        f"{r['weight_mode']:<15} | "
        f"CLIP={r['clip_mean']:.4f} ± {r['clip_std']:.4f} "
        f"(n={r['n_clip']}) | "
        f"CSR={r['csr_mean']:.4f} ± {r['csr_std']:.4f} "
        f"(n={r['n_csr']})"
    )

print("\nSaved:")
print(OUT_CSV_TASK)
print(OUT_CSV_MODE)
print(OUT_JSON_MODE)