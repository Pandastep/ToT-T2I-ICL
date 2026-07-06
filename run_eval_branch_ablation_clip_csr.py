import subprocess
from pathlib import Path

ROOT = Path("results_ablation")
EVAL_ROOT = Path("eval_branch_ablation_clip_csr")

variant_dirs = sorted([
    p for p in ROOT.glob("task*_branch_B*")
    if p.is_dir()
])

if not variant_dirs:
    raise FileNotFoundError(f"No branching ablation folders found under {ROOT}")

for variant_dir in variant_dirs:
    out_dir = EVAL_ROOT / variant_dir.name

    cmd = [
    "python", "evaluation/evaluation_clip_csr.py",
    "--results_dir", str(variant_dir),
    "--out_dir", str(out_dir),
    "--method", "tot",
    ]

    print("Running:", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print("❌ Failed:", " ".join(cmd))