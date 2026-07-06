from pathlib import Path
import re
import shutil

ROOT = Path("results_ablation")

sample_dirs = sorted(ROOT.glob("task*_branch_B*/task_*/sample_*"))

renamed = 0
copied = 0
missing = 0
already_ok = 0

for sample_dir in sample_dirs:
    m = re.search(r"sample_(\d+)", sample_dir.name)
    if not m:
        print(f"Cannot parse sample index: {sample_dir}")
        missing += 1
        continue

    sample_idx = int(m.group(1))

    expected = sample_dir / f"image_sample{sample_idx:03d}.png"
    old = sample_dir / f"image_from_prompt_sample{sample_idx:03d}.png"

    if expected.exists():
        already_ok += 1
        continue

    if old.exists():
        # safer than rename: keep old file and create expected copy
        shutil.copy2(old, expected)
        copied += 1
    else:
        print(f"Missing image: {sample_dir}")
        missing += 1

print("Done.")
print(f"Copied: {copied}")
print(f"Already OK: {already_ok}")
print(f"Missing: {missing}")