# run_ablation.py
import subprocess
import time

TASKS = [1, 3, 5, 7, 9]   # color, background, style, action, texture
NUM_SAMPLES = 6

# -----------------------
# 1. Branching (generation width)
# -----------------------
B_VALUES = [1, 2, 3, 5]

for task_id in TASKS:
    for B in B_VALUES:
        cmd = [
            "python", "ablation/main3_abl.py",
            "--task_id", str(task_id),
            "--shot", "2",
            "--num_samples", str(NUM_SAMPLES),
            "--tau", "0.08",
            "--B", str(B),
            "--selection_mode", "score",
            "--penalty_mode", "full",
            "--seed", "123",
            "--output_dir", "results_ablation",
            "--output_tag", f"task{task_id}_branch_B{B}"
        ]

        print("Running:", " ".join(cmd))

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print("❌ Failed:", " ".join(cmd))

        time.sleep(2)
