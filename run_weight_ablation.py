import subprocess
import time

TASKS = [1, 3, 5, 7, 9]   # Color-I, Background-I, Style-I, Action-I, Texture-I
NUM_SAMPLES = 6

WEIGHT_MODES = [
    "original",
    "equal",
    "no_anchor",
    "no_constraints",
]

for task_id in TASKS:
    for mode in WEIGHT_MODES:
        cmd = [
            "python", "ablation/main3_abl.py",
            "--task_id", str(task_id),
            "--shot", "2",
            "--num_samples", str(NUM_SAMPLES),
            "--tau", "0.08",
            "--B", "3",
            "--selection_mode", "score",
            "--penalty_mode", "full",
            "--weight_mode", mode,
            "--seed", "123",
            "--generate_image",
            "--output_dir", "results_ablation/weight_sensitivity",
            "--output_tag", f"task{task_id}_weight_{mode}",
        ]

        print("Running:", " ".join(cmd))

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print("❌ Failed:", " ".join(cmd))

        time.sleep(2)