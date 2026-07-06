import subprocess
import sys

TASK_IDS = list(range(1, 11))  # 1..10

COMMON_ARGS = [
    "--method", "baseline",
    "--shot", "2",
    "--data_mode", "default",
    "--max_new_tokens", "256",
    "--device", "cuda",
    "--image_device", "cuda",
    "--generate_image",
    "--max_samples", "30",
    "--output_dir", "results_final",
]

for task_id in TASK_IDS:
    cmd = [
        sys.executable, "main_bc.py",
        "--task_id", str(task_id),
        *COMMON_ARGS,
    ]

    print(f"\n=== Running BASELINE | task {task_id:03d} ===")
    print(" ".join(cmd))

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"❌ BASELINE task {task_id:03d} failed")
        break
    else:
        print(f"✅ BASELINE task {task_id:03d} done")