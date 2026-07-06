# run_main3.py

import subprocess
import sys

TASK_IDS = list(range(1, 11))

COMMON_ARGS = [
    "--shot", "2",
    "--data_mode", "default",
    "--tau", "0.08",
    "--max_new_tokens", "256",
    "--device", "cuda",
    "--image_device", "cuda",
    "--generate_image",
    "--max_samples", "30",
    "--output_dir", "results_final/tot",
]

failed_tasks = []

for task_id in TASK_IDS:
    cmd = [
        sys.executable, "main3.py",
        "--task_id", str(task_id),
        *COMMON_ARGS,
    ]

    print(f"\n=== Running ToT | task {task_id:03d} ===")
    print(" ".join(cmd))

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"❌ ToT task {task_id:03d} failed")
        failed_tasks.append(task_id)
        continue
    else:
        print(f"✅ ToT task {task_id:03d} done")

print("\n=== RUN FINISHED ===")
if failed_tasks:
    print("Failed tasks:", failed_tasks)
else:
    print("All requested tasks completed successfully.")