import os
import re
import gc
import csv
import glob
import torch
from datetime import datetime

from image_generator import ImageGenerator

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
ROOT_DIR = "results_ablation"
CSV_LOG = "generation_summary_branching.csv"
DEVICE = "cuda"
SEED_BASE = 123
SKIP_IF_EXISTS = True

# имя выходного файла рядом с prompt_sampleXXX.txt
OUTPUT_IMAGE_PREFIX = "image_from_prompt_"


# ---------------------------------------------------
# HELPERS
# ---------------------------------------------------
def extract_image_description(final_prompt: str) -> str:
    """
    Same prompt-cleaning logic as in main3_abl.py
    """
    lines = final_prompt.split("\n")
    descriptions = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if any(keyword in line.lower() for keyword in [
            "no extra", "keep identity", "binding:"
        ]):
            continue

        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip().lower()
            value = value.strip()

            if key in {"scene", "attribute", "stability", "composition"} and value:
                descriptions.append(value)
        elif len(line) > 15:
            descriptions.append(line)

    result = ". ".join(descriptions[:4]) if descriptions else final_prompt
    result = re.sub(r"\s+", " ", result).strip()
    result = re.sub(r"\.{2,}", ".", result)
    return result[:400]


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def ensure_parent(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def collect_prompt_files(root_dir: str):
    """
    Collect only branching prompt files.
    Expected path fragments contain 'branch_B'.
    """
    all_files = glob.glob(os.path.join(root_dir, "**", "prompt_sample*.txt"), recursive=True)
    selected = []

    for path in all_files:
        norm = path.replace("\\", "/").lower()
        if "branch_b" in norm:
            selected.append(path)

    return sorted(selected)


def parse_metadata_from_path(path: str):
    """
    Extract task id, B, and sample idx from path.
    """
    norm = path.replace("\\", "/")

    # e.g. results_ablation/task1_branch_B3/task_001/sample_004/prompt_sample004.txt
    task_match = re.search(r"task(\d+)_branch_b(\d+)", norm, re.IGNORECASE)
    sample_match = re.search(r"sample_(\d+)", norm, re.IGNORECASE)
    prompt_match = re.search(r"prompt_sample(\d+)\.txt", norm, re.IGNORECASE)

    task_id = int(task_match.group(1)) if task_match else None
    B = int(task_match.group(2)) if task_match else None
    sample_dir_idx = int(sample_match.group(1)) if sample_match else None
    prompt_idx = int(prompt_match.group(1)) if prompt_match else None

    return {
        "task_id": task_id,
        "B": B,
        "sample_dir_idx": sample_dir_idx,
        "prompt_idx": prompt_idx,
    }


def make_output_image_path(prompt_path: str):
    """
    Save image next to the prompt file.
    prompt_sample004.txt -> image_from_prompt_sample004.png
    """
    folder = os.path.dirname(prompt_path)
    fname = os.path.basename(prompt_path)

    m = re.match(r"prompt_(sample\d+)\.txt", fname, re.IGNORECASE)
    suffix = m.group(1) if m else os.path.splitext(fname)[0]

    return os.path.join(folder, f"{OUTPUT_IMAGE_PREFIX}{suffix}.png")


def stable_seed(task_id: int, B: int, sample_idx: int, base_seed: int = 123) -> int:
    """
    Deterministic seed per task/B/sample.
    """
    task_id = task_id if task_id is not None else 0
    B = B if B is not None else 0
    sample_idx = sample_idx if sample_idx is not None else 0
    return int(base_seed + task_id * 1000 + B * 100 + sample_idx)


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
def main():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    prompt_files = collect_prompt_files(ROOT_DIR)
    print(f"Found {len(prompt_files)} branching prompt files.")

    if not prompt_files:
        print("No branching prompt files found.")
        return

    # IMPORTANT:
    # use_seed=False -> skip SEED completely
    # use_sd=True    -> use Stable Diffusion only
    generator = ImageGenerator(device=DEVICE, use_seed=False, use_sd=True)

    csv_exists = os.path.exists(CSV_LOG)
    with open(CSV_LOG, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        if not csv_exists:
            writer.writerow([
                "timestamp",
                "task_id",
                "B",
                "sample_idx",
                "prompt_path",
                "image_path",
                "seed",
                "status"
            ])

        for i, prompt_path in enumerate(prompt_files, start=1):
            meta = parse_metadata_from_path(prompt_path)
            task_id = meta["task_id"]
            B = meta["B"]
            sample_idx = meta["sample_dir_idx"] if meta["sample_dir_idx"] is not None else meta["prompt_idx"]

            output_image_path = make_output_image_path(prompt_path)

            if SKIP_IF_EXISTS and os.path.exists(output_image_path):
                print(f"[{i}/{len(prompt_files)}] Skip existing: {output_image_path}")
                writer.writerow([
                    datetime.now().isoformat(),
                    task_id,
                    B,
                    sample_idx,
                    prompt_path,
                    output_image_path,
                    "",
                    "skipped_exists"
                ])
                continue

            try:
                full_prompt = read_text(prompt_path)
                image_prompt = extract_image_description(full_prompt)
                seed = stable_seed(task_id, B, sample_idx, base_seed=SEED_BASE)

                print(f"\n[{i}/{len(prompt_files)}] Generating | task={task_id} | B={B} | sample={sample_idx}")
                print(f"Prompt: {image_prompt[:120]}...")

                image = generator.generate(
                    prompt=image_prompt,
                    seed=seed,
                    prefer_seed=False   # IMPORTANT: force SD only
                )

                if image is None:
                    print("  -> FAILED: image is None")
                    writer.writerow([
                        datetime.now().isoformat(),
                        task_id,
                        B,
                        sample_idx,
                        prompt_path,
                        output_image_path,
                        seed,
                        "failed_none"
                    ])
                    continue

                ensure_parent(output_image_path)
                image.save(output_image_path)

                print(f"  -> saved: {output_image_path}")
                writer.writerow([
                    datetime.now().isoformat(),
                    task_id,
                    B,
                    sample_idx,
                    prompt_path,
                    output_image_path,
                    seed,
                    "ok"
                ])

            except Exception as e:
                print(f"  -> ERROR: {e}")
                writer.writerow([
                    datetime.now().isoformat(),
                    task_id,
                    B,
                    sample_idx,
                    prompt_path,
                    output_image_path,
                    "",
                    f"error: {str(e)}"
                ])

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()