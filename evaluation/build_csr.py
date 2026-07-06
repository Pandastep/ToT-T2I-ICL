#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_csr_meta_from_dataset.py

Build csr_meta.json for result folders using the original dataset loader,
not output heuristics.

This is the correct way to prepare CSR metadata because it uses ground-truth
task structure from load_dataset/task_dataframe.

Expected project usage:
  python build_csr_meta_from_dataset.py --results_dir results_final/tot
  python build_csr_meta_from_dataset.py --results_dir results_final/cot
  python build_csr_meta_from_dataset.py --results_dir results_final/baseline

Optional:
  python build_csr_meta_from_dataset.py --results_dir results_final/tot --shot 2 --data_mode default --prompt_type default --overwrite

Result in each sample folder:
  csr_meta.json

Example output:
{
  "task_id": 1,
  "task_type": "Color-I",
  "x_space": "color",
  "theta_space": "object",
  "target_object": "car",
  "target_attribute": "orange",
  "object_pool": ["car", "chair", "boat", "bottle"],
  "attribute_pool": ["red", "orange", "blue", "green"],
  "dataset_index": 0,
  "save_path": "...",
  "status": "ok",
  "notes": []
}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------
# Make project root importable
# ---------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Project imports
from load_dataset import load_dataset
from configs import task_dataframe, item2word


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

ATTRIBUTE_SPACES = {"color", "background", "style", "action", "texture"}
OBJECT_SPACES = {"object", "animal"}


def clean_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip().lower().replace("_", " ")


def normalize_list(items: List[Any]) -> List[str]:
    out = []
    seen = set()
    for item in items:
        x = clean_text(item2word.get(item, item))
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def discover_sample_dirs(results_dir: Path) -> List[Path]:
    return sorted(results_dir.glob("task_*/sample_*"))


def parse_task_id(task_dir_name: str) -> int:
    # task_001 -> 1
    return int(task_dir_name.split("_")[-1])


def parse_sample_index(sample_dir_name: str) -> int:
    # sample_000 -> 0
    return int(sample_dir_name.split("_")[-1])


def get_task_type_string(task_id: int) -> str:
    row = task_dataframe[task_id]
    return str(row.get("task_type", f"task_{task_id}"))


def extract_target_and_pools(task_id: int, item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build CSR metadata from one dataset item produced by load_dataset().

    item contains:
      - x_list       : demo x values plus query x at the end
      - theta        : selected theta value for this sample
      - x_space
      - theta_space
      - target_x
    """
    row = task_dataframe[task_id]

    x_space = clean_text(item["x_space"])
    theta_space = clean_text(item["theta_space"])

    full_x_list = normalize_list(row["x_list"])
    full_theta_list = normalize_list(row["theta_list"])

    target_x = clean_text(item.get("target_x"))
    theta = clean_text(item.get("theta"))

    if x_space in ATTRIBUTE_SPACES and theta_space in OBJECT_SPACES:
        target_attribute = target_x
        target_object = theta
        attribute_pool = full_x_list
        object_pool = full_theta_list

    elif x_space in OBJECT_SPACES and theta_space in ATTRIBUTE_SPACES:
        target_object = target_x
        target_attribute = theta
        object_pool = full_x_list
        attribute_pool = full_theta_list

    else:
        raise ValueError(
            f"Unsupported task structure for task_id={task_id}: "
            f"x_space={x_space}, theta_space={theta_space}"
        )

    # ensure targets are present
    if target_object and target_object not in object_pool:
        object_pool = [target_object] + object_pool

    if target_attribute and target_attribute not in attribute_pool:
        attribute_pool = [target_attribute] + attribute_pool

    # dedupe again preserving order
    object_pool = normalize_list(object_pool)
    attribute_pool = normalize_list(attribute_pool)

    status = "ok"
    notes: List[str] = []

    if not target_object:
        status = "partial"
        notes.append("missing target_object")

    if not target_attribute:
        status = "partial"
        notes.append("missing target_attribute")

    if len(object_pool) < 2:
        status = "partial"
        notes.append("object_pool has fewer than 2 candidates")

    if len(attribute_pool) < 2:
        status = "partial"
        notes.append("attribute_pool has fewer than 2 candidates")

    return {
        "task_id": task_id,
        "task_type": get_task_type_string(task_id),
        "x_space": x_space,
        "theta_space": theta_space,
        "target_object": target_object,
        "target_attribute": target_attribute,
        "object_pool": object_pool,
        "attribute_pool": attribute_pool,
        "status": status,
        "notes": notes,
    }


def build_meta_for_task(
    task_id: int,
    shot: int,
    prompt_type: str,
    data_mode: str,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    Rebuild the exact dataset list for one task using load_dataset().
    """
    data_loader = load_dataset(
        shot=shot,
        prompt_type=prompt_type,
        task_id=task_id,
        seed=seed,
        data_mode=data_mode,
        include_output=False,
    )

    meta_list: List[Dict[str, Any]] = []
    for idx, item in enumerate(data_loader):
        meta = extract_target_and_pools(task_id, item)
        meta["dataset_index"] = idx
        meta["save_path"] = item.get("save_path", "")
        meta["theta"] = clean_text(item.get("theta", ""))
        meta["target_x"] = clean_text(item.get("target_x", ""))
        meta["x_list_used"] = normalize_list(item.get("x_list", []))
        meta_list.append(meta)

    return meta_list


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True, help="Root directory with task_*/sample_* folders.")
    parser.add_argument("--shot", type=int, default=2, help="Shot value used for the experiment.")
    parser.add_argument("--prompt_type", type=str, default="default",
                        help="Prompt type for load_dataset reconstruction. Metadata should be identical across prompt types for default setting.")
    parser.add_argument("--data_mode", type=str, default="default", help="Dataset mode used in the experiment.")
    parser.add_argument("--seed", type=int, default=123, help="Seed used for load_dataset.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing csr_meta.json files.")
    parser.add_argument("--report_path", type=str, default="", help="Optional custom report path.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"results_dir does not exist: {results_dir}")

    sample_dirs = discover_sample_dirs(results_dir)
    if not sample_dirs:
        raise FileNotFoundError(f"No task_*/sample_* folders found under: {results_dir}")

    # Group sample dirs by task
    task_to_sample_dirs: Dict[int, List[Path]] = {}
    for sample_dir in sample_dirs:
        task_id = parse_task_id(sample_dir.parent.name)
        task_to_sample_dirs.setdefault(task_id, []).append(sample_dir)

    report: Dict[str, Any] = {
        "results_dir": str(results_dir),
        "shot": args.shot,
        "prompt_type": args.prompt_type,
        "data_mode": args.data_mode,
        "seed": args.seed,
        "n_samples": len(sample_dirs),
        "n_written": 0,
        "n_skipped_existing": 0,
        "n_ok": 0,
        "n_partial": 0,
        "n_errors": 0,
        "samples": [],
    }

    for task_id, task_sample_dirs in sorted(task_to_sample_dirs.items(), key=lambda x: x[0]):
        # Rebuild dataset metadata for this task
        try:
            task_meta_list = build_meta_for_task(
                task_id=task_id,
                shot=args.shot,
                prompt_type=args.prompt_type,
                data_mode=args.data_mode,
                seed=args.seed,
            )
        except Exception as e:
            for sample_dir in task_sample_dirs:
                report["n_errors"] += 1
                report["samples"].append({
                    "sample_dir": str(sample_dir),
                    "status": "error_rebuilding_task_dataset",
                    "task_id": task_id,
                    "error": str(e),
                })
            continue

        for sample_dir in sorted(task_sample_dirs):
            sample_idx = parse_sample_index(sample_dir.name)
            out_path = sample_dir / "csr_meta.json"

            if out_path.exists() and not args.overwrite:
                report["n_skipped_existing"] += 1
                report["samples"].append({
                    "sample_dir": str(sample_dir),
                    "task_id": task_id,
                    "sample_idx": sample_idx,
                    "status": "skipped_existing",
                })
                continue

            if sample_idx >= len(task_meta_list):
                report["n_errors"] += 1
                report["samples"].append({
                    "sample_dir": str(sample_dir),
                    "task_id": task_id,
                    "sample_idx": sample_idx,
                    "status": "index_out_of_range",
                    "max_dataset_index": len(task_meta_list) - 1,
                })
                continue

            meta = dict(task_meta_list[sample_idx])
            meta["sample_dir"] = str(sample_dir)

            try:
                write_json(out_path, meta)
                report["n_written"] += 1

                if meta["status"] == "ok":
                    report["n_ok"] += 1
                else:
                    report["n_partial"] += 1

                report["samples"].append({
                    "sample_dir": str(sample_dir),
                    "task_id": task_id,
                    "sample_idx": sample_idx,
                    "status": "written",
                    "csr_meta_status": meta["status"],
                    "target_object": meta.get("target_object"),
                    "target_attribute": meta.get("target_attribute"),
                    "object_pool_size": len(meta.get("object_pool", [])),
                    "attribute_pool_size": len(meta.get("attribute_pool", [])),
                })
            except Exception as e:
                report["n_errors"] += 1
                report["samples"].append({
                    "sample_dir": str(sample_dir),
                    "task_id": task_id,
                    "sample_idx": sample_idx,
                    "status": "write_error",
                    "error": str(e),
                })

    report_path = Path(args.report_path) if args.report_path else (results_dir / "csr_meta_report.json")
    write_json(report_path, report)

    print(f"[OK] Samples found:       {report['n_samples']}")
    print(f"[OK] Written:             {report['n_written']}")
    print(f"[OK] Skipped existing:    {report['n_skipped_existing']}")
    print(f"[OK] Status ok:           {report['n_ok']}")
    print(f"[OK] Status partial:      {report['n_partial']}")
    print(f"[OK] Errors:              {report['n_errors']}")
    print(f"[OK] Report:              {report_path}")


if __name__ == "__main__":
    main()