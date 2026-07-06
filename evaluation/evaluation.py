#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluation.py — CLIP evaluation for CoBSAT/SEED results (your folder format)

Folder format (as in your screenshot):
results/
└── task_002/
    └── sample_000/
        ├── reasoning_log.json
        ├── prompt_sample000.txt     <-- CLIP text source (YOU REQUESTED THIS)
        ├── image_sample000.png      <-- image to score
        ├── icl_used.json
        ├── icl_used.txt
        └── desc_sample000.txt

What this script does:
- Discovers samples via results_root/task_*/sample_*
- Loads text strictly from prompt_sample*.txt
- Loads image from image_sample*.{png,jpg,jpeg,webp}
- Computes CLIP cosine similarity (normalized embeddings)
- Writes:
  - clip_per_sample.csv
  - clip_by_method.csv
  - clip_by_task_method.csv
  - summary.json

Usage:
  pip install torch pillow transformers

  python evaluation.py --results_dir results --out_dir eval_out --method tot
  python evaluation.py --results_dir results --out_dir eval_out --method baseline
  python evaluation.py --results_dir results --out_dir eval_out --method cot

If you keep different methods in different roots, run it per root and compare CSVs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


# -----------------------------
# CLIP scorer
# -----------------------------

@dataclass
class ClipConfig:
    model_id: str = "openai/clip-vit-base-patch32"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32


class ClipScorer:
    def __init__(self, cfg: ClipConfig):
        self.cfg = cfg
        self.model = CLIPModel.from_pretrained(cfg.model_id).to(cfg.device)
        self.processor = CLIPProcessor.from_pretrained(cfg.model_id)
        self.model.eval()

    @torch.no_grad()
    def score_pairs(self, image_paths: List[Path], texts: List[str]) -> List[float]:
        """Cosine similarity between normalized CLIP embeddings for each (image,text) pair."""
        assert len(image_paths) == len(texts), "image_paths and texts must have same length"
        sims: List[float] = []

        bs = self.cfg.batch_size
        for i in range(0, len(image_paths), bs):
            batch_paths = image_paths[i:i + bs]
            batch_texts = texts[i:i + bs]

            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = self.processor(
                text=batch_texts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )

            # Move tensors to device
            inputs = {k: v.to(self.cfg.device) for k, v in inputs.items()}

            # IMPORTANT: pass only the expected keys
            img_feat = self.model.get_image_features(pixel_values=inputs["pixel_values"])
            txt_feat = self.model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
            )

            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

            batch_sims = (img_feat * txt_feat).sum(dim=-1).detach().cpu().tolist()
            sims.extend([float(x) for x in batch_sims])

        return sims


# -----------------------------
# IO helpers
# -----------------------------

def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: List[dict], header: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# -----------------------------
# Sample parsing (YOUR format)
# -----------------------------

def discover_sample_dirs(results_root: Path) -> List[Path]:
    """Find all task_*/sample_* folders under results_root."""
    return sorted(results_root.glob("task_*/sample_*"))


def load_clip_text_from_prompt(sample_dir: Path) -> Tuple[str, Path]:
    prompt_files = sorted(sample_dir.glob("prompt_sample*.txt"))
    if prompt_files:
        p = prompt_files[0]
    else:
        p = sample_dir / "prompt.txt"
        if not p.exists():
            raise FileNotFoundError(f"prompt_sample*.txt or prompt.txt not found in {sample_dir}")

    text = p.read_text(encoding="utf-8").strip()
    text = text.replace("[INST]", "").replace("[/INST]", "")
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        raise ValueError(f"Empty prompt text in {p}")
    return text, p




def load_image_path(sample_dir: Path) -> Path:
    """Load image_sample*.* (ToT) OR image.png (baseline/cot)."""
    imgs = [p for p in sample_dir.glob("image_sample*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    imgs = sorted(imgs, key=lambda p: p.name)
    if imgs:
        return imgs[0]

    p2 = sample_dir / "image.png"
    if p2.exists():
        return p2

    raise FileNotFoundError(f"image_sample* or image.png not found in {sample_dir}")



def safe_read_reasoning_log(sample_dir: Path) -> Optional[dict]:
    p = sample_dir / "reasoning_log.json"
    if p.exists():
        try:
            return read_json(p)
        except Exception:
            return None
    return None


def infer_method_from_reasoning_log(d: Optional[dict]) -> Optional[str]:
    """Best-effort inference; if not found, returns None."""
    if not isinstance(d, dict):
        return None
    # Common keys across variants; adjust if you store differently
    for k in ["method", "prompt_type", "reasoning_mode", "mode"]:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


# -----------------------------
# Aggregation
# -----------------------------

def mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return float(var ** 0.5)


def aggregate(rows: List[dict], key_fields: List[str]) -> List[dict]:
    groups: Dict[Tuple[str, ...], List[float]] = {}
    for r in rows:
        k = tuple(str(r[f]) for f in key_fields)
        groups.setdefault(k, []).append(float(r["clip"]))

    out: List[dict] = []
    for k, vals in sorted(groups.items(), key=lambda x: x[0]):
        rec = {key_fields[i]: k[i] for i in range(len(key_fields))}
        rec.update({
            "n": len(vals),
            "clip_mean": mean(vals),
            "clip_std": std(vals),
            "clip_min": float(min(vals)),
            "clip_max": float(max(vals)),
        })
        out.append(rec)
    return out


# -----------------------------
# Main evaluation
# -----------------------------

def evaluate(
    results_dir: Path,
    out_dir: Path,
    method_cli: str,
    infer_method: bool,
    clip_model_id: str,
    batch_size: int,
    save_text_preview: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_dirs = discover_sample_dirs(results_dir)
    if not sample_dirs:
        raise FileNotFoundError(f"No samples found under {results_dir} with pattern task_*/sample_*")

    scorer = ClipScorer(ClipConfig(model_id=clip_model_id, batch_size=batch_size))

    # Collect batch inputs
    image_paths: List[Path] = []
    texts: List[str] = []
    metas: List[dict] = []

    skipped: List[dict] = []

    for sd in sample_dirs:
        task = sd.parent.name   # task_002
        sample_id = sd.name     # sample_000

        try:
            text, prompt_path = load_clip_text_from_prompt(sd)
            img = load_image_path(sd)
        except Exception as e:
            skipped.append({"task": task, "sample_id": sample_id, "reason": str(e)})
            continue

        # method handling
        method = method_cli
        if infer_method:
            rlog = safe_read_reasoning_log(sd)
            m2 = infer_method_from_reasoning_log(rlog)
            if m2:
                method = m2
        pf = sorted(sd.glob("prompt_sample*.txt"))
        prompt_path = pf[0] if pf else (sd / "prompt.txt")
        image_paths.append(img)
        texts.append(text)
        metas.append({
            "task": task,
            "sample_id": sample_id,
            "method": method,
            "image_path": str(img),
            "prompt_path": str(prompt_path),
            "text_preview": text[:200] if save_text_preview else "",
        })

    if not image_paths:
        raise RuntimeError("All samples were skipped; nothing to score. Check file names and paths.")

    sims = scorer.score_pairs(image_paths, texts)

    rows: List[dict] = []
    for meta, sim in zip(metas, sims):
        row = {
            "task": meta["task"],
            "sample_id": meta["sample_id"],
            "method": meta["method"],
            "image_path": meta["image_path"],
            "prompt_path": meta["prompt_path"],
            "clip": float(sim),
        }
        if save_text_preview:
            row["text_preview"] = meta["text_preview"]
        rows.append(row)

    # Save per-sample
    header = ["task", "sample_id", "method", "clip", "image_path", "prompt_path"]
    if save_text_preview:
        header.append("text_preview")
    write_csv(out_dir / "clip_per_sample.csv", rows, header)

    # Aggregates
    by_method = aggregate(rows, ["method"])
    by_task_method = aggregate(rows, ["task", "method"])

    write_csv(out_dir / "clip_by_method.csv", by_method,
              ["method", "n", "clip_mean", "clip_std", "clip_min", "clip_max"])
    write_csv(out_dir / "clip_by_task_method.csv", by_task_method,
              ["task", "method", "n", "clip_mean", "clip_std", "clip_min", "clip_max"])

    summary = {
        "results_dir": str(results_dir),
        "out_dir": str(out_dir),
        "clip_model_id": clip_model_id,
        "device": ("cuda" if torch.cuda.is_available() else "cpu"),
        "method_cli": method_cli,
        "infer_method": bool(infer_method),
        "n_scored": len(rows),
        "n_skipped": len(skipped),
        "skipped": skipped[:50],  # keep file small; increase if you want
        "aggregates": {
            "by_method": by_method,
            "by_task_method": by_task_method,
        },
    }
    write_json(out_dir / "summary.json", summary)

    print(f"[OK] Scored {len(rows)} samples, skipped {len(skipped)}.")
    print(f"[OK] Wrote: {out_dir / 'clip_per_sample.csv'}")
    print(f"[OK] Wrote: {out_dir / 'clip_by_method.csv'}")
    print(f"[OK] Wrote: {out_dir / 'clip_by_task_method.csv'}")
    print(f"[OK] Wrote: {out_dir / 'summary.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, required=True,
                    help="Root results directory containing task_*/sample_* folders.")
    ap.add_argument("--out_dir", type=str, default="eval_out",
                    help="Output directory for CSV/JSON.")
    ap.add_argument("--method", type=str, default="unknown",
                    help="Method label to store in outputs (baseline/cot/tot/...).")
    ap.add_argument("--infer_method", action="store_true",
                    help="Try to infer method from reasoning_log.json if it contains a string key.")
    ap.add_argument("--clip_model_id", type=str, default="openai/clip-vit-base-patch32")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--save_text_preview", action="store_true",
                    help="Store first 200 chars of prompt text into clip_per_sample.csv for debugging.")
    args = ap.parse_args()

    evaluate(
        results_dir=Path(args.results_dir),
        out_dir=Path(args.out_dir),
        method_cli=str(args.method),
        infer_method=bool(args.infer_method),
        clip_model_id=str(args.clip_model_id),
        batch_size=int(args.batch_size),
        save_text_preview=bool(args.save_text_preview),
    )


if __name__ == "__main__":
    main()
