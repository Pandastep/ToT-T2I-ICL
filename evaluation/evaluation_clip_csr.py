#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluation.py — CLIP + CSR evaluation for CoBSAT/SEED results

Folder format:
results/
└── task_002/
    └── sample_000/
        ├── reasoning_log.json
        ├── prompt_sample000.txt
        ├── image_sample000.png
        ├── icl_used.json
        ├── icl_used.txt
        └── desc_sample000.txt

What this script does:
- Discovers samples via results_root/task_*/sample_*
- Loads text from prompt_sample*.txt
- Loads image from image_sample*.{png,jpg,jpeg,webp} or image.png
- Computes CLIP cosine similarity
- Computes CSR:
    - object_ok
    - attribute_ok
    - csr = (object_ok + attribute_ok) / 2
- Writes:
    - metrics_per_sample.csv
    - clip_by_method.csv
    - clip_by_task_method.csv
    - csr_by_method.csv
    - csr_by_task_method.csv
    - summary.json

IMPORTANT:
CSR requires target object / target attribute / candidate pools.
This script tries to read them from:
1) desc_sample*.txt   (recommended)
2) reasoning_log.json (best effort)
3) optional sidecar file csr_meta.json in sample folder

If missing, CSR fields are left empty for that sample.

Suggested desc_sample format examples:
  object: car
  attribute: orange
  object_pool: car,dog,chair,boat
  attribute_pool: red,orange,blue,green

or:
  target_object: car
  target_attribute: orange
  object_candidates: car,dog,chair,boat
  attribute_candidates: red,orange,blue,green

Usage:
  pip install torch pillow transformers

  python evaluation.py --results_dir results --out_dir eval_out --method tot
  python evaluation.py --results_dir results --out_dir eval_out --method baseline
  python evaluation.py --results_dir results --out_dir eval_out --method cot

Optional:
  python evaluation.py --results_dir results --out_dir eval_out --method tot --infer_method
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

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
            inputs = {k: v.to(self.cfg.device) for k, v in inputs.items()}

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

    @torch.no_grad()
    def score_image_text_candidates(self, image_path: Path, candidate_texts: List[str]) -> List[float]:
        """
        Score one image against multiple candidate texts.
        Returns cosine similarities in the same order as candidate_texts.
        """
        if not candidate_texts:
            return []

        image = Image.open(image_path).convert("RGB")
        images = [image for _ in candidate_texts]

        inputs = self.processor(
            text=candidate_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.cfg.device) for k, v in inputs.items()}

        img_feat = self.model.get_image_features(pixel_values=inputs["pixel_values"])
        txt_feat = self.model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
        )

        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        sims = (img_feat * txt_feat).sum(dim=-1)
        return [float(x) for x in sims.detach().cpu().tolist()]


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
# Sample parsing
# -----------------------------

def discover_sample_dirs(results_root: Path) -> List[Path]:
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
    if not isinstance(d, dict):
        return None
    for k in ["method", "prompt_type", "reasoning_mode", "mode"]:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def parse_task_id(task_name: str) -> int:
    # task_002 -> 2
    return int(task_name.split("_")[-1])


# -----------------------------
# Stats helpers
# -----------------------------

def mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return float(var ** 0.5)


def aggregate_metric(rows: List[dict], key_fields: List[str], metric: str) -> List[dict]:
    groups: Dict[Tuple[str, ...], List[float]] = {}

    for r in rows:
        val = r.get(metric, None)
        if val is None or val == "":
            continue
        try:
            val_f = float(val)
            if math.isnan(val_f):
                continue
        except Exception:
            continue

        k = tuple(str(r[f]) for f in key_fields)
        groups.setdefault(k, []).append(val_f)

    out: List[dict] = []
    for k, vals in sorted(groups.items(), key=lambda x: x[0]):
        rec = {key_fields[i]: k[i] for i in range(len(key_fields))}
        rec.update({
            "n": len(vals),
            f"{metric}_mean": mean(vals),
            f"{metric}_std": std(vals),
            f"{metric}_min": float(min(vals)),
            f"{metric}_max": float(max(vals)),
        })
        out.append(rec)
    return out


# -----------------------------
# CSR helpers
# -----------------------------

TASK_RULES = {
    1: {"type": "color", "preserve": "object", "change": "color"},
    2: {"type": "color", "preserve": "color", "change": "object"},
    3: {"type": "background", "preserve": "animal", "change": "background"},
    4: {"type": "background", "preserve": "background", "change": "animal"},
    5: {"type": "style", "preserve": "object", "change": "style"},
    6: {"type": "style", "preserve": "style", "change": "object"},
    7: {"type": "action", "preserve": "animal", "change": "action"},
    8: {"type": "action", "preserve": "action", "change": "animal"},
    9: {"type": "texture", "preserve": "object", "change": "texture"},
    10: {"type": "texture", "preserve": "texture", "change": "object"},
}


def clean_token(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def split_candidate_list(s: str) -> List[str]:
    if not s:
        return []
    parts = re.split(r"[,\n;|]+", s)
    return [clean_token(x) for x in parts if clean_token(x)]


def find_first_existing(sample_dir: Path, patterns: List[str]) -> Optional[Path]:
    for pat in patterns:
        found = sorted(sample_dir.glob(pat))
        if found:
            return found[0]
    return None


def try_load_desc_text(sample_dir: Path) -> Optional[str]:
    p = find_first_existing(sample_dir, ["desc_sample*.txt", "desc.txt"])
    if p and p.exists():
        return p.read_text(encoding="utf-8", errors="ignore")
    return None


def try_load_sidecar_meta(sample_dir: Path) -> Optional[dict]:
    p = sample_dir / "csr_meta.json"
    if p.exists():
        try:
            return read_json(p)
        except Exception:
            return None
    return None


def parse_key_value_text(text: str) -> Dict[str, str]:
    """
    Parses simple key: value lines.
    Example:
      object: car
      attribute: orange
      object_pool: car,dog,chair
      attribute_pool: red,orange,blue
    """
    out: Dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = clean_token(key)
        value = value.strip()
        if key and value:
            out[key] = value
    return out


def get_nested_str(d: Any, keys: List[str]) -> Optional[str]:
    if not isinstance(d, dict):
        return None
    for k in keys:
        if k in d and isinstance(d[k], str) and d[k].strip():
            return d[k].strip()
    return None


def get_nested_list(d: Any, keys: List[str]) -> Optional[List[str]]:
    if not isinstance(d, dict):
        return None
    for k in keys:
        if k not in d:
            continue
        v = d[k]
        if isinstance(v, list):
            cleaned = [clean_token(x) for x in v if clean_token(x)]
            if cleaned:
                return cleaned
        if isinstance(v, str):
            cleaned = split_candidate_list(v)
            if cleaned:
                return cleaned
    return None


def extract_csr_metadata(sample_dir: Path, reasoning_log: Optional[dict]) -> Dict[str, Any]:
    """
    Best-effort metadata extraction for CSR.
    Returns:
      {
        "target_object": str | None,
        "target_attribute": str | None,
        "object_pool": List[str],
        "attribute_pool": List[str]
      }
    Priority:
      1) csr_meta.json
      2) desc_sample*.txt / desc.txt
      3) reasoning_log.json
    """
    result = {
        "target_object": None,
        "target_attribute": None,
        "object_pool": [],
        "attribute_pool": [],
    }

    # 1) sidecar JSON
    meta_json = try_load_sidecar_meta(sample_dir)
    if isinstance(meta_json, dict):
        result["target_object"] = clean_token(
            get_nested_str(meta_json, ["target_object", "object", "subject", "animal"]) or ""
        ) or None
        result["target_attribute"] = clean_token(
            get_nested_str(meta_json, ["target_attribute", "attribute", "query", "value"]) or ""
        ) or None
        result["object_pool"] = get_nested_list(
            meta_json, ["object_pool", "object_candidates", "subject_pool", "subject_candidates"]
        ) or []
        result["attribute_pool"] = get_nested_list(
            meta_json, ["attribute_pool", "attribute_candidates", "value_pool", "query_candidates"]
        ) or []

    # 2) desc txt
    if not result["target_object"] or not result["target_attribute"] or not result["object_pool"] or not result["attribute_pool"]:
        desc_text = try_load_desc_text(sample_dir)
        if desc_text:
            kv = parse_key_value_text(desc_text)

            if not result["target_object"]:
                result["target_object"] = clean_token(
                    kv.get("target_object") or kv.get("object") or kv.get("subject") or kv.get("animal") or ""
                ) or None

            if not result["target_attribute"]:
                result["target_attribute"] = clean_token(
                    kv.get("target_attribute") or kv.get("attribute") or kv.get("query") or kv.get("value") or ""
                ) or None

            if not result["object_pool"]:
                result["object_pool"] = split_candidate_list(
                    kv.get("object_pool") or kv.get("object_candidates") or kv.get("subject_pool") or kv.get("subject_candidates") or ""
                )

            if not result["attribute_pool"]:
                result["attribute_pool"] = split_candidate_list(
                    kv.get("attribute_pool") or kv.get("attribute_candidates") or kv.get("value_pool") or kv.get("query_candidates") or ""
                )

    # 3) reasoning_log best effort
    if isinstance(reasoning_log, dict):
        if not result["target_object"]:
            result["target_object"] = clean_token(
                get_nested_str(reasoning_log, ["target_object", "object", "subject", "animal", "invariant"]) or ""
            ) or result["target_object"]

        if not result["target_attribute"]:
            result["target_attribute"] = clean_token(
                get_nested_str(reasoning_log, ["target_attribute", "attribute", "query", "value", "varying"]) or ""
            ) or result["target_attribute"]

        if not result["object_pool"]:
            result["object_pool"] = get_nested_list(
                reasoning_log, ["object_pool", "object_candidates", "subject_pool", "subject_candidates", "invariants"]
            ) or result["object_pool"]

        if not result["attribute_pool"]:
            result["attribute_pool"] = get_nested_list(
                reasoning_log, ["attribute_pool", "attribute_candidates", "value_pool", "query_candidates", "varyings"]
            ) or result["attribute_pool"]

    # ensure target is included in pools
    if result["target_object"] and result["target_object"] not in result["object_pool"]:
        result["object_pool"] = [result["target_object"]] + result["object_pool"]

    if result["target_attribute"] and result["target_attribute"] not in result["attribute_pool"]:
        result["attribute_pool"] = [result["target_attribute"]] + result["attribute_pool"]

    # deduplicate, preserve order
    result["object_pool"] = dedupe_preserve_order(result["object_pool"])
    result["attribute_pool"] = dedupe_preserve_order(result["attribute_pool"])

    return result


def dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        x2 = clean_token(x)
        if not x2 or x2 in seen:
            continue
        seen.add(x2)
        out.append(x2)
    return out


def build_object_candidates(object_pool: List[str]) -> List[str]:
    return [f"a photo of a {obj}" for obj in object_pool]


def build_color_candidates(target_object: str, color_pool: List[str]) -> List[str]:
    return [f"a photo of a {color} {target_object}" for color in color_pool]


def build_style_candidates(target_object: str, style_pool: List[str]) -> List[str]:
    return [f"a {style} style image of a {target_object}" for style in style_pool]


def build_action_candidates(target_subject: str, action_pool: List[str]) -> List[str]:
    return [f"a {target_subject} {action}" for action in action_pool]


def build_background_candidates(target_subject: str, bg_pool: List[str]) -> List[str]:
    return [f"a {target_subject} in {bg}" for bg in bg_pool]


def build_texture_candidates(target_object: str, texture_pool: List[str]) -> List[str]:
    return [f"a {texture} {target_object}" for texture in texture_pool]


def is_target_top1(scores: List[float], candidates: List[str], target_text: str) -> int:
    if not scores or not candidates or len(scores) != len(candidates):
        return 0
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return int(candidates[best_idx].strip().lower() == target_text.strip().lower())


def compute_csr_for_sample(
    scorer: ClipScorer,
    image_path: Path,
    task_id: int,
    target_object: str,
    target_attribute: str,
    object_pool: List[str],
    attribute_pool: List[str],
) -> Dict[str, Any]:
    """
    Returns:
      {
        "object_ok": 0/1,
        "attribute_ok": 0/1,
        "csr": float,
        "object_pred_text": str,
        "attribute_pred_text": str,
      }
    """
    if task_id not in TASK_RULES:
        raise ValueError(f"Unsupported task_id={task_id}")

    task_type = TASK_RULES[task_id]["type"]

    if not target_object or not target_attribute:
        raise ValueError("target_object and target_attribute are required for CSR")

    if not object_pool or not attribute_pool:
        raise ValueError("object_pool and attribute_pool are required for CSR")

    # object check
    object_candidates = build_object_candidates(object_pool)
    object_target_text = f"a photo of a {target_object}"

    object_scores = scorer.score_image_text_candidates(image_path, object_candidates)
    object_ok = is_target_top1(object_scores, object_candidates, object_target_text)
    object_best_idx = max(range(len(object_scores)), key=lambda i: object_scores[i])
    object_pred_text = object_candidates[object_best_idx]

    # attribute check
    if task_type == "color":
        attr_candidates = build_color_candidates(target_object, attribute_pool)
        attr_target_text = f"a photo of a {target_attribute} {target_object}"
    elif task_type == "style":
        attr_candidates = build_style_candidates(target_object, attribute_pool)
        attr_target_text = f"a {target_attribute} style image of a {target_object}"
    elif task_type == "action":
        attr_candidates = build_action_candidates(target_object, attribute_pool)
        attr_target_text = f"a {target_object} {target_attribute}"
    elif task_type == "background":
        attr_candidates = build_background_candidates(target_object, attribute_pool)
        attr_target_text = f"a {target_object} in {target_attribute}"
    elif task_type == "texture":
        attr_candidates = build_texture_candidates(target_object, attribute_pool)
        attr_target_text = f"a {target_attribute} {target_object}"
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    attr_scores = scorer.score_image_text_candidates(image_path, attr_candidates)
    attribute_ok = is_target_top1(attr_scores, attr_candidates, attr_target_text)
    attr_best_idx = max(range(len(attr_scores)), key=lambda i: attr_scores[i])
    attribute_pred_text = attr_candidates[attr_best_idx]

    csr = (object_ok + attribute_ok) / 2.0

    return {
        "object_ok": int(object_ok),
        "attribute_ok": int(attribute_ok),
        "csr": float(csr),
        "object_pred_text": object_pred_text,
        "attribute_pred_text": attribute_pred_text,
    }


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

    image_paths: List[Path] = []
    texts: List[str] = []
    metas: List[dict] = []

    skipped: List[dict] = []

    for sd in sample_dirs:
        task = sd.parent.name
        sample_id = sd.name

        try:
            text, prompt_path = load_clip_text_from_prompt(sd)
            img = load_image_path(sd)
        except Exception as e:
            skipped.append({"task": task, "sample_id": sample_id, "reason": str(e)})
            continue

        method = method_cli
        reasoning_log = None
        if infer_method:
            reasoning_log = safe_read_reasoning_log(sd)
            m2 = infer_method_from_reasoning_log(reasoning_log)
            if m2:
                method = m2
        else:
            reasoning_log = safe_read_reasoning_log(sd)

        image_paths.append(img)
        texts.append(text)
        metas.append({
            "sample_dir": str(sd),
            "task": task,
            "sample_id": sample_id,
            "method": method,
            "image_path": str(img),
            "prompt_path": str(prompt_path),
            "text_preview": text[:200] if save_text_preview else "",
            "reasoning_log": reasoning_log,
        })

    if not image_paths:
        raise RuntimeError("All samples were skipped; nothing to score. Check file names and paths.")

    sims = scorer.score_pairs(image_paths, texts)

    rows: List[dict] = []

    for meta, sim in zip(metas, sims):
        sd = Path(meta["sample_dir"])
        task = meta["task"]
        task_id = parse_task_id(task)

        row = {
            "task": meta["task"],
            "sample_id": meta["sample_id"],
            "method": meta["method"],
            "image_path": meta["image_path"],
            "prompt_path": meta["prompt_path"],
            "clip": float(sim),
            "object_ok": "",
            "attribute_ok": "",
            "csr": "",
            "target_object": "",
            "target_attribute": "",
            "object_pool_size": "",
            "attribute_pool_size": "",
            "object_pred_text": "",
            "attribute_pred_text": "",
            "csr_status": "",
        }

        if save_text_preview:
            row["text_preview"] = meta["text_preview"]

        # CSR block
        try:
            csr_meta = extract_csr_metadata(sd, meta["reasoning_log"])

            target_object = clean_token(csr_meta.get("target_object") or "")
            target_attribute = clean_token(csr_meta.get("target_attribute") or "")
            object_pool = dedupe_preserve_order(csr_meta.get("object_pool") or [])
            attribute_pool = dedupe_preserve_order(csr_meta.get("attribute_pool") or [])

            row["target_object"] = target_object
            row["target_attribute"] = target_attribute
            row["object_pool_size"] = len(object_pool)
            row["attribute_pool_size"] = len(attribute_pool)

            if not target_object or not target_attribute:
                raise ValueError("missing target_object or target_attribute")

            if len(object_pool) < 2:
                raise ValueError("object_pool has fewer than 2 candidates")

            if len(attribute_pool) < 2:
                raise ValueError("attribute_pool has fewer than 2 candidates")

            csr_info = compute_csr_for_sample(
                scorer=scorer,
                image_path=Path(meta["image_path"]),
                task_id=task_id,
                target_object=target_object,
                target_attribute=target_attribute,
                object_pool=object_pool,
                attribute_pool=attribute_pool,
            )

            row["object_ok"] = csr_info["object_ok"]
            row["attribute_ok"] = csr_info["attribute_ok"]
            row["csr"] = csr_info["csr"]
            row["object_pred_text"] = csr_info["object_pred_text"]
            row["attribute_pred_text"] = csr_info["attribute_pred_text"]
            row["csr_status"] = "ok"

        except Exception as e:
            row["csr_status"] = f"skipped: {e}"

        rows.append(row)

    # Per-sample output
    header = [
        "task",
        "sample_id",
        "method",
        "clip",
        "object_ok",
        "attribute_ok",
        "csr",
        "target_object",
        "target_attribute",
        "object_pool_size",
        "attribute_pool_size",
        "object_pred_text",
        "attribute_pred_text",
        "csr_status",
        "image_path",
        "prompt_path",
    ]
    if save_text_preview:
        header.append("text_preview")

    write_csv(out_dir / "metrics_per_sample.csv", rows, header)

    # Aggregates
    clip_by_method = aggregate_metric(rows, ["method"], "clip")
    clip_by_task_method = aggregate_metric(rows, ["task", "method"], "clip")

    csr_by_method = aggregate_metric(rows, ["method"], "csr")
    csr_by_task_method = aggregate_metric(rows, ["task", "method"], "csr")

    write_csv(
        out_dir / "clip_by_method.csv",
        clip_by_method,
        ["method", "n", "clip_mean", "clip_std", "clip_min", "clip_max"],
    )
    write_csv(
        out_dir / "clip_by_task_method.csv",
        clip_by_task_method,
        ["task", "method", "n", "clip_mean", "clip_std", "clip_min", "clip_max"],
    )
    write_csv(
        out_dir / "csr_by_method.csv",
        csr_by_method,
        ["method", "n", "csr_mean", "csr_std", "csr_min", "csr_max"],
    )
    write_csv(
        out_dir / "csr_by_task_method.csv",
        csr_by_task_method,
        ["task", "method", "n", "csr_mean", "csr_std", "csr_min", "csr_max"],
    )

    n_csr_ok = sum(1 for r in rows if r.get("csr_status") == "ok")
    n_csr_skipped = len(rows) - n_csr_ok

    summary = {
        "results_dir": str(results_dir),
        "out_dir": str(out_dir),
        "clip_model_id": clip_model_id,
        "device": ("cuda" if torch.cuda.is_available() else "cpu"),
        "method_cli": method_cli,
        "infer_method": bool(infer_method),
        "n_scored_clip": len(rows),
        "n_scored_csr": n_csr_ok,
        "n_skipped_csr": n_csr_skipped,
        "n_skipped_files": len(skipped),
        "skipped_files": skipped[:50],
        "aggregates": {
            "clip_by_method": clip_by_method,
            "clip_by_task_method": clip_by_task_method,
            "csr_by_method": csr_by_method,
            "csr_by_task_method": csr_by_task_method,
        },
    }
    write_json(out_dir / "summary.json", summary)

    print(f"[OK] CLIP scored: {len(rows)} samples")
    print(f"[OK] CSR scored:  {n_csr_ok} samples")
    print(f"[OK] CSR skipped: {n_csr_skipped} samples")
    print(f"[OK] Wrote: {out_dir / 'metrics_per_sample.csv'}")
    print(f"[OK] Wrote: {out_dir / 'clip_by_method.csv'}")
    print(f"[OK] Wrote: {out_dir / 'clip_by_task_method.csv'}")
    print(f"[OK] Wrote: {out_dir / 'csr_by_method.csv'}")
    print(f"[OK] Wrote: {out_dir / 'csr_by_task_method.csv'}")
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
                    help="Store first 200 chars of prompt text into metrics_per_sample.csv for debugging.")
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