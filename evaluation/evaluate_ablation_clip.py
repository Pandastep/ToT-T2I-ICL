# evaluate_ablation_clip.py
# Computes CLIP similarity for ablation runs:
#   results_ablation/B{B}/task_{xxx}/sample_{yyy}/
#     - image_sampleYYY.png
#     - prompt_sampleYYY.txt
#
# Outputs:
#   1) <root>/clip_scores_raw_<timestamp>.csv  (per-image)
#   2) <root>/clip_scores_agg_<timestamp>.csv  (mean/std per task,B)
#
# Usage:
#   python evaluate_ablation_clip.py --root results_ablation --device cuda
#
# Notes:
# - Tries open_clip first (recommended). If not installed, falls back to HF transformers CLIP.
# - Similarity is cosine similarity between image and text embeddings.

import os
import re
import csv
import argparse
from datetime import datetime

import torch
from PIL import Image


PROMPT_RE = re.compile(r"prompt_sample(\d{3})\.txt$", re.IGNORECASE)

def find_pairs(root: str):
    """Yield (sample_dir, prompt_path, image_path, sample_idx)."""
    for dirpath, _, filenames in os.walk(root):
        prompt_files = []
        image_files = set(f for f in filenames if f.lower().endswith(".png"))
        for fn in filenames:
            m = PROMPT_RE.match(fn)
            if m:
                prompt_files.append((fn, int(m.group(1))))
        for fn, sample_idx in prompt_files:
            prompt_path = os.path.join(dirpath, fn)
            img_name = f"image_sample{sample_idx:03d}.png"
            if img_name in image_files:
                image_path = os.path.join(dirpath, img_name)
                yield dirpath, prompt_path, image_path, sample_idx

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def parse_task_and_B(sample_dir: str):
    # sample_dir: .../<root>/B1/task_001/sample_000
    parts = sample_dir.replace("\\", "/").split("/")
    task_id = None
    b_tag = None
    for p in parts:
        if re.fullmatch(r"B\d+", p):
            b_tag = p
        if p.startswith("task_"):
            try:
                task_id = int(p.split("_")[1])
            except Exception:
                pass
    b_num = None
    if b_tag:
        try:
            b_num = int(b_tag[1:])
        except Exception:
            b_num = None
    return task_id, b_tag, b_num

def simplify_prompt_for_clip(prompt: str) -> str:
    """
    Your saved prompt is structured; CLIP text encoder is happier with a compact description.
    Keep the first line + key fields if present.
    """
    lines = [ln.strip() for ln in prompt.splitlines() if ln.strip()]
    if not lines:
        return ""
    # Remove strict instruction lines often present in your templates
    drop_kw = ("output must", "no extra", "keep identity", "text, or logos")
    kept = []
    for ln in lines:
        low = ln.lower()
        if any(k in low for k in drop_kw):
            continue
        kept.append(ln)
    if not kept:
        kept = lines

    # Prefer: "Generate an image..." + Scene/Attribute/Composition if exists
    first = kept[0]
    fields = []
    for ln in kept[1:]:
        if ":" in ln:
            k, v = ln.split(":", 1)
            k = k.strip().lower()
            v = v.strip()
            if k in ("scene", "attribute", "composition"):
                fields.append(v)
    # Build concise text
    if fields:
        txt = first
        txt += " " + ". ".join(fields[:3])
        return txt.strip()
    return " ".join(kept[:3]).strip()

class ClipScorer:
    def __init__(self, device: str = "cuda", model_name: str = None):
        self.device = device
        self.backend = None

        # Try open_clip
        try:
            import open_clip  # type: ignore
            self.backend = "open_clip"
            # Default: strong + common
            if model_name is None:
                model_name = "ViT-B-32"
            pretrained = "openai"  # robust default
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=device
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)
            self.model.eval()
            print(f"✅ Using open_clip: {model_name} ({pretrained}) on {device}")
            return
        except Exception as e:
            print(f"⚠️ open_clip not available ({e}). Falling back to transformers CLIP...")

        # Fallback: transformers CLIP
        try:
            from transformers import CLIPProcessor, CLIPModel  # type: ignore
            self.backend = "hf_clip"
            if model_name is None:
                model_name = "openai/clip-vit-base-patch32"
            self.model = CLIPModel.from_pretrained(model_name).to(device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model.eval()
            print(f"✅ Using HF CLIP: {model_name} on {device}")
        except Exception as e:
            raise RuntimeError(
                "Neither open_clip nor transformers CLIP could be loaded.\n"
                "Install one of:\n"
                "  pip install open_clip_torch\n"
                "or\n"
                "  pip install transformers\n"
                f"Original error: {e}"
            )

    @torch.no_grad()
    def score(self, image: Image.Image, text: str) -> float:
        if self.backend == "open_clip":
            img_t = self.preprocess(image).unsqueeze(0).to(self.device)
            txt_t = self.tokenizer([text]).to(self.device)

            img_f = self.model.encode_image(img_t)
            txt_f = self.model.encode_text(txt_t)

            img_f = img_f / img_f.norm(dim=-1, keepdim=True)
            txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
            sim = (img_f * txt_f).sum(dim=-1).item()
            return float(sim)

        # hf_clip
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(self.device)

        out = self.model(**inputs)
        # Use embeddings and cosine similarity
        img_f = out.image_embeds
        txt_f = out.text_embeds
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
        sim = (img_f * txt_f).sum(dim=-1).item()
        return float(sim)

def aggregate(rows):
    """
    rows: list of dict with keys task_id, B_num, clip
    returns list of dict aggregated by (task_id, B_num)
    """
    from collections import defaultdict
    import math

    buckets = defaultdict(list)
    for r in rows:
        key = (r["task_id"], r["B_num"])
        buckets[key].append(r["clip"])

    agg = []
    for (task_id, B_num), vals in sorted(buckets.items(), key=lambda x: (x[0][0], x[0][1])):
        n = len(vals)
        mean = sum(vals) / n if n else 0.0
        var = sum((v - mean) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
        std = math.sqrt(var)
        agg.append({
            "task_id": task_id,
            "B": B_num,
            "n": n,
            "clip_mean": mean,
            "clip_std": std,
            "clip_min": min(vals) if vals else 0.0,
            "clip_max": max(vals) if vals else 0.0,
        })
    return agg

def main(args):
    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Root not found: {root}")

    device = args.device
    if device.startswith("cuda") and (not torch.cuda.is_available()):
        print("⚠️ CUDA requested but not available. Switching to cpu.")
        device = "cpu"

    print(f"📂 Root: {root}")
    print(f"🖥️ Device: {device}")

    scorer = ClipScorer(device=device, model_name=args.clip_model)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_csv = os.path.join(root, f"clip_scores_raw_{ts}.csv")
    agg_csv = os.path.join(root, f"clip_scores_agg_{ts}.csv")

    rows = []
    n_total = 0
    n_ok = 0
    n_failed = 0

    pairs = list(find_pairs(root))
    if not pairs:
        raise RuntimeError("No (prompt_sampleXXX.txt, image_sampleXXX.png) pairs found under root.")

    for sample_dir, prompt_path, image_path, sample_idx in pairs:
        n_total += 1
        task_id, b_tag, b_num = parse_task_and_B(sample_dir)

        try:
            prompt_raw = read_text(prompt_path)
            prompt_txt = simplify_prompt_for_clip(prompt_raw) if args.simplify_prompt else prompt_raw
            if not prompt_txt:
                raise RuntimeError("empty prompt after preprocessing")

            img = Image.open(image_path).convert("RGB")
            clip_sim = scorer.score(img, prompt_txt)

            rows.append({
                "status": "ok",
                "task_id": task_id if task_id is not None else -1,
                "B_tag": b_tag or "",
                "B_num": b_num if b_num is not None else -1,
                "sample_idx": sample_idx,
                "prompt_path": prompt_path,
                "image_path": image_path,
                "clip": clip_sim,
                "error": "",
            })
            n_ok += 1

            if args.verbose:
                print(f"[ok] task={task_id} B={b_tag} sample={sample_idx:03d} clip={clip_sim:.4f}")

        except Exception as e:
            rows.append({
                "status": "failed",
                "task_id": task_id if task_id is not None else -1,
                "B_tag": b_tag or "",
                "B_num": b_num if b_num is not None else -1,
                "sample_idx": sample_idx,
                "prompt_path": prompt_path,
                "image_path": image_path,
                "clip": "",
                "error": repr(e),
            })
            n_failed += 1
            print(f"❌ FAILED: {image_path} | {e}")

    # write raw
    with open(raw_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "status","task_id","B_tag","B_num","sample_idx","clip","prompt_path","image_path","error"
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # aggregate ok-only
    ok_rows = [{"task_id": r["task_id"], "B_num": r["B_num"], "clip": float(r["clip"])}
               for r in rows if r["status"] == "ok"]
    agg_rows = aggregate(ok_rows)

    with open(agg_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "task_id","B","n","clip_mean","clip_std","clip_min","clip_max"
        ])
        w.writeheader()
        for r in agg_rows:
            w.writerow(r)

    print("\n==============================")
    print("✅ CLIP evaluation done.")
    print(f"total pairs: {n_total}")
    print(f"ok:          {n_ok}")
    print(f"failed:      {n_failed}")
    print(f"raw csv:     {raw_csv}")
    print(f"agg csv:     {agg_csv}")
    print("==============================")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="results_ablation",
                    help="Root folder containing B*/task_*/sample_*/(prompt,image) files")
    ap.add_argument("--device", type=str, default="cuda",
                    help="cuda or cpu")
    ap.add_argument("--clip_model", type=str, default=None,
                    help="open_clip model name (e.g., ViT-B-32) OR HF model id (e.g., openai/clip-vit-base-patch32)")
    ap.add_argument("--simplify_prompt", action="store_true",
                    help="Use a compact text for CLIP scoring (recommended for your structured prompts)")
    ap.add_argument("--verbose", action="store_true",
                    help="Print per-sample scores")
    args = ap.parse_args()
    main(args)
