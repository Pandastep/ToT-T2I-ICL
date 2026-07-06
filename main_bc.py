# main_bc.py
# Structured Baseline + CoT runner
# Fairer comparison against ToT:
# - no task metadata in prompt logic
# - no x_space usage in prompt construction
# - baseline is structured (not weak one-line)
# - CoT is single-path reasoning from ICL only
# - same structured final prompt template for both methods

import os
import sys
import json
import argparse
import gc
import re
from datetime import datetime
from typing import Optional, Dict, Any

import torch

# --------------------------------------------------
# PATH SETUP
# --------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# --------------------------------------------------
# PROJECT IMPORTS
# --------------------------------------------------
from pipeline.io_contracts import SampleIn
from reasoning.analyzer import ICLPatternAnalyzer as Analyzer
from generation.seed_runner import SeedTextRunner
from load_dataset import load_dataset

# Optional image generator
try:
    from image_generator import ImageGenerator
    HAS_IMAGE_GEN = True
except ImportError:
    HAS_IMAGE_GEN = False
    print("⚠️ ImageGenerator not found — image generation disabled")


# --------------------------------------------------
# UTILS
# --------------------------------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _clean(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s).replace("</s>", "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def get_output_dirs(output_root: str, method: str, task_id: int, sample_idx: int) -> Dict[str, str]:
    """
    output_root: e.g. "results_final"
    method: "baseline" or "cot"
    """
    base_dir = os.path.join(ROOT_DIR, output_root, method)
    task_dir = os.path.join(base_dir, f"task_{task_id:03d}")
    sample_dir = os.path.join(task_dir, f"sample_{sample_idx:03d}")
    ensure_dir(sample_dir)
    return {"base_dir": base_dir, "task_dir": task_dir, "sample_dir": sample_dir}


def format_icl_text_block(text_inputs) -> str:
    """
    CoBSAT-style:
    text_inputs = [demo1, demo2, ..., query]
    """
    if not isinstance(text_inputs, list) or len(text_inputs) == 0:
        return ""

    demos = text_inputs[:-1]
    query = text_inputs[-1]

    out = []
    out.append("=== IN-CONTEXT EXAMPLES ===")
    for i, ex in enumerate(demos):
        out.append(f"[EXAMPLE {i + 1}]")
        out.append(str(ex).strip())
        out.append("")
    out.append("=== QUERY ===")
    out.append(str(query).strip())
    return "\n".join(out).strip()


def format_icl_json_block(sample_in: SampleIn, theta: str, target_x: str) -> str:
    """
    Explicit reasoning input block.
    Still metadata-free.
    """
    demos = sample_in.text_inputs[:-1] if isinstance(sample_in.text_inputs, list) and len(sample_in.text_inputs) > 1 else []
    query = sample_in.text_inputs[-1] if isinstance(sample_in.text_inputs, list) and len(sample_in.text_inputs) > 0 else ""

    payload = {
        "examples": demos,
        "query": query,
        "theta": theta,
        "target_x": target_x,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def extract_image_description(prompt: str) -> str:
    """
    Keep only the useful lines for the image generator.
    """
    keep = []
    for ln in prompt.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if "no extra objects" in ln.lower():
            continue
        keep.append(ln)

    s = " ".join(keep)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:400]


def choose_subject_from_analysis(theta: str, analysis) -> str:
    """
    Metadata-free subject selection.
    Prefer theta if present.
    Otherwise use a non-generic invariant from analyzer.
    """
    theta = _clean(theta)
    if theta:
        return theta

    invariants = list(getattr(analysis, "common_elements", []) or [])
    generic = {"shared_context", "context", "scene", "background"}

    for inv in invariants:
        inv_clean = _clean(inv)
        if inv_clean and inv_clean.lower() not in generic:
            return inv_clean

    return "subject"


def build_structured_prompt(
    subject: str,
    target_x: str,
    scene: str,
    stability: str,
    composition: str,
) -> str:
    """
    Shared structured prompt template for BOTH baseline and CoT.
    No task metadata labels such as x_space/color/style/etc.
    """
    subject = _clean(subject) or "subject"
    target_x = _clean(target_x)
    scene = _clean(scene)
    stability = _clean(stability)
    composition = _clean(composition)

    headline = f"Generate an image that follows the demonstrated pattern for the subject '{subject}' and target value '{target_x}'."

    lines = [
        headline,
        f"Scene: {scene}",
        f"Target: apply the query value '{target_x}' to the intended aspect of '{subject}' as implied by the examples.",
        f"Stability: {stability}",
        f"Composition: {composition}",
        "No extra objects, text, or logos. Keep identity and viewpoint consistent.",
    ]
    return "\n".join(lines)


# --------------------------------------------------
# STRUCTURED BASELINE (NO LM REASONING)
# --------------------------------------------------
def run_structured_baseline(
    sample_data: dict,
    sample_in: SampleIn,
    analyzer: Analyzer,
    theta: str,
    target_x: str,
) -> Dict[str, str]:
    """
    Baseline:
    - no chain-of-thought generation
    - metadata-free
    - uses analyzer only as deterministic pattern inference
    - outputs the SAME structured prompt format as CoT
    """
    analysis = analyzer.analyze_icl_patterns(sample_data)

    subject = choose_subject_from_analysis(theta, analysis)

    invariants = list(getattr(analysis, "common_elements", []) or [])
    varyings = list(getattr(analysis, "varying_candidates", []) or [])
    transfer = _clean(getattr(analysis, "transfer_hypothesis", ""))

    if invariants:
        scene = f"Preserve the shared demonstrated context and recurring elements: {', '.join(invariants[:3])}."
    else:
        scene = "Preserve the shared demonstrated context inferred from the examples."

    if "attribute_value" in varyings:
        stability = f"Keep the subject identity consistent while applying the query value '{target_x}'."
    elif "subject_identity" in varyings:
        stability = f"Preserve the repeated context and transfer the demonstrated pattern to the query subject '{theta}'."
    else:
        stability = "Maintain the repeated structure inferred from the demonstrations without adding unrelated details."

    if transfer:
        composition = transfer
    else:
        composition = f"Combine the demonstrated shared structure with the query value '{target_x}' in a coherent single-subject image."

    final_prompt = build_structured_prompt(
        subject=subject,
        target_x=target_x,
        scene=scene,
        stability=stability,
        composition=composition,
    )

    baseline_note = (
        "Structured baseline built without language-model reasoning. "
        "It uses only metadata-free pattern analysis from the ICL examples."
    )

    return {
        "analysis_text": baseline_note,
        "final_prompt": final_prompt,
    }


# --------------------------------------------------
# CoT (SINGLE TRAJECTORY, NO METADATA)
# --------------------------------------------------
def run_cot(
    sample_data: dict,
    sample_in: SampleIn,
    analyzer: Analyzer,
    runner: Optional[SeedTextRunner], 
    max_new_tokens: int,
    theta: str,
    target_x: str,
) -> Dict[str, str]:

    analysis = analyzer.analyze_icl_patterns(sample_data)

    subject = choose_subject_from_analysis(theta, analysis)

    invariants = list(getattr(analysis, "common_elements", []) or [])
    varyings = list(getattr(analysis, "varying_candidates", []) or [])
    transfer = _clean(getattr(analysis, "transfer_hypothesis", ""))

    # ---- Deterministic single-path CoT-style reasoning ----

    scene = f"The examples consistently depict: {', '.join(invariants[:3])}." if invariants else \
            "The examples share a common visual context."

    if "attribute_value" in varyings:
        stability = f"The subject remains the same while only the attribute changes to '{target_x}'."
    elif "subject_identity" in varyings:
        stability = f"The context remains stable while the subject changes to '{theta}'."
    else:
        stability = "The structure remains consistent while applying the query change."

    if transfer:
        composition = f"Apply the pattern: {transfer}"
    else:
        composition = f"Generate '{subject}' with the query value '{target_x}' following the demonstrated pattern."

    cot_text = f"""Reasoning:
Scene: {scene}
Stability: {stability}
Composition: {composition}
"""

    final_prompt = build_structured_prompt(
        subject=subject,
        target_x=target_x,
        scene=scene,
        stability=stability,
        composition=composition,
    )

    return {"cot_text": cot_text, "final_prompt": final_prompt}


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main(args: argparse.Namespace) -> None:
    print("🚀 Running Structured Baseline / CoT pipeline")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    raw_data = load_dataset(
        shot=args.shot,
        prompt_type="default",
        task_id=args.task_id,
        data_mode=args.data_mode,
    )
    if not raw_data:
        raise ValueError("Dataset loader returned empty data.")

    raw_data = raw_data[:args.max_samples]

    analyzer = Analyzer()
    runner = None

    image_generator = None
    if args.generate_image:
        if not HAS_IMAGE_GEN:
            print("⚠️ --generate_image was set but ImageGenerator is missing. Skipping image generation.")
        else:
            image_generator = ImageGenerator(device=args.image_device,
                use_seed=False,
                use_sd=True,
                )

    task_dirs = get_output_dirs(args.output_dir, args.method, args.task_id, 0)
    ensure_dir(task_dirs["task_dir"])

    all_results = []
    n_images = 0

    for sample_idx, sample_data in enumerate(raw_data):
        dirs = get_output_dirs(args.output_dir, args.method, args.task_id, sample_idx)
        sample_dir = dirs["sample_dir"]

        sample_in = SampleIn(
            text_inputs=sample_data["text_inputs"],
            image_inputs=sample_data.get("image_inputs", []),
        )

        theta = sample_data.get("theta", "")
        target_x = sample_data.get("target_x", "")

        cot_text = None
        analysis_text = None

        if args.method == "baseline":
            out = run_structured_baseline(
                sample_data=sample_data,
                sample_in=sample_in,
                analyzer=analyzer,
                theta=theta,
                target_x=target_x,
            )
            final_prompt = out["final_prompt"]
            analysis_text = out["analysis_text"]

        elif args.method == "cot":
            out = run_cot(
                sample_data=sample_data,
                sample_in=sample_in,
                analyzer=analyzer,
                runner=runner,
                max_new_tokens=args.max_new_tokens,
                theta=theta,
                target_x=target_x,
            )
            final_prompt = out["final_prompt"]
            cot_text = out["cot_text"]

        else:
            raise ValueError("method must be baseline or cot")

        prompt_path = os.path.join(sample_dir, "prompt.txt")
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(final_prompt)

        if analysis_text is not None:
            baseline_path = os.path.join(sample_dir, "baseline_note.txt")
            with open(baseline_path, "w", encoding="utf-8") as f:
                f.write(analysis_text)

        if cot_text is not None:
            cot_path = os.path.join(sample_dir, "cot_reasoning.txt")
            with open(cot_path, "w", encoding="utf-8") as f:
                f.write(cot_text)

        icl_path = os.path.join(sample_dir, "icl_used.txt")
        with open(icl_path, "w", encoding="utf-8") as f:
            f.write(format_icl_text_block(sample_in.text_inputs))

        image_path = None
        if image_generator is not None:
            desc = extract_image_description(final_prompt)
            try:
                img = image_generator.generate(
                    prompt=desc,
                    seed=args.seed + sample_idx,
                    prefer_seed=False,
                )
            except Exception as e:
                img = None
                with open(os.path.join(sample_dir, "image_error.txt"), "w", encoding="utf-8") as f:
                    f.write(str(e))

            if img is not None:
                image_path = os.path.join(sample_dir, "image.png")
                img.save(image_path)
                n_images += 1

        all_results.append({
            "sample_idx": sample_idx,
            "theta": theta,
            "target_x": target_x,
            "prompt_file": prompt_path,
            "baseline_note_file": os.path.join(sample_dir, "baseline_note.txt") if analysis_text is not None else None,
            "cot_file": os.path.join(sample_dir, "cot_reasoning.txt") if cot_text is not None else None,
            "image_file": image_path,
        })

        print(f"✅ sample {sample_idx:03d} done | prompt: {os.path.basename(prompt_path)} | image: {bool(image_path)}")

    summary_path = os.path.join(
        ROOT_DIR, args.output_dir, args.method, f"task_{args.task_id:03d}",
        f"summary_{args.method}.json"
    )

    save_json({
        "task_id": args.task_id,
        "method": args.method,
        "shot": args.shot,
        "data_mode": args.data_mode,
        "max_samples": args.max_samples,
        "num_samples": len(all_results),
        "generated_images": n_images,
        "timestamp": datetime.now().isoformat(),
        "samples": all_results,
    }, summary_path)

    print(f"✅ Done. Summary saved to {summary_path}")


# --------------------------------------------------
# CLI
# --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Structured Baseline / CoT runner")

    parser.add_argument("--method", choices=["baseline", "cot"], required=True)
    parser.add_argument("--task_id", type=int, default=1)
    parser.add_argument("--shot", type=int, default=2)
    parser.add_argument("--data_mode", type=str, default="default")
    parser.add_argument("--max_samples", type=int, default=10)

    parser.add_argument("--max_new_tokens", type=int, default=256)

    parser.add_argument("--generate_image", action="store_true")
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="results")

    args = parser.parse_args()
    main(args)