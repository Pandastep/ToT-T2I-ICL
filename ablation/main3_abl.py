# ablation/main3_abl.py

import os
import sys
import json
import argparse
from typing import List
import gc
import torch
import re
from datetime import datetime

# ---------------------------------------------------------
# Path setup
# ---------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)

for p in (THIS_DIR, PROJECT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------
# Main project imports (read-only; we do not modify them)
# ---------------------------------------------------------
from pipeline.io_contracts import SampleIn, ReasoningLog, AnalysisLog
from pipeline.io_contracts import Hypothesis as LogHypothesis
from reasoning.analyzer import ICLPatternAnalyzer as Analyzer
from reasoning.hypothesis import HypothesisGenerator
from load_dataset import load_dataset
from generation.seed_runner import SeedTextRunner
from reasoning.render_prompt import render_final_prompt

# ---------------------------------------------------------
# Ablation-specific orchestrator import
# IMPORTANT: this should point ONLY to the ablation copy
# ---------------------------------------------------------
from ablation.pipeline_abl.orchestrator_abl import run_tot_pipeline

# ---------------------------------------------------------
# Optional image generator
# ---------------------------------------------------------
try:
    from image_generator import ImageGenerator
    HAS_IMAGE_GEN = True
except ImportError:
    print("⚠️ ImageGenerator not found, will skip image generation")
    HAS_IMAGE_GEN = False


# =========================================================
# Helpers
# =========================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_output_dirs(project_root: str, output_dir: str, task_id: int, sample_idx: int) -> dict:
    base_dir = os.path.join(project_root, output_dir)
    task_dir = os.path.join(base_dir, f"task_{task_id:03d}")
    sample_dir = os.path.join(task_dir, f"sample_{sample_idx:03d}")

    ensure_dir(sample_dir)

    return {
        "base_dir": base_dir,
        "task_dir": task_dir,
        "sample_dir": sample_dir,
    }


def save_reasoning_log(reasoning_log: ReasoningLog, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(reasoning_log.to_dict(), f, indent=2, ensure_ascii=False)


def extract_image_description(final_prompt: str) -> str:
    """
    Extract a shorter image-generation description from the final prompt.
    Kept aligned with the current working main3.py logic.
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


def extract_icl_used(sample_data: dict, shot: int, data_mode: str) -> dict:
    """
    Returns a metadata-light structure:
    {
      "info": {...},
      "examples": [...],
      "query": {...}
    }
    """
    text_inputs = sample_data.get("text_inputs", [])
    image_inputs = sample_data.get("image_inputs", [])

    examples = []
    query = {"text": None, "image": None}

    # TEXT
    if isinstance(text_inputs, list):
        for i in range(min(shot, len(text_inputs))):
            examples.append({"idx": i, "text": text_inputs[i]})
        if len(text_inputs) > shot:
            query["text"] = text_inputs[shot]
        elif len(text_inputs) > 0:
            query["text"] = text_inputs[-1]

    elif isinstance(text_inputs, dict):
        ex = text_inputs.get("examples") or text_inputs.get("demos") or []
        q = text_inputs.get("query") or text_inputs.get("prompt")
        for i in range(min(shot, len(ex))):
            examples.append({"idx": i, "text": ex[i]})
        query["text"] = q

    # IMAGES
    if isinstance(image_inputs, list) and image_inputs:
        for i in range(min(shot, len(image_inputs))):
            if i < len(examples):
                examples[i]["image"] = image_inputs[i]
            else:
                examples.append({"idx": i, "text": None, "image": image_inputs[i]})
        if len(image_inputs) > shot:
            query["image"] = image_inputs[shot]

    info = {
        "shot": shot,
        "data_mode": data_mode,
        "num_examples": len(examples),
    }

    return {"info": info, "examples": examples, "query": query}


def save_icl_log_files(sample_dir: str, icl_used: dict) -> str:
    """
    Writes:
      - sample_dir/icl_used.json
      - sample_dir/icl_used.txt
    """
    json_path = os.path.join(sample_dir, "icl_used.json")
    ensure_dir(sample_dir)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(icl_used, f, indent=2, ensure_ascii=False)

    txt_path = os.path.join(sample_dir, "icl_used.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        info = icl_used.get("info", {})
        f.write(f"shot: {info.get('shot')}\n")
        f.write(f"data_mode: {info.get('data_mode')}\n")
        f.write(f"num_examples: {info.get('num_examples')}\n")

        f.write("\n=== ICL EXAMPLES ===\n")
        for ex in icl_used.get("examples", []):
            f.write(f"\n[EX {ex.get('idx')}]\n")
            f.write(f"TEXT:\n{ex.get('text')}\n")
            if ex.get("image") is not None:
                f.write(f"IMAGE: {ex.get('image')}\n")

        f.write("\n=== QUERY ===\n")
        q = icl_used.get("query", {})
        f.write(f"TEXT:\n{q.get('text')}\n")
        if q.get("image") is not None:
            f.write(f"IMAGE: {q.get('image')}\n")

    return json_path


def build_ablation_manifest(args: argparse.Namespace) -> dict:
    """
    Save all ablation-relevant settings in one place for easier bookkeeping.
    """
    return {
        "branching_factor": args.B,
        "selection_mode": args.selection_mode,
        "penalty_mode": args.penalty_mode,
        "weight_mode": args.weight_mode,
        "tau": args.tau,
        "shot": args.shot,
        "data_mode": args.data_mode,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
    }


# =========================================================
# Main
# =========================================================
def main(args: argparse.Namespace) -> None:
    print("🚀 Starting ToT-T2I-ICL Ablation Pipeline...")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("🧹 GPU memory cleared")

    # 1. Load dataset
    print("\n📥 Loading dataset...")
    raw_data = load_dataset(
        shot=args.shot,
        prompt_type="default",
        task_id=args.task_id,
        data_mode=args.data_mode,
    )

    if not raw_data:
        raise ValueError("No data loaded from dataset")

    start = max(0, args.start_idx)
    end = min(len(raw_data), start + args.num_samples) if args.num_samples > 0 else len(raw_data)
    raw_data = raw_data[start:end]

    print(f"📦 Using samples [{start}:{end}] (n={len(raw_data)})")

    # 2. Optional image generator
    image_generator = None
    if args.generate_image and HAS_IMAGE_GEN:
        print("🎨 Initializing image generator...")
        image_generator = ImageGenerator(device=args.image_device)

    all_results = []
    last_task_dir = None
    ablation_manifest = build_ablation_manifest(args)

    for local_idx, sample_data in enumerate(raw_data):
        sample_idx = args.start_idx + local_idx

        print(f"\n{'=' * 60}")
        print(f"Processing sample {local_idx + 1}/{len(raw_data)} (global_idx={sample_idx})")
        print(f"{'=' * 60}")

        dirs = get_output_dirs(PROJECT_ROOT, args.output_dir, args.task_id, sample_idx)
        task_dir = dirs["task_dir"]
        sample_dir = dirs["sample_dir"]
        last_task_dir = task_dir

        # Save ablation config for each sample directory
        manifest_path = os.path.join(sample_dir, "ablation_config.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(ablation_manifest, f, indent=2, ensure_ascii=False)

        icl_used = extract_icl_used(
            sample_data=sample_data,
            shot=args.shot,
            data_mode=args.data_mode,
        )
        icl_log_path = save_icl_log_files(sample_dir, icl_used)
        print(f"🧾 ICL used saved: {icl_log_path}")

        image_path = None

        sample_in = SampleIn(
            text_inputs=sample_data["text_inputs"],
            image_inputs=sample_data.get("image_inputs", []),
        )
        sample_in.validate()

        print(
            f"📋 Sample loaded: {len(sample_in.text_inputs)} text inputs, "
            f"{len(sample_in.image_inputs)} image inputs"
        )

        # 3. Pattern analysis
        print("🔍 Analyzing ICL patterns...")
        analyzer = Analyzer()
        pa = analyzer.analyze_icl_patterns(sample_data)

        analysis_log = AnalysisLog(
            common=" ".join(getattr(pa, "common_elements", [])),
            varying=" ".join(getattr(pa, "support_values", [])),
            query=" ".join(getattr(pa, "query_values", [])),
            notes=f"relation={getattr(pa, 'relation_type', 'unknown')}",
        )

        # 4. Hypothesis generation
        print("💡 Generating hypothesis...")
        hypothesis_generator = HypothesisGenerator()
        hypothesis = hypothesis_generator.generate_hypothesis(sample_data)

        print(f"   Hypothesis summary: {hypothesis.summary}")
        print(f"   Rule: {getattr(hypothesis, 'rule', '')}")
        print(f"   Constraints: {getattr(hypothesis, 'constraints', [])}")

        # 5. ToT reasoning via ablation orchestrator
        print("🧠 Starting ToT Ablation Reasoning...")
        print(
            f"   Ablation config | B={args.B}, "
            f"selection_mode={args.selection_mode}, "
            f"penalty_mode={args.penalty_mode}"
        )

        runner = SeedTextRunner(device=args.device)

        def llm_generate_fn(prompt: str, k: int) -> List[str]:
            outs = []
            base_seed = args.seed if args.seed is not None else 123

            # IMPORTANT:
            # for branching-factor ablation we intentionally cap generation
            # by args.B, not only by the stage-provided k
            k_eff = max(1, min(int(args.B), int(k))) if args.respect_stage_k else max(1, int(args.B))

            for i in range(k_eff):
                out = runner.llm_generate(
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    seed=base_seed + i,
                )
                outs.append(out)

                preview = out[:80].replace("\n", " ")
                print(f"[DEBUG] branch {i + 1}/{k_eff}: {preview}")

            return outs

        orchestrator_output = run_tot_pipeline(
            hypothesis=hypothesis,
            llm_generate_fn=llm_generate_fn,
            tau=args.tau,
            pattern_analysis=pa,
            branching_factor=args.B,
            selection_mode=args.selection_mode,
            penalty_mode=args.penalty_mode,
            weight_mode=args.weight_mode,
            seed=args.seed,
        )

        final_prompt = render_final_prompt(
            hypothesis=hypothesis,
            winning_branch=orchestrator_output.winning_branch,
        )

        print("✅ ToT Ablation Reasoning completed")

        log_hypothesis = LogHypothesis(
            common=list(getattr(hypothesis, "common", [])),
            support_values=list(getattr(hypothesis, "support_values", [])),
            query=list(getattr(hypothesis, "query", [])),
            relation_type=str(getattr(hypothesis, "relation_type", "unknown")),
            rule=str(getattr(hypothesis, "rule", "unknown")),
            constraints=list(getattr(hypothesis, "constraints", [])),
            summary=str(getattr(hypothesis, "summary", "")),
        )

        reasoning_log = ReasoningLog(
            analysis=analysis_log,
            hypothesis=log_hypothesis,
            stages=orchestrator_output.stage_results,
            winning_branch=orchestrator_output.winning_branch,
            final_prompt=final_prompt,
        )

        reasoning_path = os.path.join(sample_dir, "reasoning_log.json")
        save_reasoning_log(reasoning_log, reasoning_path)
        print(f"📊 Reasoning log saved: {reasoning_path}")

        prompt_path = os.path.join(sample_dir, f"prompt_sample{sample_idx:03d}.txt")
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(final_prompt)
        print(f"📝 Final prompt saved: {prompt_path}")

        # 6. Image generation
        if args.generate_image and image_generator is not None:
            print("🎨 Generating image...")

            image_description = extract_image_description(final_prompt)
            print(f"📝 Image description: {image_description[:100]}...")

            image = image_generator.generate(
                prompt=image_description,
                seed=(args.seed + sample_idx) if args.seed is not None else (123 + sample_idx),
            )

            if image is not None:
                image_path = os.path.join(sample_dir, f"image_sample{sample_idx:03d}.png")
                image.save(image_path)
                print(f"🖼️ Image saved: {image_path}")

                desc_path = os.path.join(sample_dir, f"desc_sample{sample_idx:03d}.txt")
                with open(desc_path, "w", encoding="utf-8") as f:
                    f.write("FINAL PROMPT USED FOR IMAGE:\n")
                    f.write(final_prompt + "\n\n")
                    f.write("RAW ToT OUTPUT (debug):\n")
                    f.write(getattr(orchestrator_output, "final_prompt", "") + "\n\n")
                    f.write("IMAGE DESCRIPTION (extracted):\n")
                    f.write(image_description + "\n")
            else:
                print("❌ Image generation failed for this sample")

        del runner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        sample_result = {
            "sample_idx": sample_idx,
            "ablation": ablation_manifest,
            "hypothesis_summary": hypothesis.summary,
            "hypothesis_rule": getattr(hypothesis, "rule", ""),
            "reasoning_log": reasoning_path,
            "prompt_file": prompt_path,
            "icl_used_log": icl_log_path,
            "ablation_config_file": manifest_path,
        }

        if args.generate_image and image_path is not None:
            sample_result["image_file"] = image_path

        all_results.append(sample_result)
        print(f"✅ Sample {sample_idx + 1} completed")

    # 7. Summary report
    print(f"\n{'=' * 60}")
    print("Generating summary report...")

    if last_task_dir is None:
        raise RuntimeError("No task directory created; dataset may be empty.")

    summary_path = os.path.join(last_task_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "task_id": args.task_id,
                "shot": args.shot,
                "tau": args.tau,
                "data_mode": args.data_mode,
                "total_samples": len(raw_data),
                "generated_images": len([r for r in all_results if "image_file" in r]),
                "ablation": ablation_manifest,
                "samples": all_results,
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"📋 Summary saved: {summary_path}")
    print("\n✅ Ablation pipeline completed successfully!")
    print(f"   Processed {len(all_results)} samples")
    print(f"   Generated {len([r for r in all_results if 'image_file' in r])} images")


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ToT-T2I-ICL Ablation Pipeline")

    # Data
    parser.add_argument("--task_id", type=int, default=1, help="Task ID")
    parser.add_argument("--shot", type=int, default=2, help="Number of ICL examples")
    parser.add_argument("--data_mode", type=str, default="default", help="Data loading mode")
    parser.add_argument("--start_idx", type=int, default=0, help="Start dataset index")
    parser.add_argument("--num_samples", type=int, default=10, help="How many samples to process")

    # Reasoning
    parser.add_argument("--tau", type=float, default=0.08, help="Branching threshold")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max tokens for reasoning")

    # Ablation
    parser.add_argument("--B", type=int, default=3, help="Branching factor / number of candidate generations")
    parser.add_argument(
        "--selection_mode",
        type=str,
        default="score",
        choices=["score", "first", "random"],
        help="Branch selection mode inside ablation orchestrator",
    )
    parser.add_argument(
        "--penalty_mode",
        type=str,
        default="full",
        choices=["full", "no_unsupported", "no_redundancy", "none"],
        help="Penalty setting inside ablation scoring",
    )

    parser.add_argument(
        "--weight_mode",
        type=str,
        default="original",
        choices=["original", "equal", "no_anchor", "no_constraints"],
        help="Weight setting for Eq. 3 scoring sensitivity ablation",
    )
    
    parser.add_argument(
        "--respect_stage_k",
        action="store_true",
        help="If set, use min(B, stage_k). Otherwise force generation width to B.",
    )

    # Image generation
    parser.add_argument("--generate_image", action="store_true", help="Generate final image")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    # System
    parser.add_argument("--device", type=str, default="cuda", help="Device for reasoning")
    parser.add_argument("--image_device", type=str, default="cuda", help="Device for image generation")
    parser.add_argument("--output_dir", type=str, default="results_ablation", help="Output directory")
    parser.add_argument("--output_tag", type=str, default="", help="Extra tag to separate runs")

    args = parser.parse_args()

    if args.output_tag:
        args.output_dir = os.path.join(args.output_dir, args.output_tag)

    main(args)