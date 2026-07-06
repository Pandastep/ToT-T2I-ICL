# reasoning/format_reasoning.py

from typing import Dict, List, Any, Union
import json
from .hypothesis import Hypothesis
from .selector import SelectionResult
from .render_prompt import ReasoningBranch


def _join(items: List[str]) -> str:
    items = [str(x).strip() for x in items if str(x).strip()]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f" and {items[-1]}"


# ---------------------------
# MAIN EXPORTS
# ---------------------------

def reasoning_to_json(orch_output: Any) -> str:
    hypothesis_data = _extract_hypothesis_data(orch_output.hypothesis)

    payload = {
        "hypothesis": hypothesis_data,
        "stages": [
            {
                "stage": stage_result.stage,
                "generated_thoughts": getattr(stage_result, "generated_thoughts", []),
                "scores": getattr(stage_result, "scores", []),
                "parts": getattr(stage_result, "parts", []),
                "selection_result": _format_selection_result(
                    getattr(stage_result, "selection_result", None)
                ),
                "kept_branches": getattr(stage_result, "kept_branches", []),
                "branching_config": getattr(stage_result, "branching_config", {}),
            }
            for stage_result in getattr(orch_output, "stage_results", [])
        ],
        "winning_branch": _format_winning_branch(
            getattr(orch_output, "winning_branch", None)
        ),
        "final_prompt": getattr(orch_output, "final_prompt", ""),
        "reasoning_summary": getattr(orch_output, "reasoning_summary", ""),
    }

    return json.dumps(payload, ensure_ascii=False, indent=2)


def reasoning_pretty_text(orch_output: Any) -> str:
    hypothesis_data = _extract_hypothesis_data(orch_output.hypothesis)

    lines = ["Hypothesis:"]

    # NEW: rule-first representation
    if hypothesis_data.get("rule"):
        lines.append(f"  Rule: {hypothesis_data['rule']}")

    if hypothesis_data.get("constraints"):
        lines.append("  Constraints:")
        for c in hypothesis_data["constraints"]:
            lines.append(f"    - {c}")

    # optional debug info
    if hypothesis_data.get("summary"):
        lines.append(f"  Summary: {hypothesis_data['summary']}")

    lines.append("\nStages:")

    for i, stage_result in enumerate(getattr(orch_output, "stage_results", [])):
        sel_result = getattr(stage_result, "selection_result", None)

        if sel_result:
            lines.append(f"  {i+1}. [{sel_result.stage}]")
            lines.append(f"     Generated: {len(stage_result.generated_thoughts)}")
            lines.append(f"     Kept: {sel_result.keep_count}")
            lines.append(f"     Gap: {sel_result.gap:.3f} (τ={sel_result.tau})")

            for j, candidate in enumerate(sel_result.kept[:2]):
                text = candidate.text.replace("\n", " ").strip()
                lines.append(
                    f"     Branch {j+1}: {text[:80]}... (score: {candidate.score:.3f})"
                )
            lines.append("")

    winning_branch = getattr(orch_output, "winning_branch", None)

    if winning_branch:
        lines.append("Winning Branch:")
        lines.append(f"  Scene: {winning_branch.scene_thought}")
        lines.append(f"  Attribute: {winning_branch.attribute_thought}")
        lines.append(f"  Stability: {winning_branch.stability_thought}")
        lines.append(f"  Composition: {winning_branch.composition_thought}")
        lines.append(f"  Score: {winning_branch.branch_score:.3f}")
        lines.append("")

    lines.append("Final Prompt:")
    lines.append(f"  {getattr(orch_output, 'final_prompt', '')}")

    return "\n".join(lines)


# ---------------------------
# HELPERS
# ---------------------------

def _extract_hypothesis_data(hypothesis: Union[Dict, Hypothesis]) -> Dict:
    """
    Unified hypothesis format.
    """

    if isinstance(hypothesis, Hypothesis):
        return {
            "common": getattr(hypothesis, "common", []),
            "support_values": getattr(hypothesis, "support_values", []),
            "query": getattr(hypothesis, "query", []),
            "relation_type": getattr(hypothesis, "relation_type", "unknown"),
            "rule": getattr(hypothesis, "rule", ""),
            "constraints": getattr(hypothesis, "constraints", []),
            "summary": getattr(hypothesis, "summary", ""),
        }

    # legacy
    old_change = hypothesis.get("changing", hypothesis.get("change", []))

    return {
        "common": hypothesis.get("common", []),
        "support_values": old_change,
        "query": hypothesis.get("query", []),
        "relation_type": hypothesis.get("relation_type", "legacy-unknown"),
        "rule": hypothesis.get("rule", ""),
        "constraints": hypothesis.get("constraints", []),
        "summary": hypothesis.get("summary", ""),
    }


def _format_selection_result(selection_result: SelectionResult) -> Dict:
    if not selection_result:
        return {}

    return {
        "stage": selection_result.stage,
        "ranked_candidates": [
            {"text": c.text, "score": c.score, "parts": c.parts}
            for c in selection_result.ranked
        ],
        "kept_candidates": [
            {"text": c.text, "score": c.score, "parts": c.parts}
            for c in selection_result.kept
        ],
        "gap": selection_result.gap,
        "tau": selection_result.tau,
        "keep_count": selection_result.keep_count,
    }


def _format_winning_branch(winning_branch: ReasoningBranch) -> Dict:
    if not winning_branch:
        return {}

    return {
        "scene_thought": winning_branch.scene_thought,
        "attribute_thought": winning_branch.attribute_thought,
        "stability_thought": winning_branch.stability_thought,
        "composition_thought": winning_branch.composition_thought,
        "branch_score": winning_branch.branch_score,
    }


# ---------------------------
# LEGACY
# ---------------------------

def reasoning_to_json_legacy(orch_output: Any) -> str:
    return reasoning_to_json(orch_output)


def reasoning_pretty_text_legacy(orch_output: Any) -> str:
    return reasoning_pretty_text(orch_output)