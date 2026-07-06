# reasoning/stages.py

from typing import Dict, List, Literal, Union
import re
from .hypothesis import Hypothesis

StageName = Literal["scene", "attribute", "stability", "composition"]


def get_branching_width(stage: StageName) -> int:
    return {
        "scene": 3,
        "attribute": 3,
        "stability": 2,
        "composition": 2,
    }.get(stage, 2)


def _as_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    value = str(value).strip()
    return [value] if value else []


def _extract_hypothesis_fields(
    hypothesis: Union[Dict[str, List[str]], Hypothesis]
) -> Dict[str, Union[str, List[str]]]:
    if isinstance(hypothesis, Hypothesis):
        common = _as_list(getattr(hypothesis, "common", []))
        support_values = _as_list(getattr(hypothesis, "support_values", []))
        query = _as_list(getattr(hypothesis, "query", []))
        relation_type = str(getattr(hypothesis, "relation_type", "unknown")).strip()
        rule = str(getattr(hypothesis, "rule", "")).strip()
        constraints = _as_list(getattr(hypothesis, "constraints", []))
        summary = str(getattr(hypothesis, "summary", "")).strip()
    else:
        common = _as_list(hypothesis.get("common", []))
        support_values = _as_list(
            hypothesis.get("support_values", [])
            or hypothesis.get("changing", [])
            or hypothesis.get("change", [])
        )
        query = _as_list(hypothesis.get("query", []))
        relation_type = str(hypothesis.get("relation_type", "unknown")).strip()
        rule = str(hypothesis.get("rule", "")).strip()
        constraints = _as_list(hypothesis.get("constraints", []))
        summary = str(hypothesis.get("summary", "")).strip()

    return {
        "common": common,
        "support_values": support_values,
        "query": query,
        "relation_type": relation_type,
        "rule": rule,
        "constraints": constraints,
        "summary": summary,
    }


def _header_from_hypothesis(
    hypothesis: Union[Dict[str, List[str]], Hypothesis]
) -> str:
    """
    Keep the header informative but not overly dataset-conditioning.
    Avoid exposing support values as a lexical target to imitate.
    """
    fields = _extract_hypothesis_fields(hypothesis)

    query = fields["query"]
    relation_type = fields["relation_type"]
    rule = fields["rule"]
    constraints = fields["constraints"]
    summary = fields["summary"]

    lines: List[str] = []

    if query:
        lines.append(f"Target query: {', '.join(query)}")

    if relation_type and relation_type != "unknown":
        lines.append(f"Inferred relation: {relation_type}")

    if rule:
        lines.append(f"Inferred task rule: {rule}")
    elif summary:
        lines.append(f"Inferred task summary: {summary}")

    if constraints:
        lines.append("High-level constraints:")
        for c in constraints:
            lines.append(f"- {c}")

    return "\n".join(lines).strip()


def _core_global_guidelines() -> str:
    """
    General reasoning guidance that does NOT encode benchmark-specific
    assumptions such as 'object must stay fixed' or 'scene must not change'.
    """
    return (
        "Important guidelines:\n"
        "- Focus on the target query and the pattern inferred from the examples.\n"
        "- Avoid introducing unrelated concepts, entities, or scene details that are not justified.\n"
        "- Changes to the object, attributes, action, texture, style, or context are allowed if they are consistent with the inferred pattern.\n"
        "- Do not drift into generic image captioning or free-form storytelling.\n"
        "- Prefer concise, visually interpretable reasoning.\n"
        "- Each thought should be a plausible candidate interpretation, not a random rewrite.\n"
    )


def _output_format_block(k: int) -> str:
    return (
        f"Output exactly {k} items.\n"
        "Use the format:\n"
        + "\n".join(
            [f"Thought {i}: Describe one short idea in one sentence." for i in range(1, k + 1)]
        )
        + "\nDo not add any other lines, summaries, or justifications."
    )


def parse_thoughts(text: str, k: int = 3) -> List[str]:
    text = text.strip()

    cands = re.findall(
        r"Thought\s*(\d+)\s*:\s*(.*?)(?=\n\s*Thought\s*\d+\s*:|\Z)",
        text,
        flags=re.S | re.I,
    )
    if cands:
        cands_sorted = [t.strip() for _, t in sorted(cands, key=lambda z: int(z[0]))]
        uniq: List[str] = []
        for c in cands_sorted:
            if c and all(c.lower() != u.lower() for u in uniq):
                uniq.append(c)
        return uniq[:k]

    bullets = re.findall(r"^(?:-|\*|\d+[\).\]])\s+(.*)$", text, flags=re.M)
    bullets = [b.strip() for b in bullets if b.strip()]
    if bullets:
        uniq: List[str] = []
        for b in bullets:
            if all(b.lower() != u.lower() for u in uniq):
                uniq.append(b)
        return uniq[:k]

    lines = [ln.strip() for ln in re.split(r"[\n\.]+", text) if ln.strip()]
    uniq: List[str] = []
    for ln in lines:
        if all(ln.lower() != u.lower() for u in uniq):
            uniq.append(ln)
    return uniq[:k]


# ---------- Stage prompts ----------

def prompt_stage_scene(h, k: int = 3) -> str:
    """
    SCENE = high-level interpretation of how the target should be visually understood.
    This is NOT free background narration and NOT final composition.
    """
    hdr = _header_from_hypothesis(h)
    return (
        f"{hdr}\n\n"
        "Stage: High-level visual interpretation.\n"
        "Task: Propose several short candidate interpretations of how the target query should be understood after applying the inferred pattern.\n\n"
        + _core_global_guidelines()
        + "\n"
        "Stage-specific guidance:\n"
        "- Focus on the main subject and its overall visual identity.\n"
        "- Stay close to the target query.\n"
        "- Do not turn this stage into detailed composition, layout, or camera framing.\n"
        "- Do not turn this stage into free scene narration.\n"
        "- Do not force unnecessary object substitution.\n"
        "- Each thought must be meaningfully different.\n\n"
        "Return short, concrete, one-sentence thoughts only.\n\n"
        + _output_format_block(k)
    )


def prompt_stage_attribute(h, k: int = 3) -> str:
    """
    ATTRIBUTE = how the inferred pattern manifests in visible properties.
    """
    hdr = _header_from_hypothesis(h)
    return (
        f"{hdr}\n\n"
        "Stage: Visible transformation details.\n"
        "Task: Propose several different ways the inferred pattern could manifest in the target query's visible properties.\n\n"
        + _core_global_guidelines()
        + "\n"
        "Stage-specific guidance:\n"
        "- Focus on directly visible properties such as style, texture, material, shape, color, surface detail, or appearance.\n"
        "- Refine the target query rather than drifting into unrelated scene elements.\n"
        "- Do not simply copy the reference examples literally.\n"
        "- Each thought should express one plausible transformed version of the target query.\n"
        "- Each thought must be meaningfully different.\n\n"
        "Return short, concrete, one-sentence thoughts only.\n\n"
        + _output_format_block(k)
    )


def prompt_stage_stability(h, k: int = 2) -> str:
    """
    STABILITY = what keeps the current hypothesis coherent and recognizable.
    """
    hdr = _header_from_hypothesis(h)
    return (
        f"{hdr}\n\n"
        "Stage: Consistency and recognizability.\n"
        "Task: Propose conditions or refinements that keep the current target interpretation clear, coherent, and faithful to the inferred pattern.\n\n"
        + _core_global_guidelines()
        + "\n"
        "Stage-specific guidance:\n"
        "- Focus on recognizability, coherence, and consistency of the current interpretation.\n"
        "- Avoid contradiction, ambiguity, or random drift.\n"
        "- Do not introduce unnecessary new entities.\n"
        "- These thoughts should stabilize the interpretation rather than replace it with a new one.\n"
        "- Each thought must be meaningfully different.\n\n"
        "Return short, concrete, one-sentence thoughts only.\n\n"
        + _output_format_block(k)
    )


def prompt_stage_composition(h, k: int = 2) -> str:
    """
    COMPOSITION = how to present the already-chosen interpretation clearly.
    """
    hdr = _header_from_hypothesis(h)
    return (
        f"{hdr}\n\n"
        "Stage: Presentation and arrangement.\n"
        "Task: Propose composition choices that make the current transformed target interpretation easy to perceive.\n\n"
        + _core_global_guidelines()
        + "\n"
        "Stage-specific guidance:\n"
        "- Focus on presentation, salience, framing, placement, or layout of the chosen interpretation.\n"
        "- Do not replace the main subject with a different idea.\n"
        "- Do not revert to abstract rule explanation.\n"
        "- These thoughts should help present the target clearly, not redefine it from scratch.\n"
        "- Each thought must be meaningfully different.\n\n"
        "Return short, concrete, one-sentence thoughts only.\n\n"
        + _output_format_block(k)
    )


def _history_block(previous_thoughts: List[str] | None) -> str:
    """
    Carry forward the current branch as evolving reasoning state.
    Do not invite the model to reset or ignore prior steps.
    """
    if not previous_thoughts:
        return ""

    numbered = [f"{i+1}) {t}" for i, t in enumerate(previous_thoughts)]
    return (
        "Current reasoning branch:\n"
        + "\n".join(numbered)
        + "\n\n"
        "Use these earlier thoughts as the current hypothesis state.\n"
        "Refine, clarify, or extend them without unnecessary contradiction or reset.\n"
        "If an earlier thought is weak, improve it by making the next thought more coherent and more faithful to the target query.\n\n"
    )


def make_stage_prompt(
    h: Union[Dict[str, List[str]], Hypothesis],
    stage: StageName,
    k: int | None = None,
    previous_thoughts: List[str] | None = None,
) -> str:
    if k is None:
        k = get_branching_width(stage)

    history_block = _history_block(previous_thoughts)

    if stage == "scene":
        return history_block + prompt_stage_scene(h, k)
    if stage == "attribute":
        return history_block + prompt_stage_attribute(h, k)
    if stage == "stability":
        return history_block + prompt_stage_stability(h, k)
    if stage == "composition":
        return history_block + prompt_stage_composition(h, k)

    raise ValueError(f"Unknown stage: {stage}")