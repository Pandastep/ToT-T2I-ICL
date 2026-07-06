# reasoning/render_prompt.py

from typing import Dict, List, Any
from dataclasses import dataclass
import re

from .hypothesis import Hypothesis


@dataclass
class ReasoningBranch:
    """Full reasoning branch through all 4 stages."""
    scene_thought: str
    attribute_thought: str
    stability_thought: str
    composition_thought: str
    branch_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_thought": self.scene_thought,
            "attribute_thought": self.attribute_thought,
            "stability_thought": self.stability_thought,
            "composition_thought": self.composition_thought,
            "branch_score": self.branch_score,
        }


def _as_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(t).strip() for t in x if str(t).strip()]
    value = str(x).strip()
    return [value] if value else []


def _norm_text(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"</s>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Thought\s*\d+\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Reasoning\s*(branch|for the inferred pattern)?\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _deduplicate_phrases(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        key = item.lower().strip(" .,:;")
        if key and key not in seen:
            seen.add(key)
            out.append(item.strip(" .,:;"))
    return out


def _strip_meta_language(text: str) -> str:
    """
    Remove reasoning / meta language that is harmful for image generation.
    """
    text = _norm_text(text)

    bad_patterns = [
        r"\bconsistent with the inferred reasoning\b",
        r"\binferred rule\b",
        r"\binferred pattern\b",
        r"\bpattern illustrated by the demonstrations\b",
        r"\bhypothesis\b",
        r"\bconstraint\b",
        r"\bdemonstrations?\b",
        r"\bexamples?\b",
        r"\breference examples?\b",
        r"\btarget query\b",
        r"\btransferable pattern\b",
        r"\bstable anchor\b",
        r"\bshared anchor\b",
        r"\bpreserve the stable anchor\b",
        r"\breflect the target query\b",
        r"\bearlier demonstration values should not be copied directly unless justified by the inferred pattern\b",
        r"\bavoid irrelevant or contradictory elements\b",
        r"\bkeep the final image visually coherent and unambiguous\b",
        r"\bthe final result must\b",
        r"\buse the demonstrations as references\b",
        r"\bwithout introducing unnecessary elements\b",
        r"\bconsistent elements across the demonstrations\b",
        r"\bdimensions supported by the demonstrations\b",
        r"\bquery entity\b",
        r"\binvariant candidates?\b",
        r"\bvarying candidates?\b",
        r"\bsubject identity\b",
        r"\btransfer hypothesis\b",
        r"\bconfidence\b",
    ]

    for pat in bad_patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)

    text = re.sub(r"\s+", " ", text).strip(" .,:;")
    return text


def _extract_target_subject(hypothesis: Hypothesis) -> str:
    """
    Prefer new structured field query_entity; fallback to legacy query/common.
    """
    query_entity = str(getattr(hypothesis, "query_entity", "") or "").strip()
    if query_entity:
        return query_entity

    query = _as_list(getattr(hypothesis, "query", []))
    if query:
        return query[0]

    common = _as_list(getattr(hypothesis, "common", []))
    if common:
        return common[0]

    return "object"


def _extract_invariants(hypothesis: Hypothesis) -> List[str]:
    inv = _as_list(getattr(hypothesis, "invariant_candidates", []))
    inv = [_strip_meta_language(x) for x in inv]
    inv = [x for x in inv if x]
    return _deduplicate_phrases(inv)


def _extract_varyings(hypothesis: Hypothesis) -> List[str]:
    var = _as_list(getattr(hypothesis, "varying_candidates", []))
    var = [_strip_meta_language(x) for x in var]
    var = [x for x in var if x]
    return _deduplicate_phrases(var)


def _clean_invariant_for_prompt(x: str) -> str:
    x = _strip_meta_language(x)
    x = re.sub(r"^(subject|context|style|action|attribute)\s*:\s*", "", x, flags=re.IGNORECASE)
    x = x.strip(" .,:;")
    return x


def _build_base_prompt(hypothesis: Hypothesis) -> str:
    target = _extract_target_subject(hypothesis)
    invariants = _extract_invariants(hypothesis)

    if invariants:
        inv = ", ".join(invariants[:3])
        return f"{target}, with {inv}"

    return target


def _looks_like_meta_or_reference_only(text: str) -> bool:
    lower = text.lower()

    bad_signals = [
        "example",
        "examples",
        "demonstration",
        "demonstrations",
        "inferred",
        "pattern",
        "query entity",
        "target query",
        "reasoning",
        "constraint",
        "rule",
    ]
    return any(sig in lower for sig in bad_signals)


def _mentions_unwanted_reference_copy(text: str) -> bool:
    lower = text.lower()
    patterns = [
        "same visual style",
        "same concept family",
        "reference examples",
        "same as the examples",
        "same as the references",
    ]
    return any(p in lower for p in patterns)


def _is_too_generic(text: str) -> bool:
    lower = text.lower().strip(" .")
    generic = {
        "clear",
        "coherent",
        "visually coherent",
        "well-defined",
        "recognizable",
        "stable",
        "natural",
    }
    return lower in generic


def _stage_to_visual_phrase(stage_name: str, text: str, target_subject: str) -> str:
    text = _strip_meta_language(text)
    if not text:
        return ""

    if _looks_like_meta_or_reference_only(text):
        return ""

    if _mentions_unwanted_reference_copy(text):
        return ""

    if _is_too_generic(text):
        return ""

    target_lower = target_subject.lower().strip()
    text_clean = text.strip()

    if target_lower:
        text_clean = re.sub(
            rf"^\bthe\s+{re.escape(target_lower)}\b\s+",
            "",
            text_clean,
            flags=re.IGNORECASE,
        )
        text_clean = re.sub(
            rf"^\b{re.escape(target_lower)}\b\s+",
            "",
            text_clean,
            flags=re.IGNORECASE,
        )

    text_clean = text_clean.strip(" .,:;")
    if not text_clean:
        return ""

    return text_clean


def _filter_stage_phrases(
    phrases: List[str],
    target_subject: str,
    hypothesis: Hypothesis,
) -> List[str]:
    out: List[str] = []
    invariants = " ".join(_extract_invariants(hypothesis)).lower()
    target_lower = target_subject.lower()

    for p in phrases:
        p = _norm_text(p)
        if not p:
            continue

        lower = p.lower()

        if len(lower) < 4:
            continue

        if "reference examples" in lower or "demonstrations" in lower:
            continue

        has_target = target_lower in lower if target_lower else False
        has_invariant = any(tok in lower for tok in invariants.split()) if invariants else False

        # keep phrase if it relates either to the target OR to inferred invariants
        if not has_target and not has_invariant:
            continue

        out.append(p)

    return _deduplicate_phrases(out)


def _clean_prompt(prompt: str) -> str:
    prompt = _norm_text(prompt)

    prompt = re.sub(
        r"\b(Scene|Attribute|Stability|Composition|Visual realization)\s*:\s*",
        "",
        prompt,
        flags=re.IGNORECASE,
    )
    prompt = re.sub(r"\s*,\s*,+", ", ", prompt)
    prompt = re.sub(r"\s*\.\s*", ". ", prompt)
    prompt = re.sub(r"\s*;\s*", "; ", prompt)
    prompt = re.sub(r"\s+", " ", prompt).strip(" ,;.")

    # comma-level dedup for diffusion-style prompts
    parts = [p.strip() for p in re.split(r"\s*,\s*", prompt) if p.strip()]
    parts = _deduplicate_phrases(parts)
    prompt = ", ".join(parts).strip()

    if prompt and not prompt.endswith("."):
        prompt += "."
    return prompt


def render_final_prompt(
    hypothesis: Hypothesis,
    winning_branch: ReasoningBranch,
) -> str:
    """
    Image-oriented final prompt aligned with the new analyzer + hypothesis.

    Uses:
    - query_entity / query as target subject
    - invariant_candidates as transferable consistent factors
    - stage thoughts only as auxiliary visual cues

    Avoids:
    - support example leakage
    - reference copying language
    - reasoning/meta text
    """
    target_subject = _extract_target_subject(hypothesis)
    base_subject_phrase = _build_base_prompt(hypothesis)

    scene_phrase = _stage_to_visual_phrase(
        "scene",
        getattr(winning_branch, "scene_thought", ""),
        target_subject,
    )
    attribute_phrase = _stage_to_visual_phrase(
        "attribute",
        getattr(winning_branch, "attribute_thought", ""),
        target_subject,
    )
    stability_phrase = _stage_to_visual_phrase(
        "stability",
        getattr(winning_branch, "stability_thought", ""),
        target_subject,
    )
    composition_phrase = _stage_to_visual_phrase(
        "composition",
        getattr(winning_branch, "composition_thought", ""),
        target_subject,
    )

    stage_phrases = _filter_stage_phrases(
        [attribute_phrase, scene_phrase, stability_phrase, composition_phrase],
        target_subject=target_subject,
        hypothesis=hypothesis,
    )

    phrases: List[str] = [
        base_subject_phrase,
        *stage_phrases,
        "single main subject",
        "no text",
        "no logo",
    ]

    phrases = [p for p in phrases if p]
    phrases = _deduplicate_phrases(phrases)

    prompt = ", ".join(phrases)
    return _clean_prompt(prompt)


def render_final_prompt_legacy(
    hypothesis: Dict[str, List[str]],
    final_thought: str,
) -> str:
    """
    Legacy wrapper for older dict-style inputs.
    """
    common = hypothesis.get("common", [])
    query = hypothesis.get("query", [])
    support_values = hypothesis.get("support_values", hypothesis.get("change", []))

    new_hypothesis = Hypothesis(
        common=common if isinstance(common, list) else [str(common)],
        support_values=support_values if isinstance(support_values, list) else [str(support_values)],
        query=query if isinstance(query, list) else [str(query)],
        relation_type=hypothesis.get("relation_type", "legacy-unknown"),
        rule=hypothesis.get("rule", "unknown"),
        constraints=hypothesis.get("constraints", []),
        summary=hypothesis.get("summary", ""),
        invariant_candidates=hypothesis.get("invariant_candidates", []),
        varying_candidates=hypothesis.get("varying_candidates", []),
        query_entity=hypothesis.get("query_entity", query[0] if isinstance(query, list) and query else ""),
        transfer_hypothesis=hypothesis.get("transfer_hypothesis", ""),
        confidence=float(hypothesis.get("confidence", 0.0)),
    )

    winning_branch = ReasoningBranch(
        scene_thought="",
        attribute_thought=final_thought or "",
        stability_thought="",
        composition_thought="",
        branch_score=1.0,
    )

    return render_final_prompt(new_hypothesis, winning_branch)