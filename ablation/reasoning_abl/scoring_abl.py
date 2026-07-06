# ablation/reasoning_abl/scoring_abl.py

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import re

# IMPORTANT:
# use the main project hypothesis, not a local copy, unless you really created one
from reasoning.hypothesis import Hypothesis


# ---------------------------
# UTILS
# ---------------------------

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "being", "been",
    "to", "of", "in", "on", "at", "for", "with", "by", "from", "as",
    "and", "or", "that", "this", "these", "those", "it", "its",
    "into", "through", "over", "under", "after", "before", "than",
    "then", "now", "but", "not", "no", "do", "does", "did",
    "have", "has", "had", "can", "could", "should", "would", "may",
    "might", "must", "will", "just", "very", "more", "most",
    "object", "target", "query", "image", "visual", "result",
    "thought", "interpretation", "representation", "subject"
}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _words(s: str) -> List[str]:
    return re.findall(r"[a-z0-9\-]+", _norm(s))


def _content_words(s: str) -> List[str]:
    return [w for w in _words(s) if w not in _STOPWORDS and len(w) > 2]


def _unique_ratio(text: str) -> float:
    ws = _words(text)
    if not ws:
        return 0.0
    return len(set(ws)) / len(ws)


def _length_score(text: str, min_len: int = 5, max_len: int = 24) -> float:
    n = len(_words(text))
    if n <= 0:
        return 0.0
    if n < min_len:
        return n / float(min_len)
    if n > max_len:
        overflow = min(n - max_len, max_len)
        return max(0.0, 1.0 - overflow / float(max_len))
    return 1.0


def _token_overlap(a: str, b: str) -> float:
    wa = set(_content_words(a))
    wb = set(_content_words(b))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / float(len(wb))


def _jaccard(a: str, b: str) -> float:
    wa = set(_content_words(a))
    wb = set(_content_words(b))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / float(len(wa | wb))


def _contains_any(text: str, phrases: List[str]) -> bool:
    t = _norm(text)
    return any(_norm(p) in t for p in phrases if str(p).strip())


def _count_any(text: str, phrases: List[str]) -> int:
    t = _norm(text)
    return sum(1 for p in phrases if _norm(p) in t and str(p).strip())


def _flatten_query(hypothesis: Hypothesis) -> str:
    q = getattr(hypothesis, "query", "")
    if isinstance(q, list):
        return " ".join(str(x) for x in q if str(x).strip())
    return str(q)


def _flatten_constraints(hypothesis: Hypothesis) -> List[str]:
    c = getattr(hypothesis, "constraints", []) or []
    if isinstance(c, list):
        return [str(x) for x in c if str(x).strip()]
    return [str(c)] if str(c).strip() else []


# ---------------------------
# CONFIG
# ---------------------------

@dataclass
class ScoreConfig:
    # main weights
    w_query_anchor: float = 0.24
    w_entity_preservation: float = 0.22
    w_stage_fidelity: float = 0.18
    w_rule: float = 0.14
    w_constraints: float = 0.12
    w_quality: float = 0.10

    # penalty magnitudes
    unsupported_leap_penalty: float = 0.18
    redundancy_penalty: float = 0.15

    # ablation toggles
    enable_unsupported_penalty: bool = True
    enable_redundancy_penalty: bool = True

    vague_markers: Tuple[str, ...] = (
        "maybe", "perhaps", "somehow", "kind of", "sort of",
        "something", "nice", "good", "beautiful", "interesting"
    )

    contradiction_markers: Tuple[str, ...] = (
        "completely different",
        "random",
        "unrelated",
        "ignore the rule",
        "ignore the constraints",
        "change everything",
    )

    replacement_markers: Tuple[str, ...] = (
        "becomes",
        "become",
        "turned into",
        "transformed into",
        "is now a",
        "is now an",
        "turns into",
        "converted into",
        "changes into",
    )

    extra_entity_markers: Tuple[str, ...] = (
        "person", "people", "man", "woman", "child", "crowd",
        "house", "building", "car", "vehicle", "ball", "sports ball",
        "tree", "forest", "street", "city", "room", "sky"
    )

    scene_positive: Tuple[str, ...] = (
        "depicts", "appears", "rendered", "shown", "presented",
        "subject", "main subject", "single subject", "object-centered",
        "iconic", "minimal", "stylized", "visual form"
    )
    scene_negative: Tuple[str, ...] = (
        "foreground", "background", "left side", "right side", "top",
        "bottom", "corner", "center of the image", "layout", "framing"
    )

    attribute_positive: Tuple[str, ...] = (
        "style", "texture", "material", "surface", "appearance",
        "shape", "pattern", "color", "detail", "look", "visual property"
    )
    attribute_negative: Tuple[str, ...] = (
        "street", "forest", "room", "crowd", "person", "layout",
        "framing", "placement"
    )

    stability_positive: Tuple[str, ...] = (
        "clear", "coherent", "consistent", "stable", "readable",
        "well-defined", "unambiguous", "recognizable", "balanced", "natural"
    )
    stability_negative: Tuple[str, ...] = (
        "random", "chaotic", "unclear", "confusing", "contradictory", "blurry"
    )

    composition_positive: Tuple[str, ...] = (
        "centered", "placement", "layout", "focus", "emphasis",
        "framing", "foreground", "background", "prominent", "salient",
        "middle of the image", "inside", "around", "surrounded"
    )
    composition_negative: Tuple[str, ...] = (
        "texture", "material", "style only", "completely different object"
    )


def build_score_config_from_penalty_mode(penalty_mode: str) -> ScoreConfig:
    """
    penalty_mode:
      - full
      - no_unsupported
      - no_redundancy
      - none
    """
    cfg = ScoreConfig()

    if penalty_mode == "full":
        return cfg
    elif penalty_mode == "no_unsupported":
        cfg.enable_unsupported_penalty = False
        return cfg
    elif penalty_mode == "no_redundancy":
        cfg.enable_redundancy_penalty = False
        return cfg
    elif penalty_mode == "none":
        cfg.enable_unsupported_penalty = False
        cfg.enable_redundancy_penalty = False
        return cfg
    else:
        raise ValueError(f"Unknown penalty_mode: {penalty_mode}")

def _renormalize_weights(cfg: ScoreConfig) -> ScoreConfig:
    total = (
        cfg.w_query_anchor
        + cfg.w_entity_preservation
        + cfg.w_stage_fidelity
        + cfg.w_rule
        + cfg.w_constraints
        + cfg.w_quality
    )

    if total <= 0:
        raise ValueError("Sum of scoring weights must be positive.")

    cfg.w_query_anchor /= total
    cfg.w_entity_preservation /= total
    cfg.w_stage_fidelity /= total
    cfg.w_rule /= total
    cfg.w_constraints /= total
    cfg.w_quality /= total

    return cfg


def build_score_config(
    penalty_mode: str = "full",
    weight_mode: str = "original",
) -> ScoreConfig:
    """
    Build scoring configuration for Eq. 3 sensitivity analysis.

    weight_mode:
      - original: original manually selected Eq. 3 weights
      - equal: equal weights for all six criteria
      - no_anchor: remove query anchoring and renormalize remaining weights
      - no_constraints: remove constraint consistency and renormalize remaining weights
    """
    cfg = build_score_config_from_penalty_mode(penalty_mode)

    if weight_mode == "original":
        return cfg

    elif weight_mode == "equal":
        w = 1.0 / 6.0
        cfg.w_query_anchor = w
        cfg.w_entity_preservation = w
        cfg.w_stage_fidelity = w
        cfg.w_rule = w
        cfg.w_constraints = w
        cfg.w_quality = w
        return cfg

    elif weight_mode == "no_anchor":
        cfg.w_query_anchor = 0.0
        return _renormalize_weights(cfg)

    elif weight_mode == "no_constraints":
        cfg.w_constraints = 0.0
        return _renormalize_weights(cfg)

    else:
        raise ValueError(f"Unknown weight_mode: {weight_mode}")
    
    
# ---------------------------
# COMPONENTS
# ---------------------------

def score_query_anchor(thought: str, hypothesis: Hypothesis) -> float:
    text = _norm(thought)
    query_text = _flatten_query(hypothesis)
    q_words = set(_content_words(query_text))

    if not text:
        return 0.0

    if not q_words:
        if "target object" in text or "target" in text or "subject" in text:
            return 0.8
        return 0.5

    overlap = len(set(_content_words(text)) & q_words) / float(len(q_words))
    explicit_ref = 0.0
    if "target object" in text or "target" in text or "subject" in text:
        explicit_ref = 0.2

    score = min(1.0, overlap + explicit_ref)

    replacement_hits = _count_any(text, list(ScoreConfig().replacement_markers))
    if replacement_hits > 0 and overlap < 0.2:
        score *= 0.45

    return max(0.0, min(1.0, score))


def score_entity_preservation(thought: str, hypothesis: Hypothesis, config: ScoreConfig) -> float:
    text = _norm(thought)
    if not text:
        return 0.0

    query_text = _flatten_query(hypothesis)
    q_words = set(_content_words(query_text))
    t_words = set(_content_words(text))

    overlap = len(t_words & q_words) / float(len(q_words)) if q_words else 0.0
    replacement_hits = _count_any(text, list(config.replacement_markers))
    extra_entity_hits = _count_any(text, list(config.extra_entity_markers))

    score = 0.75

    if overlap >= 0.5 or "target object" in text or "subject" in text:
        score += 0.2

    if replacement_hits > 0 and overlap < 0.35:
        score -= 0.35

    if extra_entity_hits > 0 and overlap < 0.35:
        score -= min(0.25, 0.08 * extra_entity_hits)

    if overlap == 0.0 and replacement_hits > 0:
        score -= 0.2

    return max(0.0, min(1.0, score))


def score_rule_consistency(thought: str, hypothesis: Hypothesis) -> float:
    text = _norm(thought)
    if not text:
        return 0.0

    rule_text = _norm(getattr(hypothesis, "rule", ""))
    summary_text = _norm(getattr(hypothesis, "summary", ""))

    if not rule_text and not summary_text:
        return 0.5

    overlap_rule = _token_overlap(text, rule_text) if rule_text else 0.0
    overlap_summary = _token_overlap(text, summary_text) if summary_text else 0.0
    base = max(overlap_rule, overlap_summary)

    if _contains_any(text, list(ScoreConfig().contradiction_markers)):
        base -= 0.25

    return max(0.0, min(1.0, base))


def score_constraints_consistency(thought: str, hypothesis: Hypothesis) -> float:
    text = _norm(thought)
    constraints = _flatten_constraints(hypothesis)

    if not text:
        return 0.0
    if not constraints:
        return 0.7

    scores = []
    for c in constraints:
        c_norm = _norm(c)

        if any(neg in c_norm for neg in ["never", "must not", "do not", "should not"]):
            bad_refs = [
                "support example", "support examples",
                "previous example", "earlier example",
                "copy the example", "copied from"
            ]
            if _contains_any(text, bad_refs):
                scores.append(0.1)
            else:
                scores.append(0.8)
        else:
            overlap = _token_overlap(text, c_norm)

            if overlap >= 0.4:
                scores.append(0.85)
            elif overlap >= 0.2:
                scores.append(0.65)
            elif overlap > 0.0:
                scores.append(0.5)
            else:
                scores.append(0.35)

    return max(0.0, min(1.0, sum(scores) / len(scores)))


def score_stage_fidelity(thought: str, stage: str, config: ScoreConfig) -> float:
    text = _norm(thought)
    if not text:
        return 0.0

    score = 0.5

    if stage == "scene":
        pos = _count_any(text, list(config.scene_positive))
        neg = _count_any(text, list(config.scene_negative))
        if pos > 0:
            score += min(0.3, 0.1 * pos)
        if neg > 0:
            score -= min(0.25, 0.08 * neg)

    elif stage == "attribute":
        pos = _count_any(text, list(config.attribute_positive))
        neg = _count_any(text, list(config.attribute_negative))
        if pos > 0:
            score += min(0.3, 0.1 * pos)
        if neg > 0:
            score -= min(0.25, 0.08 * neg)

    elif stage == "stability":
        pos = _count_any(text, list(config.stability_positive))
        neg = _count_any(text, list(config.stability_negative))
        if pos > 0:
            score += min(0.35, 0.12 * pos)
        if neg > 0:
            score -= min(0.35, 0.12 * neg)

    elif stage == "composition":
        pos = _count_any(text, list(config.composition_positive))
        neg = _count_any(text, list(config.composition_negative))
        if pos > 0:
            score += min(0.35, 0.12 * pos)
        if neg > 0:
            score -= min(0.25, 0.08 * neg)

    return max(0.0, min(1.0, score))


def score_quality(thought: str, config: ScoreConfig, stage: str = "") -> float:
    text = _norm(thought)
    if not text:
        return 0.0

    uniq = _unique_ratio(text)
    len_reg = _length_score(text, min_len=5, max_len=22)

    vague_hits = _count_any(text, list(config.vague_markers))
    contradiction_hits = _count_any(text, list(config.contradiction_markers))

    vag_penalty = min(0.3, vague_hits * 0.1)
    ctr_penalty = min(0.4, contradiction_hits * 0.2)

    if stage == "stability":
        len_reg = max(len_reg, 0.75)

    score = 0.40 * uniq + 0.40 * len_reg + 0.20 * (1.0 - vag_penalty)
    score -= ctr_penalty

    return max(0.0, min(1.0, score))


def score_unsupported_leap_penalty(thought: str, hypothesis: Hypothesis, config: ScoreConfig) -> float:
    if not config.enable_unsupported_penalty:
        return 0.0

    text = _norm(thought)
    if not text:
        return 0.0

    query_text = _flatten_query(hypothesis)
    overlap = _token_overlap(text, query_text) if query_text else 0.0
    replacement_hits = _count_any(text, list(config.replacement_markers))

    if replacement_hits == 0:
        return 0.0

    if overlap == 0.0:
        return config.unsupported_leap_penalty
    if overlap < 0.2:
        return config.unsupported_leap_penalty * 0.7
    if overlap < 0.4:
        return config.unsupported_leap_penalty * 0.4
    return 0.0


def score_redundancy_penalty(thought: str, previous_thoughts: Optional[List[str]], config: ScoreConfig) -> float:
    if not config.enable_redundancy_penalty:
        return 0.0

    if not previous_thoughts:
        return 0.0

    sims = [_jaccard(thought, prev) for prev in previous_thoughts if _norm(prev)]
    if not sims:
        return 0.0

    max_sim = max(sims)

    if max_sim >= 0.85:
        return config.redundancy_penalty
    if max_sim >= 0.70:
        return config.redundancy_penalty * 0.65
    if max_sim >= 0.55:
        return config.redundancy_penalty * 0.35
    return 0.0


# ---------------------------
# TOTAL
# ---------------------------

def score_one_thought(
    thought: str,
    hypothesis: Hypothesis,
    stage: str = "",
    config: Optional[ScoreConfig] = None,
    previous_thoughts: Optional[List[str]] = None,
) -> Dict[str, float]:
    if config is None:
        config = ScoreConfig()

    A = score_query_anchor(thought, hypothesis)
    B = score_entity_preservation(thought, hypothesis, config)
    C = score_stage_fidelity(thought, stage, config)
    D = score_rule_consistency(thought, hypothesis)
    E = score_constraints_consistency(thought, hypothesis)
    F = score_quality(thought, config, stage)

    leap_pen = score_unsupported_leap_penalty(thought, hypothesis, config)
    red_pen = score_redundancy_penalty(thought, previous_thoughts, config)

    total = (
        config.w_query_anchor * A
        + config.w_entity_preservation * B
        + config.w_stage_fidelity * C
        + config.w_rule * D
        + config.w_constraints * E
        + config.w_quality * F
        - leap_pen
        - red_pen
    )
    total = max(0.0, min(1.0, total))

    return {
        "query_anchor": A,
        "entity_preservation": B,
        "stage_fidelity": C,
        "rule_consistency": D,
        "constraint_consistency": E,
        "quality": F,
        "unsupported_leap_penalty": leap_pen,
        "redundancy_penalty": red_pen,
        "total": total,
    }


def rank_thoughts(
    thoughts: List[str],
    hypothesis: Hypothesis,
    stage: str = "",
    config: Optional[ScoreConfig] = None,
    previous_thoughts: Optional[List[str]] = None,
) -> List[Tuple[int, Dict[str, float]]]:
    results = []

    for i, t in enumerate(thoughts):
        scores = score_one_thought(
            thought=t,
            hypothesis=hypothesis,
            stage=stage,
            config=config,
            previous_thoughts=previous_thoughts,
        )
        results.append((i, scores))

    results.sort(key=lambda z: z[1]["total"], reverse=True)
    return results


# ---------------------------
# INTERPRETATION
# ---------------------------

def interpret_score(total_score: float) -> str:
    if total_score >= 0.80:
        return "excellent"
    elif total_score >= 0.60:
        return "ok"
    elif total_score >= 0.40:
        return "weak"
    else:
        return "bad"


def get_score_thresholds() -> Dict[str, float]:
    return {
        "excellent_min": 0.80,
        "ok_min": 0.60,
        "weak_min": 0.40,
        "tau": 0.08,
    }