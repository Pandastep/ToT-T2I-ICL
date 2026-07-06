# reasoning/selector.py

from typing import List, Dict, Tuple, Literal, Optional
from dataclasses import dataclass, asdict

StageName = Literal["scene", "attribute", "stability", "composition"]


# ---------------------------
# STAGE ORDER / CONFIG
# ---------------------------

STAGE_ORDER: Dict[StageName, int] = {
    "scene": 1,
    "attribute": 2,
    "stability": 3,
    "composition": 4,
}


# ---------------------------
# DATA STRUCTURES
# ---------------------------

@dataclass
class Candidate:
    """
    One candidate thought for a stage.
    """
    text: str
    score: float
    parts: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class SelectionResult:
    """
    Selection result for one stage.
    """
    stage: StageName
    ranked: List[Candidate]
    kept: List[Candidate]
    gap: float
    tau: float
    keep_count: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "stage": self.stage,
            "ranked": [c.to_dict() for c in self.ranked],
            "kept": [c.to_dict() for c in self.kept],
            "gap": self.gap,
            "tau": self.tau,
            "keep_count": self.keep_count,
        }


# ---------------------------
# INTERNAL HELPERS
# ---------------------------

def _safe_strip(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _get_penalty_value(candidate: Candidate) -> float:
    """
    Lower total penalty is better.
    """
    if not candidate.parts:
        return 0.0

    unsupported = float(candidate.parts.get("unsupported_leap_penalty", 0.0))
    redundancy = float(candidate.parts.get("redundancy_penalty", 0.0))
    return unsupported + redundancy


def _effective_length(text: str) -> int:
    """
    Light length helper for tie-breaking.
    Prefer neither ultra-short nor very long thoughts.
    """
    return len(text.strip().split())


def _rank_candidates(candidates: List[Candidate]) -> List[Candidate]:
    """
    Stable descending sort by score.

    Tie-break policy:
    1. higher score
    2. lower explicit penalties
    3. prefer moderate length over ultra-short / too long
    """
    def sort_key(c: Candidate):
        penalty = _get_penalty_value(c)
        length = _effective_length(c.text)

        # closeness to a moderate target length
        target_len = 10
        len_distance = abs(length - target_len)

        return (
            c.score,          # primary: higher is better
            -penalty,         # lower penalty is better
            -len_distance,    # closer to moderate length is better
        )

    return sorted(candidates, key=sort_key, reverse=True)


def _decide_keep_count(ranked: List[Candidate], tau: float) -> Tuple[int, float]:
    """
    Decide whether to keep 1 or 2 branches.

    Rule:
    - if < 2 candidates -> keep 1
    - else compute gap = s1 - s2
      * if gap <= tau -> keep 2
      * else keep 1
    """
    if len(ranked) < 2:
        return 1, 0.0

    gap = max(0.0, ranked[0].score - ranked[1].score)
    keep_count = 2 if gap <= tau else 1
    return keep_count, gap


# ---------------------------
# MAIN SELECTION FUNCTION
# ---------------------------

def select_for_next(
    stage: StageName,
    texts: List[str],
    scores: List[float],
    parts_list: Optional[List[Optional[Dict[str, float]]]] = None,
    tau: float = 0.08,
) -> SelectionResult:
    """
    Select which candidates move to the next stage.

    Inputs:
      stage      : stage name
      texts      : candidate thoughts
      scores     : aggregated candidate scores
      parts_list : optional score decomposition for each thought
      tau        : branching threshold
    """
    if len(texts) != len(scores):
        raise ValueError("texts and scores must have одинаковую длину")

    if parts_list is not None and len(parts_list) != len(texts):
        raise ValueError("parts_list must have the same length as texts/scores")

    candidates: List[Candidate] = []
    for i, (t, s) in enumerate(zip(texts, scores)):
        parts = parts_list[i] if parts_list is not None else None
        candidates.append(
            Candidate(
                text=_safe_strip(t),
                score=float(s),
                parts=parts,
            )
        )

    ranked = _rank_candidates(candidates)
    keep_count, gap = _decide_keep_count(ranked, tau)
    kept = ranked[:keep_count]

    return SelectionResult(
        stage=stage,
        ranked=ranked,
        kept=kept,
        gap=gap,
        tau=tau,
        keep_count=keep_count,
    )


# ---------------------------
# BRANCHING HELPERS
# ---------------------------

def generation_width_for_stage(stage: StageName) -> int:
    """
    Recommended generation width for each stage.
    """
    order = STAGE_ORDER[stage]
    return 3 if order in (1, 2) else 2


def get_branching_config(stage: StageName) -> Dict[str, int]:
    """
    Branching configuration for a stage.
    """
    order = STAGE_ORDER[stage]
    return {
        "generation_width": 3 if order in (1, 2) else 2,
        "max_keep_branches": 2,
        "stage_order": order,
    }


def should_prune_branch(selection_result: SelectionResult, branch_index: int) -> bool:
    """
    Optional helper:
    prune only if the branch is last and significantly worse than the previous one.
    """
    if branch_index >= len(selection_result.ranked) - 1:
        if len(selection_result.ranked) > 1:
            last_gap = selection_result.ranked[-2].score - selection_result.ranked[-1].score
            return last_gap > selection_result.tau
    return False