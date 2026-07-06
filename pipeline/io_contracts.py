# pipeline/io_contracts.py
# Contracts/types for reasoning pipeline.
# No task-specific metadata and no benchmark-specific fields.

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional


# ---------- INPUT FOR ONE SAMPLE ----------

@dataclass
class SampleIn:
    """
    Metadata-free input for one sample.

    The pipeline should infer structure from:
    - text_inputs
    - image_inputs
    instead of receiving benchmark-specific fields such as
    theta / x_list / target_x.
    """
    text_inputs: List[str]
    image_inputs: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if not isinstance(self.text_inputs, list) or not self.text_inputs:
            raise ValueError("SampleIn: 'text_inputs' must be a non-empty list.")

        if not isinstance(self.image_inputs, list):
            raise ValueError("SampleIn: 'image_inputs' must be a list.")

        if self.image_inputs and len(self.image_inputs) not in (
            len(self.text_inputs) - 1,
            len(self.text_inputs),
        ):
            print(
                f"⚠️ Warning: image_inputs length {len(self.image_inputs)} "
                f"does not match the expected ICL structure."
            )


# ---------- REASONING ANALYSIS LOG ----------

@dataclass
class AnalysisLog:
    """
    Lightweight analysis log.
    These are descriptive fields, not benchmark metadata.
    """
    common: str
    varying: str
    query: str
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------- LOGGED HYPOTHESIS ----------

@dataclass
class Hypothesis:
    """
    Logged hypothesis aligned with reasoning/hypothesis.py.
    """
    common: List[str]
    support_values: List[str]
    query: List[str]
    relation_type: str = "unknown"
    rule: str = "unknown"
    constraints: List[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "common": self.common,
            "support_values": self.support_values,
            "query": self.query,
            "relation_type": self.relation_type,
            "rule": self.rule,
            "constraints": self.constraints,
            "summary": self.summary,
        }


# ---------- CANDIDATE / SELECTION ----------

@dataclass
class Candidate:
    """
    One candidate thought on a stage.
    """
    text: str
    score: float
    parts: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SelectionResult:
    """
    Candidate selection result for one stage.
    """
    stage: str
    ranked: List[Candidate]
    kept: List[Candidate]
    gap: float
    tau: float
    keep_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "ranked": [c.to_dict() for c in self.ranked],
            "kept": [c.to_dict() for c in self.kept],
            "gap": self.gap,
            "tau": self.tau,
            "keep_count": self.keep_count,
        }


# ---------- STAGE RESULT ----------

@dataclass
class StageResult:
    """
    Log for one reasoning stage.
    """
    stage: str
    generated_thoughts: List[str]
    scores: List[float]
    parts: List[Dict[str, float]]
    selection_result: Optional[SelectionResult] = None
    kept_branches: List[List[str]] = field(default_factory=list)
    branching_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "generated_thoughts": self.generated_thoughts,
            "scores": self.scores,
            "parts": self.parts,
            "selection_result": self.selection_result.to_dict() if self.selection_result else None,
            "kept_branches": self.kept_branches,
            "branching_config": self.branching_config,
        }


# ---------- WINNING BRANCH ----------

@dataclass
class ReasoningBranch:
    """
    Full winning branch through 4 stages.
    """
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


# ---------- FULL REASONING LOG ----------

@dataclass
class ReasoningLog:
    """
    Full reasoning process log.
    """
    analysis: AnalysisLog
    hypothesis: Hypothesis
    stages: List[StageResult] = field(default_factory=list)
    winning_branch: Optional[ReasoningBranch] = None
    final_prompt: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis": self.analysis.to_dict(),
            "hypothesis": self.hypothesis.to_dict(),
            "stages": [s.to_dict() for s in self.stages],
            "winning_branch": self.winning_branch.to_dict() if self.winning_branch else None,
            "final_prompt": self.final_prompt,
        }


# ---------- GENERATION OUTPUT ----------

@dataclass
class GenOut:
    description: str
    image_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------- ORCHESTRATOR OUTPUT ----------

@dataclass
class OrchestratorOutput:
    """
    Output of run_tot_pipeline.
    """
    stage_results: List[StageResult]
    winning_branch: ReasoningBranch
    final_prompt: str
    reasoning_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_results": [s.to_dict() for s in self.stage_results],
            "winning_branch": self.winning_branch.to_dict(),
            "final_prompt": self.final_prompt,
            "reasoning_summary": self.reasoning_summary,
        }


# ---------- EXPORTS ----------

__all__ = [
    "SampleIn",
    "AnalysisLog",
    "Hypothesis",
    "Candidate",
    "SelectionResult",
    "StageResult",
    "ReasoningBranch",
    "ReasoningLog",
    "GenOut",
    "OrchestratorOutput",
]