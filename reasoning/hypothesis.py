# reasoning/hypothesis.py

from typing import Dict, List, Any
from dataclasses import dataclass, field

from .analyzer import PatternAnalysis, ICLPatternAnalyzer


@dataclass
class Hypothesis:
    """
    Structured hypothesis for ToT reasoning.

    Design principles:
    - no task metadata fallback
    - use inferred structure from PatternAnalysis
    - preserve backward compatibility with the rest of the pipeline
    """

    # legacy-compatible core fields
    common: List[str]
    support_values: List[str]
    query: List[str]

    relation_type: str = "unknown"
    rule: str = "unknown"
    constraints: List[str] = field(default_factory=list)
    summary: str = "unknown"

    # new structured fields from analyzer
    invariant_candidates: List[str] = field(default_factory=list)
    varying_candidates: List[str] = field(default_factory=list)
    query_entity: str = ""
    transfer_hypothesis: str = "unknown"
    confidence: float = 0.0

    @classmethod
    def from_pattern(cls, pa: PatternAnalysis) -> "Hypothesis":
        # backward-compatible fields
        common = list(getattr(pa, "common_elements", []) or [])
        support_values = list(getattr(pa, "support_values", []) or [])
        query = list(getattr(pa, "query_values", []) or [])
        relation_type = str(getattr(pa, "relation_type", "unknown") or "unknown")

        # new analyzer fields
        invariant_candidates = list(getattr(pa, "invariant_candidates", []) or [])
        varying_candidates = list(getattr(pa, "varying_candidates", []) or [])
        query_entity = str(getattr(pa, "query_entity", "") or "")
        transfer_hypothesis = str(getattr(pa, "transfer_hypothesis", "") or "")
        confidence = float(getattr(pa, "confidence", 0.0) or 0.0)

        rule = cls._infer_rule(
            invariant_candidates=invariant_candidates,
            varying_candidates=varying_candidates,
            query_entity=query_entity,
            transfer_hypothesis=transfer_hypothesis,
            relation_type=relation_type,
            support_values=support_values,
            query=query,
        )

        constraints = cls._infer_constraints(
            invariant_candidates=invariant_candidates,
            varying_candidates=varying_candidates,
            query_entity=query_entity,
            relation_type=relation_type,
            support_values=support_values,
            confidence=confidence,
        )

        summary = cls._format_summary(
            invariant_candidates=invariant_candidates,
            varying_candidates=varying_candidates,
            query_entity=query_entity,
            support_values=support_values,
            relation_type=relation_type,
            transfer_hypothesis=transfer_hypothesis,
            confidence=confidence,
            rule=rule,
            constraints=constraints,
        )

        return cls(
            common=common,
            support_values=support_values,
            query=query,
            relation_type=relation_type,
            rule=rule,
            constraints=constraints,
            summary=summary,
            invariant_candidates=invariant_candidates,
            varying_candidates=varying_candidates,
            query_entity=query_entity,
            transfer_hypothesis=transfer_hypothesis,
            confidence=confidence,
        )

    @staticmethod
    def _infer_rule(
        invariant_candidates: List[str],
        varying_candidates: List[str],
        query_entity: str,
        transfer_hypothesis: str,
        relation_type: str,
        support_values: List[str],
        query: List[str],
    ) -> str:
        """
        Build the main transfer rule from inferred structure, not from metadata.
        """

        has_invariants = bool(invariant_candidates)
        has_varyings = bool(varying_candidates)
        has_query_entity = bool(query_entity)
        has_support = bool(support_values)
        has_query = bool(query)

        if transfer_hypothesis and transfer_hypothesis != "unknown":
            base = transfer_hypothesis.strip()
        else:
            base = ""

        if has_invariants and has_varyings and has_query_entity:
            invariant_str = ", ".join(invariant_candidates)

            if "subject_identity" in varying_candidates:
                return (
                    f"Keep {invariant_str} and replace the subject with {query_entity}."
                )

            return (
                f"Keep {invariant_str} and apply the observed variation to {query_entity}."
            )

        if has_invariants and has_query_entity:
            invariant_str = ", ".join(invariant_candidates)
            return (
                f"Preserve the inferred consistent elements across the demonstrations "
                f"({invariant_str}) while adapting the result to the query entity ({query_entity})."
            )

        if has_varyings and has_query_entity:
            varying_str = ", ".join(varying_candidates)
            return (
                f"Use the demonstrations to infer which dimensions vary ({varying_str}) "
                f"and transfer the appropriate variation to the query entity ({query_entity}) "
                f"without changing unrelated aspects."
            )

        if base:
            return base

        if relation_type == "structured-transfer" and has_support and has_query:
            return (
                "Infer the structured transfer pattern from the demonstrations and apply it "
                "to the target query while preserving well-supported consistent elements."
            )

        if relation_type in {"variation-transfer", "set-to-query-transfer"} and has_support and has_query:
            return (
                "Use the demonstrations to infer a coherent transformation pattern and apply it "
                "to the target query without introducing unsupported attributes, objects, or scene changes."
            )

        if has_query:
            return "Generate the target query in a way that remains consistent with the available demonstration evidence."

        if has_support:
            return "Infer a coherent pattern from the demonstrations."

        return "Pattern is under-specified."

    @staticmethod
    def _infer_constraints(
        invariant_candidates: List[str],
        varying_candidates: List[str],
        query_entity: str,
        relation_type: str,
        support_values: List[str],
        confidence: float,
    ) -> List[str]:
        """
        Build natural-language constraints from inferred structure.
        """
        constraints: List[str] = []

        if query_entity:
            constraints.append(f"The final result should clearly reflect the query entity ({query_entity}).")
        else:
            constraints.append("The final result should clearly reflect the target query.")

        if invariant_candidates:
            inv = ", ".join(invariant_candidates)
            constraints.append(
                f"Preserve the consistent elements inferred from the demonstrations ({inv}) unless there is strong evidence to change them."
            )

        if "subject_identity" in varying_candidates and query_entity:
            constraints.append(
                f"Replace the demonstrated subject identity with the query entity ({query_entity}) while preserving other supported elements."
            )

        if varying_candidates:
            var = ", ".join(varying_candidates)
            constraints.append(
                f"Only modify dimensions supported by the demonstrations as varying ({var}); avoid unrelated reinterpretations."
            )

        if support_values:
            constraints.append(
                "Use the demonstrations as references for the inferred pattern rather than copying them literally."
            )

        constraints.append(
            "Do not introduce extra objects, attributes, actions, or relations unless they are required by the inferred pattern."
        )

        constraints.append(
            "Prefer a single coherent interpretation over loosely related or decorative scene details."
        )

        if confidence < 0.5:
            constraints.append("keep scene simple")
            constraints.append("avoid complex compositions")

        return constraints

    @staticmethod
    def _format_summary(
        invariant_candidates: List[str],
        varying_candidates: List[str],
        query_entity: str,
        support_values: List[str],
        relation_type: str,
        transfer_hypothesis: str,
        confidence: float,
        rule: str,
        constraints: List[str],
    ) -> str:
        invariant_str = ", ".join(invariant_candidates) if invariant_candidates else "none"
        varying_str = ", ".join(varying_candidates) if varying_candidates else "none"
        support_str = ", ".join(support_values) if support_values else "none"
        query_str = query_entity if query_entity else "none"
        relation_str = relation_type if relation_type else "unknown"
        transfer_str = transfer_hypothesis if transfer_hypothesis else "unknown"
        constraints_str = " | ".join(constraints) if constraints else "none"

        return (
            f"Invariants: {invariant_str}; "
            f"Varyings: {varying_str}; "
            f"SupportValues: {support_str}; "
            f"QueryEntity: {query_str}; "
            f"Relation: {relation_str}; "
            f"TransferHypothesis: {transfer_str}; "
            f"Confidence: {confidence:.2f}; "
            f"Rule: {rule}; "
            f"Constraints: {constraints_str}"
        )


class HypothesisGenerator:
    """
    Hypothesis generator based only on analyzer output.
    No task metadata, no benchmark-specific fallback.
    """

    def __init__(self):
        self.pattern_analyzer = ICLPatternAnalyzer()

    def generate_hypothesis(self, data_item: Dict[str, Any]) -> Hypothesis:
        pattern_analysis = self.pattern_analyzer.analyze_icl_patterns(data_item)
        return Hypothesis.from_pattern(pattern_analysis)