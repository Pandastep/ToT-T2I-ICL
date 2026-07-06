# ablation/pipeline_abl/orchestrator_abl.py

from dataclasses import dataclass
from typing import Callable, List, Optional
import random

from pipeline.io_contracts import (
    OrchestratorOutput,
    StageResult,
    ReasoningBranch,
)

from reasoning.stages import make_stage_prompt, parse_thoughts

# IMPORTANT:
# use only ablation copies here
from ablation.reasoning_abl.selector_abl import (
    select_for_next,
    get_branching_config,
)
from ablation.reasoning_abl.scoring_abl import (
    score_one_thought,
    ScoreConfig,
    build_score_config,
)


# ---------------------------
# INTERNAL STATE
# ---------------------------

@dataclass
class _State:
    thoughts: List[str]
    score: float


# ---------------------------
# HELPERS
# ---------------------------

def _dedup_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        key = str(x).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(str(x).strip())
    return out


def _safe_stage_width(stage: str, branching_factor: int) -> int:
    """
    Use the smaller of:
    - ablation-requested branching factor B
    - stage-native generation width from current stage config
    """
    cfg = get_branching_config(stage)
    stage_gen_width = int(cfg.get("generation_width", 2))
    return max(1, min(int(branching_factor), stage_gen_width))


# ---------------------------
# MAIN PIPELINE
# ---------------------------

def run_tot_pipeline(
    hypothesis,
    llm_generate_fn: Callable[[str, int], List[str]],
    tau: float = 0.08,
    pattern_analysis=None,   # kept only for compatibility
    branching_factor: int = 3,
    selection_mode: str = "score",
    penalty_mode: str = "full",
    seed: Optional[int] = None,
    score_config: Optional[ScoreConfig] = None,
    weight_mode: str = "original",
) -> OrchestratorOutput:
    """
    Ablation ToT pipeline.

    Supports:
    - branching_factor ablation
    - selection_mode ablation
    - penalty_mode ablation

    This file is isolated from the working main pipeline.
    """

    rng = random.Random(seed if seed is not None else 0)

    if score_config is None:
        score_config = build_score_config(
            penalty_mode=penalty_mode,
            weight_mode=weight_mode,
        )

    stages = ["scene", "attribute", "stability", "composition"]

    current_states: List[_State] = [_State(thoughts=[], score=0.0)]
    stage_results: List[StageResult] = []

    print("\n🔍 Hypothesis summary:")
    print(f"   {getattr(hypothesis, 'summary', '')}")
    print(f"   Rule: {getattr(hypothesis, 'rule', '')}")
    print(f"   Constraints: {getattr(hypothesis, 'constraints', [])}")
    print(
        f"   Ablation config: "
        f"B={branching_factor}, "
        f"selection_mode={selection_mode}, "
        f"penalty_mode={penalty_mode}"
        f"weight_mode={weight_mode}"
    )

    for stage in stages:
        print(f"\n=== Stage: {stage.upper()} ===")

        new_states: List[_State] = []

        branching_cfg = get_branching_config(stage)
        gen_width = _safe_stage_width(stage, branching_factor)

        # stage-level logs across all parent states
        stage_generated: List[str] = []
        stage_scores: List[float] = []
        stage_parts: List[dict] = []
        latest_selection_result = None

        for state_idx, state in enumerate(current_states):
            print(f"\n[Parent state {state_idx + 1}] prev_score={state.score:.3f}")
            print(f"  prev_thoughts={state.thoughts}")

            prompt = make_stage_prompt(
                hypothesis,
                stage,
                k=gen_width,
                previous_thoughts=state.thoughts,
            )

            raw_outputs = llm_generate_fn(prompt, gen_width)

            parsed_candidates: List[str] = []
            for out in raw_outputs:
                parsed_candidates.extend(parse_thoughts(out, k=gen_width))

            candidates = _dedup_keep_order(parsed_candidates)

            if not candidates:
                print("⚠️ No candidates generated")
                continue

            candidate_scores: List[float] = []
            candidate_parts: List[dict] = []

            for cand in candidates:
                parts = score_one_thought(
                    thought=cand,
                    hypothesis=hypothesis,
                    stage=stage,
                    config=score_config,
                    previous_thoughts=state.thoughts,
                )
                total = float(parts["total"])

                candidate_scores.append(total)
                candidate_parts.append(parts)

                stage_generated.append(cand)
                stage_scores.append(total)
                stage_parts.append(parts)

            sel = select_for_next(
                stage=stage,
                texts=candidates,
                scores=candidate_scores,
                parts_list=candidate_parts,
                tau=tau,
                selection_mode=selection_mode,
                rng=rng,
            )
            latest_selection_result = sel

            for i, cand in enumerate(candidates):
                print(
                    f"  cand{i + 1}: {cand} | "
                    f"score={candidate_scores[i]:.3f} | "
                    f"parts={candidate_parts[i]}"
                )

            print(f"  -> keeping {len(sel.kept)} branch(es)")

            for kept in sel.kept:
                next_thoughts = state.thoughts + [kept.text]
                next_score = state.score + float(kept.score)

                new_states.append(
                    _State(
                        thoughts=next_thoughts,
                        score=next_score,
                    )
                )

        if not new_states:
            print("⚠️ All branches pruned — stopping early")
            break

        # Global pruning after stage:
        # keep at most branching_factor states, but never less than 1
        new_states.sort(key=lambda s: s.score, reverse=True)
        keep_global = max(1, int(branching_factor))
        current_states = new_states[:keep_global]

        print(f"\nTop states after {stage}:")
        for i, st in enumerate(current_states):
            print(f"  {i + 1}. score={st.score:.3f}, thoughts={st.thoughts}")

        stage_results.append(
            StageResult(
                stage=stage,
                generated_thoughts=stage_generated,
                scores=stage_scores,
                parts=stage_parts,
                selection_result=latest_selection_result,
                kept_branches=[s.thoughts for s in current_states],
                branching_config={
                    **branching_cfg,
                    "ablation_branching_factor": int(branching_factor),
                    "effective_generation_width": int(gen_width),
                    "selection_mode": selection_mode,
                    "penalty_mode": penalty_mode,
                    "global_keep_after_stage": int(keep_global),
                    "weight_mode": weight_mode,
                },
            )
        )

    # ---------------------------
    # FINAL WINNING BRANCH
    # ---------------------------

    if current_states:
        best = max(current_states, key=lambda s: s.score)
        thoughts = best.thoughts + [""] * 4

        winning_branch = ReasoningBranch(
            scene_thought=thoughts[0],
            attribute_thought=thoughts[1],
            stability_thought=thoughts[2],
            composition_thought=thoughts[3],
            branch_score=best.score,
        )
    else:
        winning_branch = ReasoningBranch(
            scene_thought="",
            attribute_thought="",
            stability_thought="",
            composition_thought="",
            branch_score=0.0,
        )

    return OrchestratorOutput(
        stage_results=stage_results,
        winning_branch=winning_branch,
        final_prompt="",
        reasoning_summary=getattr(hypothesis, "summary", ""),
    )


# ---------------------------
# LEGACY
# ---------------------------

def run_tot_pipeline_legacy(*args, **kwargs):
    raise RuntimeError("Legacy ablation pipeline is deprecated. Use run_tot_pipeline().")