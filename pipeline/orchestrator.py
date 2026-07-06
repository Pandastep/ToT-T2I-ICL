# pipeline/orchestrator.py

from dataclasses import dataclass
from typing import Callable, List, Optional

from pipeline.io_contracts import (
    OrchestratorOutput,
    StageResult,
    SelectionResult,
    ReasoningBranch,
)
from reasoning.stages import make_stage_prompt, parse_thoughts
from reasoning.selector import select_for_next, get_branching_config
from reasoning.scoring import score_one_thought, ScoreConfig


# ---------------------------
# INTERNAL STATE
# ---------------------------

@dataclass
class _State:
    thoughts: List[str]
    score: float


# ---------------------------
# MAIN PIPELINE
# ---------------------------

def run_tot_pipeline(
    hypothesis,
    llm_generate_fn: Callable[[str, int], List[str]],
    tau: float = 0.08,
    pattern_analysis=None,  # kept only for compatibility, not used
    score_config: Optional[ScoreConfig] = None,
) -> OrchestratorOutput:
    """
    Metadata-light ToT pipeline.

    Key properties:
    - does not use task_id / x_space / theta_space directly
    - relies on the hypothesis object as natural-language guidance
    - carries forward previous branch thoughts into both prompting and scoring
    """

    if score_config is None:
        score_config = ScoreConfig()

    stages = ["scene", "attribute", "stability", "composition"]

    current_states: List[_State] = [_State(thoughts=[], score=0.0)]
    stage_results: List[StageResult] = []

    print("\n🔍 Hypothesis summary:")
    print(f"   {getattr(hypothesis, 'summary', '')}")
    print(f"   Rule: {getattr(hypothesis, 'rule', '')}")
    print(f"   Constraints: {getattr(hypothesis, 'constraints', [])}")

    for stage in stages:
        print(f"\n=== Stage: {stage.upper()} ===")

        new_states: List[_State] = []
        branching_cfg = get_branching_config(stage)
        gen_width = branching_cfg["generation_width"]

        # stage-level logs across all parent states
        stage_generated: List[str] = []
        stage_scores: List[float] = []
        stage_parts: List[dict] = []
        stage_kept_branches: List[List[str]] = []
        latest_selection_result: SelectionResult | None = None

        for state in current_states:
            prompt = make_stage_prompt(
                hypothesis,
                stage,
                k=gen_width,
                previous_thoughts=state.thoughts,
            )

            raw_outputs = llm_generate_fn(prompt, gen_width)
            candidates = parse_thoughts("\n".join(raw_outputs), k=gen_width)

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
                    previous_thoughts=state.thoughts,   # ← ключевая правка
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
            )
            latest_selection_result = sel

            print(f"\n[Branch expansion] prev_score={state.score:.3f}")
            for i, cand in enumerate(candidates):
                print(
                    f"  cand{i+1}: {cand} | "
                    f"score={candidate_scores[i]:.3f} | "
                    f"parts={candidate_parts[i]}"
                )

            print(f"  -> keeping {len(sel.kept)} branch(es)")

            for kept in sel.kept:
                next_thoughts = state.thoughts + [kept.text]
                next_score = state.score + kept.score

                new_states.append(
                    _State(
                        thoughts=next_thoughts,
                        score=next_score,
                    )
                )
                stage_kept_branches.append(next_thoughts)

        if not new_states:
            print("⚠️ All branches pruned — stopping early")
            break

        # keep top branches globally after this stage
        new_states.sort(key=lambda s: s.score, reverse=True)
        current_states = new_states[:2]

        print(f"\nTop states after {stage}:")
        for i, st in enumerate(current_states):
            print(f"  {i+1}. score={st.score:.3f}, thoughts={st.thoughts}")

        stage_results.append(
            StageResult(
                stage=stage,
                generated_thoughts=stage_generated,
                scores=stage_scores,
                parts=stage_parts,
                selection_result=latest_selection_result,
                kept_branches=[s.thoughts for s in current_states],
                branching_config=branching_cfg,
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
    raise RuntimeError("Legacy pipeline is deprecated. Use run_tot_pipeline().")