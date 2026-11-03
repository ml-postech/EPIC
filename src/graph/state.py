"""State for the EPIC parse–generate–verify–refine pipeline."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Annotated
from operator import add


@dataclass
class InputState:
    """Run inputs for the fixed EPIC main method."""

    saving_files: bool = field(default=False)
    saving_path: str = field(default="outputs/examples")
    image_seed: int = field(default=42)
    vlm_max_resolution: int = field(default=1024)
    timing_log: dict = field(default_factory=dict)
    gen_num_inference_steps: int = 4
    gen_guidance_scale: float = 1.0
    gen_height: int = field(default=1024)
    gen_width: int = field(default=1024)
    vp_rewritten_prompt_count: int = field(default=8)
    vp_persistent_stuck_in_loop_auditor_streak: int = field(default=2)


@dataclass
class State(InputState):
    """Images, verifier evidence, and decisions accumulated by EPIC."""

    output_images: Annotated[list[str], add] = field(default_factory=list)
    round_best_images: Annotated[list[str], add] = field(
        default_factory=list
    )  # list of best image upto each round
    executor_outputs: Annotated[list[list[dict]], add] = field(default_factory=list)
    vp_verification_reports: Annotated[list[dict], add] = field(default_factory=list)
    vp_action_selections: Annotated[list[dict], add] = field(default_factory=list)
    vp_iteration_logs: Annotated[list[dict], add] = field(default_factory=list)
    vp_stage_timing_breakdown: dict = field(default_factory=dict)
    vp_pipeline_summary: dict = field(default_factory=dict)
    vp_verifier_thresholds: dict = field(default_factory=dict)
    vp_relation_scoring_config: dict = field(default_factory=dict)
    original_prompt: str = field(default=None)
    parsed_visual_program: dict = field(default=None)
    vp_visual_program_source: str = field(default="generated")
    vp_parser_normalization_report: dict = field(default_factory=dict)
    vp_parser_review_report: dict = field(default_factory=dict)
    vp_rewritten_prompt_generator_reports: Annotated[list[dict], add] = field(
        default_factory=list
    )
    vp_rewritten_prompt_generator_call_count: int = field(default=0)
    vp_hard_predicate_ids: Annotated[list[str], add] = field(default_factory=list)
    vp_soft_predicate_ids: Annotated[list[str], add] = field(default_factory=list)
    vp_critical_predicate_ids: Annotated[list[str], add] = field(default_factory=list)
    in_loop_auditor_output: dict = field(default=None)
    in_loop_auditor_call_count: int = field(default=0)
    final_candidate_accepted: bool = field(default=False)
    refinement_loop_vlm_calls: int = field(default=0)
    refinement_loop_called_any_vlm: bool = field(default=False)
    all_hard_predicates_satisfied: bool = field(default=False)
    all_soft_predicates_satisfied: bool = field(default=False)
    all_critical_predicates_satisfied: bool = field(default=False)
    current_round: int = field(default=1)  # current round index, starting from 1
    max_rounds: int = field(
        default=32
    )  # maximum number of rounds allowed, including the first round
    best_round: int = field(
        default=1
    )  # Initialized to 1, will be updated to the best round index (1-based)
    best_round_sample: int = field(
        default=1
    )  # Initialized to 1, will be updated to the best sample index within the best round (1-based)
