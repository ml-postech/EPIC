"""Main VP-based refinement pipeline node."""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import TYPE_CHECKING, Any, Literal, Sequence

from graph.timing import (
    build_vp_stage_timing_breakdown,
    ensure_timing_log,
    finalize_timing_log,
    record_duration,
    snapshot_timing_log,
)
from graph.visual_program import (
    AttributePredicate,
    GlobalScenePredicate,
    Predicate,
    PredicateState,
    SpatialRelationPredicate,
    VisualProgram,
)
from graph.rewritten_prompt_generator import (
    failed_rewritten_prompt_generation_result,
    generate_rewritten_prompts,
)
from graph.visual_program_action_selector import (
    ActionExecutionRecord,
    ActionExecutionStatus,
    CandidateMetadata,
    RefinementActionType,
    VisualProgramActionSelector,
)
from graph.visual_program_auditor import (
    run_in_loop_auditor,
    skipped_in_loop_audit_result,
)
from graph.visual_program_parser import parse_visual_program_from_prompt
from graph.vp_prompt_builder import (
    compose_prompt_with_scene_guidance,
    resolve_action_execution_input,
    select_initial_generation_prompt,
    select_resample_rewritten_prompt,
)
from graph.vp_partial_selection import (
    build_partial_match_rank_key,
    summarize_partial_match_priority,
)
from graph.visual_program_runtime_tools import get_visual_program_perception_toolbox
from graph.visual_program_support import (
    normalize_attribute_name,
    normalize_spatial_relation_name,
)
from graph.visual_program_verifier import (
    RelationScoringConfig,
    VerifierThresholds,
    VisualProgramVerificationReport,
    VisualProgramVerifier,
)

if TYPE_CHECKING:
    from graph.context import Context
    from graph.state import State
    from langchain_core.runnables import RunnableConfig
    from langgraph.runtime import Runtime
    from langgraph.types import Command
else:
    from langgraph.types import Command

    RunnableConfig = Any
    Runtime = Any
    State = Any


def _artifact_dir(state: State) -> str:
    return os.path.join(state.saving_path, state.original_prompt.replace(" ", "-")[:64])


def _round_debug_dir(state: State, round_index: int) -> str:
    return os.path.join(_artifact_dir(state), f"vp_round_{round_index:02d}_debug")


def resolve_generation_seed_offset(
    *,
    sample_index: int,
    round_index: int,
    force_new_seed: bool = False,
) -> int:
    """Resolve the deterministic seed offset for one generation attempt.

    Initial generation uses the sample-based offset. Resample actions can
    opt into a fresh seed by folding in the round index, which is monotonic over
    the VP loop and therefore guarantees a different seed from the initial
    sample-1 generation.
    """

    base_offset = max(0, int(sample_index) - 1)
    if not force_new_seed:
        return base_offset
    return max(base_offset, max(0, int(round_index) - 1))


def serialize_verifier_thresholds(thresholds: VerifierThresholds) -> dict[str, float]:
    """Convert verifier thresholds into a plain dict for artifacts and logs."""
    return {
        "proposal_satisfied": float(thresholds.proposal_satisfied),
        "proposal_uncertain": float(thresholds.proposal_uncertain),
        "scene_background_min_area_ratio": float(
            thresholds.scene_background_min_area_ratio
        ),
        "attribute_satisfied": float(thresholds.attribute_satisfied),
        "attribute_uncertain": float(thresholds.attribute_uncertain),
        "action_satisfied": float(thresholds.action_satisfied),
        "action_uncertain": float(thresholds.action_uncertain),
        "relation_satisfied": float(thresholds.relation_satisfied),
        "relation_uncertain": float(thresholds.relation_uncertain),
    }


def _vp_refinement_vlm_call_count(state: State) -> int:
    """Count refinement-loop VLM verifier calls across all optional MLLM tools."""
    counts = state.timing_log.get("counts", {}) if state.timing_log else {}
    return int(counts.get("text_verifier_requests", 0)) + int(
        counts.get("action_verifier_requests", 0)
    )


_HARD_ATTRIBUTE_NAMES = frozenset(
    {
        "color",
        "material",
        "shape",
        "pattern",
    }
)
_IN_LOOP_AUDITOR_ATTRIBUTE_NAMES = frozenset(
    {*_HARD_ATTRIBUTE_NAMES, "action", "pose", "state"}
)
_IN_LOOP_AUDITOR_RELATION_NAMES = frozenset({"in", "inside", "on"})


def _contrastive_size_attribute_predicate_ids(program) -> tuple[str, ...]:
    size_predicates = [
        predicate
        for predicate in program.predicates
        if isinstance(predicate, AttributePredicate)
        and normalize_attribute_name(predicate.attribute_name) == "size"
    ]
    object_ids = {predicate.target.object_id for predicate in size_predicates}
    values = {
        normalize_attribute_name(predicate.expected_value)
        for predicate in size_predicates
    }
    if len(object_ids) < 2 or len(values) < 2:
        return ()
    return tuple(predicate.predicate_id for predicate in size_predicates)


def is_in_loop_auditor_relation_predicate(predicate: Predicate) -> bool:
    """Return whether one relation is eligible for stuck in-loop auditor override."""
    return (
        isinstance(predicate, SpatialRelationPredicate)
        and normalize_spatial_relation_name(predicate.relation)
        in _IN_LOOP_AUDITOR_RELATION_NAMES
    )


def is_in_loop_auditor_global_scene_predicate(predicate: Predicate) -> bool:
    """Return whether one global-scene predicate is eligible for in-loop auditing."""
    return isinstance(predicate, GlobalScenePredicate)


def in_loop_auditor_attribute_predicate_ids(program) -> tuple[str, ...]:
    """Return attribute predicate IDs eligible for in-loop visual checks."""
    contrastive_size_ids = set(_contrastive_size_attribute_predicate_ids(program))
    eligible_attribute_ids: list[str] = []
    for predicate in program.predicates:
        if not isinstance(predicate, AttributePredicate):
            continue
        normalized_attribute = normalize_attribute_name(predicate.attribute_name)
        if normalized_attribute in _IN_LOOP_AUDITOR_ATTRIBUTE_NAMES:
            eligible_attribute_ids.append(predicate.predicate_id)
            continue
        if predicate.predicate_id in contrastive_size_ids:
            eligible_attribute_ids.append(predicate.predicate_id)
    return tuple(eligible_attribute_ids)


def in_loop_auditor_predicate_ids(program) -> tuple[str, ...]:
    """Return gating predicates eligible for one-shot in-loop auditor override."""
    eligible_ids = list(in_loop_auditor_attribute_predicate_ids(program))
    eligible_ids.extend(
        predicate.predicate_id
        for predicate in program.predicates
        if is_in_loop_auditor_relation_predicate(predicate)
    )
    eligible_ids.extend(
        predicate.predicate_id
        for predicate in program.predicates
        if is_in_loop_auditor_global_scene_predicate(predicate)
    )
    return tuple(eligible_ids)


def hard_predicate_ids(program) -> tuple[str, ...]:
    """Return every predicate ID that gates the main VP refinement loop."""
    return tuple(predicate.predicate_id for predicate in program.predicates)


def soft_predicate_ids(program) -> tuple[str, ...]:
    """Compatibility helper: soft predicates are no longer separated."""
    _ = program
    return ()


def update_in_loop_auditor_stuck_streaks(
    program,
    report: VisualProgramVerificationReport,
    *,
    previous_streaks: dict[str, int] | None = None,
    accept_uncertain: bool = False,
) -> dict[str, int]:
    """Count consecutive rounds where each auditor-eligible predicate stayed unsatisfied."""
    previous = dict(previous_streaks or {})
    eligible_ids = set(in_loop_auditor_predicate_ids(program))
    streaks: dict[str, int] = {}
    for predicate in program.predicates:
        if predicate.predicate_id not in eligible_ids:
            continue
        result = report.get_result(predicate.predicate_id)
        is_satisfied = result.state == PredicateState.SATISFIED
        if accept_uncertain and result.state == PredicateState.UNCERTAIN:
            is_satisfied = True
        if is_satisfied:
            streaks[predicate.predicate_id] = 0
        else:
            streaks[predicate.predicate_id] = (
                previous.get(predicate.predicate_id, 0) + 1
            )
    return streaks


def persistent_stuck_in_loop_auditor_ids(
    program,
    report: VisualProgramVerificationReport,
    *,
    hard_ids: tuple[str, ...],
    stuck_streaks: dict[str, int],
    min_streak: int = 2,
    accept_uncertain: bool = False,
) -> tuple[str, ...]:
    """Return blocking predicates eligible for one-shot stuck-predicate auditing."""
    if int(min_streak) <= 0:
        return ()
    eligible_ids = set(in_loop_auditor_predicate_ids(program))

    blocking_ids: list[str] = []
    for predicate_id in hard_ids:
        result = report.get_result(predicate_id)
        if result.state == PredicateState.SATISFIED:
            continue
        if accept_uncertain and result.state == PredicateState.UNCERTAIN:
            continue
        blocking_ids.append(predicate_id)

    if not blocking_ids:
        return ()
    if any(predicate_id not in eligible_ids for predicate_id in blocking_ids):
        return ()
    if any(
        int(stuck_streaks.get(predicate_id, 0)) < int(min_streak)
        for predicate_id in blocking_ids
    ):
        return ()
    return tuple(blocking_ids)


def predicate_group_satisfied(
    report: VisualProgramVerificationReport,
    *,
    predicate_ids: tuple[str, ...],
    accept_uncertain: bool = False,
) -> bool:
    """Return whether every predicate in one named group is satisfied."""
    for predicate_id in predicate_ids:
        state = report.get_result(predicate_id).state
        if state == PredicateState.SATISFIED:
            continue
        if accept_uncertain and state == PredicateState.UNCERTAIN:
            continue
        return False
    return True


def summarize_predicate_states(
    report: VisualProgramVerificationReport,
) -> list[dict[str, Any]]:
    """Summarize predicate states for logging."""
    return [
        {
            "predicate_id": result.predicate_id,
            "category": result.category.value,
            "state": result.state.value,
            "score": result.score,
            "confidence": result.confidence,
            "evidence_summary": result.evidence_summary,
        }
        for result in report.results
    ]


def summarize_blocking_predicates(
    report: VisualProgramVerificationReport,
    *,
    predicate_ids: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Summarize the predicates in one group that remain unsatisfied."""
    blocking: list[dict[str, Any]] = []
    for predicate_id in predicate_ids:
        result = report.get_result(predicate_id)
        if result.state == PredicateState.SATISFIED:
            continue
        blocking.append(
            {
                "predicate_id": result.predicate_id,
                "category": result.category.value,
                "state": result.state.value,
                "score": result.score,
                "confidence": result.confidence,
                "evidence_summary": result.evidence_summary,
            }
        )
    return blocking


def _json_ready_constraint_value(value: Any) -> Any:
    """Convert predicate constraint payloads into JSON-friendly structures."""
    if isinstance(value, tuple):
        return [_json_ready_constraint_value(item) for item in value]
    if isinstance(value, list):
        return [_json_ready_constraint_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _json_ready_constraint_value(item) for key, item in value.items()
        }
    return value


def summarize_blocking_in_loop_auditor_predicates(
    program,
    report: VisualProgramVerificationReport,
    *,
    predicate_ids: tuple[str, ...],
    stuck_streaks: dict[str, int],
) -> list[dict[str, Any]]:
    """Build a blocker-focused payload for the in-loop semantic override auditor."""
    object_labels = {obj.object_id: obj.label for obj in program.objects}
    predicate_map = {
        predicate.predicate_id: predicate for predicate in program.predicates
    }
    blocking: list[dict[str, Any]] = []
    for predicate_id in predicate_ids:
        predicate = predicate_map.get(predicate_id)
        result = report.get_result(predicate_id)
        payload: dict[str, Any] = {
            "predicate_id": result.predicate_id,
            "category": result.category.value,
            "description": getattr(predicate, "description", ""),
            "state": result.state.value,
            "score": result.score,
            "confidence": result.confidence,
            "evidence_summary": result.evidence_summary,
            "stuck_streak": int(stuck_streaks.get(predicate_id, 0)),
        }
        if isinstance(predicate, AttributePredicate):
            payload.update(
                {
                    "predicate_kind": "attribute",
                    "object_id": predicate.target.object_id,
                    "object_label": object_labels.get(
                        predicate.target.object_id,
                        predicate.target.object_id,
                    ),
                    "attribute_name": predicate.attribute_name,
                    "expected_value": _json_ready_constraint_value(
                        predicate.expected_value
                    ),
                    "action_target_object_id": (
                        predicate.action_target.object_id
                        if predicate.action_target is not None
                        else None
                    ),
                    "action_target_label": (
                        object_labels.get(
                            predicate.action_target.object_id,
                            predicate.action_target.object_id,
                        )
                        if predicate.action_target is not None
                        else None
                    ),
                }
            )
        elif isinstance(predicate, SpatialRelationPredicate):
            payload.update(
                {
                    "predicate_kind": "spatial_relation",
                    "subject_id": predicate.subject.object_id,
                    "subject_label": object_labels.get(
                        predicate.subject.object_id,
                        predicate.subject.object_id,
                    ),
                    "relation": predicate.relation,
                    "reference_id": predicate.reference.object_id,
                    "reference_label": object_labels.get(
                        predicate.reference.object_id,
                        predicate.reference.object_id,
                    ),
                }
            )
        elif isinstance(predicate, GlobalScenePredicate):
            payload.update(
                {
                    "predicate_kind": "global_scene",
                    "attribute_name": predicate.attribute_name,
                    "expected_value": _json_ready_constraint_value(
                        predicate.expected_value
                    ),
                }
            )
        blocking.append(payload)
    return blocking


def finalize_action_record(
    record: ActionExecutionRecord,
    report: VisualProgramVerificationReport,
    *,
    accept_uncertain: bool = False,
) -> ActionExecutionRecord:
    """Mark a previously applied action as succeeded or failed."""
    metadata = dict(record.metadata)
    if record.target_predicate_ids:
        state_rank = {
            PredicateState.CONTRADICTION: 0,
            PredicateState.VIOLATED: 1,
            PredicateState.UNCERTAIN: 2,
            PredicateState.SATISFIED: 3,
        }
        satisfied_count = 0
        state_rank_sum = 0
        scores: list[float] = []
        for predicate_id in record.target_predicate_ids:
            result = report.get_result(predicate_id)
            if result.state == PredicateState.SATISFIED:
                satisfied_count += 1
            state_rank_sum += state_rank.get(result.state, 0)
            if result.score is not None:
                scores.append(float(result.score))
        metadata["predicate_progress"] = {
            "satisfied_count": satisfied_count,
            "state_rank_sum": state_rank_sum,
            "average_score": (sum(scores) / len(scores)) if scores else None,
        }

    if not record.target_predicate_ids:
        return ActionExecutionRecord(
            action_type=record.action_type,
            status=ActionExecutionStatus.FAILED,
            target_predicate_ids=record.target_predicate_ids,
            target_object_ids=record.target_object_ids,
            edit_operation=record.edit_operation,
            metadata=metadata,
        )

    succeeded = True
    for predicate_id in record.target_predicate_ids:
        state = report.get_result(predicate_id).state
        if state == PredicateState.SATISFIED:
            continue
        if accept_uncertain and state == PredicateState.UNCERTAIN:
            continue
        succeeded = False
        break

    return ActionExecutionRecord(
        action_type=record.action_type,
        status=(
            ActionExecutionStatus.SUCCEEDED
            if succeeded
            else ActionExecutionStatus.FAILED
        ),
        target_predicate_ids=record.target_predicate_ids,
        target_object_ids=record.target_object_ids,
        edit_operation=record.edit_operation,
        metadata=metadata,
    )


def summarize_predicate_group_match(
    report, *, predicate_ids, accept_uncertain=False
) -> dict[str, Any]:
    """Summarize scores and strict satisfaction for a candidate's predicates."""
    results = [report.get_result(predicate_id) for predicate_id in predicate_ids]
    unsatisfied = [
        r
        for r in results
        if r.state != PredicateState.SATISFIED
        and not (accept_uncertain and r.state == PredicateState.UNCERTAIN)
    ]
    scores = [float(r.score) for r in results if r.score is not None]
    unsatisfied_scores = [
        float(r.score) if r.score is not None else 0.0 for r in unsatisfied
    ]
    return {
        "satisfied_count": len(results) - len(unsatisfied),
        "total_count": len(results),
        "all_satisfied": not unsatisfied,
        "average_score": sum(scores) / len(scores) if scores else 0.0,
        "unsatisfied_count": len(unsatisfied),
        "unsatisfied_score": sum(unsatisfied_scores) / len(unsatisfied_scores)
        if unsatisfied_scores
        else 1.0,
        "unsatisfied_predicate_ids": [r.predicate_id for r in unsatisfied],
    }


def build_verification_artifact_payload(
    report: VisualProgramVerificationReport,
    *,
    round_index: int,
    hard_ids: tuple[str, ...],
    soft_ids: tuple[str, ...] = (),
    accept_uncertain: bool = False,
    verifier_thresholds: dict[str, Any] | None = None,
    relation_scoring_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a human-readable verification artifact without dropping raw details."""
    payload = report.to_dict()
    predicate_state_summary = summarize_predicate_states(report)
    predicate_summary_by_id: dict[str, dict[str, Any]] = {}
    predicate_state_counts = {
        PredicateState.SATISFIED.value: 0,
        PredicateState.UNCERTAIN.value: 0,
        PredicateState.VIOLATED.value: 0,
    }
    for item in predicate_state_summary:
        predicate_state_counts[item["state"]] = (
            predicate_state_counts.get(item["state"], 0) + 1
        )
        predicate_summary_by_id[item["predicate_id"]] = {
            "category": item["category"],
            "state": item["state"],
            "score": item["score"],
            "confidence": item["confidence"],
            "evidence_summary": item["evidence_summary"],
            "is_gating": item["predicate_id"] in hard_ids,
            "is_hard": item["predicate_id"] in hard_ids,
            "is_soft": item["predicate_id"] in soft_ids,
            "is_critical": item["predicate_id"] in hard_ids,
        }

    hard_match = summarize_predicate_group_match(
        report,
        predicate_ids=hard_ids,
        accept_uncertain=accept_uncertain,
    )
    soft_match = summarize_predicate_group_match(
        report,
        predicate_ids=soft_ids,
        accept_uncertain=accept_uncertain,
    )
    all_predicates_satisfied = predicate_state_counts[
        PredicateState.SATISFIED.value
    ] == len(report.results)
    payload.update(
        {
            "round": round_index,
            "predicate_partition_mode": "all_gating_no_soft_split",
            "gating_predicate_ids": list(hard_ids),
            "hard_predicate_ids": list(hard_ids),
            "soft_predicate_ids": list(soft_ids),
            "critical_predicate_ids": list(hard_ids),
            "unsupported_predicate_ids": [],
            "accept_uncertain_as_satisfied": accept_uncertain,
            "verifier_thresholds": dict(verifier_thresholds or {}),
            "relation_scoring_config": dict(relation_scoring_config or {}),
            "predicate_state_counts": predicate_state_counts,
            "all_satisfied": all_predicates_satisfied,
            "all_predicates_satisfied": all_predicates_satisfied,
            "all_gating_predicates_satisfied": hard_match["all_satisfied"],
            "all_hard_predicates_satisfied": hard_match["all_satisfied"],
            "all_soft_predicates_satisfied": soft_match["all_satisfied"],
            "all_critical_predicates_satisfied": hard_match["all_satisfied"],
            "gating_match": hard_match,
            "hard_match": hard_match,
            "soft_match": soft_match,
            "critical_match": hard_match,
            "blocking_gating_predicates": summarize_blocking_predicates(
                report,
                predicate_ids=hard_ids,
            ),
            "blocking_hard_predicates": summarize_blocking_predicates(
                report,
                predicate_ids=hard_ids,
            ),
            "blocking_soft_predicates": summarize_blocking_predicates(
                report,
                predicate_ids=soft_ids,
            ),
            "blocking_critical_predicates": summarize_blocking_predicates(
                report,
                predicate_ids=hard_ids,
            ),
            "predicate_summary_by_id": predicate_summary_by_id,
        }
    )
    return payload


def build_partial_selection_candidate(
    *,
    report: VisualProgramVerificationReport,
    image_data_url: str,
    round_index: int,
    hard_ids: Sequence[str],
    soft_ids: Sequence[str] = (),
    accept_uncertain: bool = False,
) -> dict[str, Any]:
    """Keep the image and verification evidence for best-partial selection."""
    hard_ids = tuple(hard_ids)
    return {
        "round_index": round_index,
        "report": report,
        "image_data_url": image_data_url,
        "hard_ids": hard_ids,
        "soft_ids": tuple(soft_ids),
        "hard_match": summarize_predicate_group_match(
            report, predicate_ids=hard_ids, accept_uncertain=accept_uncertain
        ),
        "partial_match_priority": summarize_partial_match_priority(
            report, predicate_ids=hard_ids, accept_uncertain=accept_uncertain
        ),
    }


def select_best_partial_candidate_records(
    candidates: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Rank predicate-family coverage, satisfied count, score, then earlier round."""
    if not candidates:
        raise ValueError("Cannot rank an empty candidate pool.")
    accepted = [c for c in candidates if c["hard_match"]["all_satisfied"]]
    if accepted:
        return max(
            accepted,
            key=lambda c: (c["hard_match"]["average_score"], -c["round_index"]),
        )
    return max(
        candidates,
        key=lambda c: build_partial_match_rank_key(
            priority_summary=c["partial_match_priority"],
            satisfied_count=c["hard_match"]["satisfied_count"],
            unsatisfied_score=c["hard_match"]["unsatisfied_score"],
            round_index=c["round_index"],
        ),
    )


def _partial_selection_candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "round_index": candidate["round_index"],
        "hard_match": candidate["hard_match"],
    }


def summarize_best_partial_selection_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep the selected round and matching evidence in the run summary."""
    return {key: value for key, value in payload.items() if key != "candidates"}


def select_run_best_partial_candidate(
    *,
    candidates: Sequence[dict[str, Any]],
    selection_reason: str,
) -> dict[str, Any]:
    """Select the best image verified against this run's fixed visual program."""
    selected = select_best_partial_candidate_records(candidates)
    return {
        "selected_candidate": selected,
        "selection_payload": {
            "selection_reason": selection_reason,
            "selection_scope": "all_rounds",
            "candidate_count": len(candidates),
            "selected_round": selected["round_index"],
            "hard_match": selected["hard_match"],
            "candidates": [_partial_selection_candidate_summary(c) for c in candidates],
        },
    }


def build_vp_pipeline_summary(
    *,
    visual_program_payload: dict[str, Any],
    hard_ids: tuple[str, ...],
    soft_ids: tuple[str, ...],
    iteration_logs: list[dict[str, Any]],
    in_loop_auditor_output: dict[str, Any],
    in_loop_auditor_call_count: int,
    all_hard_predicates_satisfied: bool,
    all_soft_predicates_satisfied: bool | None,
    refinement_loop_vlm_calls: int,
    refinement_loop_called_any_vlm: bool,
    timing_breakdown: dict[str, Any],
    verifier_thresholds: dict[str, Any],
    relation_scoring_config: dict[str, Any],
    rewritten_prompt_generator_call_count: int,
    rewritten_prompt_count: int,
    hard_gate_auditor_override: dict[str, Any] | None = None,
    run_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact summary of the integrated VP pipeline run."""
    hard_gate_overridden = bool(hard_gate_auditor_override)
    final_candidate_accepted = bool(
        all_hard_predicates_satisfied or hard_gate_overridden
    )
    acceptance_decision_source = (
        "in_loop_auditor_override"
        if hard_gate_overridden
        else "gating_predicates"
        if all_hard_predicates_satisfied
        else "none"
    )
    return {
        "program_id": visual_program_payload.get("program_id"),
        "visual_program_source": "generated",
        "visual_program_parser_skipped": False,
        "rewritten_prompt_source": "generated",
        "predicate_count": len(visual_program_payload.get("predicates", [])),
        "predicate_partition_mode": "all_gating_no_soft_split",
        "gating_predicate_ids": list(hard_ids),
        "hard_predicate_ids": list(hard_ids),
        "soft_predicate_ids": list(soft_ids),
        "critical_predicate_ids": list(hard_ids),
        "unsupported_predicate_ids": [],
        "iteration_count": len(iteration_logs),
        "all_gating_predicates_satisfied": all_hard_predicates_satisfied,
        "all_hard_predicates_satisfied": all_hard_predicates_satisfied,
        "all_soft_predicates_satisfied": all_soft_predicates_satisfied,
        "all_critical_predicates_satisfied": all_hard_predicates_satisfied,
        "hard_gate_auditor_override": hard_gate_auditor_override,
        "hard_gate_overridden_by_auditor": hard_gate_overridden,
        "final_candidate_accepted": final_candidate_accepted,
        "acceptance_decision_source": acceptance_decision_source,
        "rewritten_prompt_generator_call_count": rewritten_prompt_generator_call_count,
        "rewritten_prompt_count": rewritten_prompt_count,
        "in_loop_auditor_called": bool(in_loop_auditor_output.get("called", False)),
        "in_loop_auditor_call_count": in_loop_auditor_call_count,
        "in_loop_auditor_override_allowed": bool(
            in_loop_auditor_output.get("override_allowed", False)
        ),
        "in_loop_auditor_reason": in_loop_auditor_output.get("short_reason"),
        "refinement_loop_vlm_calls": refinement_loop_vlm_calls,
        "refinement_loop_called_any_vlm": refinement_loop_called_any_vlm,
        "verifier_thresholds": dict(verifier_thresholds),
        "relation_scoring_config": dict(relation_scoring_config),
        "run_summary": run_summary,
        "timing_breakdown": timing_breakdown,
    }


async def _save_json_artifact(
    state: State, filename: str, payload: Any, *, timing_metadata=None
) -> None:
    from graph.utils import ensure_directory, json_dumps_timed, write_text_file

    if not state.saving_files:
        return
    artifact_dir = _artifact_dir(state)
    save_path = os.path.join(artifact_dir, filename)
    serialized = json_dumps_timed(
        payload,
        indent=4,
        ensure_ascii=False,
        timing_log=state.timing_log,
        timing_component=f"vp_pipeline:{filename}:serialize",
        timing_metadata=timing_metadata,
    )
    await asyncio.to_thread(
        ensure_directory,
        os.path.dirname(save_path),
        timing_log=state.timing_log,
        timing_component="vp_pipeline:mkdir",
        timing_metadata=timing_metadata,
    )
    await asyncio.to_thread(
        write_text_file,
        save_path,
        serialized,
        timing_log=state.timing_log,
        timing_component=f"vp_pipeline:{filename}:write",
        timing_metadata=timing_metadata,
    )


async def _save_text_artifact(
    state: State,
    filename: str,
    text: str,
    *,
    timing_metadata=None,
) -> None:
    from graph.utils import ensure_directory, write_text_file

    if not state.saving_files:
        return
    artifact_dir = _artifact_dir(state)
    save_path = os.path.join(artifact_dir, filename)
    await asyncio.to_thread(
        ensure_directory,
        os.path.dirname(save_path),
        timing_log=state.timing_log,
        timing_component="vp_pipeline:mkdir",
        timing_metadata=timing_metadata,
    )
    await asyncio.to_thread(
        write_text_file,
        save_path,
        text,
        timing_log=state.timing_log,
        timing_component=f"vp_pipeline:{filename}:write",
        timing_metadata=timing_metadata,
    )


async def _save_image_artifact(
    state: State, filename: str, image, *, timing_metadata=None
) -> None:
    from graph.utils import ensure_directory, save_image_file

    if not state.saving_files:
        return
    artifact_dir = _artifact_dir(state)
    save_path = os.path.join(artifact_dir, filename)
    await asyncio.to_thread(
        ensure_directory,
        os.path.dirname(save_path),
        timing_log=state.timing_log,
        timing_component="vp_pipeline:mkdir",
        timing_metadata=timing_metadata,
    )
    await asyncio.to_thread(
        save_image_file,
        image,
        save_path,
        timing_log=state.timing_log,
        timing_component=f"vp_pipeline:{filename}:write",
        timing_metadata=timing_metadata,
    )


async def _save_visual_program_artifacts(
    *,
    state: State,
    parse_result,
    visual_program_payload: dict[str, Any],
    visual_program_source: str,
    normalization_report_payload: dict[str, Any] | None = None,
    review_report_payload: dict[str, Any] | None = None,
) -> None:
    artifact_specs = [
        (
            "visual_program.json",
            visual_program_payload,
            {"stage": "visual_program"},
        ),
    ]
    if parse_result is not None:
        artifact_specs.extend(
            [
                (
                    "visual_program_parser_output_raw.json",
                    parse_result.raw_parser_output.model_dump(mode="json"),
                    {
                        "stage": "visual_program_parser_raw",
                    },
                ),
                (
                    "visual_program_parser_output_normalized.json",
                    parse_result.normalized_parser_output.model_dump(mode="json"),
                    {
                        "stage": "visual_program_parser_normalized",
                    },
                ),
                (
                    "visual_program_parser_output_final.json",
                    parse_result.final_parser_output.model_dump(mode="json"),
                    {
                        "stage": "visual_program_parser_final",
                    },
                ),
            ]
        )
    artifact_specs.extend(
        [
            (
                "visual_program_parser_normalization_report.json",
                normalization_report_payload or {},
                {
                    "stage": "visual_program_parser_normalization_report",
                },
            ),
            (
                "visual_program_parser_review_report.json",
                review_report_payload or {},
                {
                    "stage": "visual_program_parser_review_report",
                },
            ),
        ]
    )
    for filename, payload, timing_metadata in artifact_specs:
        await _save_json_artifact(
            state,
            filename,
            payload,
            timing_metadata=timing_metadata,
        )

    text_timing_metadata = {"stage": "visual_program"}
    await _save_text_artifact(
        state,
        "visual_program.py",
        visual_program_source,
        timing_metadata=text_timing_metadata,
    )


async def _save_rewritten_prompt_generator_artifacts(
    *,
    state: State,
    generation_result,
) -> None:
    raw_payload = (
        generation_result.raw_output.model_dump(mode="json")
        if generation_result.raw_output is not None
        else {"rewritten_prompts": []}
    )
    final_payload = (
        generation_result.normalized_output.model_dump(mode="json")
        if generation_result.normalized_output is not None
        else {"rewritten_prompts": []}
    )
    report_payload = generation_result.to_report()
    artifact_specs = [
        (
            "rewritten_prompt_generator_output_raw.json",
            raw_payload,
            {
                "stage": "rewritten_prompt_generator_raw",
            },
        ),
        (
            "rewritten_prompt_generator_output_final.json",
            final_payload,
            {
                "stage": "rewritten_prompt_generator_final",
            },
        ),
        (
            "rewritten_prompt_generator_report.json",
            report_payload,
            {
                "stage": "rewritten_prompt_generator_report",
            },
        ),
    ]
    for filename, payload, timing_metadata in artifact_specs:
        await _save_json_artifact(
            state,
            filename,
            payload,
            timing_metadata=timing_metadata,
        )


async def _run_generation_action(
    *,
    state: State,
    prompt: str,
    sample_index: int,
    generation_executor,
    generation_model_name: str,
    force_new_seed: bool = False,
):
    from graph.utils import image_to_data_url

    seed_offset = resolve_generation_seed_offset(
        sample_index=sample_index,
        round_index=state.current_round,
        force_new_seed=force_new_seed,
    )
    timing_metadata = {
        "round": state.current_round,
        "sample": sample_index,
        "action": "generation",
        "force_new_seed": force_new_seed,
    }
    generated_image, seed_used = await asyncio.to_thread(
        generation_executor,
        seed_offset,
        state,
        prompt,
        timing_metadata,
    )
    output_image_data_url = image_to_data_url(
        generated_image,
        timing_log=state.timing_log,
        timing_component="vp_pipeline:generation_output_encode",
        timing_metadata=timing_metadata,
    )
    await _save_image_artifact(
        state,
        f"vp_round_{state.current_round:02d}_sample_{sample_index:02d}.jpg",
        generated_image,
        timing_metadata=timing_metadata,
    )
    return (
        generated_image,
        output_image_data_url,
        {
            "model_choice": "generation",
            "generation_model": generation_model_name,
            "editing_model": None,
            "seed_used": seed_used,
            "seed_offset": seed_offset,
            "input_image_data_url": None,
            "output_image_data_url": output_image_data_url,
            "original_prompt": state.original_prompt,
            "executed_prompt": prompt,
            "image_size": generated_image.size,
            "detection_caption": None,
            "detection_boxes": None,
            "detection_labels": None,
            "detection_depths": None,
            "score": None,
        },
    )


async def _run_edit_action(
    *,
    state: State,
    prompt: str,
    input_image,
    input_image_data_url: str,
    sample_index: int,
):
    from graph.utils import image_to_data_url

    from graph.image_runtime import get_edit_model_name, import_and_execute_edit_pipe

    seed_offset = resolve_generation_seed_offset(
        sample_index=sample_index,
        round_index=state.current_round,
        force_new_seed=True,
    )
    timing_metadata = {
        "round": state.current_round,
        "sample": sample_index,
        "action": "editing",
        "force_new_seed": True,
    }
    edited_image, seed_used = await asyncio.to_thread(
        import_and_execute_edit_pipe,
        seed_offset,
        state,
        prompt,
        input_image,
        timing_metadata,
    )
    output_image_data_url = image_to_data_url(
        edited_image,
        timing_log=state.timing_log,
        timing_component="vp_pipeline:edit_output_encode",
        timing_metadata=timing_metadata,
    )
    await _save_image_artifact(
        state,
        f"vp_round_{state.current_round:02d}_sample_{sample_index:02d}.jpg",
        edited_image,
        timing_metadata=timing_metadata,
    )
    return (
        edited_image,
        output_image_data_url,
        {
            "model_choice": "editing",
            "generation_model": None,
            "editing_model": get_edit_model_name(state),
            "seed_used": seed_used,
            "seed_offset": seed_offset,
            "input_image_data_url": input_image_data_url,
            "output_image_data_url": output_image_data_url,
            "original_prompt": state.original_prompt,
            "executed_prompt": prompt,
            "image_size": edited_image.size,
            "detection_caption": None,
            "detection_boxes": None,
            "detection_labels": None,
            "detection_depths": None,
            "score": None,
        },
    )


async def vp_pipeline(
    state: State,
    config: RunnableConfig,
    runtime: Runtime[Context],
) -> Command[Literal["__end__"]]:
    """Run the VP parse-generate-verify-refine-audit pipeline."""
    from graph.image_runtime import get_generation_executor

    node_start = time.perf_counter()
    state.timing_log = ensure_timing_log(state.timing_log)

    if not state.original_prompt:
        raise ValueError("No original_prompt found in the VP pipeline input state.")

    print("vp_pipeline | starting EPIC refinement")

    action_selector = VisualProgramActionSelector()
    generation_executor, generation_model_name = get_generation_executor(state)
    verifier_thresholds = VerifierThresholds()
    verifier_thresholds_payload = serialize_verifier_thresholds(verifier_thresholds)
    relation_scoring_config = RelationScoringConfig()
    relation_scoring_config_payload = relation_scoring_config.to_dict()

    parse_result = None
    visual_program_payload: dict[str, Any] = {}
    hard_ids: tuple[str, ...] = ()
    soft_ids: tuple[str, ...] = ()
    executor_outputs: list[list[dict[str, Any]]] = []
    round_best_images: list[str] = []
    vp_verification_reports: list[dict[str, Any]] = []
    vp_action_selections: list[dict[str, Any]] = []
    vp_iteration_logs: list[dict[str, Any]] = []
    accept_uncertain = False
    in_loop_audit_result = skipped_in_loop_audit_result(
        "In-loop auditor not yet run.",
    )
    in_loop_auditor_call_count = 0
    selected_best_round = state.current_round
    all_hard_satisfied = False
    all_soft_satisfied = False
    hard_gate_auditor_override: dict[str, Any] | None = None
    current_image = None
    current_image_data_url = None
    run_partial_selection_candidates: list[dict[str, Any]] = []
    rewritten_prompt_generator_reports: list[dict[str, Any]] = []
    rewritten_prompt_generator_call_count = 0
    visual_program_source_kind = "generated"

    parse_component_name = "visual_program_parser"
    parse_result = await parse_visual_program_from_prompt(
        state.original_prompt,
        state=state,
        runtime=runtime,
        component_name=parse_component_name,
    )
    visual_program = parse_result.visual_program
    rewrite_component_name = "rewritten_prompt_generator"
    rewritten_prompt_request_count = int(state.vp_rewritten_prompt_count)
    try:
        rewritten_prompt_generation_result = await generate_rewritten_prompts(
            state.original_prompt,
            state=state,
            runtime=runtime,
            requested_count=rewritten_prompt_request_count,
            component_name=rewrite_component_name,
        )
    except Exception as exc:
        rewritten_prompt_generation_result = failed_rewritten_prompt_generation_result(
            state.original_prompt,
            requested_count=rewritten_prompt_request_count,
            component_name=rewrite_component_name,
            exc=exc,
        )
        print(
            "vp_pipeline | rewritten prompt generator failed; "
            "falling back to source prompt only: "
            f"{type(exc).__name__}: {exc}"
        )
    rewritten_prompt_generator_report = rewritten_prompt_generation_result.to_report()
    rewritten_prompt_generator_reports.append(rewritten_prompt_generator_report)
    rewritten_prompt_generator_call_count += (
        1 if rewritten_prompt_generation_result.called else 0
    )
    visual_program = VisualProgram.from_predicates(
        source_prompt=visual_program.source_prompt,
        objects=list(visual_program.objects),
        predicates=list(visual_program.predicates),
        rewritten_prompts=list(rewritten_prompt_generation_result.rewritten_prompts),
        program_id=visual_program.program_id,
    )
    visual_program_payload = visual_program.to_dict()
    normalization_report_payload = parse_result.normalization_report.model_dump(
        mode="json"
    )
    review_report_payload = parse_result.review_report.model_dump(mode="json")
    visual_program_source = visual_program.to_python_source()
    hard_ids = hard_predicate_ids(visual_program)
    soft_ids = soft_predicate_ids(visual_program)
    all_soft_satisfied = len(soft_ids) == 0

    print(
        "vp_pipeline | active visual program:\n"
        + json.dumps(visual_program_payload, ensure_ascii=False, indent=2)
    )
    await _save_visual_program_artifacts(
        state=state,
        parse_result=parse_result,
        visual_program_payload=visual_program_payload,
        visual_program_source=visual_program_source,
        normalization_report_payload=normalization_report_payload,
        review_report_payload=review_report_payload,
    )
    await _save_rewritten_prompt_generator_artifacts(
        state=state,
        generation_result=rewritten_prompt_generation_result,
    )

    run_record = {
        "program_id": visual_program_payload.get("program_id"),
        "predicate_count": len(visual_program_payload.get("predicates", [])),
        "hard_predicate_ids": list(hard_ids),
        "soft_predicate_ids": list(soft_ids),
        "visual_program_source": visual_program_source_kind,
        "normalization_report": normalization_report_payload,
        "review_report": review_report_payload,
        "rewritten_prompt_generator": rewritten_prompt_generator_report,
    }
    action_history: list[ActionExecutionRecord] = []
    pending_action_record = None
    hard_gate_auditor_override = None
    in_loop_auditor_stuck_streaks: dict[str, int] = {}
    last_persistent_stuck_signature: tuple[str, ...] | None = None
    final_report = None
    in_loop_audit_result = skipped_in_loop_audit_result(
        "In-loop auditor not triggered."
    )

    (
        selected_initial_base_prompt,
        initial_rewrite_index,
        initial_rewrite_selection,
    ) = select_initial_generation_prompt(
        program=visual_program,
        random_seed=state.image_seed,
    )
    used_rewrite_indices = list(
        initial_rewrite_selection.get("used_rewrite_indices_after_selection", [])
    )
    current_prompt, current_scene_prompt_metadata = compose_prompt_with_scene_guidance(
        program=visual_program,
        base_prompt=selected_initial_base_prompt,
    )
    current_rewrite_index = initial_rewrite_index
    current_rewrite_prompt = (
        selected_initial_base_prompt
        if initial_rewrite_index is not None
        else state.original_prompt
    )
    initial_prompt_payload = {
        "original_prompt": state.original_prompt,
        "selected_base_prompt": selected_initial_base_prompt,
        "executed_prompt": current_prompt,
        "used_rewritten_prompt": initial_rewrite_index is not None,
        "selected_rewritten_prompt_index": initial_rewrite_index,
        **current_scene_prompt_metadata,
        **initial_rewrite_selection,
    }
    run_record["initial_generation_prompt"] = initial_prompt_payload
    print(
        "vp_pipeline | initial generation prompt choice:\n"
        + json.dumps(
            initial_prompt_payload,
            ensure_ascii=False,
            indent=2,
        )
    )
    initial_generation_timing = {
        "stage": "initial_generation",
    }
    await _save_json_artifact(
        state,
        "initial_generation_prompt.json",
        initial_prompt_payload,
        timing_metadata=initial_generation_timing,
    )
    (
        current_image,
        current_image_data_url,
        executor_output,
    ) = await _run_generation_action(
        state=state,
        prompt=current_prompt,
        sample_index=1,
        generation_executor=generation_executor,
        generation_model_name=generation_model_name,
    )
    executor_output["used_rewritten_prompt_for_initial_generation"] = (
        initial_rewrite_index is not None
    )
    executor_output["selected_rewritten_prompt_index"] = initial_rewrite_index
    executor_output["rewrite_selection_pool"] = initial_rewrite_selection.get(
        "rewrite_selection_pool"
    )
    executor_output.update(current_scene_prompt_metadata)
    executor_outputs.append([executor_output])
    round_best_images.append(current_image_data_url)

    while True:
        round_debug_dir = (
            _round_debug_dir(state, state.current_round) if state.saving_files else None
        )
        toolbox = get_visual_program_perception_toolbox(
            timing_log=state.timing_log,
            text_verifier_base_url=runtime.context.base_url,
            text_verifier_model=runtime.context.model,
            text_verifier_temperature=float(0.0),
            text_verifier_seed=int(runtime.context.seed),
            text_verifier_timeout_seconds=int(180),
            text_verifier_max_resolution=int(state.vlm_max_resolution),
            action_verifier_base_url=runtime.context.base_url,
            action_verifier_model=runtime.context.model,
            action_verifier_temperature=float(0.0),
            action_verifier_seed=int(runtime.context.seed),
            action_verifier_timeout_seconds=int(180),
            action_verifier_max_resolution=int(state.vlm_max_resolution),
            action_verifier_system_prompt=runtime.context.system_prompt_action_verifier,
            debug_artifact_dir=round_debug_dir,
        )
        verifier = VisualProgramVerifier(
            tools=toolbox,
            thresholds=verifier_thresholds,
            relation_scoring=relation_scoring_config,
        )

        verification_start = time.perf_counter()
        report = await asyncio.to_thread(verifier.verify, current_image, visual_program)
        record_duration(
            state.timing_log,
            "vp_predicate_verification_seconds",
            time.perf_counter() - verification_start,
            component="vp_pipeline:verification",
            count_key="predicate_verifications",
            metadata={"round": state.current_round},
        )

        if pending_action_record is not None:
            finalized_record = finalize_action_record(
                pending_action_record,
                report,
                accept_uncertain=accept_uncertain,
            )
            action_history.append(finalized_record)
            if vp_action_selections:
                vp_action_selections[-1]["execution_status"] = (
                    finalized_record.status.value
                )
            pending_action_record = None

        predicate_state_summary = summarize_predicate_states(report)
        verification_artifact = build_verification_artifact_payload(
            report,
            round_index=state.current_round,
            hard_ids=hard_ids,
            soft_ids=soft_ids,
            accept_uncertain=accept_uncertain,
            verifier_thresholds=verifier_thresholds_payload,
            relation_scoring_config=relation_scoring_config_payload,
        )
        vp_verification_reports.append(verification_artifact)
        candidate_record = build_partial_selection_candidate(
            report=report,
            image_data_url=current_image_data_url,
            round_index=state.current_round,
            hard_ids=hard_ids,
            soft_ids=soft_ids,
            accept_uncertain=accept_uncertain,
        )
        run_partial_selection_candidates.append(candidate_record)
        final_report = report
        print(
            f"vp_pipeline | round {state.current_round} | predicate states:\n"
            + json.dumps(predicate_state_summary, ensure_ascii=False, indent=2)
        )
        await _save_json_artifact(
            state,
            f"vp_verification_round_{state.current_round}.json",
            verification_artifact,
            timing_metadata={"round": state.current_round, "stage": "verification"},
        )

        hard_satisfied = predicate_group_satisfied(
            report,
            predicate_ids=hard_ids,
            accept_uncertain=accept_uncertain,
        )
        soft_satisfied = predicate_group_satisfied(
            report,
            predicate_ids=soft_ids,
            accept_uncertain=accept_uncertain,
        )
        all_hard_satisfied = hard_satisfied
        all_soft_satisfied = soft_satisfied
        refinement_vlm_call_count = _vp_refinement_vlm_call_count(state)
        in_loop_auditor_stuck_streaks = update_in_loop_auditor_stuck_streaks(
            visual_program,
            report,
            previous_streaks=in_loop_auditor_stuck_streaks,
            accept_uncertain=accept_uncertain,
        )
        persistent_stuck_predicate_ids = persistent_stuck_in_loop_auditor_ids(
            visual_program,
            report,
            hard_ids=hard_ids,
            stuck_streaks=in_loop_auditor_stuck_streaks,
            min_streak=state.vp_persistent_stuck_in_loop_auditor_streak,
            accept_uncertain=accept_uncertain,
        )
        verification_artifact["in_loop_auditor_stuck_streaks"] = dict(
            in_loop_auditor_stuck_streaks
        )
        verification_artifact["persistent_stuck_in_loop_auditor_ids"] = list(
            persistent_stuck_predicate_ids
        )
        iteration_log = {
            "round": state.current_round,
            "predicate_partition_mode": "all_gating_no_soft_split",
            "gating_predicate_ids": hard_ids,
            "hard_predicate_ids": hard_ids,
            "soft_predicate_ids": soft_ids,
            "critical_predicate_ids": hard_ids,
            "unsupported_predicate_ids": (),
            "program_id": visual_program_payload.get("program_id"),
            "gating_predicates_satisfied": hard_satisfied,
            "hard_predicates_satisfied": hard_satisfied,
            "soft_predicates_satisfied": soft_satisfied,
            "critical_predicates_satisfied": hard_satisfied,
            "in_loop_auditor_stuck_streaks": dict(in_loop_auditor_stuck_streaks),
            "persistent_stuck_in_loop_auditor_ids": list(
                persistent_stuck_predicate_ids
            ),
            "predicate_states": predicate_state_summary,
            "refinement_loop_vlm_calls": refinement_vlm_call_count,
            "refinement_loop_called_any_vlm": refinement_vlm_call_count > 0,
            "round_debug_artifact_dir": round_debug_dir,
        }
        await _save_json_artifact(
            state,
            f"vp_round_{state.current_round:02d}_debug/timing_snapshot.json",
            build_vp_stage_timing_breakdown(state.timing_log),
            timing_metadata={"round": state.current_round, "stage": "timing_snapshot"},
        )

        if hard_satisfied:
            last_persistent_stuck_signature = None
            iteration_log["stop_reason"] = "gating_predicates_satisfied"
            vp_iteration_logs.append(iteration_log)
            break

        if persistent_stuck_predicate_ids:
            current_signature = tuple(persistent_stuck_predicate_ids)
            if current_signature != last_persistent_stuck_signature:
                blocking_predicates = summarize_blocking_in_loop_auditor_predicates(
                    visual_program,
                    report,
                    predicate_ids=current_signature,
                    stuck_streaks=in_loop_auditor_stuck_streaks,
                )
                early_audit_result = await run_in_loop_auditor(
                    state=state,
                    runtime=runtime,
                    original_prompt=state.original_prompt,
                    candidate_image=current_image,
                    blocking_predicates=blocking_predicates,
                    component_name=f"persistent_stuck_in_loop_auditor:round_{state.current_round}",
                )
                in_loop_audit_result = early_audit_result
                in_loop_auditor_call_count += 1 if early_audit_result.called else 0
                early_audit_payload = {
                    **early_audit_result.to_dict(),
                    "round": state.current_round,
                    "predicate_ids": list(current_signature),
                    "blocking_predicates": blocking_predicates,
                    "trigger_reason": (
                        "All remaining blocking hard predicates are in-loop-auditor-"
                        "eligible predicates that stayed unsatisfied for consecutive rounds."
                    ),
                }
                verification_artifact["persistent_stuck_in_loop_audit"] = (
                    early_audit_payload
                )
                iteration_log["persistent_stuck_in_loop_audit"] = early_audit_payload
                await _save_json_artifact(
                    state,
                    f"vp_persistent_stuck_in_loop_audit_round_{state.current_round}.json",
                    early_audit_payload,
                    timing_metadata={
                        "round": state.current_round,
                        "stage": "persistent_stuck_in_loop_audit",
                    },
                )
                last_persistent_stuck_signature = current_signature
                if early_audit_result.override_allowed:
                    hard_gate_auditor_override = {
                        "used": True,
                        "round": state.current_round,
                        "predicate_ids": list(current_signature),
                        "blocking_predicates": blocking_predicates,
                        "reason": early_audit_payload["trigger_reason"],
                        "in_loop_auditor_result": early_audit_result.to_dict(),
                    }
                    selected_best_round = state.current_round
                    iteration_log["stop_reason"] = (
                        "persistent_stuck_in_loop_auditor_passed"
                    )
                    vp_iteration_logs.append(iteration_log)
                    break
            else:
                iteration_log["persistent_stuck_in_loop_audit"] = {
                    "called": False,
                    "skip_reason": (
                        "Skipped duplicate persistent stuck-predicate auditor call for "
                        "the same blocking eligible-predicate signature."
                    ),
                    "round": state.current_round,
                    "predicate_ids": list(current_signature),
                }
        else:
            last_persistent_stuck_signature = None

        if state.current_round >= state.max_rounds:
            iteration_log["stop_reason"] = "max_rounds_reached"
            vp_iteration_logs.append(iteration_log)
            break

        selection_start = time.perf_counter()
        selection = action_selector.select_action(
            program=visual_program,
            verification_report=report,
            candidate_metadata=CandidateMetadata(
                candidate_id="primary_candidate",
                round_index=state.current_round,
                sample_index=1,
                current_prompt=current_prompt,
                action_history=tuple(action_history),
                metadata={
                    "current_rewrite_index": current_rewrite_index,
                    "current_rewrite_prompt": current_rewrite_prompt,
                    "current_scene_prompt_guidance": current_scene_prompt_metadata.get(
                        "scene_prompt_guidance",
                        [],
                    ),
                },
            ),
            active_predicate_ids=hard_ids,
        )
        record_duration(
            state.timing_log,
            "vp_action_selection_seconds",
            time.perf_counter() - selection_start,
            component="vp_pipeline:action_selection",
            count_key="action_selections",
            metadata={"round": state.current_round},
        )
        selected_resample_base_prompt = None
        selected_rewrite_index = None
        resample_rewrite_selection: dict[str, Any] = {
            "available_rewrite_count": len(visual_program.rewritten_prompts),
            "rewrite_selection_pool": "source_prompt_fallback",
            "used_rewrite_indices_before_selection": list(used_rewrite_indices),
            "used_rewrite_indices_after_selection": list(used_rewrite_indices),
            "candidate_rewrite_indices": [],
        }
        if selection.action_type == RefinementActionType.RESAMPLE:
            (
                selected_resample_base_prompt,
                selected_rewrite_index,
                resample_rewrite_selection,
            ) = select_resample_rewritten_prompt(
                program=visual_program,
                round_index=state.current_round,
                random_seed=state.image_seed,
                used_rewrite_indices=used_rewrite_indices,
            )

        current_action_scene_prompt_metadata: dict[str, Any] = {}
        if selection.action_type == RefinementActionType.RESAMPLE:
            execution_input_kind = "resample_prompt"
            resample_base_prompt = (
                selected_resample_base_prompt or visual_program.source_prompt
            )
            execution_input_text, current_action_scene_prompt_metadata = (
                compose_prompt_with_scene_guidance(
                    program=visual_program,
                    base_prompt=resample_base_prompt,
                )
            )
        else:
            execution_input_kind, execution_input_text = resolve_action_execution_input(
                program=visual_program,
                selection=selection,
            )
        action_payload = selection.to_dict()
        vp_action_selections.append(action_payload)
        iteration_log["action_selection"] = action_payload
        vp_iteration_logs.append(iteration_log)

        if selection.action_type == RefinementActionType.NO_OP:
            vp_iteration_logs[-1]["stop_reason"] = "selector_returned_no_op"
            break

        state.current_round += 1
        if selection.action_type == RefinementActionType.EDIT:
            assert execution_input_text is not None
            edit_prompt = execution_input_text
            current_prompt = edit_prompt
            (
                current_image,
                current_image_data_url,
                executor_output,
            ) = await _run_edit_action(
                state=state,
                prompt=edit_prompt,
                input_image=current_image,
                input_image_data_url=current_image_data_url,
                sample_index=1,
            )
            pending_action_record = ActionExecutionRecord(
                action_type=selection.action_type,
                status=ActionExecutionStatus.APPLIED,
                target_predicate_ids=selection.target_predicate_ids,
                target_object_ids=selection.action_parameters.target_object_ids,
                edit_operation=selection.action_parameters.edit_operation,
                metadata={
                    "executed_prompt": edit_prompt,
                    "reason": selection.reason,
                    "rewrite_index": current_rewrite_index,
                    "rewrite_prompt": current_rewrite_prompt,
                    "primary_object_id": selection.action_parameters.primary_object_id,
                    "expected_count": selection.action_parameters.expected_count,
                    "count_delta": selection.action_parameters.count_delta,
                    "attribute_name": selection.action_parameters.attribute_name,
                    "attribute_value": selection.action_parameters.attribute_value,
                    "action_target_object_id": selection.action_parameters.action_target_object_id,
                },
            )
        elif selection.action_type == RefinementActionType.RESAMPLE:
            assert execution_input_text is not None
            resample_prompt = execution_input_text
            current_prompt = resample_prompt
            (
                current_image,
                current_image_data_url,
                executor_output,
            ) = await _run_generation_action(
                state=state,
                prompt=resample_prompt,
                sample_index=1,
                generation_executor=generation_executor,
                generation_model_name=generation_model_name,
                force_new_seed=True,
            )
            current_rewrite_index = selected_rewrite_index
            current_rewrite_prompt = (
                selected_resample_base_prompt
                if selected_resample_base_prompt is not None
                else state.original_prompt
            )
            used_rewrite_indices = list(
                resample_rewrite_selection.get(
                    "used_rewrite_indices_after_selection", []
                )
            )
            current_scene_prompt_metadata = dict(current_action_scene_prompt_metadata)
            # A resample produces a fresh candidate image, so stuck-predicate history
            # should not carry over into the new trajectory.
            in_loop_auditor_stuck_streaks = {}
            last_persistent_stuck_signature = None
            pending_action_record = ActionExecutionRecord(
                action_type=selection.action_type,
                status=ActionExecutionStatus.APPLIED,
                target_predicate_ids=selection.target_predicate_ids,
                target_object_ids=selection.action_parameters.target_object_ids,
                edit_operation=None,
                metadata={
                    "executed_prompt": resample_prompt,
                    "reason": selection.reason,
                    "rewrite_index": current_rewrite_index,
                    "rewrite_prompt": current_rewrite_prompt,
                    **current_action_scene_prompt_metadata,
                },
            )
        else:
            raise ValueError(f"Unsupported action type: {selection.action_type}")

        executor_output["action_type"] = selection.action_type.value
        executor_output["action_reason"] = selection.reason
        if selection.action_type == RefinementActionType.RESAMPLE:
            executor_output["selected_rewritten_prompt_index"] = selected_rewrite_index
            executor_output["rewrite_selection_pool"] = resample_rewrite_selection.get(
                "rewrite_selection_pool"
            )
            executor_output.update(current_action_scene_prompt_metadata)
        executor_outputs.append([executor_output])
        round_best_images.append(current_image_data_url)

    if final_report is None:
        raise RuntimeError("VP pipeline ended without producing a verification report.")

    selected_best_round = state.current_round
    if hard_gate_auditor_override is None and not all_hard_satisfied:
        best_partial_selection_reason = (
            "max_rounds_reached_without_hard_satisfaction"
            if state.current_round >= state.max_rounds
            else "no_refinement_action_available"
        )
        best_partial_result = select_run_best_partial_candidate(
            candidates=run_partial_selection_candidates,
            selection_reason=best_partial_selection_reason,
        )
        selected_candidate = best_partial_result["selected_candidate"]
        best_partial_selection_payload = best_partial_result["selection_payload"]
        selected_best_round = selected_candidate["round_index"]
        final_report = selected_candidate["report"]
        current_image_data_url = selected_candidate["image_data_url"]
        if selected_best_round != state.current_round:
            from graph.utils import data_url_to_image

            current_image = data_url_to_image(
                current_image_data_url,
                timing_log=state.timing_log,
                timing_component="vp_pipeline:best_partial_decode",
            )
        all_hard_satisfied = bool(selected_candidate["hard_match"]["all_satisfied"])
        all_soft_satisfied = predicate_group_satisfied(
            final_report,
            predicate_ids=tuple(selected_candidate["soft_ids"]),
            accept_uncertain=accept_uncertain,
        )
        run_record["best_partial_selection"] = summarize_best_partial_selection_payload(
            best_partial_selection_payload
        )
        await _save_json_artifact(
            state,
            "vp_best_partial_selection.json",
            {
                **best_partial_selection_payload,
            },
            timing_metadata={"stage": "best_partial_selection"},
        )

    refinement_vlm_call_count = _vp_refinement_vlm_call_count(state)
    print(
        "vp_pipeline | refinement loop called any VLM inside refinement: "
        f"{refinement_vlm_call_count > 0} | call_count: "
        f"{refinement_vlm_call_count}"
    )
    run_record.update(
        {
            "final_round": state.current_round,
            "selected_best_round": selected_best_round,
            "all_hard_predicates_satisfied": all_hard_satisfied,
            "hard_gate_auditor_override": hard_gate_auditor_override,
            "in_loop_auditor": in_loop_audit_result.to_dict(),
            "accepted": bool(all_hard_satisfied or hard_gate_auditor_override),
        }
    )

    run_summary = run_record
    await _save_json_artifact(
        state,
        "vp_run_summary.json",
        run_summary,
        timing_metadata={"stage": "run_summary"},
    )

    in_loop_auditor_output = in_loop_audit_result.to_dict()
    print(
        "vp_pipeline | in-loop auditor total call count for this candidate path: "
        f"{in_loop_auditor_call_count}"
    )
    print(
        "vp_pipeline | in-loop auditor result:\n"
        + json.dumps(in_loop_auditor_output, ensure_ascii=False, indent=2)
    )
    await _save_json_artifact(
        state,
        "in_loop_auditor.json",
        in_loop_auditor_output,
        timing_metadata={"stage": "in_loop_auditor"},
    )
    if current_image is not None:
        await _save_image_artifact(
            state,
            "output_selected.jpg",
            current_image,
            timing_metadata={
                "stage": "final_selected_output",
                "selected_best_round": selected_best_round,
            },
        )
        print(
            "vp_pipeline | saved final selected image artifact to output_selected.jpg "
            f"(round {selected_best_round})."
        )

    record_duration(
        state.timing_log,
        "node_vp_pipeline_seconds",
        time.perf_counter() - node_start,
        component="vp_pipeline",
    )
    state.timing_log = finalize_timing_log(state.timing_log)
    timing_breakdown = build_vp_stage_timing_breakdown(state.timing_log)
    refinement_vlm_call_count = _vp_refinement_vlm_call_count(state)
    print(
        "vp_pipeline | stage timing breakdown:\n"
        + json.dumps(timing_breakdown, ensure_ascii=False, indent=2)
    )
    pipeline_summary = build_vp_pipeline_summary(
        visual_program_payload=visual_program_payload,
        hard_ids=hard_ids,
        soft_ids=soft_ids,
        iteration_logs=vp_iteration_logs,
        in_loop_auditor_output=in_loop_auditor_output,
        in_loop_auditor_call_count=in_loop_auditor_call_count,
        all_hard_predicates_satisfied=all_hard_satisfied,
        all_soft_predicates_satisfied=all_soft_satisfied,
        refinement_loop_vlm_calls=refinement_vlm_call_count,
        refinement_loop_called_any_vlm=refinement_vlm_call_count > 0,
        timing_breakdown=timing_breakdown,
        verifier_thresholds=verifier_thresholds_payload,
        relation_scoring_config=relation_scoring_config_payload,
        rewritten_prompt_generator_call_count=rewritten_prompt_generator_call_count,
        rewritten_prompt_count=len(visual_program_payload.get("rewritten_prompts", [])),
        hard_gate_auditor_override=hard_gate_auditor_override,
        run_summary=run_summary,
    )
    print(
        "vp_pipeline | pipeline summary:\n"
        + json.dumps(pipeline_summary, ensure_ascii=False, indent=2)
    )
    await _save_json_artifact(
        state,
        "vp_pipeline_summary.json",
        pipeline_summary,
        timing_metadata={"stage": "pipeline_summary"},
    )
    await _save_json_artifact(
        state,
        "vp_timing_breakdown.json",
        timing_breakdown,
        timing_metadata={"stage": "timing_breakdown"},
    )
    finalized_timing_log = snapshot_timing_log(state.timing_log)
    await _save_json_artifact(
        state,
        "timing_log.json",
        finalized_timing_log,
        timing_metadata={"stage": "timing_log"},
    )

    update_dict = {
        "executor_outputs": executor_outputs,
        "round_best_images": round_best_images,
        "output_images": [current_image_data_url],
        "parsed_visual_program": visual_program_payload,
        "vp_visual_program_source": visual_program_source_kind,
        "vp_parser_normalization_report": normalization_report_payload,
        "vp_parser_review_report": review_report_payload,
        "vp_rewritten_prompt_generator_reports": rewritten_prompt_generator_reports,
        "vp_rewritten_prompt_generator_call_count": (
            rewritten_prompt_generator_call_count
        ),
        "vp_hard_predicate_ids": list(hard_ids),
        "vp_soft_predicate_ids": list(soft_ids),
        "vp_critical_predicate_ids": list(hard_ids),
        "vp_verification_reports": vp_verification_reports,
        "vp_action_selections": vp_action_selections,
        "vp_iteration_logs": vp_iteration_logs,
        "vp_stage_timing_breakdown": timing_breakdown,
        "vp_pipeline_summary": pipeline_summary,
        "vp_verifier_thresholds": verifier_thresholds_payload,
        "vp_relation_scoring_config": relation_scoring_config_payload,
        "in_loop_auditor_output": in_loop_auditor_output,
        "in_loop_auditor_call_count": in_loop_auditor_call_count,
        "final_candidate_accepted": pipeline_summary["final_candidate_accepted"],
        "refinement_loop_vlm_calls": pipeline_summary["refinement_loop_vlm_calls"],
        "refinement_loop_called_any_vlm": pipeline_summary[
            "refinement_loop_called_any_vlm"
        ],
        "all_hard_predicates_satisfied": all_hard_satisfied,
        "all_soft_predicates_satisfied": all_soft_satisfied,
        "all_critical_predicates_satisfied": all_hard_satisfied,
        "best_round": selected_best_round,
        "best_round_sample": 1,
        "current_round": state.current_round,
        "timing_log": finalized_timing_log,
    }

    return Command(update=update_dict, goto="__end__")
