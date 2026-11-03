"""Utilities for ranking partially-satisfied VP candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from graph.visual_program import PredicateCategory, PredicateState
from graph.visual_program_verifier import VisualProgramVerificationReport

PARTIAL_SELECTION_CATEGORY_PRIORITY: tuple[str, ...] = (
    PredicateCategory.OBJECT_COUNT.value,
    PredicateCategory.SPATIAL_RELATION.value,
    PredicateCategory.TEXT_CONSTRAINT.value,
    PredicateCategory.EXCLUSION.value,
    PredicateCategory.ATTRIBUTE_CONSTRAINT.value,
    PredicateCategory.GLOBAL_SCENE_CONSTRAINT.value,
)
_PARTIAL_SELECTION_CATEGORY_RANK = {
    category: index
    for index, category in enumerate(PARTIAL_SELECTION_CATEGORY_PRIORITY)
}
_OTHER_CATEGORY_LABEL = "__other__"


@dataclass(frozen=True)
class PartialMatchPrioritySummary:
    """Summarize which predicate families are already satisfied."""

    category_order: tuple[str, ...]
    satisfied_priority_counts: tuple[int, ...]


def summarize_partial_match_priority(
    report: VisualProgramVerificationReport,
    *,
    predicate_ids: Sequence[str],
    accept_uncertain: bool = False,
) -> PartialMatchPrioritySummary:
    """Summarize satisfied predicate families using a human-oriented priority order."""
    category_count = len(PARTIAL_SELECTION_CATEGORY_PRIORITY) + 1
    satisfied_priority_counts = [0] * category_count

    for predicate_id in predicate_ids:
        result = report.get_result(predicate_id)
        bucket = _PARTIAL_SELECTION_CATEGORY_RANK.get(
            result.category.value,
            len(PARTIAL_SELECTION_CATEGORY_PRIORITY),
        )

        is_satisfied = result.state == PredicateState.SATISFIED
        if accept_uncertain and result.state == PredicateState.UNCERTAIN:
            is_satisfied = True

        if is_satisfied:
            satisfied_priority_counts[bucket] += 1
            continue

    return PartialMatchPrioritySummary(
        category_order=tuple(
            list(PARTIAL_SELECTION_CATEGORY_PRIORITY) + [_OTHER_CATEGORY_LABEL]
        ),
        satisfied_priority_counts=tuple(satisfied_priority_counts),
    )


def build_partial_match_rank_key(
    *,
    priority_summary: PartialMatchPrioritySummary,
    satisfied_count: int,
    unsatisfied_score: float,
    round_index: int,
) -> tuple[tuple[int, ...], int, float, int]:
    """Rank partial candidates by high-priority predicate coverage before raw scores."""
    return (
        tuple(priority_summary.satisfied_priority_counts),
        int(satisfied_count),
        float(unsatisfied_score),
        -int(round_index),
    )
