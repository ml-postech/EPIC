"""Rule-based action selection for visual-program refinement.

This module chooses the next refinement action from predicate-level verifier
results without calling a VLM. It is intentionally lightweight and easy to
extend: the initial implementation uses explicit predicate-type rules and
simple action history checks.

Examples:
    >>> from graph.visual_program import CountPredicate, ObjectSpec, VisualProgram
    >>> from graph.visual_program_verifier import PredicateVerificationResult, VisualProgramVerificationReport
    >>> cat = ObjectSpec.named("cat", object_id="cat_1")
    >>> program = VisualProgram.from_predicates(
    ...     source_prompt="a cat",
    ...     objects=[cat],
    ...     predicates=[CountPredicate.exactly(cat, 1)],
    ... )
    >>> report = VisualProgramVerificationReport(
    ...     program_id=program.program_id,
    ...     results=(
    ...         PredicateVerificationResult(
    ...             predicate_id="count:cat_1:exactly:1",
    ...             category=program.predicates[0].category,
    ...             state=program.initial_state().statuses["count:cat_1:exactly:1"].state.VIOLATED,
    ...             score=0.1,
    ...             confidence=0.1,
    ...             evidence_summary="No cat candidate found.",
    ...             evidence={"object_id": "cat_1"},
    ...         ),
    ...     ),
    ...     program_state=program.initial_state(),
    ...     cache_summary={},
    ... )
    >>> selection = VisualProgramActionSelector().select_action(
    ...     program=program,
    ...     verification_report=report,
    ...     candidate_metadata=CandidateMetadata(candidate_id="sample-1"),
    ... )
    >>> selection.action_type.value
    'edit'
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Sequence


from graph.visual_program import (
    AttributePredicate,
    CountComparator,
    CountPredicate,
    ExclusionPredicate,
    GlobalScenePredicate,
    Predicate,
    PredicateCategory,
    PredicateState,
    SpatialRelationPredicate,
    TextPredicate,
    VisualProgram,
)
from graph.visual_program_support import normalize_attribute_name
from graph.visual_program_verifier import (
    PredicateVerificationResult,
    VisualProgramVerificationReport,
)


def _serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            item.name: _serialize(getattr(value, item.name)) for item in fields(value)
        }
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    return value


def _predicate_family_priority(predicate: Predicate) -> int:
    """Rank failures by how localized and actionable their fixes usually are."""
    if isinstance(predicate, CountPredicate):
        return 0
    if isinstance(predicate, SpatialRelationPredicate):
        return 1
    if isinstance(predicate, AttributePredicate):
        return 2
    if isinstance(predicate, ExclusionPredicate):
        return 3
    if isinstance(predicate, GlobalScenePredicate):
        return 4
    if isinstance(predicate, TextPredicate):
        return 5
    return 6


class RefinementActionType(str, Enum):
    """Enumerate supported high-level refinement actions."""

    NO_OP = "no_op"
    EDIT = "edit"
    RESAMPLE = "resample"


class EditOperationType(str, Enum):
    """Enumerate deterministic edit intents."""

    ADD_OBJECT = "add_object"
    REMOVE_OBJECT = "remove_object"
    CHANGE_ATTRIBUTE = "change_attribute"
    CHANGE_GLOBAL_SCENE = "change_global_scene"


class ResampleStrategy(str, Enum):
    """Enumerate initial resampling strategies."""

    GLOBAL_RESAMPLE = "global_resample"
    RESAMPLE_FOR_MISSING_OBJECTS = "resample_for_missing_objects"
    RESAMPLE_FOR_EXCESS_OBJECTS = "resample_for_excess_objects"
    RESAMPLE_FOR_LAYOUT = "resample_for_layout"
    RESAMPLE_FOR_ACTION = "resample_for_action"
    RESAMPLE_FOR_ATTRIBUTE = "resample_for_attribute"


class ActionExecutionStatus(str, Enum):
    """Track how a previously selected action turned out."""

    PROPOSED = "proposed"
    APPLIED = "applied"
    FAILED = "failed"
    SUCCEEDED = "succeeded"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class EditActionParameters:
    """Structured parameters for an edit action."""

    edit_operation: EditOperationType
    target_object_ids: tuple[str, ...]
    target_predicate_ids: tuple[str, ...]
    primary_object_id: Optional[str] = None
    expected_count: Optional[int] = None
    count_delta: Optional[int] = None
    attribute_name: Optional[str] = None
    attribute_value: Optional[Any] = None
    action_target_object_id: Optional[str] = None
    template_key: Optional[str] = None
    template_inputs: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the action parameters into plain Python data."""
        return _serialize(self)


@dataclass(frozen=True)
class ResampleActionParameters:
    """Structured parameters for a resample action."""

    strategy: ResampleStrategy
    target_predicate_ids: tuple[str, ...]
    target_object_ids: tuple[str, ...] = ()
    template_key: Optional[str] = None
    template_inputs: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the action parameters into plain Python data."""
        return _serialize(self)


@dataclass(frozen=True)
class ActionSelection:
    """Decision output from the action selector."""

    action_type: RefinementActionType
    action_parameters: Optional[EditActionParameters | ResampleActionParameters] = None
    reason: str = ""
    target_predicate_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the action selection into plain Python data."""
        return _serialize(self)


@dataclass(frozen=True)
class ActionExecutionRecord:
    """One historical action attempt for a candidate image."""

    action_type: RefinementActionType
    status: ActionExecutionStatus
    target_predicate_ids: tuple[str, ...] = ()
    target_object_ids: tuple[str, ...] = ()
    edit_operation: Optional[EditOperationType] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def matches(
        self,
        *,
        action_type: RefinementActionType,
        target_predicate_ids: Sequence[str] = (),
        target_object_ids: Sequence[str] = (),
        edit_operation: Optional[EditOperationType] = None,
    ) -> bool:
        """Return whether this historical record targets the same failure surface."""
        if self.action_type != action_type:
            return False
        if edit_operation is not None and self.edit_operation != edit_operation:
            return False
        if target_predicate_ids and not set(self.target_predicate_ids).intersection(
            target_predicate_ids
        ):
            return False
        if target_object_ids and not set(self.target_object_ids).intersection(
            target_object_ids
        ):
            return False
        return True


@dataclass(frozen=True)
class CandidateMetadata:
    """Current candidate context available to the action selector."""

    candidate_id: str
    round_index: Optional[int] = None
    sample_index: Optional[int] = None
    score: Optional[float] = None
    current_prompt: Optional[str] = None
    action_history: tuple[ActionExecutionRecord, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionSelectorContext:
    """Bundle selector inputs for easy future expansion."""

    program: VisualProgram
    verification_report: VisualProgramVerificationReport
    candidate_metadata: CandidateMetadata


@dataclass(frozen=True)
class FailedPredicate:
    """A failed or unresolved predicate paired with its typed spec."""

    predicate: Predicate
    result: PredicateVerificationResult

    @property
    def priority(self) -> tuple[int, int, float]:
        """Priority ordering: severity, predicate family, then low score first."""
        severity_order = {
            PredicateState.CONTRADICTION: 0,
            PredicateState.VIOLATED: 1,
            PredicateState.UNCERTAIN: 2,
        }
        score = self.result.score if self.result.score is not None else 1.0
        family_order = _predicate_family_priority(self.predicate)
        return (severity_order.get(self.result.state, 3), family_order, score)


@dataclass
class VisualProgramActionSelector:
    """Rule-based selector for the next VP refinement action."""

    edit_retry_score_delta: float = 0.12

    def select_action(
        self,
        *,
        program: VisualProgram,
        verification_report: VisualProgramVerificationReport,
        candidate_metadata: CandidateMetadata,
        active_predicate_ids: Optional[Sequence[str]] = None,
    ) -> ActionSelection:
        """Select the next action from predicate-level verification results."""
        context = ActionSelectorContext(
            program=program,
            verification_report=verification_report,
            candidate_metadata=candidate_metadata,
        )

        failures = self._collect_failures(
            context,
            active_predicate_ids=active_predicate_ids,
        )
        if not failures:
            return ActionSelection(
                action_type=RefinementActionType.NO_OP,
                action_parameters=None,
                reason="All currently supported predicates are satisfied; no refinement action is needed.",
                target_predicate_ids=(),
            )

        prioritized = sorted(failures, key=lambda item: item.priority)
        return self._select_first_available(context, prioritized)

    def _select_first_available(
        self,
        context: ActionSelectorContext,
        failures: Sequence[FailedPredicate],
    ) -> ActionSelection:
        for failed_predicate in failures:
            selection = self._select_for_failure(context, failed_predicate)
            if selection is not None:
                return selection
        return self._global_resample_for_failures(failures)

    def _global_resample_for_failures(
        self,
        failures: Sequence[FailedPredicate],
    ) -> ActionSelection:
        failed_ids = tuple(item.predicate.predicate_id for item in failures)
        return ActionSelection(
            action_type=RefinementActionType.RESAMPLE,
            action_parameters=ResampleActionParameters(
                strategy=ResampleStrategy.GLOBAL_RESAMPLE,
                target_predicate_ids=failed_ids,
                template_key="global_resample",
                template_inputs={"failed_predicate_ids": failed_ids},
            ),
            reason="No specialized rule matched the remaining failed predicates, so the selector falls back to global resampling.",
            target_predicate_ids=failed_ids,
        )

    def _collect_failures(
        self,
        context: ActionSelectorContext,
        *,
        active_predicate_ids: Optional[Sequence[str]] = None,
    ) -> list[FailedPredicate]:
        allowed_ids = (
            None if active_predicate_ids is None else set(active_predicate_ids)
        )
        failures = []
        for predicate in context.program.predicates:
            if allowed_ids is not None and predicate.predicate_id not in allowed_ids:
                continue
            result = self._result_or_placeholder(
                context.verification_report,
                predicate_id=predicate.predicate_id,
                category=predicate.category,
            )
            if result.state in {
                PredicateState.VIOLATED,
                PredicateState.UNCERTAIN,
                PredicateState.CONTRADICTION,
            }:
                failures.append(FailedPredicate(predicate=predicate, result=result))
        return failures

    def _select_for_failure(
        self,
        context: ActionSelectorContext,
        failed_predicate: FailedPredicate,
    ) -> Optional[ActionSelection]:
        predicate = failed_predicate.predicate
        if isinstance(predicate, ExclusionPredicate):
            return self._select_exclusion_action(context, failed_predicate)
        if isinstance(predicate, CountPredicate):
            return self._select_count_action(context, failed_predicate)
        if isinstance(predicate, SpatialRelationPredicate):
            return self._select_relation_action(context, failed_predicate)
        if isinstance(predicate, AttributePredicate):
            return self._select_attribute_action(context, failed_predicate)
        if isinstance(predicate, GlobalScenePredicate):
            return self._select_global_scene_action(context, failed_predicate)
        return None

    def _select_exclusion_action(
        self,
        context: ActionSelectorContext,
        failed_predicate: FailedPredicate,
    ) -> ActionSelection:
        predicate = failed_predicate.predicate
        assert isinstance(predicate, ExclusionPredicate)

        predicate_ids = (predicate.predicate_id,)
        object_ids = (predicate.target.object_id,)
        current_rewrite_attempts = self._current_rewrite_edit_attempts(
            context,
            target_predicate_ids=predicate_ids,
            target_object_ids=object_ids,
            edit_operation=EditOperationType.REMOVE_OBJECT,
        )
        if current_rewrite_attempts == 0:
            reason = f"Exclusion failed for {predicate.target.object_id}, so the selector tries one removal edit for the current rewritten image."
            return ActionSelection(
                action_type=RefinementActionType.EDIT,
                action_parameters=EditActionParameters(
                    edit_operation=EditOperationType.REMOVE_OBJECT,
                    target_object_ids=object_ids,
                    target_predicate_ids=predicate_ids,
                    primary_object_id=predicate.target.object_id,
                    template_key="remove_object",
                    template_inputs={"object_id": predicate.target.object_id},
                ),
                reason=reason,
                target_predicate_ids=predicate_ids,
            )

        return ActionSelection(
            action_type=RefinementActionType.RESAMPLE,
            action_parameters=ResampleActionParameters(
                strategy=ResampleStrategy.RESAMPLE_FOR_EXCESS_OBJECTS,
                target_predicate_ids=predicate_ids,
                target_object_ids=object_ids,
                template_key="resample_for_exclusion_failure",
                template_inputs={"object_id": predicate.target.object_id},
            ),
            reason=(
                f"Exclusion still fails for {predicate.target.object_id} after this rewritten image already used its one removal edit attempt, so the selector escalates to resampling."
            ),
            target_predicate_ids=predicate_ids,
        )

    def _select_count_action(
        self,
        context: ActionSelectorContext,
        failed_predicate: FailedPredicate,
    ) -> ActionSelection:
        predicate = failed_predicate.predicate
        assert isinstance(predicate, CountPredicate)

        object_id = predicate.target.object_id
        current_count = self._infer_current_count(failed_predicate.result)
        predicate_ids = self._related_object_failure_ids(
            context.program,
            object_id,
            predicate_classes=(CountPredicate,),
        )
        object_ids = (object_id,)
        weak_count = self._infer_weak_count(failed_predicate.result)
        if (
            predicate.comparator == CountComparator.EXACTLY
            and current_count == predicate.expected_count
            and weak_count is not None
            and weak_count > predicate.expected_count
        ):
            weak_overcount_delta = weak_count - predicate.expected_count
            if not self._all_required_positive_objects_have_minimum_presence(context):
                return ActionSelection(
                    action_type=RefinementActionType.RESAMPLE,
                    action_parameters=ResampleActionParameters(
                        strategy=ResampleStrategy.RESAMPLE_FOR_MISSING_OBJECTS,
                        target_predicate_ids=predicate_ids,
                        target_object_ids=object_ids,
                        template_key="resample_before_remove_until_all_objects_present",
                        template_inputs={
                            "object_id": object_id,
                            "expected_count": predicate.expected_count,
                            "observed_count": weak_count,
                            "strong_count": current_count,
                        },
                    ),
                    reason=(
                        f"Weak count evidence for {object_id} suggests extra plausible instances, but removal is deferred until every required object category has at least one visible instance."
                    ),
                    target_predicate_ids=predicate_ids,
                )
            if weak_overcount_delta != 1:
                return ActionSelection(
                    action_type=RefinementActionType.RESAMPLE,
                    action_parameters=ResampleActionParameters(
                        strategy=ResampleStrategy.RESAMPLE_FOR_EXCESS_OBJECTS,
                        target_predicate_ids=predicate_ids,
                        target_object_ids=object_ids,
                        template_key="resample_for_large_overcount",
                        template_inputs={
                            "object_id": object_id,
                            "expected_count": predicate.expected_count,
                            "observed_count": weak_count,
                            "strong_count": current_count,
                            "count_delta": weak_overcount_delta,
                        },
                    ),
                    reason=(
                        f"Strong count for {object_id} matches the target, but weak candidates suggest {weak_overcount_delta} extra plausible instances, so the selector resamples instead of trying to remove multiple uncertain instances."
                    ),
                    target_predicate_ids=predicate_ids,
                )
            current_rewrite_attempts = self._current_rewrite_edit_attempts(
                context,
                target_predicate_ids=predicate_ids,
                target_object_ids=object_ids,
                edit_operation=EditOperationType.REMOVE_OBJECT,
            )
            if current_rewrite_attempts == 0:
                return ActionSelection(
                    action_type=RefinementActionType.EDIT,
                    action_parameters=EditActionParameters(
                        edit_operation=EditOperationType.REMOVE_OBJECT,
                        target_object_ids=object_ids,
                        target_predicate_ids=predicate_ids,
                        primary_object_id=object_id,
                        expected_count=predicate.expected_count,
                        count_delta=weak_overcount_delta,
                        template_key="remove_object_to_reach_count",
                        template_inputs={
                            "object_id": object_id,
                            "expected_count": predicate.expected_count,
                            "observed_count": weak_count,
                            "strong_count": current_count,
                            "count_delta": weak_overcount_delta,
                        },
                    ),
                    reason=(
                        f"Strong count for {object_id} matches the target, but weak candidates suggest one extra plausible instance, so the selector tries one careful removal edit."
                    ),
                    target_predicate_ids=predicate_ids,
                )

            return ActionSelection(
                action_type=RefinementActionType.RESAMPLE,
                action_parameters=ResampleActionParameters(
                    strategy=ResampleStrategy.RESAMPLE_FOR_EXCESS_OBJECTS,
                    target_predicate_ids=predicate_ids,
                    target_object_ids=object_ids,
                    template_key="resample_for_overcount",
                    template_inputs={
                        "object_id": object_id,
                        "expected_count": predicate.expected_count,
                        "observed_count": weak_count,
                        "strong_count": current_count,
                    },
                ),
                reason=(
                    f"Weak count evidence for {object_id} still suggests an extra plausible instance after a removal edit attempt, so the selector escalates to resampling."
                ),
                target_predicate_ids=predicate_ids,
            )

        if (
            predicate.comparator == CountComparator.EXACTLY
            and current_count is not None
            and current_count < predicate.expected_count
            and weak_count is not None
            and weak_count < predicate.expected_count
        ):
            count_delta = predicate.expected_count - current_count
            if not self._all_required_positive_objects_have_minimum_presence(context):
                return ActionSelection(
                    action_type=RefinementActionType.RESAMPLE,
                    action_parameters=ResampleActionParameters(
                        strategy=ResampleStrategy.RESAMPLE_FOR_MISSING_OBJECTS,
                        target_predicate_ids=predicate_ids,
                        target_object_ids=object_ids,
                        template_key="resample_before_add_until_all_objects_present",
                        template_inputs={
                            "object_id": object_id,
                            "expected_count": predicate.expected_count,
                            "observed_count": current_count,
                        },
                    ),
                    reason=(
                        f"Add-object editing for {object_id} is deferred until every required object category has at least one visible instance, so the selector resamples first."
                    ),
                    target_predicate_ids=predicate_ids,
                )
            current_rewrite_attempts = self._current_rewrite_edit_attempts(
                context,
                target_predicate_ids=predicate_ids,
                target_object_ids=object_ids,
                edit_operation=EditOperationType.ADD_OBJECT,
            )
            retry_reason = self._retry_add_object_reason_in_current_rewrite(
                context,
                target_predicate_ids=predicate_ids,
                target_object_ids=object_ids,
                current_count=current_count,
                expected_count=predicate.expected_count,
            )
            if current_rewrite_attempts == 0 or retry_reason is not None:
                if current_rewrite_attempts == 0:
                    reason = f"Exact count for {object_id} is below target ({current_count} < {predicate.expected_count}), so the selector chooses one object-addition edit for the current rewritten image."
                else:
                    reason = f"Exact count for {object_id} still misses the target, but {retry_reason}, so the selector retries object addition once more before escalating to resampling."
                return ActionSelection(
                    action_type=RefinementActionType.EDIT,
                    action_parameters=EditActionParameters(
                        edit_operation=EditOperationType.ADD_OBJECT,
                        target_object_ids=object_ids,
                        target_predicate_ids=predicate_ids,
                        primary_object_id=object_id,
                        expected_count=predicate.expected_count,
                        count_delta=count_delta,
                        template_key="add_object_to_reach_count",
                        template_inputs={
                            "object_id": object_id,
                            "expected_count": predicate.expected_count,
                            "count_delta": count_delta,
                        },
                    ),
                    reason=reason,
                    target_predicate_ids=predicate_ids,
                )

            return ActionSelection(
                action_type=RefinementActionType.RESAMPLE,
                action_parameters=ResampleActionParameters(
                    strategy=ResampleStrategy.RESAMPLE_FOR_MISSING_OBJECTS,
                    target_predicate_ids=predicate_ids,
                    target_object_ids=object_ids,
                    template_key="resample_for_undercount",
                    template_inputs={
                        "object_id": object_id,
                        "expected_count": predicate.expected_count,
                        "observed_count": current_count,
                    },
                ),
                reason=(
                    f"Count for {object_id} is still below target without local count-gap progress in the current rewritten image, so the selector escalates to resampling."
                ),
                target_predicate_ids=predicate_ids,
            )

        if (
            predicate.comparator == CountComparator.EXACTLY
            and current_count is not None
            and current_count > predicate.expected_count
        ):
            overcount_delta = current_count - predicate.expected_count
            if not self._all_required_positive_objects_have_minimum_presence(context):
                return ActionSelection(
                    action_type=RefinementActionType.RESAMPLE,
                    action_parameters=ResampleActionParameters(
                        strategy=ResampleStrategy.RESAMPLE_FOR_MISSING_OBJECTS,
                        target_predicate_ids=predicate_ids,
                        target_object_ids=object_ids,
                        template_key="resample_before_remove_until_all_objects_present",
                        template_inputs={
                            "object_id": object_id,
                            "expected_count": predicate.expected_count,
                            "observed_count": current_count,
                        },
                    ),
                    reason=(
                        f"Remove-object editing for {object_id} is deferred until every required object category has at least one visible instance, so the selector resamples first."
                    ),
                    target_predicate_ids=predicate_ids,
                )
            if overcount_delta != 1:
                return ActionSelection(
                    action_type=RefinementActionType.RESAMPLE,
                    action_parameters=ResampleActionParameters(
                        strategy=ResampleStrategy.RESAMPLE_FOR_EXCESS_OBJECTS,
                        target_predicate_ids=predicate_ids,
                        target_object_ids=object_ids,
                        template_key="resample_for_large_overcount",
                        template_inputs={
                            "object_id": object_id,
                            "expected_count": predicate.expected_count,
                            "observed_count": current_count,
                            "count_delta": overcount_delta,
                        },
                    ),
                    reason=(
                        f"Remove-object editing for {object_id} is only attempted when the scene exceeds the target count by exactly one instance, so the selector resamples instead of trying to remove {overcount_delta} instances."
                    ),
                    target_predicate_ids=predicate_ids,
                )
            current_rewrite_attempts = self._current_rewrite_edit_attempts(
                context,
                target_predicate_ids=predicate_ids,
                target_object_ids=object_ids,
                edit_operation=EditOperationType.REMOVE_OBJECT,
            )
            if current_rewrite_attempts == 0:
                reason = f"Exact count for {object_id} is above target ({current_count} > {predicate.expected_count}), so the selector chooses one removal edit for the current rewritten image."
                return ActionSelection(
                    action_type=RefinementActionType.EDIT,
                    action_parameters=EditActionParameters(
                        edit_operation=EditOperationType.REMOVE_OBJECT,
                        target_object_ids=object_ids,
                        target_predicate_ids=predicate_ids,
                        primary_object_id=object_id,
                        expected_count=predicate.expected_count,
                        count_delta=overcount_delta,
                        template_key="remove_object_to_reach_count",
                        template_inputs={
                            "object_id": object_id,
                            "expected_count": predicate.expected_count,
                            "count_delta": overcount_delta,
                        },
                    ),
                    reason=reason,
                    target_predicate_ids=predicate_ids,
                )

            return ActionSelection(
                action_type=RefinementActionType.RESAMPLE,
                action_parameters=ResampleActionParameters(
                    strategy=ResampleStrategy.RESAMPLE_FOR_EXCESS_OBJECTS,
                    target_predicate_ids=predicate_ids,
                    target_object_ids=object_ids,
                    template_key="resample_for_overcount",
                    template_inputs={
                        "object_id": object_id,
                        "expected_count": predicate.expected_count,
                        "observed_count": current_count,
                    },
                ),
                reason=(
                    f"Count for {object_id} is still above target after the current rewritten image already used its one removal edit attempt, so the selector escalates to resampling."
                ),
                target_predicate_ids=predicate_ids,
            )

        return ActionSelection(
            action_type=RefinementActionType.RESAMPLE,
            action_parameters=ResampleActionParameters(
                strategy=ResampleStrategy.GLOBAL_RESAMPLE,
                target_predicate_ids=predicate_ids,
                target_object_ids=object_ids,
                template_key="global_resample_for_count",
                template_inputs={
                    "object_id": object_id,
                    "expected_count": predicate.expected_count,
                },
            ),
            reason=(
                f"Count failure for {object_id} could not be resolved into a clear add/remove edit, so the selector falls back to resampling."
            ),
            target_predicate_ids=predicate_ids,
        )

    def _select_relation_action(
        self,
        context: ActionSelectorContext,
        failed_predicate: FailedPredicate,
    ) -> Optional[ActionSelection]:
        predicate = failed_predicate.predicate
        assert isinstance(predicate, SpatialRelationPredicate)

        predicate_ids = (predicate.predicate_id,)
        object_ids = (predicate.subject.object_id, predicate.reference.object_id)
        if not self._objects_have_minimum_presence(context, object_ids):
            return None

        reason = f"Spatial relation {predicate.relation} failed between {predicate.subject.object_id} and {predicate.reference.object_id}, so the selector uses layout-focused resampling instead of a local reposition edit."
        return ActionSelection(
            action_type=RefinementActionType.RESAMPLE,
            action_parameters=ResampleActionParameters(
                strategy=ResampleStrategy.RESAMPLE_FOR_LAYOUT,
                target_predicate_ids=predicate_ids,
                target_object_ids=object_ids,
                template_key="resample_for_relation",
                template_inputs={
                    "subject_object_id": predicate.subject.object_id,
                    "reference_object_id": predicate.reference.object_id,
                    "relation": predicate.relation,
                },
            ),
            reason=reason,
            target_predicate_ids=predicate_ids,
        )

    def _select_attribute_action(
        self,
        context: ActionSelectorContext,
        failed_predicate: FailedPredicate,
    ) -> Optional[ActionSelection]:
        predicate = failed_predicate.predicate
        assert isinstance(predicate, AttributePredicate)

        predicate_ids = (predicate.predicate_id,)
        action_target_object_id = (
            None
            if predicate.action_target is None
            else predicate.action_target.object_id
        )
        object_ids = (
            (predicate.target.object_id,)
            if action_target_object_id is None
            else (predicate.target.object_id, action_target_object_id)
        )
        normalized_attribute = normalize_attribute_name(predicate.attribute_name)
        is_action_attribute = normalized_attribute == "action"
        if not self._objects_have_stable_positive_constraints(context, object_ids):
            return None
        current_rewrite_attempts = self._current_rewrite_edit_attempts(
            context,
            target_predicate_ids=predicate_ids,
            target_object_ids=object_ids,
            edit_operation=EditOperationType.CHANGE_ATTRIBUTE,
        )
        retry_reason = self._predicate_progress_retry_reason_in_current_rewrite(
            context,
            target_predicate_ids=predicate_ids,
            target_object_ids=object_ids,
            edit_operation=EditOperationType.CHANGE_ATTRIBUTE,
        )
        if current_rewrite_attempts == 0 or retry_reason is not None:
            if current_rewrite_attempts == 0:
                if is_action_attribute:
                    if action_target_object_id is None:
                        reason = (
                            f"Action {predicate.expected_value} failed for {predicate.target.object_id}, "
                            "so the selector chooses one action-focused edit for the current rewritten image."
                        )
                    else:
                        reason = (
                            f"Action {predicate.expected_value} from {predicate.target.object_id} "
                            f"toward {action_target_object_id} failed, so the selector chooses one action-focused edit for the current rewritten image."
                        )
                else:
                    reason = f"Attribute {predicate.attribute_name} failed for {predicate.target.object_id}, so the selector chooses one attribute edit for the current rewritten image."
            else:
                if is_action_attribute:
                    reason = (
                        f"Action {predicate.expected_value} still fails for {predicate.target.object_id}, "
                        f"but {retry_reason}, so the selector retries an action-focused edit once more."
                    )
                else:
                    reason = (
                        f"Attribute {predicate.attribute_name} still fails for {predicate.target.object_id}, "
                        f"but {retry_reason}, so the selector retries an attribute edit once more."
                    )
            return ActionSelection(
                action_type=RefinementActionType.EDIT,
                action_parameters=EditActionParameters(
                    edit_operation=EditOperationType.CHANGE_ATTRIBUTE,
                    target_object_ids=object_ids,
                    target_predicate_ids=predicate_ids,
                    primary_object_id=predicate.target.object_id,
                    attribute_name=predicate.attribute_name,
                    attribute_value=predicate.expected_value,
                    action_target_object_id=action_target_object_id,
                    template_key=(
                        "change_action" if is_action_attribute else "change_attribute"
                    ),
                    template_inputs={
                        "object_id": predicate.target.object_id,
                        "attribute_name": predicate.attribute_name,
                        "attribute_value": predicate.expected_value,
                        "action_target_object_id": action_target_object_id,
                    },
                ),
                reason=reason,
                target_predicate_ids=predicate_ids,
            )

        return ActionSelection(
            action_type=RefinementActionType.RESAMPLE,
            action_parameters=ResampleActionParameters(
                strategy=(
                    ResampleStrategy.RESAMPLE_FOR_ACTION
                    if is_action_attribute
                    else ResampleStrategy.RESAMPLE_FOR_ATTRIBUTE
                ),
                target_predicate_ids=predicate_ids,
                target_object_ids=object_ids,
                template_key=(
                    "resample_for_action"
                    if is_action_attribute
                    else "resample_for_attribute"
                ),
                template_inputs={
                    "object_id": predicate.target.object_id,
                    "attribute_name": predicate.attribute_name,
                    "attribute_value": predicate.expected_value,
                    "action_target_object_id": action_target_object_id,
                },
            ),
            reason=(
                f"Action {predicate.expected_value} still fails for {predicate.target.object_id} without local attribute progress in the current rewritten image, so the selector escalates to resampling."
                if is_action_attribute
                else f"Attribute {predicate.attribute_name} still fails for {predicate.target.object_id} without local attribute progress in the current rewritten image, so the selector escalates to resampling."
            ),
            target_predicate_ids=predicate_ids,
        )

    def _select_global_scene_action(
        self,
        context: ActionSelectorContext,
        failed_predicate: FailedPredicate,
    ) -> Optional[ActionSelection]:
        predicate = failed_predicate.predicate
        assert isinstance(predicate, GlobalScenePredicate)

        predicate_ids = (predicate.predicate_id,)
        object_ids: tuple[str, ...] = ()
        if not self._scene_level_edit_dependencies_stable(context):
            return None
        current_rewrite_attempts = self._current_rewrite_edit_attempts(
            context,
            target_predicate_ids=predicate_ids,
            target_object_ids=object_ids,
            edit_operation=EditOperationType.CHANGE_GLOBAL_SCENE,
        )
        retry_reason = self._predicate_progress_retry_reason_in_current_rewrite(
            context,
            target_predicate_ids=predicate_ids,
            target_object_ids=object_ids,
            edit_operation=EditOperationType.CHANGE_GLOBAL_SCENE,
        )
        if current_rewrite_attempts == 0 or retry_reason is not None:
            if current_rewrite_attempts == 0:
                reason = (
                    "The scene-level background/environment constraint failed, "
                    "so the selector chooses one background-focused edit for the current rewritten image."
                )
            else:
                reason = (
                    "The scene-level background/environment constraint still fails, "
                    f"but {retry_reason}, so the selector retries a background-focused edit once more."
                )
            return ActionSelection(
                action_type=RefinementActionType.EDIT,
                action_parameters=EditActionParameters(
                    edit_operation=EditOperationType.CHANGE_GLOBAL_SCENE,
                    target_object_ids=object_ids,
                    target_predicate_ids=predicate_ids,
                    attribute_name=predicate.attribute_name,
                    attribute_value=predicate.expected_value,
                    template_key="change_global_scene",
                    template_inputs={
                        "attribute_name": predicate.attribute_name,
                        "attribute_value": predicate.expected_value,
                    },
                ),
                reason=reason,
                target_predicate_ids=predicate_ids,
            )

        return ActionSelection(
            action_type=RefinementActionType.RESAMPLE,
            action_parameters=ResampleActionParameters(
                strategy=ResampleStrategy.GLOBAL_RESAMPLE,
                target_predicate_ids=predicate_ids,
                target_object_ids=object_ids,
                template_key="global_resample_for_scene_constraint",
                template_inputs={
                    "attribute_name": predicate.attribute_name,
                    "attribute_value": predicate.expected_value,
                },
            ),
            reason=(
                "The scene-level background/environment constraint still fails without "
                "local progress in the current rewritten image, so the selector escalates to resampling."
            ),
            target_predicate_ids=predicate_ids,
        )

    def _record_matches_current_rewrite(
        self,
        context: ActionSelectorContext,
        record: ActionExecutionRecord,
    ) -> bool:
        """Return whether an action record belongs to the current rewritten image."""
        current_rewrite_index = context.candidate_metadata.metadata.get(
            "current_rewrite_index"
        )
        current_rewrite_prompt = context.candidate_metadata.metadata.get(
            "current_rewrite_prompt"
        )
        record_rewrite_index = record.metadata.get("rewrite_index")
        record_rewrite_prompt = record.metadata.get("rewrite_prompt")
        if current_rewrite_index is not None or record_rewrite_index is not None:
            return current_rewrite_index == record_rewrite_index
        if isinstance(current_rewrite_prompt, str) or isinstance(
            record_rewrite_prompt, str
        ):
            return current_rewrite_prompt == record_rewrite_prompt
        return True

    def _edit_attempt_records(
        self,
        context: ActionSelectorContext,
        *,
        target_predicate_ids: Sequence[str],
        target_object_ids: Sequence[str],
        edit_operation: EditOperationType,
        current_rewrite_only: bool = False,
    ) -> tuple[ActionExecutionRecord, ...]:
        """Return historical edit attempts for the same failure surface."""
        return tuple(
            record
            for record in context.candidate_metadata.action_history
            if record.action_type == RefinementActionType.EDIT
            and record.status != ActionExecutionStatus.PROPOSED
            and record.matches(
                action_type=RefinementActionType.EDIT,
                target_predicate_ids=target_predicate_ids,
                target_object_ids=target_object_ids,
                edit_operation=edit_operation,
            )
            and (
                not current_rewrite_only
                or self._record_matches_current_rewrite(context, record)
            )
        )

    def _failed_edit_records(
        self,
        context: ActionSelectorContext,
        *,
        target_predicate_ids: Sequence[str],
        target_object_ids: Sequence[str],
        edit_operation: EditOperationType,
        current_rewrite_only: bool = False,
    ) -> tuple[ActionExecutionRecord, ...]:
        """Return failed edit attempts for the same failure surface."""
        return tuple(
            record
            for record in self._edit_attempt_records(
                context,
                target_predicate_ids=target_predicate_ids,
                target_object_ids=target_object_ids,
                edit_operation=edit_operation,
                current_rewrite_only=current_rewrite_only,
            )
            if record.status == ActionExecutionStatus.FAILED
        )

    def _current_rewrite_edit_attempts(
        self,
        context: ActionSelectorContext,
        *,
        target_predicate_ids: Sequence[str],
        target_object_ids: Sequence[str],
        edit_operation: EditOperationType,
    ) -> int:
        """Count same-surface edit attempts for the current rewritten image."""
        return len(
            self._edit_attempt_records(
                context,
                target_predicate_ids=target_predicate_ids,
                target_object_ids=target_object_ids,
                edit_operation=edit_operation,
                current_rewrite_only=True,
            )
        )

    @staticmethod
    def _result_or_placeholder(
        report: VisualProgramVerificationReport,
        *,
        predicate_id: str,
        category: Any,
    ) -> PredicateVerificationResult:
        """Return a report result when present, or a safe placeholder when absent."""
        try:
            return report.get_result(predicate_id)
        except KeyError:
            return PredicateVerificationResult(
                predicate_id=predicate_id,
                category=category,
                state=PredicateState.CONTRADICTION,
                score=0.0,
                confidence=0.0,
                evidence_summary="Missing predicate verification result.",
                evidence={},
            )

    @staticmethod
    def _predicate_progress_snapshot(
        report: VisualProgramVerificationReport,
        *,
        target_predicate_ids: Sequence[str],
    ) -> dict[str, Any]:
        """Summarize how much progress the current image makes on one failure surface."""
        state_rank = {
            PredicateState.CONTRADICTION: 0,
            PredicateState.VIOLATED: 1,
            PredicateState.UNCERTAIN: 2,
            PredicateState.SATISFIED: 3,
        }
        satisfied_count = 0
        state_rank_sum = 0
        scores: list[float] = []
        for predicate_id in target_predicate_ids:
            result = VisualProgramActionSelector._result_or_placeholder(
                report,
                predicate_id=predicate_id,
                category=PredicateCategory.OBJECT_COUNT,
            )
            if result.state == PredicateState.SATISFIED:
                satisfied_count += 1
            state_rank_sum += state_rank.get(result.state, 0)
            if result.score is not None:
                scores.append(float(result.score))
        average_score = sum(scores) / len(scores) if scores else None
        return {
            "satisfied_count": satisfied_count,
            "state_rank_sum": state_rank_sum,
            "average_score": average_score,
        }

    def _predicate_progress_retry_reason_since_record(
        self,
        *,
        record: ActionExecutionRecord,
        report: VisualProgramVerificationReport,
        target_predicate_ids: Sequence[str],
    ) -> Optional[str]:
        """Return a retry reason when the target predicate surface improved locally."""
        prior_progress = record.metadata.get("predicate_progress")
        current_progress = self._predicate_progress_snapshot(
            report,
            target_predicate_ids=target_predicate_ids,
        )
        if not isinstance(prior_progress, Mapping):
            return None

        reasons: list[str] = []
        prior_satisfied = int(prior_progress.get("satisfied_count", 0))
        current_satisfied = int(current_progress.get("satisfied_count", 0))
        if current_satisfied > prior_satisfied:
            reasons.append(
                "more target predicates are satisfied than after the last failed edit"
            )
        prior_state_rank = int(prior_progress.get("state_rank_sum", 0))
        current_state_rank = int(current_progress.get("state_rank_sum", 0))
        if current_state_rank > prior_state_rank:
            reasons.append(
                "the target predicate states improved since the last failed edit"
            )
        prior_average_score = prior_progress.get("average_score")
        current_average_score = current_progress.get("average_score")
        if (
            isinstance(prior_average_score, (int, float))
            and isinstance(current_average_score, (int, float))
            and current_average_score
            >= prior_average_score + self.edit_retry_score_delta
        ):
            reasons.append(
                "the target verification score improved meaningfully since the last failed edit"
            )
        if not reasons:
            return None
        return " and ".join(dict.fromkeys(reasons))

    def _predicate_progress_retry_reason_in_current_rewrite(
        self,
        context: ActionSelectorContext,
        *,
        target_predicate_ids: Sequence[str],
        target_object_ids: Sequence[str],
        edit_operation: EditOperationType,
    ) -> Optional[str]:
        """Allow one same-rewrite retry only when the target surface improved locally."""
        failed_records = self._failed_edit_records(
            context,
            target_predicate_ids=target_predicate_ids,
            target_object_ids=target_object_ids,
            edit_operation=edit_operation,
            current_rewrite_only=True,
        )
        if len(failed_records) != 1:
            return None
        return self._predicate_progress_retry_reason_since_record(
            record=failed_records[-1],
            report=context.verification_report,
            target_predicate_ids=target_predicate_ids,
        )

    @staticmethod
    def _related_object_failure_ids(
        program: VisualProgram,
        object_id: str,
        *,
        predicate_classes: tuple[type[Predicate], ...],
    ) -> tuple[str, ...]:
        """Return failure-relevant predicate IDs for one object."""
        return tuple(
            predicate.predicate_id
            for predicate in program.predicates
            if isinstance(predicate, predicate_classes)
            and object_id in predicate.referenced_object_ids()
        )

    @staticmethod
    def _infer_current_count(result: PredicateVerificationResult) -> Optional[int]:
        """Infer observed object count from verifier evidence when available."""
        evidence = dict(result.evidence)
        if "strong_count" in evidence:
            return int(evidence["strong_count"])
        if "candidate_count" in evidence:
            return int(evidence["candidate_count"])
        candidate_regions = evidence.get("candidate_regions")
        if isinstance(candidate_regions, Sequence):
            return len(candidate_regions)
        return None

    @staticmethod
    def _infer_weak_count(result: PredicateVerificationResult) -> Optional[int]:
        """Infer the weak proposal count from verifier evidence when available."""
        evidence = dict(result.evidence)
        if "weak_count" in evidence:
            return int(evidence["weak_count"])
        candidate_regions = evidence.get("candidate_regions")
        if isinstance(candidate_regions, Sequence):
            return len(candidate_regions)
        return None

    def _retry_add_object_reason_in_current_rewrite(
        self,
        context: ActionSelectorContext,
        *,
        target_predicate_ids: Sequence[str],
        target_object_ids: Sequence[str],
        current_count: Optional[int],
        expected_count: Optional[int],
    ) -> Optional[str]:
        """Allow one same-rewrite add retry only when the missing-count gap shrank."""
        if (
            current_count is None
            or expected_count is None
            or current_count >= expected_count
        ):
            return None
        failed_records = self._failed_edit_records(
            context,
            target_predicate_ids=target_predicate_ids,
            target_object_ids=target_object_ids,
            edit_operation=EditOperationType.ADD_OBJECT,
            current_rewrite_only=True,
        )
        if len(failed_records) != 1:
            return None
        prior_missing_count = failed_records[-1].metadata.get("count_delta")
        if not isinstance(prior_missing_count, int) or prior_missing_count <= 0:
            return None
        current_missing_count = expected_count - current_count
        if current_missing_count >= prior_missing_count:
            return None
        return (
            f"the missing-count gap improved from {prior_missing_count} to "
            f"{current_missing_count} in the current rewritten image"
        )

    @staticmethod
    def _is_failed_state(state: PredicateState) -> bool:
        """Return whether a predicate state still needs refinement."""
        return state in {
            PredicateState.VIOLATED,
            PredicateState.UNCERTAIN,
            PredicateState.CONTRADICTION,
        }

    def _object_has_stable_positive_constraints(
        self,
        context: ActionSelectorContext,
        object_id: str,
    ) -> bool:
        """Return whether positive count predicates for one object are all satisfied."""
        for predicate in context.program.predicates:
            if not isinstance(predicate, CountPredicate):
                continue
            if object_id not in predicate.referenced_object_ids():
                continue
            result = self._result_or_placeholder(
                context.verification_report,
                predicate_id=predicate.predicate_id,
                category=predicate.category,
            )
            if result.state != PredicateState.SATISFIED:
                return False
        return True

    def _objects_have_stable_positive_constraints(
        self,
        context: ActionSelectorContext,
        object_ids: Sequence[str],
    ) -> bool:
        """Return whether every referenced object has stable positive-count support."""
        return all(
            self._object_has_stable_positive_constraints(context, object_id)
            for object_id in object_ids
        )

    def _scene_level_edit_dependencies_stable(
        self,
        context: ActionSelectorContext,
    ) -> bool:
        """Gate global-scene edits until all object-level constraints are settled."""
        blocking_predicate_types = (
            CountPredicate,
            AttributePredicate,
            SpatialRelationPredicate,
            ExclusionPredicate,
        )
        for predicate in context.program.predicates:
            if not isinstance(predicate, blocking_predicate_types):
                continue
            result = self._result_or_placeholder(
                context.verification_report,
                predicate_id=predicate.predicate_id,
                category=predicate.category,
            )
            if self._is_failed_state(result.state):
                return False
        return True

    @staticmethod
    def _required_positive_object_ids(program: VisualProgram) -> tuple[str, ...]:
        """Return object IDs that the scene positively requires to appear."""
        object_ids: list[str] = []
        for predicate in program.predicates:
            if not isinstance(predicate, CountPredicate):
                continue
            if (
                predicate.comparator
                not in {
                    CountComparator.EXACTLY,
                    CountComparator.AT_LEAST,
                }
                or predicate.expected_count <= 0
            ):
                continue
            object_id = predicate.target.object_id
            if object_id not in object_ids:
                object_ids.append(object_id)
        return tuple(object_ids)

    @staticmethod
    def _presence_count(result: PredicateVerificationResult) -> Optional[int]:
        """Infer whether a predicate result shows at least one plausible instance."""
        evidence = dict(result.evidence)
        counts: list[int] = []
        for key in ("candidate_count", "weak_count", "strong_count"):
            value = evidence.get(key)
            if isinstance(value, (int, float)):
                counts.append(int(value))
        candidate_regions = evidence.get("candidate_regions")
        if isinstance(candidate_regions, Sequence):
            counts.append(len(candidate_regions))
        if not counts:
            return None
        return max(counts)

    def _all_required_positive_objects_have_minimum_presence(
        self,
        context: ActionSelectorContext,
    ) -> bool:
        """Return whether every required object category has at least one plausible instance."""
        required_object_ids = self._required_positive_object_ids(context.program)
        return self._objects_have_minimum_presence(context, required_object_ids)

    def _max_presence_for_object(
        self,
        context: ActionSelectorContext,
        object_id: str,
    ) -> int:
        """Return maximum plausible presence count inferred for one object ID."""
        max_presence_count = 0
        for predicate in context.program.predicates:
            if not isinstance(predicate, CountPredicate):
                continue
            if predicate.target.object_id != object_id:
                continue
            if (
                predicate.comparator
                not in {
                    CountComparator.EXACTLY,
                    CountComparator.AT_LEAST,
                }
                or predicate.expected_count <= 0
            ):
                continue
            presence_count = self._presence_count(
                self._result_or_placeholder(
                    context.verification_report,
                    predicate_id=predicate.predicate_id,
                    category=predicate.category,
                )
            )
            if presence_count is not None:
                max_presence_count = max(max_presence_count, presence_count)
        return max_presence_count

    def _objects_have_minimum_presence(
        self,
        context: ActionSelectorContext,
        object_ids: Sequence[str],
    ) -> bool:
        """Return whether each object has at least one plausible observed instance."""
        return all(
            self._max_presence_for_object(context, object_id) >= 1
            for object_id in object_ids
        )


__all__ = [
    "ActionExecutionRecord",
    "ActionExecutionStatus",
    "ActionSelection",
    "CandidateMetadata",
    "EditActionParameters",
    "EditOperationType",
    "RefinementActionType",
    "ResampleActionParameters",
    "ResampleStrategy",
    "VisualProgramActionSelector",
]
