"""Compile each source prompt into a fixed visual program.

Normalization repairs schema-level issues, and the reviewer runs only when
normalization or lint findings warrant review. The resulting program stays
fixed throughout image refinement.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import traceback
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Optional, Union, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from graph.prompts import (
    SYSTEM_PROMPT_VISUAL_PROGRAM_PARSER,
    SYSTEM_PROMPT_VISUAL_PROGRAM_REVIEWER,
)
from graph.timing import ensure_timing_log, record_duration
from graph.visual_program_support import (
    SUPPORTED_ATTRIBUTE_NAMES,
    SUPPORTED_GLOBAL_SCENE_ATTRIBUTE_NAMES,
    SUPPORTED_SPATIAL_RELATIONS,
    normalize_spatial_relation_name,
)
from graph.visual_program import (
    AttributePredicate,
    CountPredicate,
    ExclusionPredicate,
    GlobalScenePredicate,
    ObjectSpec,
    Predicate,
    SpatialRelationPredicate,
    TextMatchType,
    TextPredicate,
    VisualProgram,
)

if TYPE_CHECKING:
    from langgraph.runtime import Runtime

    from graph.context import Context
    from graph.state import State


_SIMPLE_COLOR_EXCLUSION_LABELS = {
    "black",
    "white",
    "gray",
    "grey",
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "pink",
    "brown",
}
_SIMPLE_COLOR_PREFIXES = {"light", "dark", "bright", "deep", "pale"}
_SCENE_FRAMING_PREFIXES = (
    "a photo of ",
    "an image of ",
    "a photograph of ",
    "photo of ",
    "image of ",
    "photograph of ",
)
_EXPLICIT_DEPICTION_OBJECT_PREFIXES = (
    "a photo ",
    "an image ",
    "a picture ",
    "a photograph ",
    "a poster ",
    "a postcard ",
    "a polaroid ",
    "a printout ",
    "a framed photo ",
    "a printed photo ",
    "the photo ",
    "the image ",
    "the picture ",
    "the photograph ",
)
_EXPLICIT_DEPICTION_OBJECT_CUES = (
    "physical photo",
    "printed photo",
    "framed photo",
    "photo print",
    "picture frame",
    "printout",
    "polaroid",
    "poster",
    "postcard",
    "screenshot",
)
_DEPICTION_OBJECT_LABELS = {"photo", "image", "picture", "photograph"}
_CONTAINMENT_RELATION_NAMES = {"inside", "in", "within"}


class VisualProgramParsingError(RuntimeError):
    """Raised when parser output cannot be compiled into a valid visual program."""

    def __init__(
        self,
        message: str,
        *,
        parser_output: Any = None,
        visual_program_candidate: Any = None,
        timing_log: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.parser_output = parser_output
        self.visual_program_candidate = visual_program_candidate
        self.timing_log = timing_log


def _artifact_dir_for_parser_state(state: "State") -> str:
    prompt_key = (state.original_prompt or "unknown_prompt").replace(" ", "-")[:64]
    return os.path.join(state.saving_path, prompt_key)


async def _save_parser_failure_artifacts(
    *,
    state: "State",
    prompt: str,
    component_name: str,
    parser_output: Any,
    visual_program_candidate: Any,
    exc: Exception,
    extra_payload: dict[str, Any] | None = None,
) -> None:
    if not getattr(state, "saving_files", False):
        return

    from graph.utils import ensure_directory, json_dumps_timed, write_text_file

    artifact_dir = _artifact_dir_for_parser_state(state)
    await asyncio.to_thread(ensure_directory, artifact_dir)

    failure_payload = {
        "component": component_name,
        "prompt": prompt,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
        "visual_program_candidate": visual_program_candidate,
        "parser_output": (
            parser_output.model_dump(mode="json")
            if hasattr(parser_output, "model_dump")
            else parser_output
        ),
    }
    if extra_payload:
        failure_payload.update(extra_payload)
    serialized = json_dumps_timed(
        failure_payload,
        indent=4,
        ensure_ascii=False,
        timing_log=None,
        timing_component=f"{component_name}:failure_artifact:serialize",
    )
    await asyncio.to_thread(
        write_text_file,
        os.path.join(artifact_dir, "visual_program_parser_failure.json"),
        serialized,
        timing_log=None,
    )
    if visual_program_candidate is not None:
        candidate_serialized = json_dumps_timed(
            visual_program_candidate,
            indent=4,
            ensure_ascii=False,
            timing_log=None,
            timing_component=f"{component_name}:candidate_artifact:serialize",
        )
        await asyncio.to_thread(
            write_text_file,
            os.path.join(artifact_dir, "visual_program_candidate.json"),
            candidate_serialized,
            timing_log=None,
        )


def _normalize_parser_label(value: str) -> str:
    return " ".join(
        str(value).strip().lower().replace("_", " ").replace("-", " ").split()
    )


def _normalize_object_aliases(
    label: str,
    aliases: list[str] | tuple[str, ...] | None,
) -> list[str]:
    normalized_label = _normalize_parser_label(label)
    seen = {normalized_label} if normalized_label else set()
    normalized_aliases: list[str] = []
    for alias in aliases or ():
        cleaned = " ".join(str(alias).strip().split())
        alias_key = _normalize_parser_label(cleaned)
        if not cleaned or not alias_key or alias_key in seen:
            continue
        normalized_aliases.append(cleaned)
        seen.add(alias_key)
    return normalized_aliases


def _append_alias_if_new(parsed_object: "ParsedObjectSpec", alias: str) -> bool:
    normalized_aliases = _normalize_object_aliases(
        parsed_object.label,
        [*parsed_object.aliases, alias],
    )
    if normalized_aliases == list(parsed_object.aliases):
        return False
    parsed_object.aliases = normalized_aliases
    return True


def _is_exclusion_only_object(
    parser_output: "VisualProgramParserOutput",
    object_id: str,
) -> bool:
    """Return whether one object is referenced only by exclusion constraints."""
    if object_id not in {
        constraint.object_id for constraint in parser_output.exclusion_constraints
    }:
        return False
    if object_id in {
        constraint.object_id for constraint in parser_output.exact_count_constraints
    }:
        return False
    if object_id in {
        constraint.object_id for constraint in parser_output.at_least_count_constraints
    }:
        return False
    for relation in parser_output.relation_constraints:
        if object_id in {relation.subject_object_id, relation.reference_object_id}:
            return False
    for constraint in parser_output.attribute_constraints:
        if object_id == constraint.subject_object_id:
            return False
        if object_id == constraint.action_target_object_id:
            return False
    for text_constraint in parser_output.text_constraints:
        if object_id == text_constraint.target_object_id:
            return False
    return True


def _contrastive_size_subject_ids(
    parser_output: "VisualProgramParserOutput",
) -> set[str]:
    size_constraints = [
        constraint
        for constraint in parser_output.attribute_constraints
        if constraint.attribute_name == "size"
        and constraint.attribute_value is not None
    ]
    subject_ids = {constraint.subject_object_id for constraint in size_constraints}
    normalized_values = {
        _canonicalize_predicate_name(str(constraint.attribute_value))
        for constraint in size_constraints
    }
    if len(subject_ids) < 2 or len(normalized_values) < 2:
        return set()
    return subject_ids


def _is_simple_color_exclusion_label(label: str) -> bool:
    normalized = _normalize_parser_label(label)
    if not normalized:
        return False
    if normalized in _SIMPLE_COLOR_EXCLUSION_LABELS:
        return True
    tokens = normalized.split()
    return (
        len(tokens) == 2
        and tokens[0] in _SIMPLE_COLOR_PREFIXES
        and tokens[1] in _SIMPLE_COLOR_EXCLUSION_LABELS
    )


def _strip_scene_framing_prefix(prompt: str) -> str | None:
    normalized_prompt = str(prompt).strip()
    lowered_prompt = normalized_prompt.lower()
    for prefix in _SCENE_FRAMING_PREFIXES:
        if lowered_prompt.startswith(prefix):
            remainder = normalized_prompt[len(prefix) :].strip()
            return remainder or None
    return None


def _prompt_explicitly_requests_depiction_object(prompt_body: str) -> bool:
    lowered_body = str(prompt_body).strip().lower()
    if not lowered_body:
        return False
    if any(
        lowered_body.startswith(prefix)
        for prefix in _EXPLICIT_DEPICTION_OBJECT_PREFIXES
    ):
        return True
    return any(cue in lowered_body for cue in _EXPLICIT_DEPICTION_OBJECT_CUES)


def _uses_scene_framing_only(prompt: str) -> bool:
    prompt_body = _strip_scene_framing_prefix(prompt)
    if prompt_body is None:
        return False
    return not _prompt_explicitly_requests_depiction_object(prompt_body)


def _is_depiction_object_label(label: str) -> bool:
    return _normalize_parser_label(label) in _DEPICTION_OBJECT_LABELS


def _object_label_for_id(
    object_id: str,
    *,
    object_map: dict[str, "ParsedObjectSpec"],
) -> str:
    parsed_object = object_map.get(object_id)
    if parsed_object is None:
        return object_id.replace("_", " ")
    return parsed_object.label


def _relation_description(
    subject_object_id: str,
    relation: str,
    reference_object_id: str,
    *,
    object_map: dict[str, "ParsedObjectSpec"],
) -> str:
    subject_label = _object_label_for_id(subject_object_id, object_map=object_map)
    reference_label = _object_label_for_id(reference_object_id, object_map=object_map)
    return (
        f"The {subject_label} should be {relation.replace('_', ' ')} "
        f"the {reference_label}."
    )


def _action_description(
    subject_object_id: str,
    action_value: str,
    target_object_id: str | None,
    *,
    object_map: dict[str, "ParsedObjectSpec"],
) -> str:
    subject_label = _object_label_for_id(subject_object_id, object_map=object_map)
    action_text = str(action_value).strip().replace("_", " ")
    if target_object_id is None:
        return f"The {subject_label} should be {action_text}."
    target_label = _object_label_for_id(target_object_id, object_map=object_map)
    return f"The {subject_label} should be {action_text} the {target_label}."


def _canonicalize_predicate_name(value: str) -> str:
    return "_".join(
        str(value).strip().lower().replace("-", " ").replace("_", " ").split()
    )


def _guess_label_from_object_id(object_id: str) -> str:
    normalized = _canonicalize_predicate_name(object_id)
    tokens = [token for token in normalized.split("_") if token]
    while tokens and tokens[-1].isdigit():
        tokens.pop()
    return " ".join(tokens) or normalized.replace("_", " ") or "object"


def _object_presence_descriptions(object_id: str) -> tuple[str, str]:
    label = _guess_label_from_object_id(object_id)
    return (
        f"Added missing object declaration for {label} because it was referenced by another predicate.",
        f"At least one {label} should appear in the scene.",
    )


class VisualProgramNormalizationReport(BaseModel):
    """Summarize deterministic parser normalization and linting results."""

    changed: bool = Field(default=False)
    should_request_review: bool = Field(default=False)
    applied_fixes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    missing_declared_object_ids_added: list[str] = Field(default_factory=list)
    missing_at_least_count_constraints_added: list[str] = Field(default_factory=list)
    unsupported_relation_names: list[str] = Field(default_factory=list)
    unsupported_attribute_names: list[str] = Field(default_factory=list)
    unsupported_global_scene_attribute_names: list[str] = Field(default_factory=list)

    def add_fix(self, message: str) -> None:
        self.changed = True
        self.applied_fixes.append(" ".join(str(message).strip().split()))

    def add_warning(self, message: str) -> None:
        self.should_request_review = True
        self.warnings.append(" ".join(str(message).strip().split()))


class VisualProgramReviewOutput(BaseModel):
    """Structured output for one review-and-repair pass over a VP candidate."""

    approved_candidate: bool = Field(
        ...,
        description="Whether the provided candidate visual program is already correct and verifier-compatible.",
    )
    review_reasoning: str = Field(
        ...,
        description="Short reasoning explaining why the candidate was approved or changed.",
    )
    detected_issues: list[str] = Field(
        default_factory=list,
        description="Short issue list describing the key problems found in the candidate.",
    )
    reviewed_program: "VisualProgramParserOutput" = Field(
        ...,
        description="The reviewed visual program. If approved_candidate is true, this should be equivalent to the input candidate.",
    )


class VisualProgramReviewReport(BaseModel):
    """Persisted metadata about the optional parser review stage."""

    called: bool = Field(default=False)
    trigger_reason: str = Field(default="")
    approved_candidate: Optional[bool] = Field(default=None)
    review_reasoning: Optional[str] = Field(default=None)
    detected_issues: list[str] = Field(default_factory=list)
    used_corrected_program: bool = Field(default=False)
    reviewed_program: Optional["VisualProgramParserOutput"] = Field(default=None)
    post_review_normalization_report: Optional[VisualProgramNormalizationReport] = (
        Field(default=None)
    )


@dataclass(frozen=True)
class ParsedVisualProgramResult:
    """Return the final VP plus parser-stage metadata."""

    visual_program: VisualProgram
    raw_parser_output: "VisualProgramParserOutput"
    normalized_parser_output: "VisualProgramParserOutput"
    final_parser_output: "VisualProgramParserOutput"
    normalization_report: VisualProgramNormalizationReport
    review_report: VisualProgramReviewReport


class ParsedObjectSpec(BaseModel):
    """Structured object declaration emitted by the prompt parser."""

    object_id: str = Field(
        ...,
        description=(
            "Stable snake_case object identifier used by constraints, for example "
            "'cat_1', 'dog_1', 'umbrella_1', or 'apple'."
        ),
    )
    label: str = Field(
        ...,
        description="Canonical singular noun phrase for the object, such as 'cat' or 'stop sign'.",
    )
    proposal_text: Optional[str] = Field(
        default=None,
        description=(
            "Optional short detector-facing grounding phrase for region proposal, such as "
            "'green book' or 'striped mug'. Keep `label` canonical and put detector-helpful "
            "visual modifiers here when they are useful."
        ),
    )
    aliases: list[str] = Field(
        default_factory=list,
        description="Optional alternate names or surface forms for this object.",
    )
    description: Optional[str] = Field(
        default=None,
        description="Optional short note clarifying object identity when needed.",
    )

    def to_object_spec(self) -> ObjectSpec:
        """Convert the parsed object declaration into the fixed schema type."""
        return ObjectSpec(
            object_id=self.object_id,
            label=self.label,
            proposal_text=self.proposal_text,
            aliases=tuple(self.aliases),
            description=self.description,
        )


class AtLeastCountConstraint(BaseModel):
    """Require at least a minimum number of instances for one declared object."""

    object_id: str = Field(
        ..., description="Referenced object ID from the objects list."
    )
    count: int = Field(
        ...,
        ge=1,
        description="Minimum required count for the referenced object.",
    )
    description: str = Field(
        ...,
        description="Short human-readable explanation of the lower-bound count requirement.",
    )

    def to_predicate(self) -> Predicate:
        """Convert the parsed lower-bound count constraint into a predicate."""
        return CountPredicate.at_least(
            self.object_id,
            self.count,
            description=self.description,
        )


class ExclusionConstraint(BaseModel):
    """Require a declared object not to appear in the scene."""

    object_id: str = Field(
        ..., description="Referenced object ID from the objects list."
    )
    description: str = Field(
        ...,
        description="Short human-readable explanation of the required absence.",
    )

    def to_predicate(self) -> Predicate:
        """Convert the parsed exclusion constraint into a predicate."""
        return ExclusionPredicate.absent(
            self.object_id,
            description=self.description,
        )


class ExactCountConstraint(BaseModel):
    """Require an exact number of instances for a declared object."""

    object_id: str = Field(
        ..., description="Referenced object ID from the objects list."
    )
    count: int = Field(
        ...,
        ge=0,
        description="Exact required count for the referenced object.",
    )
    description: str = Field(
        ...,
        description="Short human-readable explanation of the count requirement.",
    )

    def to_predicate(self) -> Predicate:
        """Convert the parsed exact-count constraint into a predicate."""
        return CountPredicate.exactly(
            self.object_id,
            self.count,
            description=self.description,
        )


class RelationConstraint(BaseModel):
    """Require a spatial relation between two declared objects."""

    subject_object_id: str = Field(
        ..., description="Object ID for the relation subject."
    )
    relation: str = Field(
        ...,
        description=(
            "Canonical relation name in snake_case, such as 'left_of', 'right_of', "
            "'on', 'inside', or 'behind'."
        ),
    )
    reference_object_id: str = Field(
        ...,
        description="Object ID for the relation reference target.",
    )
    description: str = Field(
        ...,
        description="Short human-readable explanation of the relation constraint.",
    )

    def to_predicate(self) -> Predicate:
        """Convert the parsed spatial relation into a predicate."""
        return SpatialRelationPredicate.relate(
            self.subject_object_id,
            self.relation,
            self.reference_object_id,
            description=self.description,
        )


class AttributeConstraint(BaseModel):
    """Represent an object-local attribute, optionally with an action target."""

    model_config = ConfigDict(populate_by_name=True)

    subject_object_id: str = Field(
        ..., description="Object ID for the constraint subject."
    )
    attribute_name: str = Field(
        ...,
        description=(
            "Attribute name for the subject object, for example 'color', 'material', "
            "'shape', 'pose', 'state', or 'action'."
        ),
    )
    attribute_value: Union[str, int, float, bool] = Field(
        ...,
        description="Attribute value for the subject object.",
    )
    action_target_object_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional target object ID when attribute_name='action' and the action is "
            "semantically directed at another declared object. Leave null for non-action "
            "attributes or for standalone actions with no specific target."
        ),
    )
    description: str = Field(
        ...,
        description=("Short human-readable explanation of the attribute constraint."),
    )

    @model_validator(mode="before")
    @classmethod
    def _validate_action_target_usage(
        cls,
        data: Any,
    ) -> Any:
        if not isinstance(data, dict):
            return data
        attribute_name = _canonicalize_predicate_name(
            str(data.get("attribute_name", ""))
        )
        if (
            data.get("action_target_object_id") is not None
            and attribute_name != "action"
        ):
            raise ValueError(
                "action_target_object_id may be used only when attribute_name='action'."
            )
        return data

    def to_predicate(self) -> Predicate:
        """Convert the parsed attribute constraint into a predicate."""
        return AttributePredicate.equals(
            self.subject_object_id,
            self.attribute_name,
            self.attribute_value,
            action_target=self.action_target_object_id,
            description=self.description,
        )


class GlobalSceneConstraint(BaseModel):
    """Require a holistic scene-level property such as environment or background."""

    attribute_name: str = Field(
        ...,
        description=(
            "Scene-level attribute name, such as 'environment', 'background', "
            "or 'setting'."
        ),
    )
    expected_value: Union[str, int, float, bool] = Field(
        ...,
        description="Expected scene-level value, such as 'underwater' or 'desert'.",
    )
    description: str = Field(
        ...,
        description="Short human-readable explanation of the scene-level constraint.",
    )

    def to_predicate(self) -> Predicate:
        """Convert the parsed scene-level requirement into a predicate."""
        return GlobalScenePredicate.equals(
            self.attribute_name,
            self.expected_value,
            description=self.description,
        )


class ParsedTextConstraint(BaseModel):
    """Require literal visible text, optionally tied to a declared object."""

    expected_text: str = Field(
        ...,
        description="Visible text that should appear in the image.",
    )
    match_type: Literal["exact", "contains", "starts_with", "ends_with"] = Field(
        default="exact",
        description="How the text should be matched during verification.",
    )
    target_object_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional object ID when the text belongs to a specific object, such as a sign or shirt."
        ),
    )
    case_sensitive: bool = Field(
        default=False,
        description="Whether text matching should preserve case.",
    )
    description: str = Field(
        ...,
        description="Short human-readable explanation of the text constraint.",
    )

    def to_predicate(self) -> Predicate:
        """Convert the parsed text constraint into a predicate."""
        target = self.target_object_id
        if self.match_type == "exact":
            return TextPredicate.exactly(
                self.expected_text,
                target=target,
                case_sensitive=self.case_sensitive,
                description=self.description,
            )
        if self.match_type == "contains":
            return TextPredicate.contains(
                self.expected_text,
                target=target,
                case_sensitive=self.case_sensitive,
                description=self.description,
            )

        return TextPredicate(
            predicate_id=(
                f"text:{'scene' if target is None else target}:{self.match_type}:"
                f"{self.expected_text.lower().replace(' ', '_')}"
            ),
            description=self.description,
            expected_text=self.expected_text,
            match_type=TextMatchType(self.match_type),
            target=target,
            case_sensitive=self.case_sensitive,
        )


class VisualProgramParserOutput(BaseModel):
    """Structured parser output that can be compiled into a fixed visual program."""

    parser_reasoning: str = Field(
        ...,
        description="Short reasoning summary describing how the prompt was decomposed.",
    )
    source_prompt: str = Field(
        ...,
        description="Original user prompt being compiled.",
    )
    objects: list[ParsedObjectSpec] = Field(
        default_factory=list,
        description=(
            "Canonical object declarations referenced by the constraint buckets. Keep one "
            "shared object declaration per counted category unless the prompt explicitly "
            "distinguishes separate instances."
        ),
    )
    at_least_count_constraints: list[AtLeastCountConstraint] = Field(
        default_factory=list,
        description=(
            "Lower-bound count constraints for required objects whose multiplicity is "
            "not otherwise fixed exactly."
        ),
    )
    exclusion_constraints: list[ExclusionConstraint] = Field(
        default_factory=list,
        description="Explicit absence constraints.",
    )
    exact_count_constraints: list[ExactCountConstraint] = Field(
        default_factory=list,
        description=(
            "Exact object count constraints. Use these to represent multiplicity instead of "
            "inventing per-instance objects for otherwise identical repeated items."
        ),
    )
    relation_constraints: list[RelationConstraint] = Field(
        default_factory=list,
        description=(
            "Spatial relation constraints. Prefer group-level relations between canonical "
            "counted objects unless the prompt explicitly specifies per-instance pairings."
        ),
    )
    attribute_constraints: list[AttributeConstraint] = Field(
        default_factory=list,
        description=(
            "Object-local attribute constraints. Use attribute_name='action' with "
            "optional action_target_object_id when an action is directed at another "
            "declared object."
        ),
    )
    global_scene_constraints: list[GlobalSceneConstraint] = Field(
        default_factory=list,
        description=(
            "Holistic environment or background constraints that apply to the whole "
            "scene rather than to one declared object."
        ),
    )
    text_constraints: list[ParsedTextConstraint] = Field(
        default_factory=list,
        description="Visible text constraints.",
    )

    def to_visual_program(self, *, program_id: str = "visual_program") -> VisualProgram:
        """Compile the parser output into the fixed visual-program schema.

        The resulting program is immutable. Later stages should mutate only a
        companion :class:`graph.visual_program.VisualProgramState`.
        """
        normalized_output = self._normalized_for_scene_framing()
        kept_exclusion_constraints = normalized_output._filtered_exclusion_constraints()
        referenced_object_ids = normalized_output._referenced_object_ids(
            exclusion_constraints=kept_exclusion_constraints
        )

        deduped_objects: dict[str, ObjectSpec] = {}
        for parsed_object in normalized_output.objects:
            object_spec = parsed_object.to_object_spec()
            if object_spec.object_id not in referenced_object_ids:
                continue
            deduped_objects.setdefault(object_spec.object_id, object_spec)

        predicates: list[Predicate] = []
        seen_predicate_ids: set[str] = set()
        for parsed_constraint in normalized_output._iter_constraints(
            exclusion_constraints=kept_exclusion_constraints
        ):
            predicate = parsed_constraint.to_predicate()
            if predicate.predicate_id in seen_predicate_ids:
                continue
            predicates.append(predicate)
            seen_predicate_ids.add(predicate.predicate_id)

        return VisualProgram.from_predicates(
            source_prompt=normalized_output.source_prompt,
            objects=list(deduped_objects.values()),
            predicates=predicates,
            program_id=program_id,
        )

    def to_visual_program_candidate(
        self,
        *,
        program_id: str = "visual_program",
    ) -> dict[str, Any]:
        """Build a best-effort serialized VP payload without final validation.

        This is useful for debugging parser failures where the fully validated
        `VisualProgram` cannot be constructed, for example when predicates
        reference undeclared objects.
        """
        normalized_output = self._normalized_for_scene_framing()
        kept_exclusion_constraints = normalized_output._filtered_exclusion_constraints()
        referenced_object_ids = normalized_output._referenced_object_ids(
            exclusion_constraints=kept_exclusion_constraints
        )

        deduped_objects: dict[str, ObjectSpec] = {}
        for parsed_object in normalized_output.objects:
            object_spec = parsed_object.to_object_spec()
            if object_spec.object_id not in referenced_object_ids:
                continue
            deduped_objects.setdefault(object_spec.object_id, object_spec)

        predicates: list[Predicate] = []
        seen_predicate_ids: set[str] = set()
        predicate_compilation_errors: list[dict[str, str]] = []
        for parsed_constraint in normalized_output._iter_constraints(
            exclusion_constraints=kept_exclusion_constraints
        ):
            try:
                predicate = parsed_constraint.to_predicate()
            except Exception as exc:
                predicate_compilation_errors.append(
                    {
                        "constraint_type": type(parsed_constraint).__name__,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
                continue
            if predicate.predicate_id in seen_predicate_ids:
                continue
            predicates.append(predicate)
            seen_predicate_ids.add(predicate.predicate_id)

        declared_object_ids = set(deduped_objects.keys())
        missing_reference_object_ids = sorted(
            {
                object_id
                for predicate in predicates
                for object_id in predicate.referenced_object_ids()
                if object_id not in declared_object_ids
            }
        )

        return {
            "program_id": program_id,
            "source_prompt": normalized_output.source_prompt,
            "parser_reasoning": normalized_output.parser_reasoning,
            "objects": [
                object_spec.to_dict() for object_spec in deduped_objects.values()
            ],
            "predicates": [predicate.to_dict() for predicate in predicates],
            "declared_object_ids": sorted(declared_object_ids),
            "missing_reference_object_ids": missing_reference_object_ids,
            "predicate_compilation_errors": predicate_compilation_errors,
        }

    def _filtered_exclusion_constraints(self) -> list[ExclusionConstraint]:
        object_map = {
            parsed_object.object_id: parsed_object for parsed_object in self.objects
        }
        kept: list[ExclusionConstraint] = []
        for exclusion in self.exclusion_constraints:
            parsed_object = object_map.get(exclusion.object_id)
            if parsed_object is not None and _is_simple_color_exclusion_label(
                parsed_object.label
            ):
                continue
            kept.append(exclusion)
        return kept

    def _referenced_object_ids(
        self,
        *,
        exclusion_constraints: list[ExclusionConstraint],
    ) -> set[str]:
        referenced: set[str] = set()
        for at_least_count in self.at_least_count_constraints:
            referenced.add(at_least_count.object_id)
        for exclusion in exclusion_constraints:
            referenced.add(exclusion.object_id)
        for exact_count in self.exact_count_constraints:
            referenced.add(exact_count.object_id)
        for relation in self.relation_constraints:
            referenced.add(relation.subject_object_id)
            referenced.add(relation.reference_object_id)
        for attribute_constraint in self.attribute_constraints:
            referenced.add(attribute_constraint.subject_object_id)
            if attribute_constraint.action_target_object_id is not None:
                referenced.add(attribute_constraint.action_target_object_id)
        for text_constraint in self.text_constraints:
            if text_constraint.target_object_id is not None:
                referenced.add(text_constraint.target_object_id)
        return referenced

    def _iter_constraints(
        self,
        *,
        exclusion_constraints: list[ExclusionConstraint] | None = None,
    ) -> list[Any]:
        """Flatten constraint buckets in a deterministic compilation order."""
        if exclusion_constraints is None:
            exclusion_constraints = self.exclusion_constraints
        return [
            *self.at_least_count_constraints,
            *exclusion_constraints,
            *self.exact_count_constraints,
            *self.relation_constraints,
            *self.attribute_constraints,
            *self.global_scene_constraints,
            *self.text_constraints,
        ]

    def _normalized_for_scene_framing(self) -> "VisualProgramParserOutput":
        if not _uses_scene_framing_only(self.source_prompt):
            return self

        depiction_object_ids = {
            parsed_object.object_id
            for parsed_object in self.objects
            if _is_depiction_object_label(parsed_object.label)
        }
        if not depiction_object_ids:
            return self

        containment_children_by_depiction: dict[str, list[str]] = {}
        depiction_has_external_relation: dict[str, bool] = {
            object_id: False for object_id in depiction_object_ids
        }
        for relation in self.relation_constraints:
            if (
                relation.reference_object_id in depiction_object_ids
                and relation.relation in _CONTAINMENT_RELATION_NAMES
            ):
                containment_children_by_depiction.setdefault(
                    relation.reference_object_id, []
                ).append(relation.subject_object_id)
                continue
            if relation.subject_object_id in depiction_has_external_relation:
                depiction_has_external_relation[relation.subject_object_id] = True
            if relation.reference_object_id in depiction_has_external_relation:
                depiction_has_external_relation[relation.reference_object_id] = True

        depiction_remap: dict[str, str] = {}
        repairable_depiction_object_ids: set[str] = set()
        for depiction_object_id in depiction_object_ids:
            contained_children = containment_children_by_depiction.get(
                depiction_object_id, []
            )
            if not contained_children:
                continue
            if depiction_has_external_relation[depiction_object_id]:
                unique_children = list(dict.fromkeys(contained_children))
                if len(unique_children) != 1:
                    return self
                depiction_remap[depiction_object_id] = unique_children[0]
            repairable_depiction_object_ids.add(depiction_object_id)

        if not repairable_depiction_object_ids:
            return self

        normalized_output = self.model_copy(deep=True)
        original_object_map = {
            parsed_object.object_id: parsed_object
            for parsed_object in normalized_output.objects
        }
        normalized_output.objects = [
            parsed_object
            for parsed_object in normalized_output.objects
            if parsed_object.object_id not in repairable_depiction_object_ids
        ]
        normalized_object_map = {
            parsed_object.object_id: parsed_object
            for parsed_object in normalized_output.objects
        }

        normalized_output.at_least_count_constraints = [
            constraint
            for constraint in normalized_output.at_least_count_constraints
            if constraint.object_id not in repairable_depiction_object_ids
        ]
        normalized_output.exclusion_constraints = [
            constraint
            for constraint in normalized_output.exclusion_constraints
            if constraint.object_id not in repairable_depiction_object_ids
        ]
        normalized_output.exact_count_constraints = [
            constraint
            for constraint in normalized_output.exact_count_constraints
            if constraint.object_id not in repairable_depiction_object_ids
        ]

        repaired_relations: list[RelationConstraint] = []
        for relation in normalized_output.relation_constraints:
            if (
                relation.reference_object_id in repairable_depiction_object_ids
                and relation.relation in _CONTAINMENT_RELATION_NAMES
            ):
                continue

            if (
                relation.subject_object_id in repairable_depiction_object_ids
                and relation.subject_object_id not in depiction_remap
            ):
                continue
            if (
                relation.reference_object_id in repairable_depiction_object_ids
                and relation.reference_object_id not in depiction_remap
            ):
                continue

            remapped_subject = depiction_remap.get(
                relation.subject_object_id, relation.subject_object_id
            )
            remapped_reference = depiction_remap.get(
                relation.reference_object_id, relation.reference_object_id
            )
            repaired_relations.append(
                RelationConstraint(
                    subject_object_id=remapped_subject,
                    relation=relation.relation,
                    reference_object_id=remapped_reference,
                    description=_relation_description(
                        remapped_subject,
                        relation.relation,
                        remapped_reference,
                        object_map=normalized_object_map,
                    ),
                )
            )
        normalized_output.relation_constraints = repaired_relations

        repaired_attribute_constraints: list[AttributeConstraint] = []
        for constraint in normalized_output.attribute_constraints:
            if (
                constraint.subject_object_id in repairable_depiction_object_ids
                and constraint.subject_object_id not in depiction_remap
            ):
                continue
            if (
                constraint.action_target_object_id in repairable_depiction_object_ids
                and constraint.action_target_object_id not in depiction_remap
            ):
                continue

            remapped_subject = depiction_remap.get(
                constraint.subject_object_id, constraint.subject_object_id
            )
            remapped_target = (
                depiction_remap.get(
                    constraint.action_target_object_id,
                    constraint.action_target_object_id,
                )
                if constraint.action_target_object_id is not None
                else None
            )
            description = constraint.description
            if constraint.attribute_name == "action":
                description = _action_description(
                    remapped_subject,
                    str(constraint.attribute_value),
                    remapped_target,
                    object_map=normalized_object_map,
                )
            repaired_attribute_constraints.append(
                AttributeConstraint(
                    subject_object_id=remapped_subject,
                    attribute_name=constraint.attribute_name,
                    attribute_value=constraint.attribute_value,
                    action_target_object_id=remapped_target,
                    description=description,
                )
            )
        normalized_output.attribute_constraints = repaired_attribute_constraints

        normalized_output.text_constraints = [
            constraint
            for constraint in normalized_output.text_constraints
            if constraint.target_object_id not in repairable_depiction_object_ids
        ]

        if original_object_map.keys() != normalized_object_map.keys():
            normalized_output.parser_reasoning = (
                f"{normalized_output.parser_reasoning} "
                "Leading photo/image framing was normalized as scene-style context rather than a literal depiction object."
            ).strip()

        return normalized_output


def vp_lint_and_normalize(
    parser_output: VisualProgramParserOutput,
) -> tuple[VisualProgramParserOutput, VisualProgramNormalizationReport]:
    """Apply deterministic parser repairs and collect review-worthy warnings."""
    normalized_output = parser_output._normalized_for_scene_framing().model_copy(
        deep=True
    )
    report = VisualProgramNormalizationReport()

    deduped_objects: list[ParsedObjectSpec] = []
    seen_object_ids: set[str] = set()
    for parsed_object in normalized_output.objects:
        if parsed_object.object_id in seen_object_ids:
            report.add_fix(
                f"Removed duplicate object declaration for object_id={parsed_object.object_id}."
            )
            continue
        seen_object_ids.add(parsed_object.object_id)
        deduped_objects.append(parsed_object)
    normalized_output.objects = deduped_objects
    for parsed_object in normalized_output.objects:
        if parsed_object.proposal_text is not None:
            normalized_proposal_text = " ".join(
                str(parsed_object.proposal_text).strip().split()
            )
            if normalized_proposal_text != parsed_object.proposal_text:
                report.add_fix(
                    f"Normalized proposal_text for object_id={parsed_object.object_id}."
                )
                parsed_object.proposal_text = normalized_proposal_text or None
        normalized_aliases = _normalize_object_aliases(
            parsed_object.label,
            parsed_object.aliases,
        )
        if normalized_aliases != list(parsed_object.aliases):
            report.add_fix(
                f"Normalized aliases for object_id={parsed_object.object_id}."
            )
            parsed_object.aliases = normalized_aliases

    object_map = {
        parsed_object.object_id: parsed_object
        for parsed_object in normalized_output.objects
    }

    for relation in normalized_output.relation_constraints:
        canonical_relation = _canonicalize_predicate_name(relation.relation)
        if canonical_relation != relation.relation:
            report.add_fix(
                f"Canonicalized relation name {relation.relation!r} to {canonical_relation!r}."
            )
            relation.relation = canonical_relation

    for constraint in normalized_output.attribute_constraints:
        canonical_attribute = _canonicalize_predicate_name(constraint.attribute_name)
        if canonical_attribute != constraint.attribute_name:
            report.add_fix(
                f"Canonicalized attribute name {constraint.attribute_name!r} to {canonical_attribute!r}."
            )
            constraint.attribute_name = canonical_attribute
        if isinstance(constraint.attribute_value, str):
            normalized_value = " ".join(str(constraint.attribute_value).strip().split())
            if normalized_value != constraint.attribute_value:
                report.add_fix(
                    f"Normalized attribute value for object_id={constraint.subject_object_id} and attribute_name={constraint.attribute_name!r}."
                )
                constraint.attribute_value = normalized_value

    filtered_relations: list[RelationConstraint] = []
    for relation in normalized_output.relation_constraints:
        if relation.subject_object_id == relation.reference_object_id:
            report.add_fix(
                f"Removed self-relation {relation.relation!r} on object_id={relation.subject_object_id}."
            )
            continue
        filtered_relations.append(relation)
    normalized_output.relation_constraints = filtered_relations

    contrastive_size_subject_ids = _contrastive_size_subject_ids(normalized_output)
    filtered_attribute_constraints: list[AttributeConstraint] = []
    for constraint in normalized_output.attribute_constraints:
        if (
            constraint.attribute_name == "action"
            and constraint.action_target_object_id == constraint.subject_object_id
        ):
            report.add_fix(
                "Removed self-targeted action from "
                f"object_id={constraint.subject_object_id}."
            )
            continue
        if constraint.attribute_name == "type":
            parsed_object = object_map.get(constraint.subject_object_id)
            if (
                parsed_object is not None
                and constraint.attribute_value is not None
                and _append_alias_if_new(parsed_object, str(constraint.attribute_value))
            ):
                report.add_fix(
                    f"Preserved removed type wording as an alias for object_id={constraint.subject_object_id}."
                )
            report.add_fix(
                f"Removed type attribute from object_id={constraint.subject_object_id}; keep identity in the canonical label and aliases instead."
            )
            continue
        if (
            constraint.attribute_name == "size"
            and constraint.subject_object_id not in contrastive_size_subject_ids
        ):
            report.add_fix(
                f"Removed non-contrastive size attribute from object_id={constraint.subject_object_id}."
            )
            continue
        filtered_attribute_constraints.append(constraint)
    normalized_output.attribute_constraints = filtered_attribute_constraints

    for scene_constraint in normalized_output.global_scene_constraints:
        canonical_attribute = _canonicalize_predicate_name(
            scene_constraint.attribute_name
        )
        if canonical_attribute != scene_constraint.attribute_name:
            report.add_fix(
                f"Canonicalized global scene attribute name {scene_constraint.attribute_name!r} to {canonical_attribute!r}."
            )
            scene_constraint.attribute_name = canonical_attribute

    kept_exclusion_constraints = normalized_output._filtered_exclusion_constraints()
    if len(kept_exclusion_constraints) != len(normalized_output.exclusion_constraints):
        report.add_fix(
            "Dropped simple color-word exclusions that should not be modeled as standalone absent objects."
        )
        normalized_output.exclusion_constraints = kept_exclusion_constraints

    referenced_object_ids = normalized_output._referenced_object_ids(
        exclusion_constraints=normalized_output.exclusion_constraints
    )
    declared_object_ids = {
        parsed_object.object_id for parsed_object in normalized_output.objects
    }
    missing_object_ids = sorted(referenced_object_ids - declared_object_ids)
    for missing_object_id in missing_object_ids:
        description, _ = _object_presence_descriptions(missing_object_id)
        normalized_output.objects.append(
            ParsedObjectSpec(
                object_id=missing_object_id,
                label=_guess_label_from_object_id(missing_object_id),
                aliases=[],
                description=description,
            )
        )
        report.add_fix(
            f"Added missing object declaration for referenced object_id={missing_object_id}."
        )
        report.missing_declared_object_ids_added.append(missing_object_id)

    presence_or_absence_object_ids = {
        constraint.object_id
        for constraint in normalized_output.at_least_count_constraints
    }
    presence_or_absence_object_ids.update(
        constraint.object_id for constraint in normalized_output.exclusion_constraints
    )
    presence_or_absence_object_ids.update(
        constraint.object_id for constraint in normalized_output.exact_count_constraints
    )
    candidate_positive_object_ids = set(referenced_object_ids)
    candidate_positive_object_ids.update(
        parsed_object.object_id
        for parsed_object in normalized_output.objects
        if not _is_exclusion_only_object(normalized_output, parsed_object.object_id)
    )
    for object_id in sorted(candidate_positive_object_ids):
        if object_id in presence_or_absence_object_ids:
            continue
        _, positive_description = _object_presence_descriptions(object_id)
        normalized_output.at_least_count_constraints.append(
            AtLeastCountConstraint(
                object_id=object_id,
                count=1,
                description=positive_description,
            )
        )
        report.add_fix(
            "Added missing at-least count constraint "
            f"for referenced object_id={object_id}."
        )
        report.missing_at_least_count_constraints_added.append(object_id)

    exact_count_object_ids = {
        constraint.object_id for constraint in normalized_output.exact_count_constraints
    }
    filtered_at_least_counts: list[AtLeastCountConstraint] = []
    for constraint in normalized_output.at_least_count_constraints:
        if constraint.object_id in exact_count_object_ids:
            report.add_fix(
                "Removed lower-bound count constraint for "
                f"object_id={constraint.object_id} because an exact count constraint already exists."
            )
            continue
        filtered_at_least_counts.append(constraint)
    normalized_output.at_least_count_constraints = filtered_at_least_counts

    positive_count_object_ids = set(exact_count_object_ids)
    positive_count_object_ids.update(
        constraint.object_id
        for constraint in normalized_output.at_least_count_constraints
        if constraint.count > 0
    )
    required_object_ids = {
        constraint.object_id
        for constraint in normalized_output.at_least_count_constraints
        if constraint.count > 0
    }
    required_object_ids.update(positive_count_object_ids)
    filtered_exclusions: list[ExclusionConstraint] = []
    for exclusion in normalized_output.exclusion_constraints:
        if exclusion.object_id in positive_count_object_ids:
            report.add_fix(
                "Removed exclusion for object_id="
                f"{exclusion.object_id} because a positive count constraint already exists."
            )
            continue
        exclusion_object = object_map.get(exclusion.object_id)
        exclusion_label = (
            _normalize_parser_label(exclusion_object.label)
            if exclusion_object is not None
            else ""
        )
        conflicting_required_object_ids = sorted(
            object_id
            for object_id in required_object_ids
            if object_id != exclusion.object_id
            and object_id in object_map
            and _normalize_parser_label(object_map[object_id].label) == exclusion_label
        )
        if (
            exclusion_label
            and conflicting_required_object_ids
            and _is_exclusion_only_object(normalized_output, exclusion.object_id)
        ):
            report.add_fix(
                "Removed exclusion for object_id="
                f"{exclusion.object_id} because it duplicates required object label "
                f"{exclusion_object.label!r} already used by object_id(s)="
                f"{', '.join(conflicting_required_object_ids)}."
            )
            continue
        filtered_exclusions.append(exclusion)
    normalized_output.exclusion_constraints = filtered_exclusions

    object_map = {
        parsed_object.object_id: parsed_object
        for parsed_object in normalized_output.objects
    }
    normalized_output.relation_constraints = [
        RelationConstraint(
            subject_object_id=relation.subject_object_id,
            relation=relation.relation,
            reference_object_id=relation.reference_object_id,
            description=_relation_description(
                relation.subject_object_id,
                relation.relation,
                relation.reference_object_id,
                object_map=object_map,
            ),
        )
        for relation in normalized_output.relation_constraints
    ]
    rebuilt_attribute_constraints: list[AttributeConstraint] = []
    for constraint in normalized_output.attribute_constraints:
        description = constraint.description
        if constraint.attribute_name == "action":
            description = _action_description(
                constraint.subject_object_id,
                str(constraint.attribute_value),
                constraint.action_target_object_id,
                object_map=object_map,
            )
        rebuilt_attribute_constraints.append(
            AttributeConstraint(
                subject_object_id=constraint.subject_object_id,
                attribute_name=constraint.attribute_name,
                attribute_value=constraint.attribute_value,
                action_target_object_id=constraint.action_target_object_id,
                description=description,
            )
        )
    normalized_output.attribute_constraints = rebuilt_attribute_constraints

    for relation in normalized_output.relation_constraints:
        normalized_relation = normalize_spatial_relation_name(relation.relation)
        if normalized_relation not in SUPPORTED_SPATIAL_RELATIONS:
            if relation.relation not in report.unsupported_relation_names:
                report.unsupported_relation_names.append(relation.relation)
            report.add_warning(
                f"Unsupported spatial relation name {relation.relation!r} remains in the candidate program."
            )

    for constraint in normalized_output.attribute_constraints:
        if constraint.attribute_name not in SUPPORTED_ATTRIBUTE_NAMES:
            if constraint.attribute_name not in report.unsupported_attribute_names:
                report.unsupported_attribute_names.append(constraint.attribute_name)
            report.add_warning(
                f"Non-preferred object attribute name {constraint.attribute_name!r} remains in the candidate program."
            )

    for scene_constraint in normalized_output.global_scene_constraints:
        if (
            scene_constraint.attribute_name
            not in SUPPORTED_GLOBAL_SCENE_ATTRIBUTE_NAMES
        ):
            if (
                scene_constraint.attribute_name
                not in report.unsupported_global_scene_attribute_names
            ):
                report.unsupported_global_scene_attribute_names.append(
                    scene_constraint.attribute_name
                )
            report.add_warning(
                f"Unsupported global scene attribute name {scene_constraint.attribute_name!r} remains in the candidate program."
            )

    report.should_request_review = bool(report.should_request_review or report.changed)
    return normalized_output, report


def build_visual_program_review_input(
    *,
    prompt: str,
    candidate_output: VisualProgramParserOutput,
    normalization_report: VisualProgramNormalizationReport,
    system_prompt: str = SYSTEM_PROMPT_VISUAL_PROGRAM_REVIEWER,
) -> list[Any]:
    """Build the chat payload for reviewing one normalized VP candidate."""
    candidate_json = json.dumps(
        candidate_output.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
    )
    normalization_json = json.dumps(
        normalization_report.model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
    )
    inputs = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Review the following normalized visual-program candidate. "
                "Decide whether it is already correct and verifier-compatible. "
                "If not, return a repaired reviewed_program.\n\n"
                f"original_prompt:\n{prompt}\n\n"
                f"normalization_report:\n{normalization_json}\n\n"
                f"candidate_visual_program:\n{candidate_json}"
            ),
        },
    ]
    return inputs


def build_visual_program_parser_input(
    prompt: str,
    *,
    system_prompt: str = SYSTEM_PROMPT_VISUAL_PROGRAM_PARSER,
) -> list[Any]:
    """Build the chat input payload for prompt-to-program compilation.

    Examples:
        >>> messages = build_visual_program_parser_input("a red cat left of a dog")
        >>> messages[0]["role"]
        'system'
    """
    inputs = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Compile the following user prompt into the structured visual "
                "program schema.\n\n"
                f"user_prompt:\n{prompt}"
            ),
        },
    ]
    return inputs


VisualProgramReviewOutput.model_rebuild()
VisualProgramReviewReport.model_rebuild()


async def parse_visual_program_from_prompt(
    prompt: str,
    *,
    state: State,
    runtime: Runtime[Context],
    program_id: str = "visual_program",
    timeout: int = 180,
    component_name: str = "visual_program_parser",
) -> ParsedVisualProgramResult:
    """Compile a prompt into a fixed visual program through the configured vLLM server.

    Called at pipeline startup, this returns the fixed visual program and
    parser-stage metadata.
    """
    from graph.models import ainvoke_with_timeout_retry

    node_start = time.perf_counter()
    state.timing_log = ensure_timing_log(state.timing_log)

    prompt_prep_start = time.perf_counter()
    parser_input = build_visual_program_parser_input(
        prompt,
        system_prompt=runtime.context.system_prompt_visual_program_parser,
    )
    record_duration(
        state.timing_log,
        "prompt_preparation_seconds",
        time.perf_counter() - prompt_prep_start,
        component=component_name,
        count_key="prompt_preparations",
        metadata={"prompt_chars": len(prompt)},
    )

    try:
        parser_output = cast(
            VisualProgramParserOutput,
            await ainvoke_with_timeout_retry(
                state=state,
                runtime=runtime,
                inp=parser_input,
                output_class=VisualProgramParserOutput,
                timeout=timeout,
                component_name=component_name,
            ),
        )
    except Exception as exc:
        await _save_parser_failure_artifacts(
            state=state,
            prompt=prompt,
            component_name=component_name,
            parser_output=None,
            visual_program_candidate=None,
            exc=exc,
            extra_payload={
                "stage": "structured_output_validation",
                "message": (
                    "The parser did not produce a valid VisualProgramParserOutput "
                    "after the configured retries."
                ),
            },
        )
        raise VisualProgramParsingError(
            "Structured visual-program parser output failed validation after "
            f"configured retries: {type(exc).__name__}: {exc}",
            parser_output=None,
            visual_program_candidate=None,
            timing_log=deepcopy(state.timing_log),
        ) from exc

    postprocess_start = time.perf_counter()
    normalized_output, normalization_report = vp_lint_and_normalize(parser_output)
    review_report = VisualProgramReviewReport(called=False, trigger_reason="not_needed")
    final_parser_output = normalized_output
    should_review = normalization_report.should_request_review
    if should_review:
        review_input = build_visual_program_review_input(
            prompt=prompt,
            candidate_output=normalized_output,
            normalization_report=normalization_report,
            system_prompt=runtime.context.system_prompt_visual_program_reviewer,
        )
        review_output = cast(
            VisualProgramReviewOutput,
            await ainvoke_with_timeout_retry(
                state=state,
                runtime=runtime,
                inp=review_input,
                output_class=VisualProgramReviewOutput,
                timeout=timeout,
                component_name=f"{component_name}_review",
            ),
        )
        reviewed_normalized_output, post_review_normalization_report = (
            vp_lint_and_normalize(review_output.reviewed_program)
        )
        review_report = VisualProgramReviewReport(
            called=True,
            trigger_reason="normalization_or_lint_needed",
            approved_candidate=bool(review_output.approved_candidate),
            review_reasoning=" ".join(
                str(review_output.review_reasoning).strip().split()
            ),
            detected_issues=[
                " ".join(str(issue).strip().split())
                for issue in review_output.detected_issues
                if str(issue).strip()
            ],
            used_corrected_program=not bool(review_output.approved_candidate),
            reviewed_program=review_output.reviewed_program,
            post_review_normalization_report=post_review_normalization_report,
        )
        if not review_output.approved_candidate:
            final_parser_output = reviewed_normalized_output

    visual_program_candidate = None
    try:
        visual_program_candidate = final_parser_output.to_visual_program_candidate(
            program_id=program_id
        )
    except Exception:
        visual_program_candidate = None
    try:
        visual_program = final_parser_output.to_visual_program(program_id=program_id)
    except Exception as exc:
        await _save_parser_failure_artifacts(
            state=state,
            prompt=prompt,
            component_name=component_name,
            parser_output=final_parser_output,
            visual_program_candidate=visual_program_candidate,
            exc=exc,
            extra_payload={
                "raw_parser_output": parser_output.model_dump(mode="json"),
                "normalized_parser_output": normalized_output.model_dump(mode="json"),
                "normalization_report": normalization_report.model_dump(mode="json"),
                "review_report": review_report.model_dump(mode="json"),
            },
        )
        raise VisualProgramParsingError(
            str(exc),
            parser_output=final_parser_output,
            visual_program_candidate=visual_program_candidate,
            timing_log=deepcopy(state.timing_log),
        ) from exc
    record_duration(
        state.timing_log,
        "output_postprocessing_seconds",
        time.perf_counter() - postprocess_start,
        component=component_name,
        count_key="output_postprocesses",
        metadata={
            "object_count": len(visual_program.objects),
            "predicate_count": len(visual_program.predicates),
        },
    )

    record_duration(
        state.timing_log,
        "vp_visual_program_parser_seconds",
        time.perf_counter() - node_start,
        component=component_name,
        count_key="visual_program_parses",
    )
    return ParsedVisualProgramResult(
        visual_program=visual_program,
        raw_parser_output=parser_output,
        normalized_parser_output=normalized_output,
        final_parser_output=final_parser_output,
        normalization_report=normalization_report,
        review_report=review_report,
    )


__all__ = [
    "AtLeastCountConstraint",
    "AttributeConstraint",
    "ExactCountConstraint",
    "ExclusionConstraint",
    "GlobalSceneConstraint",
    "ParsedObjectSpec",
    "ParsedTextConstraint",
    "ParsedVisualProgramResult",
    "RelationConstraint",
    "VisualProgramNormalizationReport",
    "VisualProgramParserOutput",
    "VisualProgramParsingError",
    "VisualProgramReviewOutput",
    "VisualProgramReviewReport",
    "build_visual_program_parser_input",
    "build_visual_program_review_input",
    "parse_visual_program_from_prompt",
    "vp_lint_and_normalize",
]
