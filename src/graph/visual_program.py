"""Define a fixed visual-program schema for prompt-derived constraints.

This module is intentionally independent from any VLM or LLM runtime code.
The visual program is compiled once from the user prompt and should remain
stable throughout refinement. Runtime code is expected to update only the
predicate states in :class:`VisualProgramState`.

Examples:
    >>> cat = ObjectSpec.named("cat", index=1)
    >>> dog = ObjectSpec.named("dog", index=1)
    >>> program = VisualProgram.from_predicates(
    ...     source_prompt="a red cat left of a dog",
    ...     objects=[cat, dog],
    ...     predicates=[
    ...         CountPredicate.at_least(cat, 1),
    ...         CountPredicate.at_least(dog, 1),
    ...         AttributePredicate.equals(cat, "color", "red"),
    ...         SpatialRelationPredicate.relate(cat, "left_of", dog),
    ...     ],
    ... )
    >>> state = program.initial_state()
    >>> state.statuses["count:cat_1:at_least:1"].state.value
    'uncertain'
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any, Mapping, Sequence, Union


ConstraintValue = Union[str, int, float, bool]


def _normalize_text(value: str, field_name: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value


def _slugify(value: str) -> str:
    slug = []
    previous_was_separator = False
    for char in value.lower().strip():
        if char.isalnum():
            slug.append(char)
            previous_was_separator = False
            continue
        if not previous_was_separator:
            slug.append("_")
            previous_was_separator = True
    normalized = "".join(slug).strip("_")
    return normalized or "item"


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


def _python_literal(value: Any) -> str:
    if isinstance(value, Enum):
        return _python_literal(value.value)
    return repr(value)


def _object_var_name(object_id: str) -> str:
    return f"obj_{_slugify(object_id)}"


def _indent_lines(lines: Sequence[str], prefix: str) -> list[str]:
    return [f"{prefix}{line}" if line else line for line in lines]


def _render_predicate_python(
    predicate: Predicate,
    *,
    object_var_names: dict[str, str],
) -> list[str]:
    if isinstance(predicate, ExclusionPredicate):
        return [
            "ExclusionPredicate.absent(",
            f"    {object_var_names[predicate.target.object_id]},",
            f"    predicate_id={_python_literal(predicate.predicate_id)},",
            f"    description={_python_literal(predicate.description)},",
            "),",
        ]

    if isinstance(predicate, CountPredicate):
        helper_name = {
            CountComparator.EXACTLY: "exactly",
            CountComparator.AT_LEAST: "at_least",
            CountComparator.AT_MOST: "at_most",
        }.get(predicate.comparator)
        if helper_name is not None:
            return [
                f"CountPredicate.{helper_name}(",
                f"    {object_var_names[predicate.target.object_id]},",
                f"    {predicate.expected_count},",
                f"    predicate_id={_python_literal(predicate.predicate_id)},",
                f"    description={_python_literal(predicate.description)},",
                "),",
            ]
        return [
            "CountPredicate(",
            f"    predicate_id={_python_literal(predicate.predicate_id)},",
            f"    description={_python_literal(predicate.description)},",
            f"    target={object_var_names[predicate.target.object_id]},",
            f"    comparator=CountComparator.{predicate.comparator.name},",
            f"    expected_count={predicate.expected_count},",
            "),",
        ]

    if isinstance(predicate, AttributePredicate):
        if predicate.operator == "equals":
            lines = [
                "AttributePredicate.equals(",
                f"    {object_var_names[predicate.target.object_id]},",
                f"    {_python_literal(predicate.attribute_name)},",
                f"    {_python_literal(predicate.expected_value)},",
            ]
            if predicate.action_target is not None:
                lines.append(
                    "    action_target="
                    f"{object_var_names[predicate.action_target.object_id]},"
                )
            lines.extend(
                [
                    f"    predicate_id={_python_literal(predicate.predicate_id)},",
                    f"    description={_python_literal(predicate.description)},",
                    "),",
                ]
            )
            return lines
        lines = [
            "AttributePredicate(",
            f"    predicate_id={_python_literal(predicate.predicate_id)},",
            f"    description={_python_literal(predicate.description)},",
            f"    target={object_var_names[predicate.target.object_id]},",
            f"    attribute_name={_python_literal(predicate.attribute_name)},",
            f"    expected_value={_python_literal(predicate.expected_value)},",
        ]
        if predicate.action_target is not None:
            lines.append(
                "    action_target="
                f"{object_var_names[predicate.action_target.object_id]},"
            )
        lines.extend(
            [
                f"    operator={_python_literal(predicate.operator)},",
                "),",
            ]
        )
        return lines

    if isinstance(predicate, SpatialRelationPredicate):
        return [
            "SpatialRelationPredicate.relate(",
            f"    {object_var_names[predicate.subject.object_id]},",
            f"    {_python_literal(predicate.relation)},",
            f"    {object_var_names[predicate.reference.object_id]},",
            f"    predicate_id={_python_literal(predicate.predicate_id)},",
            f"    description={_python_literal(predicate.description)},",
            "),",
        ]

    if isinstance(predicate, GlobalScenePredicate):
        if predicate.operator == "equals":
            return [
                "GlobalScenePredicate.equals(",
                f"    {_python_literal(predicate.attribute_name)},",
                f"    {_python_literal(predicate.expected_value)},",
                f"    predicate_id={_python_literal(predicate.predicate_id)},",
                f"    description={_python_literal(predicate.description)},",
                "),",
            ]
        return [
            "GlobalScenePredicate(",
            f"    predicate_id={_python_literal(predicate.predicate_id)},",
            f"    description={_python_literal(predicate.description)},",
            f"    attribute_name={_python_literal(predicate.attribute_name)},",
            f"    expected_value={_python_literal(predicate.expected_value)},",
            f"    operator={_python_literal(predicate.operator)},",
            "),",
        ]

    if isinstance(predicate, TextPredicate):
        target_expr = (
            "None"
            if predicate.target is None
            else object_var_names[predicate.target.object_id]
        )
        if predicate.match_type == TextMatchType.EXACT:
            return [
                "TextPredicate.exactly(",
                f"    {_python_literal(predicate.expected_text)},",
                f"    target={target_expr},",
                f"    case_sensitive={predicate.case_sensitive},",
                f"    predicate_id={_python_literal(predicate.predicate_id)},",
                f"    description={_python_literal(predicate.description)},",
                "),",
            ]
        if predicate.match_type == TextMatchType.CONTAINS:
            return [
                "TextPredicate.contains(",
                f"    {_python_literal(predicate.expected_text)},",
                f"    target={target_expr},",
                f"    case_sensitive={predicate.case_sensitive},",
                f"    predicate_id={_python_literal(predicate.predicate_id)},",
                f"    description={_python_literal(predicate.description)},",
                "),",
            ]
        return [
            "TextPredicate(",
            f"    predicate_id={_python_literal(predicate.predicate_id)},",
            f"    description={_python_literal(predicate.description)},",
            f"    expected_text={_python_literal(predicate.expected_text)},",
            f"    match_type=TextMatchType.{predicate.match_type.name},",
            f"    target={target_expr},",
            f"    case_sensitive={predicate.case_sensitive},",
            "),",
        ]

    raise TypeError(f"Unsupported predicate type: {type(predicate).__name__}")


@dataclass(frozen=True)
class ObjectRef:
    """Reference a canonical object declared in a visual program."""

    object_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "object_id", _normalize_text(self.object_id, "object_id")
        )

    @classmethod
    def from_value(cls, value: ObjectRef | ObjectSpec | str) -> ObjectRef:
        """Coerce an object reference from a spec, ref, or object ID string."""
        if isinstance(value, cls):
            return value
        if isinstance(value, ObjectSpec):
            return value.ref()
        return cls(object_id=value)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the reference into plain Python types."""
        return _serialize(self)


@dataclass(frozen=True)
class ObjectSpec:
    """Declare an object that may be referenced by predicates.

    Attributes:
        object_id: Stable identifier used by predicates and verifier outputs.
        label: Canonical noun phrase for the object, such as ``cat`` or ``stop sign``.
        proposal_text: Optional detector-facing grounding phrase for region proposal.
        aliases: Optional alternate surface forms for matching or parsing.
        description: Optional free-form note for downstream parser/verifier modules.

    Examples:
        >>> ObjectSpec.named("cat", index=2).object_id
        'cat_2'
    """

    object_id: str
    label: str
    proposal_text: str | None = None
    aliases: tuple[str, ...] = ()
    description: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "object_id", _normalize_text(self.object_id, "object_id")
        )
        object.__setattr__(self, "label", _normalize_text(self.label, "label"))
        if self.proposal_text is not None:
            object.__setattr__(
                self,
                "proposal_text",
                _normalize_text(self.proposal_text, "proposal_text"),
            )
        object.__setattr__(
            self,
            "aliases",
            tuple(alias.strip() for alias in self.aliases if alias and alias.strip()),
        )
        if self.description is not None:
            object.__setattr__(
                self,
                "description",
                _normalize_text(self.description, "description"),
            )

    @classmethod
    def named(
        cls,
        label: str,
        *,
        object_id: str | None = None,
        index: int | None = None,
        proposal_text: str | None = None,
        aliases: Sequence[str] = (),
        description: str | None = None,
    ) -> ObjectSpec:
        """Build an object spec with a predictable ID.

        Args:
            label: Canonical noun phrase for the object.
            object_id: Optional explicit stable identifier.
            index: Optional numeric suffix when the prompt contains repeated objects.
            proposal_text: Optional detector-facing grounding phrase.
            aliases: Optional alternate labels.
            description: Optional free-form annotation.
        """
        label = _normalize_text(label, "label")
        if index is not None and index < 1:
            raise ValueError("index must be at least 1 when provided.")
        default_id = _slugify(label)
        if index is not None:
            default_id = f"{default_id}_{index}"
        return cls(
            object_id=object_id or default_id,
            label=label,
            proposal_text=proposal_text,
            aliases=tuple(aliases),
            description=description,
        )

    def ref(self) -> ObjectRef:
        """Return a stable reference that predicates can use."""
        return ObjectRef(object_id=self.object_id)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the object spec into plain Python types."""
        return _serialize(self)


class PredicateState(str, Enum):
    """Describe the verification state of a predicate."""

    SATISFIED = "satisfied"
    VIOLATED = "violated"
    UNCERTAIN = "uncertain"
    CONTRADICTION = "contradiction"


class PredicateCategory(str, Enum):
    """Enumerate supported predicate families in the visual program."""

    OBJECT_COUNT = "object_count"
    ATTRIBUTE_CONSTRAINT = "attribute_constraint"
    SPATIAL_RELATION = "spatial_relation"
    GLOBAL_SCENE_CONSTRAINT = "global_scene_constraint"
    EXCLUSION = "exclusion"
    TEXT_CONSTRAINT = "text_constraint"


class CountComparator(str, Enum):
    """Enumerate numeric comparisons for count predicates."""

    EXACTLY = "exactly"
    AT_LEAST = "at_least"
    AT_MOST = "at_most"


class TextMatchType(str, Enum):
    """Enumerate supported text-matching styles."""

    EXACT = "exact"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"


@dataclass(frozen=True)
class Predicate:
    """Base class for visual-program predicates."""

    predicate_id: str
    description: str
    category: PredicateCategory = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "predicate_id", _normalize_text(self.predicate_id, "predicate_id")
        )
        object.__setattr__(
            self, "description", _normalize_text(self.description, "description")
        )

    def referenced_object_ids(self) -> tuple[str, ...]:
        """Return all object IDs touched by the predicate."""
        return ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the predicate into plain Python types."""
        return _serialize(self)


@dataclass(frozen=True)
class CountPredicate(Predicate):
    """Constrain how many instances of an object should appear."""

    target: ObjectRef
    comparator: CountComparator
    expected_count: int
    category: PredicateCategory = field(
        default=PredicateCategory.OBJECT_COUNT,
        init=False,
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "target", ObjectRef.from_value(self.target))
        if self.expected_count < 0:
            raise ValueError("expected_count must be non-negative.")

    @classmethod
    def exactly(
        cls,
        target: ObjectRef | ObjectSpec | str,
        expected_count: int,
        *,
        predicate_id: str | None = None,
        description: str | None = None,
    ) -> CountPredicate:
        """Construct an exact-count predicate."""
        target_ref = ObjectRef.from_value(target)
        return cls(
            predicate_id=predicate_id
            or f"count:{target_ref.object_id}:exactly:{expected_count}",
            description=description
            or f"There must be exactly {expected_count} instance(s) of {target_ref.object_id}.",
            target=target_ref,
            comparator=CountComparator.EXACTLY,
            expected_count=expected_count,
        )

    @classmethod
    def at_least(
        cls,
        target: ObjectRef | ObjectSpec | str,
        expected_count: int,
        *,
        predicate_id: str | None = None,
        description: str | None = None,
    ) -> CountPredicate:
        """Construct a lower-bound count predicate."""
        target_ref = ObjectRef.from_value(target)
        return cls(
            predicate_id=predicate_id
            or f"count:{target_ref.object_id}:at_least:{expected_count}",
            description=description
            or f"There must be at least {expected_count} instance(s) of {target_ref.object_id}.",
            target=target_ref,
            comparator=CountComparator.AT_LEAST,
            expected_count=expected_count,
        )

    @classmethod
    def at_most(
        cls,
        target: ObjectRef | ObjectSpec | str,
        expected_count: int,
        *,
        predicate_id: str | None = None,
        description: str | None = None,
    ) -> CountPredicate:
        """Construct an upper-bound count predicate."""
        target_ref = ObjectRef.from_value(target)
        return cls(
            predicate_id=predicate_id
            or f"count:{target_ref.object_id}:at_most:{expected_count}",
            description=description
            or f"There must be at most {expected_count} instance(s) of {target_ref.object_id}.",
            target=target_ref,
            comparator=CountComparator.AT_MOST,
            expected_count=expected_count,
        )

    def referenced_object_ids(self) -> tuple[str, ...]:
        return (self.target.object_id,)


@dataclass(frozen=True)
class AttributePredicate(Predicate):
    """Constrain an attribute value for a declared object."""

    target: ObjectRef
    attribute_name: str
    expected_value: ConstraintValue
    action_target: ObjectRef | None = None
    operator: str = "equals"
    category: PredicateCategory = field(
        default=PredicateCategory.ATTRIBUTE_CONSTRAINT,
        init=False,
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "target", ObjectRef.from_value(self.target))
        object.__setattr__(
            self,
            "attribute_name",
            _normalize_text(self.attribute_name, "attribute_name"),
        )
        if self.action_target is not None:
            object.__setattr__(
                self,
                "action_target",
                ObjectRef.from_value(self.action_target),
            )
        object.__setattr__(self, "operator", _normalize_text(self.operator, "operator"))
        if self.action_target is not None and self.attribute_name != "action":
            raise ValueError(
                "action_target may be set only when attribute_name='action'."
            )

    @classmethod
    def equals(
        cls,
        target: ObjectRef | ObjectSpec | str,
        attribute_name: str,
        expected_value: ConstraintValue,
        *,
        action_target: ObjectRef | ObjectSpec | str | None = None,
        predicate_id: str | None = None,
        description: str | None = None,
    ) -> AttributePredicate:
        """Construct an attribute-equality predicate."""
        target_ref = ObjectRef.from_value(target)
        attribute_name = _normalize_text(attribute_name, "attribute_name")
        action_target_ref = (
            None if action_target is None else ObjectRef.from_value(action_target)
        )
        predicate_key = (
            f"attribute:{target_ref.object_id}:{_slugify(attribute_name)}:"
            f"{_slugify(str(expected_value))}"
        )
        if action_target_ref is not None:
            predicate_key = f"{predicate_key}:{action_target_ref.object_id}"
        if description is None:
            if attribute_name == "action" and action_target_ref is not None:
                description = (
                    f"{target_ref.object_id} should be {expected_value} "
                    f"{action_target_ref.object_id}."
                )
            elif attribute_name == "action":
                description = f"{target_ref.object_id} should be {expected_value}."
            else:
                description = f"{target_ref.object_id}.{attribute_name} must equal {expected_value!r}."
        return cls(
            predicate_id=predicate_id or predicate_key,
            description=description,
            target=target_ref,
            attribute_name=attribute_name,
            expected_value=expected_value,
            action_target=action_target_ref,
        )

    def referenced_object_ids(self) -> tuple[str, ...]:
        if self.action_target is None:
            return (self.target.object_id,)
        return (self.target.object_id, self.action_target.object_id)


@dataclass(frozen=True)
class SpatialRelationPredicate(Predicate):
    """Constrain a spatial relation between two declared objects."""

    subject: ObjectRef
    relation: str
    reference: ObjectRef
    category: PredicateCategory = field(
        default=PredicateCategory.SPATIAL_RELATION,
        init=False,
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "subject", ObjectRef.from_value(self.subject))
        object.__setattr__(self, "reference", ObjectRef.from_value(self.reference))
        object.__setattr__(self, "relation", _normalize_text(self.relation, "relation"))

    @classmethod
    def relate(
        cls,
        subject: ObjectRef | ObjectSpec | str,
        relation: str,
        reference: ObjectRef | ObjectSpec | str,
        *,
        predicate_id: str | None = None,
        description: str | None = None,
    ) -> SpatialRelationPredicate:
        """Construct a spatial relation predicate."""
        subject_ref = ObjectRef.from_value(subject)
        reference_ref = ObjectRef.from_value(reference)
        relation = _normalize_text(relation, "relation")
        return cls(
            predicate_id=predicate_id
            or f"spatial:{subject_ref.object_id}:{_slugify(relation)}:{reference_ref.object_id}",
            description=description
            or f"{subject_ref.object_id} must be {relation} {reference_ref.object_id}.",
            subject=subject_ref,
            relation=relation,
            reference=reference_ref,
        )

    def referenced_object_ids(self) -> tuple[str, ...]:
        return (self.subject.object_id, self.reference.object_id)


@dataclass(frozen=True)
class GlobalScenePredicate(Predicate):
    """Constrain a holistic scene-level property such as environment or background."""

    attribute_name: str
    expected_value: ConstraintValue
    operator: str = "equals"
    category: PredicateCategory = field(
        default=PredicateCategory.GLOBAL_SCENE_CONSTRAINT,
        init=False,
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(
            self,
            "attribute_name",
            _normalize_text(self.attribute_name, "attribute_name"),
        )
        object.__setattr__(self, "operator", _normalize_text(self.operator, "operator"))

    @classmethod
    def equals(
        cls,
        attribute_name: str,
        expected_value: ConstraintValue,
        *,
        predicate_id: str | None = None,
        description: str | None = None,
    ) -> GlobalScenePredicate:
        """Construct a global scene predicate."""
        attribute_name = _normalize_text(attribute_name, "attribute_name")
        return cls(
            predicate_id=predicate_id
            or f"global_scene:{_slugify(attribute_name)}:{_slugify(str(expected_value))}",
            description=description
            or f"The scene-level {attribute_name} must equal {expected_value!r}.",
            attribute_name=attribute_name,
            expected_value=expected_value,
        )


@dataclass(frozen=True)
class ExclusionPredicate(Predicate):
    """Require an object not to exist in the generated scene."""

    target: ObjectRef
    category: PredicateCategory = field(
        default=PredicateCategory.EXCLUSION,
        init=False,
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "target", ObjectRef.from_value(self.target))

    @classmethod
    def absent(
        cls,
        target: ObjectRef | ObjectSpec | str,
        *,
        predicate_id: str | None = None,
        description: str | None = None,
    ) -> ExclusionPredicate:
        """Construct an exclusion predicate."""
        target_ref = ObjectRef.from_value(target)
        return cls(
            predicate_id=predicate_id or f"exclude:{target_ref.object_id}",
            description=description
            or f"{target_ref.object_id} must not appear in the scene.",
            target=target_ref,
        )

    def referenced_object_ids(self) -> tuple[str, ...]:
        return (self.target.object_id,)


@dataclass(frozen=True)
class TextPredicate(Predicate):
    """Constrain visible text in the image, optionally on a specific object."""

    expected_text: str
    match_type: TextMatchType = TextMatchType.EXACT
    target: ObjectRef | None = None
    case_sensitive: bool = False
    category: PredicateCategory = field(
        default=PredicateCategory.TEXT_CONSTRAINT,
        init=False,
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(
            self,
            "expected_text",
            _normalize_text(self.expected_text, "expected_text"),
        )
        if self.target is not None:
            object.__setattr__(self, "target", ObjectRef.from_value(self.target))

    @classmethod
    def exactly(
        cls,
        expected_text: str,
        *,
        target: ObjectRef | ObjectSpec | str | None = None,
        case_sensitive: bool = False,
        predicate_id: str | None = None,
        description: str | None = None,
    ) -> TextPredicate:
        """Construct an exact text predicate."""
        expected_text = _normalize_text(expected_text, "expected_text")
        target_ref = None if target is None else ObjectRef.from_value(target)
        text_key = _slugify(expected_text)
        target_key = "scene" if target_ref is None else target_ref.object_id
        return cls(
            predicate_id=predicate_id or f"text:{target_key}:exact:{text_key}",
            description=description
            or f"Visible text for {target_key} must equal {expected_text!r}.",
            expected_text=expected_text,
            match_type=TextMatchType.EXACT,
            target=target_ref,
            case_sensitive=case_sensitive,
        )

    @classmethod
    def contains(
        cls,
        expected_text: str,
        *,
        target: ObjectRef | ObjectSpec | str | None = None,
        case_sensitive: bool = False,
        predicate_id: str | None = None,
        description: str | None = None,
    ) -> TextPredicate:
        """Construct a substring text predicate."""
        expected_text = _normalize_text(expected_text, "expected_text")
        target_ref = None if target is None else ObjectRef.from_value(target)
        text_key = _slugify(expected_text)
        target_key = "scene" if target_ref is None else target_ref.object_id
        return cls(
            predicate_id=predicate_id or f"text:{target_key}:contains:{text_key}",
            description=description
            or f"Visible text for {target_key} must contain {expected_text!r}.",
            expected_text=expected_text,
            match_type=TextMatchType.CONTAINS,
            target=target_ref,
            case_sensitive=case_sensitive,
        )

    def referenced_object_ids(self) -> tuple[str, ...]:
        if self.target is None:
            return ()
        return (self.target.object_id,)


@dataclass
class PredicateStatus:
    """Track the mutable verification state for one predicate."""

    predicate_id: str
    state: PredicateState = PredicateState.UNCERTAIN
    rationale: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.predicate_id = _normalize_text(self.predicate_id, "predicate_id")
        if self.rationale is not None:
            self.rationale = _normalize_text(self.rationale, "rationale")

    @classmethod
    def pending(
        cls,
        predicate: Predicate,
        *,
        state: PredicateState = PredicateState.UNCERTAIN,
        rationale: str | None = None,
    ) -> PredicateStatus:
        """Construct a status record for a known predicate."""
        return cls(
            predicate_id=predicate.predicate_id, state=state, rationale=rationale
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the predicate status into plain Python types."""
        return _serialize(self)


@dataclass
class VisualProgramState:
    """Store mutable predicate states while the visual program stays fixed."""

    statuses: dict[str, PredicateStatus] = field(default_factory=dict)

    @classmethod
    def for_program(
        cls,
        program: VisualProgram,
        *,
        default_state: PredicateState = PredicateState.UNCERTAIN,
    ) -> VisualProgramState:
        """Initialize an all-predicates state view for a program."""
        return cls(
            statuses={
                predicate.predicate_id: PredicateStatus.pending(
                    predicate,
                    state=default_state,
                )
                for predicate in program.predicates
            }
        )

    def update(
        self,
        predicate_id: str,
        state: PredicateState,
        *,
        rationale: str | None = None,
        evidence: dict[str, Any] | None = None,
    ) -> None:
        """Update the state of a known predicate."""
        predicate_id = _normalize_text(predicate_id, "predicate_id")
        if predicate_id not in self.statuses:
            raise KeyError(f"Unknown predicate_id: {predicate_id}")

        current = self.statuses[predicate_id]
        self.statuses[predicate_id] = PredicateStatus(
            predicate_id=predicate_id,
            state=state,
            rationale=rationale if rationale is not None else current.rationale,
            evidence=dict(current.evidence if evidence is None else evidence),
        )

    def unresolved_predicate_ids(self) -> tuple[str, ...]:
        """Return predicates that are not currently satisfied."""
        return tuple(
            predicate_id
            for predicate_id, status in self.statuses.items()
            if status.state is not PredicateState.SATISFIED
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the program state into plain Python types."""
        return _serialize(self)


@dataclass(frozen=True)
class VisualProgram:
    """Represent a fixed, prompt-derived visual specification.

    Attributes:
        source_prompt: Original user prompt from which the program was compiled.
        objects: Canonical object declarations referenced by predicates.
        predicates: Predicate list that stays fixed during refinement.
        rewritten_prompts: Optional resample-safe prompt rewrites that preserve the
            same scored scene semantics while varying wording or adding non-scoring detail.
        program_id: Optional stable identifier for storing or logging the program.

    Examples:
        >>> program = VisualProgram.example()
        >>> program.program_id
        'visual_program'
    """

    source_prompt: str
    objects: tuple[ObjectSpec, ...]
    predicates: tuple[Predicate, ...]
    rewritten_prompts: tuple[str, ...] = ()
    program_id: str = "visual_program"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_prompt",
            _normalize_text(self.source_prompt, "source_prompt"),
        )
        normalized_rewrites: list[str] = []
        seen_rewrites: set[str] = set()
        for index, rewrite in enumerate(self.rewritten_prompts):
            normalized = _normalize_text(rewrite, f"rewritten_prompts[{index}]")
            if normalized == self.source_prompt or normalized in seen_rewrites:
                continue
            normalized_rewrites.append(normalized)
            seen_rewrites.add(normalized)
        object.__setattr__(self, "rewritten_prompts", tuple(normalized_rewrites))
        object.__setattr__(
            self, "program_id", _normalize_text(self.program_id, "program_id")
        )
        object.__setattr__(self, "objects", tuple(self.objects))
        object.__setattr__(self, "predicates", tuple(self.predicates))

        if not self.predicates:
            raise ValueError("predicates must contain at least one predicate.")

        object_ids = [obj.object_id for obj in self.objects]
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("objects must use unique object_id values.")

        predicate_ids = [predicate.predicate_id for predicate in self.predicates]
        if len(predicate_ids) != len(set(predicate_ids)):
            raise ValueError("predicates must use unique predicate_id values.")

        known_object_ids = set(object_ids)
        for predicate in self.predicates:
            missing_ids = set(predicate.referenced_object_ids()) - known_object_ids
            if missing_ids:
                raise ValueError(
                    f"Predicate {predicate.predicate_id!r} references unknown objects: "
                    f"{sorted(missing_ids)!r}"
                )

    @classmethod
    def from_predicates(
        cls,
        *,
        source_prompt: str,
        objects: Sequence[ObjectSpec],
        predicates: Sequence[Predicate],
        rewritten_prompts: Sequence[str] = (),
        program_id: str = "visual_program",
    ) -> VisualProgram:
        """Build a program from sequence inputs."""
        return cls(
            source_prompt=source_prompt,
            objects=tuple(objects),
            predicates=tuple(predicates),
            rewritten_prompts=tuple(rewritten_prompts),
            program_id=program_id,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> VisualProgram:
        """Deserialize one exact canonical runtime visual-program payload.

        The loader is intentionally strict: unknown or missing keys, invalid
        primitive types, unsupported predicate categories, and payloads that do
        not round-trip to the canonical serialization are rejected.  The only
        compatibility exception is surrounding whitespace in ``source_prompt``
        from frozen legacy results; the runtime object still normalizes it.
        """

        def require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
            if not isinstance(value, Mapping):
                raise ValueError(f"{field_name} must be a mapping.")
            return value

        def require_exact_keys(
            value: Mapping[str, Any], expected: set[str], field_name: str
        ) -> None:
            actual = set(value)
            if actual != expected:
                missing = sorted(expected - actual)
                extra = sorted(actual - expected)
                raise ValueError(
                    f"{field_name} keys do not match the runtime schema; "
                    f"missing={missing}, extra={extra}."
                )

        def require_string(value: Any, field_name: str) -> str:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string.")
            return value

        def require_constraint_value(value: Any, field_name: str) -> ConstraintValue:
            if not isinstance(value, (str, int, float, bool)):
                raise ValueError(
                    f"{field_name} must be a string, integer, float, or boolean."
                )
            return value

        def object_ref(value: Any, field_name: str) -> ObjectRef:
            mapping = require_mapping(value, field_name)
            require_exact_keys(mapping, {"object_id"}, field_name)
            return ObjectRef(
                require_string(mapping["object_id"], f"{field_name}.object_id")
            )

        def optional_object_ref(value: Any, field_name: str) -> ObjectRef | None:
            if value is None:
                return None
            return object_ref(value, field_name)

        program = require_mapping(payload, "visual_program")
        require_exact_keys(
            program,
            {
                "source_prompt",
                "objects",
                "predicates",
                "rewritten_prompts",
                "program_id",
            },
            "visual_program",
        )

        object_values = program["objects"]
        if not isinstance(object_values, list):
            raise ValueError("visual_program.objects must be a list.")
        objects: list[ObjectSpec] = []
        for index, value in enumerate(object_values):
            field_name = f"visual_program.objects[{index}]"
            mapping = require_mapping(value, field_name)
            require_exact_keys(
                mapping,
                {"object_id", "label", "proposal_text", "aliases", "description"},
                field_name,
            )
            aliases = mapping["aliases"]
            if not isinstance(aliases, list) or any(
                not isinstance(alias, str) for alias in aliases
            ):
                raise ValueError(f"{field_name}.aliases must be a list of strings.")
            proposal_text = mapping["proposal_text"]
            description = mapping["description"]
            if proposal_text is not None and not isinstance(proposal_text, str):
                raise ValueError(
                    f"{field_name}.proposal_text must be a string or null."
                )
            if description is not None and not isinstance(description, str):
                raise ValueError(f"{field_name}.description must be a string or null.")
            objects.append(
                ObjectSpec(
                    object_id=require_string(
                        mapping["object_id"], f"{field_name}.object_id"
                    ),
                    label=require_string(mapping["label"], f"{field_name}.label"),
                    proposal_text=proposal_text,
                    aliases=tuple(aliases),
                    description=description,
                )
            )

        predicate_values = program["predicates"]
        if not isinstance(predicate_values, list):
            raise ValueError("visual_program.predicates must be a list.")
        predicates: list[Predicate] = []
        for index, value in enumerate(predicate_values):
            field_name = f"visual_program.predicates[{index}]"
            mapping = require_mapping(value, field_name)
            category = mapping.get("category")
            common = {
                "predicate_id": require_string(
                    mapping.get("predicate_id"), f"{field_name}.predicate_id"
                ),
                "description": require_string(
                    mapping.get("description"), f"{field_name}.description"
                ),
            }

            if category == PredicateCategory.OBJECT_COUNT.value:
                require_exact_keys(
                    mapping,
                    {
                        "predicate_id",
                        "description",
                        "category",
                        "target",
                        "comparator",
                        "expected_count",
                    },
                    field_name,
                )
                expected_count = mapping["expected_count"]
                if not isinstance(expected_count, int) or isinstance(
                    expected_count, bool
                ):
                    raise ValueError(f"{field_name}.expected_count must be an integer.")
                try:
                    comparator = CountComparator(mapping["comparator"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{field_name}.comparator is not supported."
                    ) from exc
                predicates.append(
                    CountPredicate(
                        **common,
                        target=object_ref(mapping["target"], f"{field_name}.target"),
                        comparator=comparator,
                        expected_count=expected_count,
                    )
                )
                continue

            if category == PredicateCategory.ATTRIBUTE_CONSTRAINT.value:
                require_exact_keys(
                    mapping,
                    {
                        "predicate_id",
                        "description",
                        "category",
                        "target",
                        "attribute_name",
                        "expected_value",
                        "action_target",
                        "operator",
                    },
                    field_name,
                )
                predicates.append(
                    AttributePredicate(
                        **common,
                        target=object_ref(mapping["target"], f"{field_name}.target"),
                        attribute_name=require_string(
                            mapping["attribute_name"], f"{field_name}.attribute_name"
                        ),
                        expected_value=require_constraint_value(
                            mapping["expected_value"], f"{field_name}.expected_value"
                        ),
                        action_target=optional_object_ref(
                            mapping["action_target"], f"{field_name}.action_target"
                        ),
                        operator=require_string(
                            mapping["operator"], f"{field_name}.operator"
                        ),
                    )
                )
                continue

            if category == PredicateCategory.SPATIAL_RELATION.value:
                require_exact_keys(
                    mapping,
                    {
                        "predicate_id",
                        "description",
                        "category",
                        "subject",
                        "relation",
                        "reference",
                    },
                    field_name,
                )
                predicates.append(
                    SpatialRelationPredicate(
                        **common,
                        subject=object_ref(mapping["subject"], f"{field_name}.subject"),
                        relation=require_string(
                            mapping["relation"], f"{field_name}.relation"
                        ),
                        reference=object_ref(
                            mapping["reference"], f"{field_name}.reference"
                        ),
                    )
                )
                continue

            if category == PredicateCategory.GLOBAL_SCENE_CONSTRAINT.value:
                require_exact_keys(
                    mapping,
                    {
                        "predicate_id",
                        "description",
                        "category",
                        "attribute_name",
                        "expected_value",
                        "operator",
                    },
                    field_name,
                )
                predicates.append(
                    GlobalScenePredicate(
                        **common,
                        attribute_name=require_string(
                            mapping["attribute_name"], f"{field_name}.attribute_name"
                        ),
                        expected_value=require_constraint_value(
                            mapping["expected_value"], f"{field_name}.expected_value"
                        ),
                        operator=require_string(
                            mapping["operator"], f"{field_name}.operator"
                        ),
                    )
                )
                continue

            if category == PredicateCategory.EXCLUSION.value:
                require_exact_keys(
                    mapping,
                    {"predicate_id", "description", "category", "target"},
                    field_name,
                )
                predicates.append(
                    ExclusionPredicate(
                        **common,
                        target=object_ref(mapping["target"], f"{field_name}.target"),
                    )
                )
                continue

            if category == PredicateCategory.TEXT_CONSTRAINT.value:
                require_exact_keys(
                    mapping,
                    {
                        "predicate_id",
                        "description",
                        "category",
                        "expected_text",
                        "match_type",
                        "target",
                        "case_sensitive",
                    },
                    field_name,
                )
                try:
                    match_type = TextMatchType(mapping["match_type"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{field_name}.match_type is not supported."
                    ) from exc
                case_sensitive = mapping["case_sensitive"]
                if not isinstance(case_sensitive, bool):
                    raise ValueError(f"{field_name}.case_sensitive must be boolean.")
                predicates.append(
                    TextPredicate(
                        **common,
                        expected_text=require_string(
                            mapping["expected_text"], f"{field_name}.expected_text"
                        ),
                        match_type=match_type,
                        target=optional_object_ref(
                            mapping["target"], f"{field_name}.target"
                        ),
                        case_sensitive=case_sensitive,
                    )
                )
                continue

            raise ValueError(f"{field_name}.category is invalid: {category!r}.")

        rewritten_prompts = program["rewritten_prompts"]
        if not isinstance(rewritten_prompts, list) or any(
            not isinstance(prompt, str) for prompt in rewritten_prompts
        ):
            raise ValueError(
                "visual_program.rewritten_prompts must be a list of strings."
            )

        runtime_program = cls.from_predicates(
            source_prompt=require_string(
                program["source_prompt"], "visual_program.source_prompt"
            ),
            objects=objects,
            predicates=predicates,
            rewritten_prompts=rewritten_prompts,
            program_id=require_string(
                program["program_id"], "visual_program.program_id"
            ),
        )
        canonical = runtime_program.to_dict()
        if canonical != dict(program):
            prompt_preserving_canonical = dict(canonical)
            if (
                str(canonical.get("source_prompt", "")).strip()
                == str(program.get("source_prompt", "")).strip()
            ):
                prompt_preserving_canonical["source_prompt"] = program["source_prompt"]
            if prompt_preserving_canonical != dict(program):
                raise ValueError(
                    "visual_program is not the exact canonical runtime serialization."
                )
        return runtime_program

    @classmethod
    def example(cls) -> VisualProgram:
        """Build a small example program useful in docs and tests."""
        cat = ObjectSpec.named("cat", index=1)
        mat = ObjectSpec.named("mat", index=1)
        return cls.from_predicates(
            source_prompt="a red cat sitting on a mat",
            objects=[cat, mat],
            predicates=[
                CountPredicate.at_least(cat, 1),
                CountPredicate.at_least(mat, 1),
                AttributePredicate.equals(cat, "color", "red"),
                SpatialRelationPredicate.relate(cat, "on", mat),
            ],
        )

    def get_object(self, object_id: str) -> ObjectSpec:
        """Look up a declared object by ID."""
        object_id = _normalize_text(object_id, "object_id")
        for obj in self.objects:
            if obj.object_id == object_id:
                return obj
        raise KeyError(f"Unknown object_id: {object_id}")

    def get_predicate(self, predicate_id: str) -> Predicate:
        """Look up a predicate by ID."""
        predicate_id = _normalize_text(predicate_id, "predicate_id")
        for predicate in self.predicates:
            if predicate.predicate_id == predicate_id:
                return predicate
        raise KeyError(f"Unknown predicate_id: {predicate_id}")

    def initial_state(
        self,
        *,
        default_state: PredicateState = PredicateState.UNCERTAIN,
    ) -> VisualProgramState:
        """Initialize mutable predicate states for this fixed program."""
        return VisualProgramState.for_program(self, default_state=default_state)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the visual program into plain Python types."""
        return _serialize(self)

    def to_python_source(self) -> str:
        """Render the visual program as readable Python source."""
        object_var_names = {
            obj.object_id: _object_var_name(obj.object_id) for obj in self.objects
        }

        lines = [
            '"""Auto-generated readable view of the compiled visual program."""',
            "",
            "from graph.visual_program import (",
            "    AttributePredicate,",
            "    CountComparator,",
            "    CountPredicate,",
            "    ExclusionPredicate,",
            "    GlobalScenePredicate,",
            "    ObjectSpec,",
            "    SpatialRelationPredicate,",
            "    TextMatchType,",
            "    TextPredicate,",
            "    VisualProgram,",
            ")",
            "",
        ]

        if self.objects:
            lines.append("# Objects")
            for obj in self.objects:
                lines.extend(
                    [
                        f"{object_var_names[obj.object_id]} = ObjectSpec(",
                        f"    object_id={_python_literal(obj.object_id)},",
                        f"    label={_python_literal(obj.label)},",
                        f"    proposal_text={_python_literal(obj.proposal_text)},",
                        f"    aliases={_python_literal(obj.aliases)},",
                        f"    description={_python_literal(obj.description)},",
                        ")",
                        "",
                    ]
                )

        lines.extend(
            [
                "visual_program = VisualProgram.from_predicates(",
                f"    source_prompt={_python_literal(self.source_prompt)},",
                "    objects=[",
            ]
        )
        for obj in self.objects:
            lines.append(f"        {object_var_names[obj.object_id]},")
        lines.extend(
            [
                "    ],",
                "    predicates=[",
            ]
        )
        for predicate in self.predicates:
            lines.extend(
                _indent_lines(
                    _render_predicate_python(
                        predicate,
                        object_var_names=object_var_names,
                    ),
                    "        ",
                )
            )
        lines.extend(
            [
                "    ],",
                f"    rewritten_prompts={_python_literal(self.rewritten_prompts)},",
                f"    program_id={_python_literal(self.program_id)},",
                ")",
            ]
        )
        return "\n".join(lines)


__all__ = [
    "AttributePredicate",
    "ConstraintValue",
    "CountComparator",
    "CountPredicate",
    "ExclusionPredicate",
    "GlobalScenePredicate",
    "ObjectRef",
    "ObjectSpec",
    "Predicate",
    "PredicateCategory",
    "PredicateState",
    "PredicateStatus",
    "SpatialRelationPredicate",
    "TextMatchType",
    "TextPredicate",
    "VisualProgram",
    "VisualProgramState",
]
