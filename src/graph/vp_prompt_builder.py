"""Prompt and edit-instruction builders for the VP refinement loop."""

from __future__ import annotations

import random
from typing import Any, Sequence

from graph.visual_program import (
    AttributePredicate,
    CountComparator,
    CountPredicate,
    SpatialRelationPredicate,
)
from graph.visual_program_action_selector import (
    ActionSelection,
    EditActionParameters,
    EditOperationType,
    RefinementActionType,
)
from graph.visual_program_support import (
    normalize_attribute_name,
    normalize_spatial_relation_name,
)

_MULTI_PANEL_LAYOUT_CUES = (
    "split screen",
    "split screen layout",
    "split-screen",
    "collage",
    "contact sheet",
    "diptych",
    "triptych",
    "comic panel",
    "comic panels",
    "panel layout",
    "multi panel",
    "multi-panel",
    "grid layout",
    "tiled grid",
    "comparison board",
    "before and after",
    "before-after",
)


def _humanize_relation(relation: str) -> str:
    normalized_relation = normalize_spatial_relation_name(relation)
    mapping = {
        "left_of": "to the left of",
        "right_of": "to the right of",
        "above": "above",
        "below": "below",
        "inside": "inside",
        "in": "in",
        "overlapping": "overlapping",
        "intersecting": "intersecting",
        "near": "near",
        "on": "on",
        "in_front_of": "in front of",
        "behind": "behind",
    }
    return mapping.get(normalized_relation, normalized_relation.replace("_", " "))


def _inverse_relation(relation: str) -> str | None:
    normalized_relation = normalize_spatial_relation_name(relation)
    mapping = {
        "left_of": "right_of",
        "right_of": "left_of",
        "above": "below",
        "below": "above",
        "in_front_of": "behind",
        "behind": "in_front_of",
        "near": "near",
        "overlapping": "overlapping",
    }
    return mapping.get(normalized_relation)


def _humanize_count(count: int) -> str:
    words = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
    }
    return words.get(int(count), str(count))


def _pluralize_label(label: str, count: int) -> str:
    if count == 1:
        return label
    parts = label.split()
    if not parts:
        return label
    last = parts[-1]
    lower = last.lower()
    if lower.endswith(("s", "x", "z", "ch", "sh")):
        plural_last = f"{last}es"
    elif (
        len(last) > 1
        and lower.endswith("y")
        and lower[-2] not in {"a", "e", "i", "o", "u"}
    ):
        plural_last = f"{last[:-1]}ies"
    else:
        plural_last = f"{last}s"
    parts[-1] = plural_last
    return " ".join(parts)


def _ensure_sentence(text: str) -> str:
    sentence = " ".join(text.strip().split())
    if not sentence:
        return ""
    if sentence[-1] in {".", "!", "?"}:
        return sentence
    return f"{sentence}."


def _normalize_prompt_cue_text(text: str) -> str:
    return " ".join(str(text).strip().lower().replace("-", " ").split())


def _prompt_requests_multi_panel_layout(text: str | None) -> bool:
    normalized = _normalize_prompt_cue_text(text or "")
    if not normalized:
        return False
    return any(cue in normalized for cue in _MULTI_PANEL_LAYOUT_CUES)


def _dedupe_nonempty_parts(parts: Sequence[str]) -> tuple[str, ...]:
    unique_parts: list[str] = []
    for part in parts:
        if part and part not in unique_parts:
            unique_parts.append(part)
    return tuple(unique_parts)


def _join_nonempty_parts(parts: Sequence[str]) -> str:
    return " ".join(part for part in parts if part)


def _build_scene_prompt_guidance(
    *,
    program,
    base_prompt: str,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    guidance: list[str] = []
    explicit_layout_requested = _prompt_requests_multi_panel_layout(
        program.source_prompt
    ) or _prompt_requests_multi_panel_layout(base_prompt)
    if not explicit_layout_requested:
        guidance.append("The image is a single coherent scene in one frame.")

    unique_guidance = _dedupe_nonempty_parts(guidance)
    metadata = {
        "scene_prompt_guidance": list(unique_guidance),
        "anti_grid_guidance_applied": not explicit_layout_requested,
    }
    return unique_guidance, metadata


def compose_prompt_with_scene_guidance(
    *,
    program,
    base_prompt: str,
    tail_parts: Sequence[str] = (),
) -> tuple[str, dict[str, Any]]:
    guidance, metadata = _build_scene_prompt_guidance(
        program=program,
        base_prompt=base_prompt,
    )
    prompt_parts = [
        _ensure_sentence(base_prompt),
        *guidance,
        *(_ensure_sentence(part) for part in tail_parts if part and str(part).strip()),
    ]
    return _join_nonempty_parts(_dedupe_nonempty_parts(prompt_parts)), metadata


def _count_requirement_fragment(
    label: str, expected_count: int, comparator: CountComparator
) -> str:
    noun_phrase = (
        f"{_humanize_count(expected_count)} {_pluralize_label(label, expected_count)}"
    )
    verb = "is" if expected_count == 1 else "are"
    if comparator == CountComparator.EXACTLY:
        return f"exactly {noun_phrase} {verb} visible"
    if comparator == CountComparator.AT_LEAST:
        return f"at least {noun_phrase} {verb} visible"
    return f"at most {noun_phrase} {verb} visible"


def _expected_object_count(program, object_id: str) -> int | None:
    for predicate in program.predicates:
        if not isinstance(predicate, CountPredicate):
            continue
        if predicate.target.object_id != object_id:
            continue
        if predicate.comparator not in {
            CountComparator.EXACTLY,
            CountComparator.AT_LEAST,
        }:
            continue
        return predicate.expected_count
    return None


def _descriptive_object_label(program, object_id: str) -> str:
    label = program.get_object(object_id).label
    attribute_values: dict[str, str] = {}
    attribute_priority = {
        "size": 0,
        "color": 1,
        "material": 2,
        "pattern": 3,
        "shape": 4,
        "type": 5,
    }
    for predicate in program.predicates:
        if not isinstance(predicate, AttributePredicate):
            continue
        if predicate.target.object_id != object_id:
            continue
        if predicate.attribute_name not in attribute_priority:
            continue
        attribute_values[predicate.attribute_name] = str(
            predicate.expected_value
        ).strip()

    modifiers = [
        value
        for _, value in sorted(
            (
                (attribute_priority[name], value)
                for name, value in attribute_values.items()
                if value
            ),
            key=lambda item: item[0],
        )
    ]
    if not modifiers:
        return label
    return " ".join([*modifiers, label]).strip()


def _definite_object_phrase(program, object_id: str) -> str:
    label = _descriptive_object_label(program, object_id)
    expected_count = _expected_object_count(program, object_id)
    if expected_count is not None and expected_count > 1:
        return f"the {_pluralize_label(label, expected_count)}"
    return f"the {label}"


def _natural_language_join(items: Sequence[str]) -> str:
    phrases = [item.strip() for item in items if item and item.strip()]
    if not phrases:
        return ""
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    return f"{', '.join(phrases[:-1])}, and {phrases[-1]}"


def _required_positive_object_ids(program) -> tuple[str, ...]:
    object_ids: list[str] = []
    for predicate in program.predicates:
        if not isinstance(predicate, CountPredicate):
            continue
        if predicate.comparator not in {
            CountComparator.EXACTLY,
            CountComparator.AT_LEAST,
        }:
            continue
        if predicate.expected_count <= 0:
            continue
        object_id = predicate.target.object_id
        if object_id in object_ids:
            continue
        object_ids.append(object_id)
    return tuple(object_ids)


def _preserve_other_objects_sentence(
    *,
    program,
    excluded_object_ids: Sequence[str],
) -> str:
    excluded = set(excluded_object_ids)
    preserved_phrases = [
        _definite_object_phrase(program, object_id)
        for object_id in _required_positive_object_ids(program)
        if object_id not in excluded
    ]
    preserved_text = _natural_language_join(preserved_phrases)
    if not preserved_text:
        return ""
    return f"Keep {preserved_text} unchanged."


def _remove_object_focus_sentence(
    *,
    program,
    object_id: str | None,
    expected_count: int | None,
) -> str:
    if object_id is None:
        return ""
    label = _descriptive_object_label(program, object_id)
    if expected_count == 1:
        return (
            f"Keep one clear {label} unchanged as the main subject and remove a "
            "secondary or background duplicate instead."
        )
    if expected_count is not None and expected_count > 1:
        return (
            f"Keep the remaining {_humanize_count(expected_count)} "
            f"{_pluralize_label(label, expected_count)} unchanged and remove only "
            "one excess instance."
        )
    return f"Remove only a duplicate {label}, not the primary subject."


def _add_object_relation_guidance_sentences(
    *,
    program,
    primary_object_id: str | None,
) -> tuple[str, ...]:
    if primary_object_id is None:
        return ()
    primary_label = _descriptive_object_label(program, primary_object_id)
    guidance: list[str] = []
    for predicate in program.predicates:
        if not isinstance(predicate, SpatialRelationPredicate):
            continue
        if predicate.subject.object_id == primary_object_id:
            reference_phrase = _definite_object_phrase(
                program, predicate.reference.object_id
            )
            guidance.append(
                f"Place the added {primary_label} so it is clearly "
                f"{_humanize_relation(predicate.relation)} {reference_phrase}."
            )
            continue
        if predicate.reference.object_id != primary_object_id:
            continue
        subject_phrase = _definite_object_phrase(program, predicate.subject.object_id)
        inverse_relation = _inverse_relation(predicate.relation)
        if inverse_relation is not None:
            guidance.append(
                f"Place the added {primary_label} so it is clearly "
                f"{_humanize_relation(inverse_relation)} {subject_phrase}."
            )
            continue
        guidance.append(
            f"Place the added {primary_label} so {subject_phrase} remains clearly "
            f"{_humanize_relation(predicate.relation)} the added {primary_label}."
        )
    return _dedupe_nonempty_parts(guidance)


def select_resample_rewritten_prompt(
    *,
    program,
    round_index: int | None = None,
    random_seed: int | None = None,
    used_rewrite_indices: Sequence[int] | None = None,
) -> tuple[str, int | None, dict[str, Any]]:
    """Choose one rewrite candidate for resampling."""
    if not program.rewritten_prompts:
        return (
            program.source_prompt,
            None,
            {
                "available_rewrite_count": 0,
                "rewrite_selection_pool": "source_prompt_fallback",
                "used_rewrite_indices_before_selection": [],
                "used_rewrite_indices_after_selection": [],
                "candidate_rewrite_indices": [],
            },
        )

    available_count = len(program.rewritten_prompts)
    used_indices_before_selection = [
        int(index)
        for index in (used_rewrite_indices or ())
        if isinstance(index, int) and 0 <= int(index) < available_count
    ]
    seen_used = set(used_indices_before_selection)
    candidate_indices = [
        index for index in range(available_count) if index not in seen_used
    ]
    selection_pool = "unused_rewrites"
    if not candidate_indices:
        candidate_indices = list(range(available_count))
        selection_pool = "all_rewrites_after_exhaustion"

    if round_index is None and random_seed is None:
        rewrite_index = candidate_indices[0]
    else:
        rng = random.Random(
            f"{program.program_id}:{random_seed if random_seed is not None else 'none'}:"
            f"{round_index if round_index is not None else 'none'}:"
            f"{available_count}:{selection_pool}:{','.join(map(str, used_indices_before_selection))}"
        )
        rewrite_index = candidate_indices[rng.randrange(len(candidate_indices))]

    return (
        program.rewritten_prompts[rewrite_index],
        rewrite_index,
        {
            "available_rewrite_count": available_count,
            "rewrite_selection_pool": selection_pool,
            "candidate_rewrite_indices": list(candidate_indices),
            "used_rewrite_indices_before_selection": list(
                used_indices_before_selection
            ),
            "used_rewrite_indices_after_selection": [
                *used_indices_before_selection,
                rewrite_index,
            ],
        },
    )


def select_initial_generation_prompt(
    *,
    program,
    random_seed: int | None = None,
) -> tuple[str, int | None, dict[str, Any]]:
    """Choose one rewrite candidate for the first generation attempt."""
    if not program.rewritten_prompts:
        return (
            program.source_prompt,
            None,
            {
                "available_rewrite_count": 0,
                "rewrite_selection_pool": "source_prompt_fallback",
                "used_rewrite_indices_before_selection": [],
                "used_rewrite_indices_after_selection": [],
                "candidate_rewrite_indices": [],
            },
        )
    if random_seed is None:
        return (
            program.rewritten_prompts[0],
            0,
            {
                "available_rewrite_count": len(program.rewritten_prompts),
                "rewrite_selection_pool": "rewritten_prompts",
                "candidate_rewrite_indices": list(
                    range(len(program.rewritten_prompts))
                ),
                "used_rewrite_indices_before_selection": [],
                "used_rewrite_indices_after_selection": [0],
            },
        )

    rng = random.Random(
        f"initial:{program.program_id}:{random_seed}:{len(program.rewritten_prompts)}"
    )
    rewrite_index = rng.randrange(len(program.rewritten_prompts))
    return (
        program.rewritten_prompts[rewrite_index],
        rewrite_index,
        {
            "available_rewrite_count": len(program.rewritten_prompts),
            "rewrite_selection_pool": "rewritten_prompts",
            "candidate_rewrite_indices": list(range(len(program.rewritten_prompts))),
            "used_rewrite_indices_before_selection": [],
            "used_rewrite_indices_after_selection": [rewrite_index],
        },
    )


def _resolve_count_target_fragment(
    *,
    program,
    selection: ActionSelection,
    primary_object_id: str | None,
) -> str | None:
    if primary_object_id is None:
        return None
    label = _descriptive_object_label(program, primary_object_id)
    for predicate_id in selection.target_predicate_ids:
        predicate = program.get_predicate(predicate_id)
        if (
            isinstance(predicate, CountPredicate)
            and predicate.target.object_id == primary_object_id
        ):
            return _count_requirement_fragment(
                label,
                predicate.expected_count,
                predicate.comparator,
            )
    return None


def build_edit_instruction(
    *,
    program,
    selection: ActionSelection,
) -> str:
    """Build a FLUX-friendly editing prompt focused on the exact transformation."""
    assert isinstance(selection.action_parameters, EditActionParameters)
    params = selection.action_parameters
    primary_label = (
        None
        if params.primary_object_id is None
        else program.get_object(params.primary_object_id).label
    )
    descriptive_primary_label = (
        None
        if params.primary_object_id is None
        else _descriptive_object_label(program, params.primary_object_id)
    )
    target_count_fragment = _resolve_count_target_fragment(
        program=program,
        selection=selection,
        primary_object_id=params.primary_object_id,
    )

    if params.edit_operation == EditOperationType.ADD_OBJECT:
        count_delta = params.count_delta if params.count_delta is not None else 1
        added_object_phrase = (
            descriptive_primary_label
            if count_delta == 1
            else _pluralize_label(descriptive_primary_label, count_delta)
        )
        change_sentence = (
            f"Add {_humanize_count(count_delta)} more {added_object_phrase} so that "
            f"{target_count_fragment}."
            if target_count_fragment
            else f"Add {_humanize_count(count_delta)} more {added_object_phrase}."
        )
        relation_guidance = _join_nonempty_parts(
            _add_object_relation_guidance_sentences(
                program=program,
                primary_object_id=params.primary_object_id,
            )
        )
        prompt_parts = (
            change_sentence,
            relation_guidance,
            "Use the input image as the foundation and change only what is needed.",
            "Keep the existing subjects, framing, background, and lighting consistent.",
        )
        return _join_nonempty_parts(prompt_parts)

    if params.edit_operation == EditOperationType.REMOVE_OBJECT:
        remove_label = descriptive_primary_label or primary_label or "object"
        change_sentence = (
            f"Remove only the extra {remove_label}, preferably a secondary or "
            f"background instance, so that {target_count_fragment}."
            if target_count_fragment
            else f"Remove only the extra {remove_label}, preferably a secondary or "
            "background instance."
        )
        focus_sentence = _remove_object_focus_sentence(
            program=program,
            object_id=params.primary_object_id,
            expected_count=params.expected_count,
        )
        preserve_sentence = _preserve_other_objects_sentence(
            program=program,
            excluded_object_ids=params.target_object_ids,
        )
        prompt_parts = (
            change_sentence,
            focus_sentence,
            preserve_sentence,
            "Use the input image as the foundation and change only what is needed.",
            "Preserve the identity, placement, and scale of the remaining "
            "subjects, and keep the framing, background, and lighting consistent.",
        )
        return _join_nonempty_parts(prompt_parts)

    if params.edit_operation == EditOperationType.CHANGE_ATTRIBUTE:
        normalized_attribute = normalize_attribute_name(params.attribute_name)
        attribute_text = normalized_attribute.replace("_", " ")
        action_target_phrase = (
            None
            if params.action_target_object_id is None
            else _definite_object_phrase(program, params.action_target_object_id)
        )
        if normalized_attribute in {"color", "material", "shape", "size"}:
            change_sentence = (
                f"Change the {primary_label} so that it is {params.attribute_value}."
            )
        elif normalized_attribute == "pattern":
            change_sentence = (
                f"Change the {primary_label} so that it has a "
                f"{params.attribute_value} pattern."
            )
        elif normalized_attribute == "action":
            if action_target_phrase is None:
                change_sentence = (
                    f"Change the {primary_label} so that it is clearly "
                    f"{params.attribute_value}."
                )
            else:
                change_sentence = (
                    f"Change the {primary_label} so that it is clearly "
                    f"{params.attribute_value} {action_target_phrase}."
                )
        elif normalized_attribute == "pose":
            change_sentence = (
                f"Change the {primary_label}'s pose so that it is clearly "
                f"{params.attribute_value}."
            )
        elif normalized_attribute == "state":
            change_sentence = (
                f"Change the {primary_label} so that its visible state clearly "
                f"reads as {params.attribute_value}."
            )
        else:
            change_sentence = (
                f"Change the {primary_label}'s {attribute_text} so that it is "
                f"{params.attribute_value}."
            )
        preserve_sentence = _preserve_other_objects_sentence(
            program=program,
            excluded_object_ids=params.target_object_ids,
        )
        prompt_parts = (
            change_sentence,
            preserve_sentence,
            "Use the input image as the foundation and change only this target "
            "attribute.",
            f"Keep the {primary_label}'s identity, placement, background, and "
            "lighting consistent.",
        )
        return _join_nonempty_parts(prompt_parts)

    if params.edit_operation == EditOperationType.CHANGE_GLOBAL_SCENE:
        attribute_name = (params.attribute_name or "environment").replace("_", " ")
        if params.attribute_name in {
            None,
            "background",
            "environment",
            "scene",
            "setting",
        }:
            change_sentence = (
                "Change only the background and surrounding environment so the "
                f"overall scene clearly reads as {params.attribute_value}."
            )
        else:
            change_sentence = (
                f"Change only the scene-level {attribute_name} so it clearly "
                f"reads as {params.attribute_value}."
            )
        preserve_sentence = _preserve_other_objects_sentence(
            program=program,
            excluded_object_ids=(),
        )
        prompt_parts = (
            change_sentence,
            preserve_sentence,
            "Do not add, remove, reposition, or redesign the existing subjects.",
            "Use the input image as the foundation and change only what is needed.",
            "Keep the existing subjects' identity, pose, layout, scale, and "
            "lighting as consistent as possible.",
        )
        return _join_nonempty_parts(prompt_parts)

    raise ValueError(f"Unsupported edit operation: {params.edit_operation}")


def resolve_action_execution_input(
    *,
    program,
    selection: ActionSelection,
) -> tuple[str | None, str | None]:
    """Resolve the exact text that will be sent for the selected action."""
    if selection.action_type == RefinementActionType.EDIT:
        return (
            "edit_instruction",
            build_edit_instruction(program=program, selection=selection),
        )
    return (None, None)


__all__ = [
    "build_edit_instruction",
    "compose_prompt_with_scene_guidance",
    "resolve_action_execution_input",
    "select_initial_generation_prompt",
    "select_resample_rewritten_prompt",
]
