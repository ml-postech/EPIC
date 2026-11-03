"""Shared support vocabulary for VP parsing and review."""

from __future__ import annotations

CANONICAL_SPATIAL_RELATIONS: tuple[str, ...] = (
    "left_of",
    "right_of",
    "above",
    "below",
    "inside",
    "in",
    "overlapping",
    "intersecting",
    "near",
    "on",
    "in_front_of",
    "behind",
)

SPATIAL_RELATION_ALIASES: dict[str, str] = {
    "under": "below",
    "underneath": "below",
    "beneath": "below",
    "over": "above",
    "overhead": "above",
    "on_top_of": "on",
    "atop": "on",
}

SUPPORTED_SPATIAL_RELATIONS: tuple[str, ...] = (
    *CANONICAL_SPATIAL_RELATIONS,
    *SPATIAL_RELATION_ALIASES.keys(),
)

SUPPORTED_ATTRIBUTE_NAMES: tuple[str, ...] = (
    "color",
    "material",
    "pattern",
    "shape",
    "size",
    "pose",
    "action",
    "state",
)

SUPPORTED_GLOBAL_SCENE_ATTRIBUTE_NAMES: tuple[str, ...] = (
    "environment",
    "background",
)


def normalize_spatial_relation_name(value: object) -> str:
    """Return the verifier-facing canonical name for a relation or accepted alias."""
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return SPATIAL_RELATION_ALIASES.get(normalized, normalized)


def normalize_attribute_name(value: object) -> str:
    """Normalize attribute names across parser, selector, and runtime helpers."""
    return "_".join(
        str(value).strip().lower().replace("-", " ").replace("_", " ").split()
    )


def render_visual_program_support_contract() -> str:
    """Render the verifier-facing vocabulary contract for parser prompts."""
    relation_list = ", ".join(CANONICAL_SPATIAL_RELATIONS)
    relation_alias_list = ", ".join(
        f"{alias}->{target}" for alias, target in SPATIAL_RELATION_ALIASES.items()
    )
    attribute_list = ", ".join(SUPPORTED_ATTRIBUTE_NAMES)
    scene_attribute_list = ", ".join(SUPPORTED_GLOBAL_SCENE_ATTRIBUTE_NAMES)
    return (
        "Supported verifier-facing vocabulary:\n"
        f"- Canonical spatial relation names: {relation_list}.\n"
        f"- Accepted spatial relation aliases: {relation_alias_list}.\n"
        f"- Object attribute names: {attribute_list}.\n"
        f"- Global scene attribute names: {scene_attribute_list}.\n"
        "- Action values are free text; use an action target when the action is directed at a declared object."
    )


__all__ = [
    "CANONICAL_SPATIAL_RELATIONS",
    "SPATIAL_RELATION_ALIASES",
    "SUPPORTED_ATTRIBUTE_NAMES",
    "SUPPORTED_GLOBAL_SCENE_ATTRIBUTE_NAMES",
    "SUPPORTED_SPATIAL_RELATIONS",
    "normalize_attribute_name",
    "normalize_spatial_relation_name",
    "render_visual_program_support_contract",
]
