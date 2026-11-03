"""Typed predicate verification for fixed visual programs.

Count, attribute, and relation checks reuse SAM3 proposals, SigLIP scores,
and depth estimates. Visible text uses a VLM; action checks use the VLM only
after non-action predicates pass.

The implementation is intentionally structured so multiple predicates can reuse
the same tool outputs. For example, one set of object-region proposals can
support count, relation, action, and attribute checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from math import sqrt
import re
from typing import Any, Mapping, Optional, Protocol, Sequence

import numpy as np
from PIL import Image

from graph.visual_program import (
    AttributePredicate,
    CountComparator,
    CountPredicate,
    ExclusionPredicate,
    GlobalScenePredicate,
    ObjectSpec,
    Predicate,
    PredicateState,
    PredicateCategory,
    SpatialRelationPredicate,
    TextPredicate,
    VisualProgram,
    VisualProgramState,
)
from graph.visual_program_tools import (
    BoundingBox,
    DetectedRegion,
    PerceptionCache,
    PerceptionToolbox,
)
from graph.visual_program_support import normalize_spatial_relation_name


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


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


@dataclass(frozen=True)
class VerifierThresholds:
    """Thresholds that convert heuristic scores into predicate states."""

    proposal_satisfied: float = 0.65
    proposal_uncertain: float = 0.35
    scene_background_min_area_ratio: float = 0.10
    attribute_satisfied: float = 0.60
    attribute_uncertain: float = 0.35
    action_satisfied: float = 0.55
    action_uncertain: float = 0.35
    relation_satisfied: float = 0.60
    relation_uncertain: float = 0.35


@dataclass(frozen=True)
class RelationScoringConfig:
    """Hyperparameters controlling deterministic spatial-relation scoring."""

    directional_centroid_weight: float = 0.45
    directional_gap_weight: float = 0.35
    directional_orthogonal_weight: float = 0.20
    directional_gap_scale: float = 0.12
    directional_gap_tolerance_ratio: float = 0.02
    directional_centroid_scale: float = 0.50
    directional_orthogonal_scale: float = 0.55
    directional_orthogonal_overlap_weight: float = 0.75
    directional_orthogonal_center_weight: float = 0.25
    directional_axis_overlap_penalty_weight: float = 0.05
    directional_overlap_penalty_weight: float = 0.45
    directional_containment_penalty_weight: float = 0.30
    directional_on_penalty_weight: float = 0.35

    near_centroid_weight: float = 0.45
    near_edge_weight: float = 0.55
    near_diagonal_scale: float = 0.18

    inside_containment_weight: float = 0.80
    inside_centroid_weight: float = 0.20
    in_bbox_coverage_weight: float = 0.80
    in_centroid_bbox_weight: float = 0.20

    overlap_iou_weight: float = 0.65
    overlap_intersection_weight: float = 0.35

    on_contact_weight: float = 0.55
    on_horizontal_overlap_weight: float = 0.25
    on_above_weight: float = 0.20
    on_inside_penalty_weight: float = 0.25
    on_gap_tolerance_ratio: float = 0.02

    depth_scale: float = 28.0

    def __post_init__(self) -> None:
        positive_fields = {
            "directional_gap_scale",
            "directional_gap_tolerance_ratio",
            "directional_centroid_scale",
            "directional_orthogonal_scale",
            "near_diagonal_scale",
            "on_gap_tolerance_ratio",
            "depth_scale",
        }
        for field_info in fields(self):
            value = float(getattr(self, field_info.name))
            if field_info.name in positive_fields:
                if value <= 0.0:
                    raise ValueError(f"{field_info.name} must be > 0.")
                continue
            if value < 0.0:
                raise ValueError(f"{field_info.name} must be >= 0.")
        if (
            float(self.directional_orthogonal_overlap_weight)
            + float(self.directional_orthogonal_center_weight)
            <= 0.0
        ):
            raise ValueError(
                "directional_orthogonal_overlap_weight + "
                "directional_orthogonal_center_weight must be > 0."
            )

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True)
class ActionScoringConfig:
    """Hyperparameters controlling action scoring crops."""

    target_joint_crop_padding_ratio: float = 0.10

    def __post_init__(self) -> None:
        if float(self.target_joint_crop_padding_ratio) < 0.0:
            raise ValueError("target_joint_crop_padding_ratio must be >= 0.")

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(frozen=True)
class ObjectGroupEvidence:
    """Aggregate a set of proposed regions into one object-level summary."""

    object_id: str
    regions: tuple[DetectedRegion, ...]
    bbox: BoundingBox
    confidence: float
    average_depth: Optional[float] = None

    @classmethod
    def from_regions(
        cls,
        object_id: str,
        regions: Sequence[DetectedRegion],
        cache: PerceptionCache,
    ) -> Optional[ObjectGroupEvidence]:
        """Build a group summary from one or more regions."""
        if not regions:
            return None

        tuple_regions = tuple(regions)
        bbox = BoundingBox.union([region.bbox for region in tuple_regions])
        confidences = [float(region.confidence or 0.0) for region in tuple_regions]
        depths = [cache.get_region_average_depth(region) for region in tuple_regions]
        valid_depths = [depth for depth in depths if depth is not None]
        return cls(
            object_id=object_id,
            regions=tuple_regions,
            bbox=bbox,
            confidence=sum(confidences) / max(1, len(confidences)),
            average_depth=(
                None if not valid_depths else sum(valid_depths) / len(valid_depths)
            ),
        )

    def evidence_dict(self) -> dict[str, Any]:
        """Serialize the aggregate summary for logs or tests."""
        return {
            "object_id": self.object_id,
            "confidence": self.confidence,
            "average_depth": self.average_depth,
            "bbox": self.bbox.to_dict(),
            "regions": [region.evidence_dict() for region in self.regions],
        }


@dataclass(frozen=True)
class ObjectFootprint:
    """Mask-aware object geometry used by relation scoring."""

    object_id: str
    bbox: BoundingBox
    mask: np.ndarray
    area: float
    centroid_x: float
    centroid_y: float
    average_depth: Optional[float]
    uses_segmentation_mask: bool

    def evidence_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "bbox": self.bbox.to_dict(),
            "area": self.area,
            "centroid": {
                "x": self.centroid_x,
                "y": self.centroid_y,
            },
            "average_depth": self.average_depth,
            "uses_segmentation_mask": self.uses_segmentation_mask,
        }


@dataclass(frozen=True)
class PredicateVerificationResult:
    """Structured verification output for one predicate."""

    predicate_id: str
    category: PredicateCategory
    state: PredicateState
    score: Optional[float]
    confidence: Optional[float]
    evidence_summary: str
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result into plain Python types."""
        return _serialize(self)


@dataclass(frozen=True)
class VisualProgramVerificationReport:
    """Aggregate predicate-level results for one image/program pair."""

    program_id: str
    results: tuple[PredicateVerificationResult, ...]
    program_state: VisualProgramState
    cache_summary: Mapping[str, Any]

    def get_result(self, predicate_id: str) -> PredicateVerificationResult:
        """Look up one predicate result by ID."""
        for result in self.results:
            if result.predicate_id == predicate_id:
                return result
        raise KeyError(f"Unknown predicate_id: {predicate_id}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report into plain Python types."""
        return _serialize(self)


@dataclass
class VerificationSession:
    """Hold shared caches and configuration during one verification pass."""

    image: Image.Image
    program: VisualProgram
    cache: PerceptionCache
    thresholds: VerifierThresholds = field(default_factory=VerifierThresholds)
    relation_scoring: RelationScoringConfig = field(
        default_factory=RelationScoringConfig
    )
    _object_group_cache: dict[str, Optional[ObjectGroupEvidence]] = field(
        default_factory=dict
    )
    _object_footprint_cache: dict[str, Optional[ObjectFootprint]] = field(
        default_factory=dict
    )
    _exclusion_reference_cache: dict[str, tuple[str, ...]] = field(default_factory=dict)
    _selected_raw_object_region_cache: dict[str, tuple[DetectedRegion, ...]] = field(
        default_factory=dict
    )
    _label_fallback_region_cache: dict[str, tuple[DetectedRegion, ...]] = field(
        default_factory=dict
    )
    _filtered_object_region_cache: dict[str, tuple[DetectedRegion, ...]] = field(
        default_factory=dict
    )
    _scene_text_score_cache: dict[tuple[str, str], Optional[float]] = field(
        default_factory=dict
    )
    _scene_region_cache: dict[str, tuple[DetectedRegion, ...]] = field(
        default_factory=dict
    )
    action_scoring: ActionScoringConfig = field(default_factory=ActionScoringConfig)
    action_vllm_enabled: bool = True
    action_vllm_requested: bool = False
    action_vllm_gate_blocking_predicate_ids: tuple[str, ...] = field(
        default_factory=tuple
    )

    def object_regions(self, object_id: str) -> tuple[DetectedRegion, ...]:
        """Return candidate regions for one object."""
        if object_id not in self._filtered_object_region_cache:
            raw_regions = self._selected_raw_object_regions(object_id)
            filtered_regions = tuple(
                region
                for region in raw_regions
                if not self._is_cross_label_duplicate(object_id, region)
            )
            self._filtered_object_region_cache[object_id] = filtered_regions
        return self._filtered_object_region_cache[object_id]

    def _selected_raw_object_regions(
        self, object_id: str
    ) -> tuple[DetectedRegion, ...]:
        """Choose one raw proposal set for an object before cross-label filtering."""
        if object_id in self._selected_raw_object_region_cache:
            return self._selected_raw_object_region_cache[object_id]

        object_spec = self.program.get_object(object_id)
        base_regions = tuple(self.cache.get_object_regions(object_id))
        selected_regions = base_regions
        expected_count = self.exact_count_for_object(object_id)
        normalized_proposal_text = (
            _normalized_label(object_spec.proposal_text)
            if object_spec.proposal_text is not None
            else ""
        )
        normalized_label = _normalized_label(object_spec.label)
        fallback_allowed = (
            bool(normalized_proposal_text)
            and normalized_proposal_text != normalized_label
            and expected_count is not None
            and expected_count > 0
            and not self._object_has_shared_label(object_id)
        )
        if fallback_allowed:
            base_strong_count, _ = _proposal_count_stats(
                base_regions,
                thresholds=self.thresholds,
            )
            if base_strong_count < expected_count:
                fallback_regions = self._label_fallback_regions(object_id)
                if _count_region_selection_rank(
                    fallback_regions,
                    expected_count=expected_count,
                    thresholds=self.thresholds,
                ) > _count_region_selection_rank(
                    base_regions,
                    expected_count=expected_count,
                    thresholds=self.thresholds,
                ):
                    selected_regions = fallback_regions
            elif base_strong_count == expected_count:
                fallback_regions = self._label_fallback_regions(object_id)
                fallback_strong_count, fallback_weak_count = _proposal_count_stats(
                    fallback_regions,
                    thresholds=self.thresholds,
                )
                if (
                    fallback_strong_count > expected_count
                    or fallback_weak_count > expected_count
                ):
                    selected_regions = fallback_regions
        self._selected_raw_object_region_cache[object_id] = selected_regions
        return selected_regions

    def _label_fallback_regions(self, object_id: str) -> tuple[DetectedRegion, ...]:
        """Run and cache a label-only fallback proposal query for one object."""
        if object_id in self._label_fallback_region_cache:
            return self._label_fallback_region_cache[object_id]

        object_spec = self.program.get_object(object_id)
        fallback_spec = ObjectSpec(
            object_id=f"{object_spec.object_id}__label_fallback",
            label=object_spec.label,
            proposal_text=object_spec.label,
            aliases=(),
            description=object_spec.description,
        )
        raw_regions = tuple(
            self.cache.tools.region_proposal_tool.propose_regions(
                self.image,
                fallback_spec,
            )
        )
        self._label_fallback_region_cache[object_id] = tuple(
            self._retag_region_for_object_variant(
                object_id=object_id,
                variant_name="label_fallback",
                region=region,
                index=index,
                query_text=object_spec.label,
            )
            for index, region in enumerate(raw_regions)
        )
        return self._label_fallback_region_cache[object_id]

    def _retag_region_for_object_variant(
        self,
        *,
        object_id: str,
        variant_name: str,
        region: DetectedRegion,
        index: int,
        query_text: str,
    ) -> DetectedRegion:
        """Retag fallback regions so variant queries never collide in caches."""
        metadata = dict(region.metadata)
        metadata["query"] = query_text
        metadata["query_variant"] = variant_name
        metadata["variant_original_region_id"] = region.region_id
        return DetectedRegion(
            region_id=f"{object_id}:{variant_name}:{index}",
            bbox=region.bbox,
            label=region.label,
            confidence=region.confidence,
            average_depth=region.average_depth,
            mask=region.mask,
            attributes=dict(region.attributes),
            metadata=metadata,
        )

    def _object_has_shared_label(self, object_id: str) -> bool:
        """Return whether another declared object reuses the same canonical label."""
        label_key = _normalized_label(self.program.get_object(object_id).label)
        for other_object in self.program.objects:
            if other_object.object_id == object_id:
                continue
            if _normalized_label(other_object.label) == label_key:
                return True
        return False

    def exact_count_for_object(self, object_id: str) -> Optional[int]:
        """Return an exact expected count when the program contains one."""
        for predicate in self.program.predicates:
            if (
                isinstance(predicate, CountPredicate)
                and predicate.target.object_id == object_id
                and predicate.comparator == CountComparator.EXACTLY
            ):
                return predicate.expected_count
        return None

    def selected_regions(self, object_id: str) -> tuple[DetectedRegion, ...]:
        """Return the most relevant regions for an object, respecting exact counts."""
        regions = tuple(
            sorted(
                self.object_regions(object_id),
                key=lambda region: (float(region.confidence or 0.0), region.region_id),
                reverse=True,
            )
        )
        expected_count = self.exact_count_for_object(object_id)
        if expected_count is None or expected_count <= 0:
            return regions[:1]
        return regions[:expected_count]

    def object_group(self, object_id: str) -> Optional[ObjectGroupEvidence]:
        """Return an aggregated object summary built from selected regions."""
        if object_id not in self._object_group_cache:
            self._object_group_cache[object_id] = ObjectGroupEvidence.from_regions(
                object_id,
                self.selected_regions(object_id),
                self.cache,
            )
        return self._object_group_cache[object_id]

    def object_footprint(self, object_id: str) -> Optional[ObjectFootprint]:
        """Return a mask-aware footprint for one object."""
        if object_id not in self._object_footprint_cache:
            object_group = self.object_group(object_id)
            if object_group is None:
                self._object_footprint_cache[object_id] = None
            else:
                self._object_footprint_cache[object_id] = _build_object_footprint(
                    object_group,
                    image_size=self.image.size,
                )
        return self._object_footprint_cache[object_id]

    def exclusion_reference_object_ids(self, object_id: str) -> tuple[str, ...]:
        """Return positive-object IDs that can explain away noun-like modifier exclusions."""
        if object_id not in self._exclusion_reference_cache:
            object_spec = self.program.get_object(object_id)
            object_tokens = set(_object_text_tokens(object_spec))
            linked_object_ids: list[str] = []
            for predicate in self.program.predicates:
                if not isinstance(predicate, AttributePredicate):
                    continue
                expected_tokens = set(_normalized_tokens(predicate.expected_value))
                if object_tokens.intersection(expected_tokens):
                    linked_object_ids.append(predicate.target.object_id)
            self._exclusion_reference_cache[object_id] = tuple(
                dict.fromkeys(linked_object_ids)
            )
        return self._exclusion_reference_cache[object_id]

    def _is_cross_label_duplicate(self, object_id: str, region: DetectedRegion) -> bool:
        current_confidence = float(region.confidence or 0.0)
        for other_object in self.program.objects:
            other_object_id = other_object.object_id
            if other_object_id == object_id:
                continue
            for other_region in self._selected_raw_object_regions(other_object_id):
                if not _cross_label_region_should_dominate(
                    current_object_id=object_id,
                    current_region=region,
                    other_object_id=other_object_id,
                    other_region=other_region,
                    current_confidence=current_confidence,
                    image_size=self.image.size,
                ):
                    continue
                return True
        return False

    def score_scene_text(
        self,
        *,
        variant_key: str,
        image: Image.Image,
        text: str,
    ) -> Optional[float]:
        """Score an arbitrary scene image against one text query, with lightweight caching."""
        normalized_text = text.strip().lower()
        cache_key = (variant_key, normalized_text)
        if cache_key in self._scene_text_score_cache:
            return self._scene_text_score_cache[cache_key]

        scoring_tool = self.cache.tools.region_text_scoring_tool
        if scoring_tool is None:
            self._scene_text_score_cache[cache_key] = None
            return None

        scene_region = DetectedRegion(
            region_id=variant_key,
            bbox=BoundingBox(0.0, 0.0, float(image.width), float(image.height)),
            label="scene",
            confidence=1.0,
            metadata={"region_source": "scene_variant"},
        )
        score = scoring_tool.score_region_text(image, scene_region, text)
        self._scene_text_score_cache[cache_key] = (
            None if score is None else _clamp01(score)
        )
        return self._scene_text_score_cache[cache_key]

    def scene_query_regions(self, query: str) -> tuple[DetectedRegion, ...]:
        """Run and cache an ephemeral scene-level region proposal query."""
        normalized_query = " ".join(str(query).strip().lower().split())
        if not normalized_query:
            return ()
        if normalized_query in self._scene_region_cache:
            return self._scene_region_cache[normalized_query]

        scene_object = ObjectSpec.named(
            normalized_query,
            object_id=f"scene_query_{_normalized_label(normalized_query)}",
            description="ephemeral scene-level query",
        )
        regions = tuple(
            self.cache.tools.region_proposal_tool.propose_regions(
                self.image,
                scene_object,
            )
        )
        self._scene_region_cache[normalized_query] = regions
        return regions


class PredicateEvaluator(Protocol):
    """Typed evaluator for one predicate family."""

    def evaluate(
        self,
        predicate: Predicate,
        session: VerificationSession,
    ) -> PredicateVerificationResult:
        """Evaluate one predicate against the session."""


def _normalized_label(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _normalized_tokens(value: Any) -> tuple[str, ...]:
    normalized = _normalized_label(value)
    return tuple(token for token in re.split(r"[^a-z0-9]+|_+", normalized) if token)


def _object_text_tokens(object_spec: ObjectSpec) -> tuple[str, ...]:
    tokens: list[str] = []
    for value in (object_spec.label, *object_spec.aliases):
        for token in _normalized_tokens(value):
            if token not in tokens:
                tokens.append(token)
    return tuple(tokens)


def _state_from_score(
    score: float,
    *,
    satisfied_threshold: float,
    uncertain_threshold: float,
) -> PredicateState:
    if score >= satisfied_threshold:
        return PredicateState.SATISFIED
    if score >= uncertain_threshold:
        return PredicateState.UNCERTAIN
    return PredicateState.VIOLATED


def _safe_mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


_PREDICATE_STATE_RANK = {
    PredicateState.VIOLATED: 0,
    PredicateState.UNCERTAIN: 1,
    PredicateState.SATISFIED: 2,
}


def _proposal_count_stats(
    regions: Sequence[DetectedRegion],
    *,
    thresholds: VerifierThresholds,
) -> tuple[int, int]:
    strong_count = sum(
        1
        for region in regions
        if float(region.confidence or 0.0) >= thresholds.proposal_satisfied
    )
    weak_count = sum(
        1
        for region in regions
        if float(region.confidence or 0.0) >= thresholds.proposal_uncertain
    )
    return strong_count, weak_count


def _exact_count_state_from_counts(
    *,
    strong_count: int,
    weak_count: int,
    expected_count: int,
) -> PredicateState:
    if strong_count == expected_count and weak_count == expected_count:
        return PredicateState.SATISFIED
    if strong_count <= expected_count <= weak_count:
        return PredicateState.UNCERTAIN
    return PredicateState.VIOLATED


def _exact_count_score_from_counts(
    *,
    strong_count: int,
    weak_count: int,
    expected_count: int,
) -> float:
    """Score exact counts with weak overcount treated as residual uncertainty."""
    if strong_count > expected_count:
        count_error = strong_count - expected_count
    elif weak_count < expected_count:
        count_error = expected_count - weak_count
    else:
        count_error = (expected_count - strong_count) + max(
            0,
            weak_count - expected_count,
        )
    return _clamp01(1.0 - (count_error / max(1.0, float(expected_count))))


def _count_region_selection_rank(
    regions: Sequence[DetectedRegion],
    *,
    expected_count: int,
    thresholds: VerifierThresholds,
) -> tuple[int, int, int, int, float, int]:
    strong_count, weak_count = _proposal_count_stats(
        regions,
        thresholds=thresholds,
    )
    state = _exact_count_state_from_counts(
        strong_count=strong_count,
        weak_count=weak_count,
        expected_count=expected_count,
    )
    best_confidence = max(
        (float(region.confidence or 0.0) for region in regions),
        default=0.0,
    )
    return (
        _PREDICATE_STATE_RANK[state],
        -abs(strong_count - expected_count),
        min(strong_count, expected_count),
        -max(0, weak_count - expected_count),
        best_confidence,
        -len(regions),
    )


def _region_boolean_mask(
    region: DetectedRegion,
    image_size: tuple[int, int],
) -> np.ndarray:
    width, height = image_size
    shape = (int(height), int(width))
    if region.mask is not None and np.asarray(region.mask).shape == shape:
        return np.asarray(region.mask, dtype=bool)

    mask = np.zeros(shape, dtype=bool)
    x_min = int(max(0, np.floor(region.bbox.x_min)))
    y_min = int(max(0, np.floor(region.bbox.y_min)))
    x_max = int(min(width, np.ceil(region.bbox.x_max)))
    y_max = int(min(height, np.ceil(region.bbox.y_max)))
    if x_max > x_min and y_max > y_min:
        mask[y_min:y_max, x_min:x_max] = True
    return mask


def _overlap_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    union = int(np.logical_or(mask_a, mask_b).sum())
    if union <= 0:
        return 0.0
    intersection = int(np.logical_and(mask_a, mask_b).sum())
    return float(intersection / union)


def _overlap_intersection_over_min(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    area_a = int(mask_a.sum())
    area_b = int(mask_b.sum())
    if area_a <= 0 or area_b <= 0:
        return 0.0
    intersection = int(np.logical_and(mask_a, mask_b).sum())
    return float(intersection / max(1, min(area_a, area_b)))


def _region_overlap_metrics(
    left: DetectedRegion,
    right: DetectedRegion,
    *,
    image_size: tuple[int, int],
) -> dict[str, float]:
    left_mask = _region_boolean_mask(left, image_size)
    right_mask = _region_boolean_mask(right, image_size)
    left_area = int(left_mask.sum())
    right_area = int(right_mask.sum())
    area_similarity = (
        0.0
        if left_area <= 0 or right_area <= 0
        else float(min(left_area, right_area) / max(1, max(left_area, right_area)))
    )
    return {
        "intersection_over_min_area": _overlap_intersection_over_min(
            left_mask, right_mask
        ),
        "iou": _overlap_iou(left_mask, right_mask),
        "area_similarity": area_similarity,
        "left_area": float(left_area),
        "right_area": float(right_area),
    }


def _cross_label_region_should_dominate(
    *,
    current_object_id: str,
    current_region: DetectedRegion,
    other_object_id: str,
    other_region: DetectedRegion,
    current_confidence: float,
    image_size: tuple[int, int],
) -> bool:
    other_confidence = float(other_region.confidence or 0.0)
    if other_confidence < current_confidence:
        return False
    if other_confidence == current_confidence and (
        other_object_id,
        other_region.region_id,
    ) >= (current_object_id, current_region.region_id):
        return False

    overlap_metrics = _region_overlap_metrics(
        current_region,
        other_region,
        image_size=image_size,
    )
    return (
        overlap_metrics["intersection_over_min_area"] >= 0.95
        and overlap_metrics["iou"] >= 0.90
        and overlap_metrics["area_similarity"] >= 0.85
    )


def _residual_area_ratio_after_overlap_explanations(
    candidate: DetectedRegion,
    reference_regions: Sequence[DetectedRegion],
    *,
    image_size: tuple[int, int],
) -> float:
    candidate_mask = _region_boolean_mask(candidate, image_size)
    candidate_area = int(candidate_mask.sum())
    if candidate_area <= 0:
        return 0.0
    explained_mask = np.zeros_like(candidate_mask, dtype=bool)
    for reference_region in reference_regions:
        explained_mask |= _region_boolean_mask(reference_region, image_size)
    residual_mask = np.logical_and(candidate_mask, np.logical_not(explained_mask))
    return float(int(residual_mask.sum()) / max(1, candidate_area))


def _build_object_footprint(
    group: ObjectGroupEvidence,
    *,
    image_size: tuple[int, int],
) -> ObjectFootprint:
    width, height = int(image_size[0]), int(image_size[1])
    shape = (height, width)
    merged_mask = np.zeros(shape, dtype=bool)
    uses_segmentation_mask = False
    for region in group.regions:
        merged_mask |= _region_boolean_mask(region, image_size)
        if region.mask is not None and np.asarray(region.mask).shape == shape:
            uses_segmentation_mask = True

    if not merged_mask.any():
        fallback_region = DetectedRegion(
            region_id=f"{group.object_id}:footprint_fallback",
            bbox=group.bbox,
            label=group.object_id,
        )
        merged_mask = _region_boolean_mask(fallback_region, image_size)

    if merged_mask.any():
        ys, xs = np.nonzero(merged_mask)
        bbox = BoundingBox(
            x_min=float(xs.min()),
            y_min=float(ys.min()),
            x_max=float(xs.max() + 1),
            y_max=float(ys.max() + 1),
        )
        centroid_x = float(xs.mean() + 0.5)
        centroid_y = float(ys.mean() + 0.5)
        area = float(int(merged_mask.sum()))
    else:
        bbox = group.bbox
        centroid_x, centroid_y = bbox.center
        area = float(max(1.0, bbox.area))

    return ObjectFootprint(
        object_id=group.object_id,
        bbox=bbox,
        mask=merged_mask,
        area=area,
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        average_depth=group.average_depth,
        uses_segmentation_mask=uses_segmentation_mask,
    )


def _interval_overlap_ratio(
    min_a: float,
    max_a: float,
    min_b: float,
    max_b: float,
) -> float:
    overlap = max(0.0, min(max_a, max_b) - max(min_a, min_b))
    base = max(1.0, min(max_a - min_a, max_b - min_b))
    return _clamp01(overlap / base)


def _footprint_overlap_stats(
    subject: ObjectFootprint,
    reference: ObjectFootprint,
) -> dict[str, float]:
    intersection = int(np.logical_and(subject.mask, reference.mask).sum())
    union = int(np.logical_or(subject.mask, reference.mask).sum())
    subject_area = max(1, int(subject.mask.sum()))
    reference_area = max(1, int(reference.mask.sum()))
    min_area = max(1, min(subject_area, reference_area))
    return {
        "intersection_pixels": float(intersection),
        "union_pixels": float(union),
        "iou": 0.0 if union <= 0 else float(intersection / union),
        "intersection_over_min": float(intersection / min_area),
        "subject_coverage_by_reference": float(intersection / subject_area),
        "reference_coverage_by_subject": float(intersection / reference_area),
    }


def _point_inside_bbox(
    bbox: BoundingBox,
    x: float,
    y: float,
) -> bool:
    return bbox.contains(BoundingBox(x, y, x, y))


def _edge_distance_between_footprints(
    subject: ObjectFootprint,
    reference: ObjectFootprint,
) -> tuple[float, float, float]:
    horizontal_gap = max(
        0.0,
        subject.bbox.x_min - reference.bbox.x_max,
        reference.bbox.x_min - subject.bbox.x_max,
    )
    vertical_gap = max(
        0.0,
        subject.bbox.y_min - reference.bbox.y_max,
        reference.bbox.y_min - subject.bbox.y_max,
    )
    edge_distance = sqrt((horizontal_gap**2) + (vertical_gap**2))
    return horizontal_gap, vertical_gap, edge_distance


def _support_contact_score(
    subject: ObjectFootprint,
    reference: ObjectFootprint,
    *,
    image_size: tuple[int, int],
    config: RelationScoringConfig,
) -> tuple[float, dict[str, Any]]:
    image_height = max(1.0, float(image_size[1]))
    tolerance_pixels = max(1.0, image_height * config.on_gap_tolerance_ratio)
    x_start = int(max(0, np.floor(max(subject.bbox.x_min, reference.bbox.x_min))))
    x_end = int(
        min(image_size[0], np.ceil(min(subject.bbox.x_max, reference.bbox.x_max)))
    )
    overlap_span = max(0, x_end - x_start)
    if overlap_span <= 0:
        return 0.0, {
            "score": 0.0,
            "tolerance_pixels": tolerance_pixels,
            "overlap_span_columns": overlap_span,
            "valid_contact_columns": 0,
            "column_coverage": 0.0,
            "mean_gap_pixels": None,
            "mean_closeness": 0.0,
        }

    gaps: list[float] = []
    closeness_scores: list[float] = []
    valid_columns = 0
    for x_coord in range(x_start, x_end):
        subject_rows = np.flatnonzero(subject.mask[:, x_coord])
        if subject_rows.size == 0:
            continue
        reference_rows = np.flatnonzero(reference.mask[:, x_coord])
        if reference_rows.size == 0:
            continue

        subject_bottom = int(subject_rows.max())
        reference_candidates = reference_rows[
            reference_rows >= (subject_bottom - tolerance_pixels)
        ]
        if reference_candidates.size == 0:
            continue

        gap_pixels = float(reference_candidates.min() - subject_bottom)
        gaps.append(gap_pixels)
        closeness_scores.append(
            _clamp01(1.0 - (abs(gap_pixels) / max(1.0, tolerance_pixels)))
        )
        valid_columns += 1

    column_coverage = float(valid_columns / max(1, overlap_span))
    mean_gap_pixels = _safe_mean(gaps)
    mean_closeness = _safe_mean(closeness_scores) or 0.0
    score = _clamp01(0.70 * mean_closeness + 0.30 * column_coverage)
    return score, {
        "score": score,
        "tolerance_pixels": tolerance_pixels,
        "overlap_span_columns": overlap_span,
        "valid_contact_columns": valid_columns,
        "column_coverage": column_coverage,
        "mean_gap_pixels": mean_gap_pixels,
        "mean_closeness": mean_closeness,
    }


def _near_relation_score(
    subject: ObjectFootprint,
    reference: ObjectFootprint,
    *,
    image_size: tuple[int, int],
    config: RelationScoringConfig,
) -> tuple[float, dict[str, Any]]:
    diagonal = sqrt((float(image_size[0]) ** 2) + (float(image_size[1]) ** 2))
    _, _, edge_distance = _edge_distance_between_footprints(subject, reference)
    centroid_distance = sqrt(
        ((subject.centroid_x - reference.centroid_x) ** 2)
        + ((subject.centroid_y - reference.centroid_y) ** 2)
    )
    normalizer = max(1.0, diagonal * config.near_diagonal_scale)
    centroid_score = _clamp01(1.0 - (centroid_distance / normalizer))
    edge_score = _clamp01(1.0 - (edge_distance / normalizer))
    weight_sum = config.near_centroid_weight + config.near_edge_weight
    if weight_sum <= 0.0:
        score = 0.0
    else:
        score = _clamp01(
            (
                config.near_centroid_weight * centroid_score
                + config.near_edge_weight * edge_score
            )
            / weight_sum
        )
    return score, {
        "score": score,
        "centroid_distance": centroid_distance,
        "edge_distance": edge_distance,
        "centroid_score": centroid_score,
        "edge_score": edge_score,
        "normalizer": normalizer,
    }


def _depth_relation_support_score(
    *,
    planar_alignment: float,
    near_score: float,
) -> float:
    """Return a bounded 2D support term for front/behind depth relations."""
    return _clamp01((0.60 * planar_alignment) + (0.40 * near_score))


def _depth_order_relation_score(
    *,
    depth_delta: float,
    planar_alignment: float,
    near_score: float,
    config: RelationScoringConfig,
) -> tuple[float, float, float]:
    """Score front/behind only from positive depth evidence, with 2D support."""
    depth_score = _clamp01(depth_delta / config.depth_scale)
    support_score = _depth_relation_support_score(
        planar_alignment=planar_alignment,
        near_score=near_score,
    )
    score = _clamp01(depth_score * (0.80 + (0.20 * support_score)))
    return score, depth_score, support_score


def _inside_relation_score(
    subject: ObjectFootprint,
    reference: ObjectFootprint,
    *,
    config: RelationScoringConfig,
) -> tuple[float, dict[str, Any]]:
    subject_bbox_area = max(1.0, float(subject.bbox.area))
    bbox_intersection = float(subject.bbox.intersection_area(reference.bbox))
    bbox_coverage = _clamp01(bbox_intersection / subject_bbox_area)
    centroid_inside_bbox = float(
        _point_inside_bbox(reference.bbox, subject.centroid_x, subject.centroid_y)
    )
    weight_sum = config.inside_containment_weight + config.inside_centroid_weight
    if weight_sum <= 0.0:
        score = 0.0
    else:
        score = _clamp01(
            (
                config.inside_containment_weight * bbox_coverage
                + config.inside_centroid_weight * centroid_inside_bbox
            )
            / weight_sum
        )

    return score, {
        "score": score,
        "bbox_coverage_by_reference": bbox_coverage,
        "subject_centroid_inside_reference_bbox": bool(centroid_inside_bbox),
        "bbox_contained": reference.bbox.contains(subject.bbox),
        "subject_bbox_area": subject_bbox_area,
        "bbox_intersection_area": bbox_intersection,
    }


def _in_relation_score(
    subject: ObjectFootprint,
    reference: ObjectFootprint,
    *,
    config: RelationScoringConfig,
) -> tuple[float, dict[str, Any]]:
    subject_bbox_area = max(1.0, float(subject.bbox.area))
    bbox_intersection = float(subject.bbox.intersection_area(reference.bbox))
    bbox_coverage = _clamp01(bbox_intersection / subject_bbox_area)
    centroid_inside_bbox = float(
        _point_inside_bbox(reference.bbox, subject.centroid_x, subject.centroid_y)
    )
    weight_sum = config.in_bbox_coverage_weight + config.in_centroid_bbox_weight
    if weight_sum <= 0.0:
        score = 0.0
    else:
        score = _clamp01(
            (
                config.in_bbox_coverage_weight * bbox_coverage
                + config.in_centroid_bbox_weight * centroid_inside_bbox
            )
            / weight_sum
        )

    return score, {
        "score": score,
        "bbox_coverage_by_reference": bbox_coverage,
        "subject_centroid_inside_reference_bbox": bool(centroid_inside_bbox),
        "bbox_contained": reference.bbox.contains(subject.bbox),
        "subject_bbox_area": subject_bbox_area,
        "bbox_intersection_area": bbox_intersection,
    }


def _overlap_relation_score(
    subject: ObjectFootprint,
    reference: ObjectFootprint,
    *,
    config: RelationScoringConfig,
) -> tuple[float, dict[str, Any]]:
    overlap_stats = _footprint_overlap_stats(subject, reference)
    weight_sum = config.overlap_iou_weight + config.overlap_intersection_weight
    if weight_sum <= 0.0:
        score = 0.0
    else:
        score = _clamp01(
            (
                config.overlap_iou_weight * overlap_stats["iou"]
                + config.overlap_intersection_weight
                * overlap_stats["intersection_over_min"]
            )
            / weight_sum
        )
    return score, {
        "score": score,
        **overlap_stats,
    }


def _directional_relation_score(
    relation: str,
    subject: ObjectFootprint,
    reference: ObjectFootprint,
    *,
    image_size: tuple[int, int],
    config: RelationScoringConfig,
) -> tuple[float, dict[str, Any]]:
    image_width = max(1.0, float(image_size[0]))
    image_height = max(1.0, float(image_size[1]))
    overlap_stats = _footprint_overlap_stats(subject, reference)

    normalized_relation = normalize_spatial_relation_name(_normalized_label(relation))
    if normalized_relation == "left_of":
        centroid_delta = reference.centroid_x - subject.centroid_x
        edge_gap = reference.bbox.x_min - subject.bbox.x_max
        axis_overlap = _interval_overlap_ratio(
            subject.bbox.x_min,
            subject.bbox.x_max,
            reference.bbox.x_min,
            reference.bbox.x_max,
        )
        orthogonal_delta = subject.centroid_y - reference.centroid_y
        orthogonal_overlap = _interval_overlap_ratio(
            subject.bbox.y_min,
            subject.bbox.y_max,
            reference.bbox.y_min,
            reference.bbox.y_max,
        )
        axis_size = image_width
        orthogonal_size = image_height
        on_penalty_used = True
    elif normalized_relation == "right_of":
        centroid_delta = subject.centroid_x - reference.centroid_x
        edge_gap = subject.bbox.x_min - reference.bbox.x_max
        axis_overlap = _interval_overlap_ratio(
            subject.bbox.x_min,
            subject.bbox.x_max,
            reference.bbox.x_min,
            reference.bbox.x_max,
        )
        orthogonal_delta = subject.centroid_y - reference.centroid_y
        orthogonal_overlap = _interval_overlap_ratio(
            subject.bbox.y_min,
            subject.bbox.y_max,
            reference.bbox.y_min,
            reference.bbox.y_max,
        )
        axis_size = image_width
        orthogonal_size = image_height
        on_penalty_used = True
    elif normalized_relation == "above":
        centroid_delta = reference.centroid_y - subject.centroid_y
        edge_gap = reference.bbox.y_min - subject.bbox.y_max
        axis_overlap = _interval_overlap_ratio(
            subject.bbox.y_min,
            subject.bbox.y_max,
            reference.bbox.y_min,
            reference.bbox.y_max,
        )
        orthogonal_delta = subject.centroid_x - reference.centroid_x
        orthogonal_overlap = _interval_overlap_ratio(
            subject.bbox.x_min,
            subject.bbox.x_max,
            reference.bbox.x_min,
            reference.bbox.x_max,
        )
        axis_size = image_height
        orthogonal_size = image_width
        on_penalty_used = False
    elif normalized_relation == "below":
        centroid_delta = subject.centroid_y - reference.centroid_y
        edge_gap = subject.bbox.y_min - reference.bbox.y_max
        axis_overlap = _interval_overlap_ratio(
            subject.bbox.y_min,
            subject.bbox.y_max,
            reference.bbox.y_min,
            reference.bbox.y_max,
        )
        orthogonal_delta = subject.centroid_x - reference.centroid_x
        orthogonal_overlap = _interval_overlap_ratio(
            subject.bbox.x_min,
            subject.bbox.x_max,
            reference.bbox.x_min,
            reference.bbox.x_max,
        )
        axis_size = image_height
        orthogonal_size = image_width
        on_penalty_used = False
    else:
        return 0.0, {
            "score": 0.0,
            "error": f"Unsupported directional relation: {relation}",
        }

    centroid_score = _clamp01(
        0.5 + (centroid_delta / max(1.0, axis_size * config.directional_centroid_scale))
    )
    # Small overlaps or touching objects can still satisfy a directional relation,
    # so the gap term grants partial credit within a short tolerance band.
    gap_tolerance = axis_size * config.directional_gap_tolerance_ratio
    gap_score = _clamp01(
        (edge_gap + gap_tolerance)
        / max(
            1.0,
            axis_size
            * (config.directional_gap_scale + config.directional_gap_tolerance_ratio),
        )
    )
    orthogonal_center_score = _clamp01(
        1.0
        - (
            abs(orthogonal_delta)
            / max(1.0, orthogonal_size * config.directional_orthogonal_scale)
        )
    )
    # Orthogonal overlap is more reliable than centroid alignment for natural
    # diagonal layouts, so bias this blend toward overlap evidence.
    orthogonal_weight_sum = (
        config.directional_orthogonal_overlap_weight
        + config.directional_orthogonal_center_weight
    )
    if orthogonal_weight_sum <= 0.0:
        orthogonal_score = 0.0
    else:
        orthogonal_score = _clamp01(
            (
                config.directional_orthogonal_overlap_weight * orthogonal_overlap
                + config.directional_orthogonal_center_weight * orthogonal_center_score
            )
            / orthogonal_weight_sum
        )
    base_weight_sum = (
        config.directional_centroid_weight
        + config.directional_gap_weight
        + config.directional_orthogonal_weight
    )
    if base_weight_sum <= 0.0:
        base_score = 0.0
    else:
        base_score = _clamp01(
            (
                config.directional_centroid_weight * centroid_score
                + config.directional_gap_weight * gap_score
                + config.directional_orthogonal_weight * orthogonal_score
            )
            / base_weight_sum
        )

    containment_conflict = max(
        overlap_stats["subject_coverage_by_reference"],
        overlap_stats["reference_coverage_by_subject"],
    )
    support_contact_score, support_evidence = _support_contact_score(
        subject,
        reference,
        image_size=image_size,
        config=config,
    )
    penalty_weight_sum = (
        config.directional_axis_overlap_penalty_weight
        + config.directional_overlap_penalty_weight
        + config.directional_containment_penalty_weight
        + (config.directional_on_penalty_weight if on_penalty_used else 0.0)
    )
    if penalty_weight_sum <= 0.0:
        penalty_score = 0.0
    else:
        penalty_score = _clamp01(
            (
                config.directional_axis_overlap_penalty_weight * axis_overlap
                + config.directional_overlap_penalty_weight
                * overlap_stats["intersection_over_min"]
                + config.directional_containment_penalty_weight * containment_conflict
                + (
                    config.directional_on_penalty_weight * support_contact_score
                    if on_penalty_used
                    else 0.0
                )
            )
            / penalty_weight_sum
        )
    score = _clamp01(base_score * (1.0 - penalty_score))
    return score, {
        "score": score,
        "base_score": base_score,
        "penalty_score": penalty_score,
        "axis_overlap": axis_overlap,
        "centroid_score": centroid_score,
        "gap_score": gap_score,
        "orthogonal_overlap": orthogonal_overlap,
        "orthogonal_center_score": orthogonal_center_score,
        "orthogonal_score": orthogonal_score,
        "gap_tolerance_pixels": gap_tolerance,
        "orthogonal_overlap_weight": config.directional_orthogonal_overlap_weight,
        "orthogonal_center_weight": config.directional_orthogonal_center_weight,
        "edge_gap": edge_gap,
        "centroid_delta": centroid_delta,
        "orthogonal_delta": orthogonal_delta,
        "containment_conflict": containment_conflict,
        "support_contact_score": support_contact_score,
        "support_contact": support_evidence,
        **overlap_stats,
    }


def _relation_score(
    relation: str,
    subject: ObjectFootprint,
    reference: ObjectFootprint,
    image_size: tuple[int, int],
    *,
    config: RelationScoringConfig,
    depth_nearer_is_smaller: Optional[bool] = None,
) -> tuple[float, dict[str, Any]]:
    subject_bbox = subject.bbox
    reference_bbox = reference.bbox
    horizontal_overlap = _interval_overlap_ratio(
        subject_bbox.x_min,
        subject_bbox.x_max,
        reference_bbox.x_min,
        reference_bbox.x_max,
    )
    vertical_overlap = _interval_overlap_ratio(
        subject_bbox.y_min,
        subject_bbox.y_max,
        reference_bbox.y_min,
        reference_bbox.y_max,
    )
    center_distance = sqrt(
        ((subject.centroid_x - reference.centroid_x) ** 2)
        + ((subject.centroid_y - reference.centroid_y) ** 2)
    )

    normalized_relation = normalize_spatial_relation_name(_normalized_label(relation))
    if normalized_relation == "left_of":
        score, relation_components = _directional_relation_score(
            normalized_relation,
            subject,
            reference,
            image_size=image_size,
            config=config,
        )
    elif normalized_relation == "right_of":
        score, relation_components = _directional_relation_score(
            normalized_relation,
            subject,
            reference,
            image_size=image_size,
            config=config,
        )
    elif normalized_relation == "above":
        score, relation_components = _directional_relation_score(
            normalized_relation,
            subject,
            reference,
            image_size=image_size,
            config=config,
        )
    elif normalized_relation == "below":
        score, relation_components = _directional_relation_score(
            normalized_relation,
            subject,
            reference,
            image_size=image_size,
            config=config,
        )
    elif normalized_relation == "inside":
        score, relation_components = _inside_relation_score(
            subject,
            reference,
            config=config,
        )
    elif normalized_relation == "in":
        score, relation_components = _in_relation_score(
            subject,
            reference,
            config=config,
        )
    elif normalized_relation in {"overlapping", "intersecting"}:
        score, relation_components = _overlap_relation_score(
            subject,
            reference,
            config=config,
        )
    elif normalized_relation == "near":
        score, relation_components = _near_relation_score(
            subject,
            reference,
            image_size=image_size,
            config=config,
        )
    elif normalized_relation == "on":
        support_contact_score, support_evidence = _support_contact_score(
            subject,
            reference,
            image_size=image_size,
            config=config,
        )
        above_score, above_evidence = _directional_relation_score(
            "above",
            subject,
            reference,
            image_size=image_size,
            config=config,
        )
        overlap_stats = _footprint_overlap_stats(subject, reference)
        base_weight_sum = (
            config.on_contact_weight
            + config.on_horizontal_overlap_weight
            + config.on_above_weight
        )
        if base_weight_sum <= 0.0:
            base_score = 0.0
        else:
            base_score = _clamp01(
                (
                    config.on_contact_weight * support_contact_score
                    + config.on_horizontal_overlap_weight * horizontal_overlap
                    + config.on_above_weight * above_score
                )
                / base_weight_sum
            )
        inside_conflict = overlap_stats["subject_coverage_by_reference"]
        score = _clamp01(
            base_score - (config.on_inside_penalty_weight * inside_conflict)
        )
        relation_components = {
            "score": score,
            "base_score": base_score,
            "inside_conflict": inside_conflict,
            "support_contact_score": support_contact_score,
            "support_contact": support_evidence,
            "above_score": above_score,
            "above_relation": above_evidence,
            **overlap_stats,
        }
    elif normalized_relation == "in_front_of":
        if subject.average_depth is None or reference.average_depth is None:
            score = 0.0
            planar_alignment = max(horizontal_overlap, vertical_overlap)
            relation_components = {
                "score": score,
                "depth_score": score,
                "planar_alignment": planar_alignment,
                "near_score": 0.0,
                "support_score": _depth_relation_support_score(
                    planar_alignment=planar_alignment,
                    near_score=0.0,
                ),
                "depth_available": False,
            }
        else:
            depth_delta = (
                (reference.average_depth - subject.average_depth)
                if depth_nearer_is_smaller
                else (subject.average_depth - reference.average_depth)
            )
            near_score, near_evidence = _near_relation_score(
                subject,
                reference,
                image_size=image_size,
                config=config,
            )
            planar_alignment = max(horizontal_overlap, vertical_overlap)
            score, depth_score, support_score = _depth_order_relation_score(
                depth_delta=depth_delta,
                planar_alignment=planar_alignment,
                near_score=near_score,
                config=config,
            )
            relation_components = {
                "score": score,
                "depth_score": depth_score,
                "depth_delta": depth_delta,
                "planar_alignment": planar_alignment,
                "near_score": near_score,
                "support_score": support_score,
                "near_relation": near_evidence,
                "depth_available": True,
            }
    elif normalized_relation == "behind":
        if subject.average_depth is None or reference.average_depth is None:
            score = 0.0
            planar_alignment = max(horizontal_overlap, vertical_overlap)
            relation_components = {
                "score": score,
                "depth_score": score,
                "planar_alignment": planar_alignment,
                "near_score": 0.0,
                "support_score": _depth_relation_support_score(
                    planar_alignment=planar_alignment,
                    near_score=0.0,
                ),
                "depth_available": False,
            }
        else:
            depth_delta = (
                (subject.average_depth - reference.average_depth)
                if depth_nearer_is_smaller
                else (reference.average_depth - subject.average_depth)
            )
            near_score, near_evidence = _near_relation_score(
                subject,
                reference,
                image_size=image_size,
                config=config,
            )
            planar_alignment = max(horizontal_overlap, vertical_overlap)
            score, depth_score, support_score = _depth_order_relation_score(
                depth_delta=depth_delta,
                planar_alignment=planar_alignment,
                near_score=near_score,
                config=config,
            )
            relation_components = {
                "score": score,
                "depth_score": depth_score,
                "depth_delta": depth_delta,
                "planar_alignment": planar_alignment,
                "near_score": near_score,
                "support_score": support_score,
                "near_relation": near_evidence,
                "depth_available": True,
            }
    else:
        score = 0.0
        relation_components = {
            "score": score,
            "error": f"Unsupported relation: {relation}",
        }

    return score, {
        "relation": normalized_relation,
        "subject_bbox": subject_bbox.to_dict(),
        "reference_bbox": reference_bbox.to_dict(),
        "subject_footprint": subject.evidence_dict(),
        "reference_footprint": reference.evidence_dict(),
        "subject_average_depth": subject.average_depth,
        "reference_average_depth": reference.average_depth,
        "depth_nearer_is_smaller": depth_nearer_is_smaller,
        "horizontal_overlap": horizontal_overlap,
        "vertical_overlap": vertical_overlap,
        "center_distance": center_distance,
        "uses_segmentation_mask": {
            "subject": subject.uses_segmentation_mask,
            "reference": reference.uses_segmentation_mask,
        },
        "components": relation_components,
    }


def _padded_union_bbox(
    boxes: Sequence[BoundingBox],
    *,
    image_size: tuple[int, int],
    padding_ratio: float,
) -> BoundingBox:
    joint_bbox = BoundingBox.union(tuple(boxes))
    image_width = float(max(1, int(image_size[0])))
    image_height = float(max(1, int(image_size[1])))
    pad_x = joint_bbox.width * float(padding_ratio)
    pad_y = joint_bbox.height * float(padding_ratio)
    x_min = max(0.0, joint_bbox.x_min - pad_x)
    y_min = max(0.0, joint_bbox.y_min - pad_y)
    x_max = min(image_width, joint_bbox.x_max + pad_x)
    y_max = min(image_height, joint_bbox.y_max + pad_y)

    if x_max <= x_min:
        center_x = min(max(0.0, joint_bbox.center[0]), image_width - 1.0)
        x_min = max(0.0, center_x)
        x_max = min(image_width, x_min + 1.0)
    if y_max <= y_min:
        center_y = min(max(0.0, joint_bbox.center[1]), image_height - 1.0)
        y_min = max(0.0, center_y)
        y_max = min(image_height, y_min + 1.0)

    return BoundingBox(
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
    )


def _build_action_target_scoring_region(
    *,
    subject_region: DetectedRegion,
    target_footprint: ObjectFootprint,
    target_object_id: str,
    image_size: tuple[int, int],
    config: ActionScoringConfig,
) -> DetectedRegion:
    joint_bbox = _padded_union_bbox(
        (subject_region.bbox, target_footprint.bbox),
        image_size=image_size,
        padding_ratio=config.target_joint_crop_padding_ratio,
    )
    metadata = {
        "region_source": "action_joint_crop",
        "subject_region_id": subject_region.region_id,
        "target_object_id": target_object_id,
        "target_bbox": target_footprint.bbox.to_dict(),
        "padding_ratio": float(config.target_joint_crop_padding_ratio),
    }
    return DetectedRegion(
        region_id=f"{subject_region.region_id}:action_target:{target_object_id}",
        bbox=joint_bbox,
        label=subject_region.label,
        confidence=subject_region.confidence,
        average_depth=subject_region.average_depth,
        mask=None,
        attributes=dict(subject_region.attributes),
        metadata=metadata,
    )


def _crop_image_to_region_bbox(
    image: Image.Image,
    region: DetectedRegion,
) -> Image.Image:
    left = int(max(0, np.floor(region.bbox.x_min)))
    upper = int(max(0, np.floor(region.bbox.y_min)))
    right = int(min(image.width, np.ceil(region.bbox.x_max)))
    lower = int(min(image.height, np.ceil(region.bbox.y_max)))
    if right <= left or lower <= upper:
        return image.convert("RGB")
    return image.convert("RGB").crop((left, upper, right, lower))


def _build_attribute_queries(
    object_spec: ObjectSpec,
    attribute_name: str,
    expected_value: Any,
    *,
    action_target_label: str | None = None,
) -> tuple[str, ...]:
    value_text = str(expected_value).strip()
    normalized_attribute = _normalized_label(attribute_name)
    return _build_attribute_template_queries(
        object_spec.label,
        normalized_attribute=normalized_attribute,
        value_text=value_text,
        action_target_label=action_target_label,
    )


def _build_attribute_template_queries(
    label: str,
    *,
    normalized_attribute: str,
    value_text: str,
    action_target_label: str | None = None,
) -> tuple[str, ...]:
    label = str(label).strip()
    if normalized_attribute == "action":
        target_text = (
            None if action_target_label is None else str(action_target_label).strip()
        )
        if target_text:
            return _dedupe_texts(
                (
                    f"{label} {value_text} {target_text}",
                    f"{label} that is {value_text} {target_text}",
                    f"{label} {value_text} a {target_text}",
                    f"{label} {value_text} the {target_text}",
                    f"{label} while {value_text} {target_text}",
                )
            )
        return (
            f"{label} {value_text}",
            f"{label} that is {value_text}",
            f"{label} whose action is {value_text}",
        )
    if normalized_attribute in {"pose", "state"}:
        return (
            f"{label} {value_text}",
            f"{label} that is {value_text}",
            f"{label} whose {normalized_attribute} is {value_text}",
        )
    if normalized_attribute == "color":
        return _dedupe_texts(
            (
                f"{value_text} {label}",
                f"{label} with {value_text} color",
                f"{label} whose color is {value_text}",
                f"{label} that is {value_text}",
            )
        )
    if normalized_attribute == "material":
        return _dedupe_texts(
            (
                f"{value_text} {label}",
                f"{label} made of {value_text}",
                f"{label} with {value_text} material",
            )
        )
    attribute_text = normalized_attribute.replace("_", " ")
    return _dedupe_texts(
        (
            f"{value_text} {label}",
            f"{label} with {attribute_text} {value_text}",
            f"{label} whose {attribute_text} is {value_text}",
            f"{attribute_text} {value_text} {label}",
        )
    )


def _dedupe_texts(values: Sequence[str]) -> tuple[str, ...]:
    unique: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if normalized and normalized not in unique:
            unique.append(normalized)
    return tuple(unique)


def _build_generic_object_queries(object_spec: ObjectSpec) -> tuple[str, ...]:
    label = str(object_spec.label).strip()
    candidates = [label, *[str(alias).strip() for alias in object_spec.aliases]]
    generic_queries: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        generic_queries.append(candidate)
        article = "an" if candidate[0].lower() in {"a", "e", "i", "o", "u"} else "a"
        generic_queries.append(f"{article} {candidate}")
    return _dedupe_texts(generic_queries)


def _best_query_score(
    session: VerificationSession,
    region,
    queries: Sequence[str],
) -> tuple[Optional[float], Optional[str], Mapping[str, float]]:
    if session.cache.tools.region_text_scoring_tool is None:
        return None, None, {}
    query_scores = {
        query: session.cache.score_region_text(region, query) for query in queries
    }
    available_scores = {
        query: float(value)
        for query, value in query_scores.items()
        if value is not None
    }
    if not available_scores:
        return None, None, {}
    best_query = max(available_scores, key=available_scores.get)
    return available_scores[best_query], best_query, available_scores


def _article_for_phrase(phrase: str) -> str:
    stripped = str(phrase).strip()
    if not stripped:
        return "a"
    return "an" if stripped[0].lower() in {"a", "e", "i", "o", "u"} else "a"


def _build_global_scene_queries(
    attribute_name: str,
    expected_value: Any,
) -> tuple[str, ...]:
    value_text = " ".join(str(expected_value).strip().split())
    if not value_text:
        return ()

    normalized_attribute = _normalized_label(attribute_name)

    if normalized_attribute == "background":
        return _dedupe_texts(
            (
                value_text,
                f"{value_text} background",
                f"scene with {value_text} background",
                f"{value_text} backdrop",
            )
        )

    article = _article_for_phrase(value_text)
    return _dedupe_texts(
        (
            value_text,
            f"{value_text} scene",
            f"{value_text} environment",
            f"scene in {value_text}",
            f"scene in the {value_text}",
            f"scene with {article} {value_text} background",
        )
    )


def _best_scene_query_score(
    session: VerificationSession,
    *,
    variant_key: str,
    image: Image.Image,
    queries: Sequence[str],
) -> tuple[Optional[float], Optional[str], Mapping[str, float]]:
    query_scores = {
        query: session.score_scene_text(
            variant_key=variant_key,
            image=image,
            text=query,
        )
        for query in queries
    }
    available_scores = {
        query: float(value)
        for query, value in query_scores.items()
        if value is not None
    }
    if not available_scores:
        return None, None, {}
    best_query = max(available_scores, key=available_scores.get)
    return available_scores[best_query], best_query, available_scores


def _region_area_ratio(
    region: DetectedRegion,
    image: Image.Image,
) -> float:
    total_area = float(max(1, image.width * image.height))
    return _clamp01(float(region.area) / total_area)


def _region_background_area_ratio(
    region: DetectedRegion,
    image: Image.Image,
    foreground_mask: Optional[np.ndarray],
) -> float:
    raw_area_ratio = _region_area_ratio(region, image)
    if foreground_mask is None or region.mask is None:
        return raw_area_ratio

    region_mask = np.asarray(region.mask, dtype=bool)
    foreground = np.asarray(foreground_mask, dtype=bool)
    if region_mask.shape != foreground.shape or not region_mask.any():
        return raw_area_ratio

    background_pixels = np.logical_and(region_mask, np.logical_not(foreground))
    return _clamp01(float(background_pixels.mean()))


def _best_scene_query_proposal(
    session: VerificationSession,
    *,
    queries: Sequence[str],
    foreground_mask: Optional[np.ndarray],
    min_area_ratio: float,
) -> tuple[Optional[float], Optional[str], Optional[float], Mapping[str, Any]]:
    best_score = None
    best_query = None
    best_confidence = None
    query_evidence: dict[str, Any] = {}

    for query in queries:
        regions = session.scene_query_regions(query)
        region_evidence: list[dict[str, Any]] = []
        query_best_score = None
        query_best_region_id = None
        query_best_confidence = None
        query_best_raw_area_ratio = None
        query_best_background_area_ratio = None

        for region in regions:
            raw_area_ratio = _region_area_ratio(region, session.image)
            background_area_ratio = _region_background_area_ratio(
                region,
                session.image,
                foreground_mask,
            )
            qualifies = background_area_ratio >= float(min_area_ratio)
            proposal_score = None
            if qualifies:
                proposal_score = _clamp01(
                    0.0 if region.confidence is None else float(region.confidence)
                )
                if query_best_score is None or proposal_score > query_best_score:
                    query_best_score = proposal_score
                    query_best_region_id = region.region_id
                    query_best_confidence = (
                        None if region.confidence is None else float(region.confidence)
                    )
                    query_best_raw_area_ratio = raw_area_ratio
                    query_best_background_area_ratio = background_area_ratio

            region_evidence.append(
                {
                    "region_id": region.region_id,
                    "confidence": (
                        None if region.confidence is None else float(region.confidence)
                    ),
                    "raw_area_ratio": raw_area_ratio,
                    "background_area_ratio": background_area_ratio,
                    "qualified": qualifies,
                    "proposal_score": proposal_score,
                    "bbox": region.bbox.to_dict(),
                }
            )

        query_evidence[query] = {
            "region_count": len(regions),
            "best_score": query_best_score,
            "best_region_id": query_best_region_id,
            "best_region_confidence": query_best_confidence,
            "best_raw_area_ratio": query_best_raw_area_ratio,
            "best_background_area_ratio": query_best_background_area_ratio,
            "regions": region_evidence[:3],
        }
        if query_best_score is not None and (
            best_score is None or query_best_score > best_score
        ):
            best_score = query_best_score
            best_query = query
            best_confidence = query_best_confidence

    return best_score, best_query, best_confidence, query_evidence


def _scene_foreground_object_ids(program: VisualProgram) -> tuple[str, ...]:
    object_ids: list[str] = []
    for predicate in program.predicates:
        if isinstance(predicate, ExclusionPredicate):
            continue
        for object_id in predicate.referenced_object_ids():
            if object_id not in object_ids:
                object_ids.append(object_id)
    return tuple(object_ids)


def _dilate_boolean_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    if radius <= 0:
        return np.asarray(mask, dtype=bool)

    boolean_mask = np.asarray(mask, dtype=bool)
    padded = np.pad(boolean_mask, radius, mode="constant", constant_values=False)
    dilated = np.zeros_like(boolean_mask, dtype=bool)
    for y_offset in range((2 * radius) + 1):
        for x_offset in range((2 * radius) + 1):
            dilated |= padded[
                y_offset : y_offset + boolean_mask.shape[0],
                x_offset : x_offset + boolean_mask.shape[1],
            ]
    return dilated


def _merge_foreground_mask(
    session: VerificationSession,
    object_ids: Sequence[str],
) -> Optional[np.ndarray]:
    merged_mask = np.zeros((session.image.height, session.image.width), dtype=bool)
    found_any = False
    for object_id in object_ids:
        footprint = session.object_footprint(object_id)
        if footprint is None:
            continue
        merged_mask |= np.asarray(footprint.mask, dtype=bool)
        found_any = True
    if not found_any:
        return None
    return merged_mask


def _make_background_residual_image(
    image: Image.Image,
    background_mask: np.ndarray,
) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    mask = np.asarray(background_mask, dtype=bool)
    if mask.shape != rgb.shape[:2] or not mask.any():
        return image.convert("RGB")

    background_pixels = rgb[mask]
    if background_pixels.size == 0:
        fill_color = np.array([127, 127, 127], dtype=np.uint8)
    else:
        fill_color = np.clip(np.round(background_pixels.mean(axis=0)), 0, 255).astype(
            np.uint8
        )
    residual = rgb.copy()
    residual[~mask] = fill_color
    return Image.fromarray(residual)


class CountPredicateEvaluator:
    """Evaluate object counts from cached region proposals."""

    def evaluate(
        self,
        predicate: Predicate,
        session: VerificationSession,
    ) -> PredicateVerificationResult:
        assert isinstance(predicate, CountPredicate)
        regions = tuple(
            sorted(
                session.object_regions(predicate.target.object_id),
                key=lambda region: float(region.confidence or 0.0),
                reverse=True,
            )
        )
        strong_count, weak_count = _proposal_count_stats(
            regions,
            thresholds=session.thresholds,
        )

        if predicate.comparator == CountComparator.EXACTLY:
            state = _exact_count_state_from_counts(
                strong_count=strong_count,
                weak_count=weak_count,
                expected_count=predicate.expected_count,
            )
            score = _exact_count_score_from_counts(
                strong_count=strong_count,
                weak_count=weak_count,
                expected_count=predicate.expected_count,
            )
        elif predicate.comparator == CountComparator.AT_LEAST:
            if strong_count >= predicate.expected_count:
                state = PredicateState.SATISFIED
            elif weak_count >= predicate.expected_count:
                state = PredicateState.UNCERTAIN
            else:
                state = PredicateState.VIOLATED
            score = _clamp01(strong_count / max(1.0, float(predicate.expected_count)))
        else:
            if weak_count <= predicate.expected_count:
                state = (
                    PredicateState.SATISFIED
                    if strong_count <= predicate.expected_count
                    else PredicateState.UNCERTAIN
                )
            else:
                state = PredicateState.VIOLATED
            score = _clamp01(
                1.0
                - (
                    max(0.0, weak_count - predicate.expected_count)
                    / max(1.0, float(predicate.expected_count))
                )
            )

        summary = (
            f"Counted {strong_count} strong and {weak_count} weak candidate region(s) for "
            f"{predicate.target.object_id}; expected {predicate.comparator.value} {predicate.expected_count}."
        )
        return PredicateVerificationResult(
            predicate_id=predicate.predicate_id,
            category=predicate.category,
            state=state,
            score=score,
            confidence=_safe_mean(
                [
                    float(region.confidence or 0.0)
                    for region in regions[: max(1, predicate.expected_count)]
                ]
            ),
            evidence_summary=summary,
            evidence={
                "object_id": predicate.target.object_id,
                "expected_count": predicate.expected_count,
                "comparator": predicate.comparator.value,
                "strong_count": strong_count,
                "weak_count": weak_count,
                "weak_overcount_delta": max(
                    0,
                    weak_count - predicate.expected_count,
                ),
                "candidate_regions": [region.evidence_dict() for region in regions],
            },
        )


class AttributePredicateEvaluator:
    """Evaluate colors or other object attributes on selected regions."""

    def evaluate(
        self,
        predicate: Predicate,
        session: VerificationSession,
    ) -> PredicateVerificationResult:
        assert isinstance(predicate, AttributePredicate)
        object_spec = session.program.get_object(predicate.target.object_id)
        action_target_spec = (
            None
            if predicate.action_target is None
            else session.program.get_object(predicate.action_target.object_id)
        )
        regions = session.selected_regions(predicate.target.object_id)
        if not regions:
            return PredicateVerificationResult(
                predicate_id=predicate.predicate_id,
                category=predicate.category,
                state=PredicateState.VIOLATED,
                score=0.0,
                confidence=0.0,
                evidence_summary=f"No candidate regions available for {predicate.target.object_id}.",
                evidence={
                    "object_id": predicate.target.object_id,
                    "candidate_regions": [],
                },
            )

        normalized_attribute = _normalized_label(predicate.attribute_name)
        per_region = []
        queries = _build_attribute_queries(
            object_spec,
            predicate.attribute_name,
            predicate.expected_value,
            action_target_label=(
                None if action_target_spec is None else action_target_spec.label
            ),
        )
        generic_queries = _build_generic_object_queries(object_spec)
        target_footprint = None
        action_tool = (
            session.cache.tools.action_verification_tool
            if session.action_vllm_enabled
            else None
        )
        if normalized_attribute == "action" and predicate.action_target is not None:
            target_footprint = session.object_footprint(
                predicate.action_target.object_id
            )
        for region in regions:
            metadata_value = region.attributes.get(predicate.attribute_name)
            metadata_score = None
            if metadata_value is not None:
                metadata_score = (
                    1.0
                    if _normalized_label(metadata_value)
                    == _normalized_label(predicate.expected_value)
                    else 0.0
                )

            scoring_region = region
            action_scoring_strategy = None
            if normalized_attribute == "action" and predicate.action_target is not None:
                if target_footprint is not None:
                    scoring_region = _build_action_target_scoring_region(
                        subject_region=region,
                        target_footprint=target_footprint,
                        target_object_id=predicate.action_target.object_id,
                        image_size=session.image.size,
                        config=session.action_scoring,
                    )
                    action_scoring_strategy = (
                        "joint_crop_vlm"
                        if action_tool is not None
                        else "joint_crop_siglip"
                    )
                else:
                    action_scoring_strategy = (
                        "subject_region_vlm"
                        if action_tool is not None
                        else "subject_region_siglip_fallback"
                    )
            text_score = None
            matched_query = None
            available_scores: Mapping[str, float] = {}
            generic_score = None
            generic_query = None
            generic_available_scores: Mapping[str, float] = {}
            action_verification_outcome = None

            if normalized_attribute == "action" and action_tool is not None:
                action_crop = _crop_image_to_region_bbox(session.image, scoring_region)
                action_verification_outcome = action_tool.verify_action(
                    action_crop,
                    subject_label=object_spec.label,
                    action_text=str(predicate.expected_value),
                    target_label=(
                        None if action_target_spec is None else action_target_spec.label
                    ),
                    artifact_name=(
                        f"{predicate.predicate_id}_{scoring_region.region_id}"
                    ),
                )
                if action_verification_outcome.ok is True:
                    score = 1.0
                    source = "action_vllm"
                elif action_verification_outcome.ok is False:
                    score = 0.0
                    source = "action_vllm"
                else:
                    score = 0.5
                    source = "action_vllm_uncertain"
            else:
                text_score, matched_query, available_scores = _best_query_score(
                    session,
                    scoring_region,
                    queries,
                )
                generic_score, generic_query, generic_available_scores = (
                    _best_query_score(
                        session,
                        region,
                        generic_queries,
                    )
                )

            if metadata_score == 1.0 and not (
                normalized_attribute == "action" and action_tool is not None
            ):
                score = 1.0
                source = "metadata"
            elif normalized_attribute == "action" and action_tool is not None:
                pass
            elif (
                normalized_attribute == "action" and predicate.action_target is not None
            ):
                if text_score is not None:
                    score = text_score
                    source = (
                        "siglip_joint_crop"
                        if action_scoring_strategy == "joint_crop_siglip"
                        else "siglip"
                    )
                elif metadata_score is not None:
                    score = metadata_score
                    source = "metadata"
                else:
                    score = 0.0
                    source = "no_signal"
            elif text_score is not None:
                score = text_score
                source = "siglip"
            elif metadata_score is not None:
                score = metadata_score
                source = "metadata"
            else:
                score = 0.0
                source = "no_signal"

            per_region.append(
                {
                    "region": region.evidence_dict(),
                    "scoring_region": scoring_region.evidence_dict(),
                    "source": source,
                    "metadata_value": metadata_value,
                    "matched_query": matched_query,
                    "matched_query_score": text_score,
                    "generic_query": generic_query,
                    "generic_query_score": generic_score,
                    "query_scores": dict(available_scores),
                    "generic_query_scores": dict(generic_available_scores),
                    "action_scoring_strategy": action_scoring_strategy,
                    "action_verification_outcome": (
                        None
                        if action_verification_outcome is None
                        else action_verification_outcome.to_dict()
                    ),
                    "action_verification_method": (
                        "vlm_crop"
                        if normalized_attribute == "action" and action_tool is not None
                        else "siglip_legacy"
                        if normalized_attribute == "action"
                        else None
                    ),
                    "action_target_object_id": (
                        None
                        if predicate.action_target is None
                        else predicate.action_target.object_id
                    ),
                    "score": score,
                }
            )

        scores = [float(item["score"]) for item in per_region]
        aggregate_score = min(scores)
        if normalized_attribute == "action" and action_tool is not None:
            region_states = []
            for item in per_region:
                outcome = item.get("action_verification_outcome") or {}
                ok = outcome.get("ok")
                if ok is True:
                    region_states.append(PredicateState.SATISFIED)
                elif ok is False:
                    region_states.append(PredicateState.VIOLATED)
                else:
                    region_states.append(PredicateState.UNCERTAIN)
            if any(
                state_item == PredicateState.VIOLATED for state_item in region_states
            ):
                state = PredicateState.VIOLATED
            elif any(
                state_item == PredicateState.UNCERTAIN for state_item in region_states
            ):
                state = PredicateState.UNCERTAIN
            else:
                state = PredicateState.SATISFIED
        else:
            if normalized_attribute == "action":
                satisfied_threshold = session.thresholds.action_satisfied
                uncertain_threshold = session.thresholds.action_uncertain
            else:
                satisfied_threshold = session.thresholds.attribute_satisfied
                uncertain_threshold = session.thresholds.attribute_uncertain
            state = _state_from_score(
                aggregate_score,
                satisfied_threshold=satisfied_threshold,
                uncertain_threshold=uncertain_threshold,
            )
        if normalized_attribute == "action" and action_tool is not None:
            action_confidences = [
                float(item["action_verification_outcome"]["confidence"])
                for item in per_region
                if item.get("action_verification_outcome") is not None
                and item["action_verification_outcome"].get("confidence") is not None
            ]
            confidence = _safe_mean(action_confidences)
            if confidence is None:
                confidence = _safe_mean(
                    [float(region.confidence or 0.0) for region in regions]
                )
        else:
            confidence = _safe_mean(
                [float(region.confidence or 0.0) for region in regions]
            )
        summary = (
            f"Checked {predicate.attribute_name}={predicate.expected_value!r} on "
            f"{len(regions)} selected region(s) for {predicate.target.object_id}; "
            f"aggregate score={aggregate_score:.2f}."
        )
        if predicate.action_target is not None:
            summary = (
                f"Checked {predicate.attribute_name}={predicate.expected_value!r} from "
                f"{predicate.target.object_id} toward {predicate.action_target.object_id} "
                f"using {len(regions)} selected region(s); aggregate score={aggregate_score:.2f}."
            )
        if normalized_attribute == "action" and action_tool is not None:
            summary += " Action score comes from the VLM crop-based action verifier."
        return PredicateVerificationResult(
            predicate_id=predicate.predicate_id,
            category=predicate.category,
            state=state,
            score=aggregate_score,
            confidence=confidence,
            evidence_summary=summary,
            evidence={
                "object_id": predicate.target.object_id,
                "attribute_name": predicate.attribute_name,
                "expected_value": predicate.expected_value,
                "action_target_object_id": (
                    None
                    if predicate.action_target is None
                    else predicate.action_target.object_id
                ),
                "selected_region_count": len(regions),
                "action_target_scoring": (
                    None
                    if predicate.action_target is None
                    else {
                        "strategy": action_scoring_strategy,
                        "target_footprint": (
                            None
                            if target_footprint is None
                            else target_footprint.evidence_dict()
                        ),
                        "padding_ratio": float(
                            session.action_scoring.target_joint_crop_padding_ratio
                        ),
                    }
                ),
                "per_region": per_region,
            },
        )


class SpatialRelationPredicateEvaluator:
    """Evaluate simple 2D spatial relations using reused object summaries."""

    def evaluate(
        self,
        predicate: Predicate,
        session: VerificationSession,
    ) -> PredicateVerificationResult:
        assert isinstance(predicate, SpatialRelationPredicate)
        subject_group = session.object_group(predicate.subject.object_id)
        reference_group = session.object_group(predicate.reference.object_id)
        subject_footprint = session.object_footprint(predicate.subject.object_id)
        reference_footprint = session.object_footprint(predicate.reference.object_id)
        if (
            subject_group is None
            or reference_group is None
            or subject_footprint is None
            or reference_footprint is None
        ):
            missing = []
            if subject_group is None or subject_footprint is None:
                missing.append(predicate.subject.object_id)
            if reference_group is None or reference_footprint is None:
                missing.append(predicate.reference.object_id)
            return PredicateVerificationResult(
                predicate_id=predicate.predicate_id,
                category=predicate.category,
                state=PredicateState.VIOLATED,
                score=0.0,
                confidence=0.0,
                evidence_summary=f"Missing object evidence for {', '.join(missing)}.",
                evidence={"missing_object_ids": missing},
            )

        depth_estimate = session.cache.get_depth_estimate()
        score, relation_evidence = _relation_score(
            predicate.relation,
            subject_footprint,
            reference_footprint,
            session.image.size,
            config=session.relation_scoring,
            depth_nearer_is_smaller=(
                None if depth_estimate is None else depth_estimate.nearer_is_smaller
            ),
        )
        state = _state_from_score(
            score,
            satisfied_threshold=session.thresholds.relation_satisfied,
            uncertain_threshold=session.thresholds.relation_uncertain,
        )
        confidence = _safe_mean([subject_group.confidence, reference_group.confidence])
        summary = (
            f"Relation {predicate.relation} between {predicate.subject.object_id} and "
            f"{predicate.reference.object_id} scored {score:.2f}."
        )
        return PredicateVerificationResult(
            predicate_id=predicate.predicate_id,
            category=predicate.category,
            state=state,
            score=score,
            confidence=confidence,
            evidence_summary=summary,
            evidence={
                "subject": subject_group.evidence_dict(),
                "reference": reference_group.evidence_dict(),
                "relation": relation_evidence,
            },
        )


class GlobalScenePredicateEvaluator:
    """Evaluate holistic scene-level environment or background constraints."""

    foreground_dilation_ratio: float = 0.01
    foreground_dilation_min_pixels: int = 4
    full_image_weight: float = 0.45
    residual_background_weight: float = 0.55

    def evaluate(
        self,
        predicate: Predicate,
        session: VerificationSession,
    ) -> PredicateVerificationResult:
        assert isinstance(predicate, GlobalScenePredicate)
        queries = _build_global_scene_queries(
            predicate.attribute_name,
            predicate.expected_value,
        )
        if not queries:
            return PredicateVerificationResult(
                predicate_id=predicate.predicate_id,
                category=predicate.category,
                state=PredicateState.UNCERTAIN,
                score=None,
                confidence=None,
                evidence_summary=(
                    f"No valid scene queries could be built for "
                    f"{predicate.attribute_name}={predicate.expected_value!r}."
                ),
                evidence={
                    "attribute_name": predicate.attribute_name,
                    "expected_value": predicate.expected_value,
                    "scene_queries": [],
                },
            )

        foreground_object_ids = _scene_foreground_object_ids(session.program)
        foreground_mask = _merge_foreground_mask(session, foreground_object_ids)
        dilated_foreground_mask = None
        proposal_score = None
        proposal_query = None
        proposal_confidence = None
        proposal_query_evidence: Mapping[str, Any] = {}
        residual_score = None
        residual_query = None
        residual_query_scores: Mapping[str, float] = {}
        full_score = None
        full_query = None
        full_query_scores: Mapping[str, float] = {}
        background_area_ratio = None
        foreground_area_ratio = None
        dilation_radius = None

        if foreground_mask is not None:
            foreground_area_ratio = float(
                np.asarray(foreground_mask, dtype=bool).mean()
            )
            dilation_radius = max(
                self.foreground_dilation_min_pixels,
                int(
                    round(
                        min(session.image.width, session.image.height)
                        * self.foreground_dilation_ratio
                    )
                ),
            )
            dilated_foreground_mask = _dilate_boolean_mask(
                foreground_mask,
                radius=dilation_radius,
            )
            background_mask = np.logical_not(dilated_foreground_mask)
            background_area_ratio = float(background_mask.mean())
        proposal_score, proposal_query, proposal_confidence, proposal_query_evidence = (
            _best_scene_query_proposal(
                session,
                queries=queries,
                foreground_mask=dilated_foreground_mask,
                min_area_ratio=session.thresholds.scene_background_min_area_ratio,
            )
        )

        if proposal_score is None:
            full_score, full_query, full_query_scores = _best_scene_query_score(
                session,
                variant_key="scene_full",
                image=session.image,
                queries=queries,
            )
            if (
                dilated_foreground_mask is not None
                and background_area_ratio is not None
                and background_area_ratio
                >= session.thresholds.scene_background_min_area_ratio
            ):
                residual_image = _make_background_residual_image(
                    session.image,
                    np.logical_not(dilated_foreground_mask),
                )
                (
                    residual_score,
                    residual_query,
                    residual_query_scores,
                ) = _best_scene_query_score(
                    session,
                    variant_key="scene_background_residual",
                    image=residual_image,
                    queries=queries,
                )

        aggregate_score = proposal_score
        aggregate_source = "sam3_scene_presence" if proposal_score is not None else None
        weighted_components: list[tuple[str, float, float]] = []

        if aggregate_score is None:
            if full_score is not None:
                weighted_components.append(
                    ("full_image_siglip", float(full_score), self.full_image_weight)
                )
            if residual_score is not None:
                weighted_components.append(
                    (
                        "background_residual_siglip",
                        float(residual_score),
                        self.residual_background_weight,
                    )
                )
            if weighted_components:
                total_weight = sum(weight for _, _, weight in weighted_components)
                aggregate_score = _clamp01(
                    sum(score * weight for _, score, weight in weighted_components)
                    / max(1e-8, total_weight)
                )
                aggregate_source = "siglip_scene_text"

        if aggregate_score is None:
            return PredicateVerificationResult(
                predicate_id=predicate.predicate_id,
                category=predicate.category,
                state=PredicateState.UNCERTAIN,
                score=None,
                confidence=None,
                evidence_summary=(
                    f"No scene-level background evidence was available for "
                    f"{predicate.attribute_name}={predicate.expected_value!r}."
                ),
                evidence={
                    "attribute_name": predicate.attribute_name,
                    "expected_value": predicate.expected_value,
                    "scene_queries": list(queries),
                    "foreground_object_ids": list(foreground_object_ids),
                    "foreground_mask_available": foreground_mask is not None,
                    "foreground_area_ratio": foreground_area_ratio,
                    "background_area_ratio": background_area_ratio,
                    "dilation_radius": dilation_radius,
                    "scene_proposal": {
                        "best_query": proposal_query,
                        "best_score": proposal_score,
                        "best_region_confidence": proposal_confidence,
                        "query_evidence": dict(proposal_query_evidence),
                        "scene_background_min_area_ratio": (
                            session.thresholds.scene_background_min_area_ratio
                        ),
                    },
                    "full_image": {
                        "best_query": full_query,
                        "best_score": full_score,
                        "query_scores": dict(full_query_scores),
                    },
                    "background_residual": {
                        "best_query": residual_query,
                        "best_score": residual_score,
                        "query_scores": dict(residual_query_scores),
                    },
                },
            )
        if aggregate_source == "sam3_scene_presence":
            satisfied_threshold = session.thresholds.proposal_satisfied
            uncertain_threshold = session.thresholds.proposal_uncertain
        else:
            satisfied_threshold = session.thresholds.attribute_satisfied
            uncertain_threshold = session.thresholds.attribute_uncertain
        state = _state_from_score(
            aggregate_score,
            satisfied_threshold=satisfied_threshold,
            uncertain_threshold=uncertain_threshold,
        )

        foreground_confidences = []
        for object_id in foreground_object_ids:
            object_group = session.object_group(object_id)
            if object_group is None:
                continue
            foreground_confidences.append(float(object_group.confidence))
        confidence = proposal_confidence
        if confidence is None:
            confidence = _safe_mean(foreground_confidences)
        summary = (
            f"Scene-level check for {predicate.attribute_name}={predicate.expected_value!r} "
            f"scored {aggregate_score:.2f} via {aggregate_source}"
        )
        if proposal_query is not None:
            summary += f" using SAM3 query {proposal_query!r}"
        if aggregate_source == "siglip_scene_text" and full_score is not None:
            summary += f" using full-image score {float(full_score):.2f}"
        if aggregate_source == "siglip_scene_text" and residual_score is not None:
            summary += f" and residual-background score {float(residual_score):.2f}"
        summary += "."

        return PredicateVerificationResult(
            predicate_id=predicate.predicate_id,
            category=predicate.category,
            state=state,
            score=aggregate_score,
            confidence=confidence,
            evidence_summary=summary,
            evidence={
                "attribute_name": predicate.attribute_name,
                "expected_value": predicate.expected_value,
                "scene_queries": list(queries),
                "foreground_object_ids": list(foreground_object_ids),
                "foreground_mask_available": foreground_mask is not None,
                "foreground_area_ratio": foreground_area_ratio,
                "background_area_ratio": background_area_ratio,
                "dilation_radius": dilation_radius,
                "scene_background_min_area_ratio": (
                    session.thresholds.scene_background_min_area_ratio
                ),
                "aggregate_source": aggregate_source,
                "scene_proposal": {
                    "best_query": proposal_query,
                    "best_score": proposal_score,
                    "best_region_confidence": proposal_confidence,
                    "query_evidence": dict(proposal_query_evidence),
                    "scene_background_min_area_ratio": (
                        session.thresholds.scene_background_min_area_ratio
                    ),
                },
                "weighted_components": [
                    {
                        "source": source,
                        "score": score,
                        "weight": weight,
                    }
                    for source, score, weight in weighted_components
                ],
                "full_image": {
                    "best_query": full_query,
                    "best_score": full_score,
                    "query_scores": dict(full_query_scores),
                },
                "background_residual": {
                    "best_query": residual_query,
                    "best_score": residual_score,
                    "query_scores": dict(residual_query_scores),
                },
            },
        )


class ExclusionPredicateEvaluator:
    """Evaluate object absence from cached region proposals."""

    duplicate_intersection_threshold: float = 0.90
    duplicate_area_similarity_threshold: float = 0.70
    duplicate_iou_threshold: float = 0.75
    residual_ignore_threshold: float = 0.10

    def evaluate(
        self,
        predicate: Predicate,
        session: VerificationSession,
    ) -> PredicateVerificationResult:
        assert isinstance(predicate, ExclusionPredicate)
        regions = tuple(session.cache.get_object_regions(predicate.target.object_id))
        reference_object_ids = session.exclusion_reference_object_ids(
            predicate.target.object_id
        )
        per_region_analysis: list[dict[str, Any]] = []
        unexplained_regions: list[DetectedRegion] = []
        unexplained_confidences: list[float] = []

        for region in regions:
            overlap_explanations: list[dict[str, Any]] = []
            matched_reference_regions: list[DetectedRegion] = []
            for reference_object_id in reference_object_ids:
                for reference_region in session.selected_regions(reference_object_id):
                    overlap_metrics = _region_overlap_metrics(
                        region,
                        reference_region,
                        image_size=session.image.size,
                    )
                    if (
                        overlap_metrics["intersection_over_min_area"]
                        >= self.duplicate_intersection_threshold
                        and overlap_metrics["area_similarity"]
                        >= self.duplicate_area_similarity_threshold
                        and overlap_metrics["iou"] >= self.duplicate_iou_threshold
                    ):
                        matched_reference_regions.append(reference_region)
                        overlap_explanations.append(
                            {
                                "reference_object_id": reference_object_id,
                                "reference_region": reference_region.evidence_dict(),
                                **overlap_metrics,
                            }
                        )

            raw_confidence = float(region.confidence or 0.0)
            residual_area_ratio = 1.0
            if matched_reference_regions:
                residual_area_ratio = _residual_area_ratio_after_overlap_explanations(
                    region,
                    matched_reference_regions,
                    image_size=session.image.size,
                )
                if residual_area_ratio <= self.residual_ignore_threshold:
                    effective_confidence = 0.0
                else:
                    effective_confidence = raw_confidence * residual_area_ratio
            else:
                effective_confidence = raw_confidence

            analysis = {
                "region": region.evidence_dict(),
                "raw_confidence": raw_confidence,
                "effective_confidence": effective_confidence,
                "residual_area_ratio": residual_area_ratio,
                "overlap_explanations": overlap_explanations,
            }
            per_region_analysis.append(analysis)
            if effective_confidence > 0.0:
                unexplained_regions.append(region)
                unexplained_confidences.append(effective_confidence)

        best_confidence = max(unexplained_confidences, default=0.0)
        if best_confidence < session.thresholds.proposal_uncertain:
            state = PredicateState.SATISFIED
        elif best_confidence < session.thresholds.proposal_satisfied:
            state = PredicateState.UNCERTAIN
        else:
            state = PredicateState.VIOLATED

        absence_score = 1.0 - best_confidence
        summary = (
            f"Absence check for {predicate.target.object_id} saw {len(regions)} raw candidate region(s); "
            f"{len(unexplained_regions)} remained after overlap filtering; "
            f"best unexplained confidence={best_confidence:.2f}."
        )
        return PredicateVerificationResult(
            predicate_id=predicate.predicate_id,
            category=predicate.category,
            state=state,
            score=absence_score,
            confidence=best_confidence if unexplained_regions else 1.0,
            evidence_summary=summary,
            evidence={
                "object_id": predicate.target.object_id,
                "reference_object_ids": list(reference_object_ids),
                "raw_candidate_count": len(regions),
                "candidate_count": len(unexplained_regions),
                "candidate_regions": [
                    region.evidence_dict() for region in unexplained_regions
                ],
                "per_region_analysis": per_region_analysis,
            },
        )


class TextPredicateEvaluator:
    """Evaluate visible text constraints using the configured VLM text verifier."""

    def evaluate(
        self,
        predicate: Predicate,
        session: VerificationSession,
    ) -> PredicateVerificationResult:
        assert isinstance(predicate, TextPredicate)
        text_tool = session.cache.tools.text_verification_tool
        if text_tool is None:
            return PredicateVerificationResult(
                predicate_id=predicate.predicate_id,
                category=predicate.category,
                state=PredicateState.UNCERTAIN,
                score=None,
                confidence=None,
                evidence_summary="No text verification tool is configured.",
                evidence={
                    "expected_text": predicate.expected_text,
                    "match_type": predicate.match_type.value,
                    "case_sensitive": predicate.case_sensitive,
                    "target_object_id": (
                        None if predicate.target is None else predicate.target.object_id
                    ),
                },
            )

        target_object_id = (
            None if predicate.target is None else predicate.target.object_id
        )
        target_label = None
        if target_object_id is not None:
            try:
                target_label = session.program.get_object(target_object_id).label
            except KeyError:
                target_label = target_object_id

        outcome = text_tool.verify_text(
            session.image,
            expected_text=predicate.expected_text,
            match_type=predicate.match_type.value,
            target_label=target_label,
            case_sensitive=predicate.case_sensitive,
        )
        if outcome.ok is True:
            state = PredicateState.SATISFIED
            score = 1.0
            summary_state = "satisfied"
        elif outcome.ok is False:
            state = PredicateState.VIOLATED
            score = 0.0
            summary_state = "violated"
        else:
            state = PredicateState.UNCERTAIN
            score = None
            summary_state = "uncertain"

        visible_text = outcome.visible_text.strip()
        target_phrase = f" on {target_label!r}" if target_label is not None else ""
        summary = (
            f"Text verifier {summary_state} {predicate.match_type.value} "
            f"text {predicate.expected_text!r}{target_phrase}."
        )
        if visible_text:
            summary += f" Read visible text: {visible_text!r}."
        if outcome.error:
            summary += f" Error: {outcome.error}."

        return PredicateVerificationResult(
            predicate_id=predicate.predicate_id,
            category=predicate.category,
            state=state,
            score=score,
            confidence=outcome.confidence,
            evidence_summary=summary,
            evidence={
                "expected_text": predicate.expected_text,
                "match_type": predicate.match_type.value,
                "case_sensitive": predicate.case_sensitive,
                "target_object_id": target_object_id,
                "target_label": target_label,
                "outcome": outcome.to_dict(),
            },
        )


class UnsupportedPredicateEvaluator:
    """Fallback evaluator for predicate types that are not implemented yet."""

    def evaluate(
        self,
        predicate: Predicate,
        session: VerificationSession,
    ) -> PredicateVerificationResult:
        return PredicateVerificationResult(
            predicate_id=predicate.predicate_id,
            category=predicate.category,
            state=PredicateState.UNCERTAIN,
            score=None,
            confidence=None,
            evidence_summary=f"No deterministic evaluator is implemented yet for {type(predicate).__name__}.",
            evidence={},
        )


@dataclass
class VisualProgramVerifier:
    """Evaluate all predicates in a fixed visual program against one image."""

    tools: PerceptionToolbox
    thresholds: VerifierThresholds = field(default_factory=VerifierThresholds)
    relation_scoring: RelationScoringConfig = field(
        default_factory=RelationScoringConfig
    )
    evaluators: Mapping[type[Predicate], PredicateEvaluator] = field(
        default_factory=dict
    )
    action_scoring: ActionScoringConfig = field(default_factory=ActionScoringConfig)

    def __post_init__(self) -> None:
        if not self.evaluators:
            self.evaluators = {
                CountPredicate: CountPredicateEvaluator(),
                AttributePredicate: AttributePredicateEvaluator(),
                SpatialRelationPredicate: SpatialRelationPredicateEvaluator(),
                GlobalScenePredicate: GlobalScenePredicateEvaluator(),
                ExclusionPredicate: ExclusionPredicateEvaluator(),
                TextPredicate: TextPredicateEvaluator(),
            }

    def verify(
        self,
        image: Image.Image,
        program: VisualProgram,
    ) -> VisualProgramVerificationReport:
        """Evaluate every predicate in the program using shared cached evidence."""
        cache = PerceptionCache(image=image, program=program, tools=self.tools)
        return self._verify_programmatic(image=image, program=program, cache=cache)

    def _verify_programmatic(
        self,
        *,
        image: Image.Image,
        program: VisualProgram,
        cache: PerceptionCache,
    ) -> VisualProgramVerificationReport:
        """Evaluate every predicate using the existing programmatic tool stack."""
        session = VerificationSession(
            image=image,
            program=program,
            cache=cache,
            thresholds=self.thresholds,
            relation_scoring=self.relation_scoring,
            action_scoring=self.action_scoring,
        )

        program_state = program.initial_state()
        results_by_id: dict[str, PredicateVerificationResult] = {}
        non_action_predicates: list[Predicate] = []
        action_predicates: list[Predicate] = []
        for predicate in program.predicates:
            if (
                isinstance(predicate, AttributePredicate)
                and _normalized_label(predicate.attribute_name) == "action"
            ):
                action_predicates.append(predicate)
            else:
                non_action_predicates.append(predicate)

        for predicate in non_action_predicates:
            evaluator = self._get_evaluator(predicate)
            result = evaluator.evaluate(predicate, session)
            results_by_id[predicate.predicate_id] = result

        requested_action_tool = session.cache.tools.action_verification_tool is not None
        non_action_blocking_predicate_ids = tuple(
            predicate.predicate_id
            for predicate in non_action_predicates
            if results_by_id[predicate.predicate_id].state != PredicateState.SATISFIED
        )
        session.action_vllm_requested = requested_action_tool
        session.action_vllm_gate_blocking_predicate_ids = (
            non_action_blocking_predicate_ids
        )
        session.action_vllm_enabled = bool(
            requested_action_tool and not non_action_blocking_predicate_ids
        )

        for predicate in action_predicates:
            evaluator = self._get_evaluator(predicate)
            result = evaluator.evaluate(predicate, session)
            results_by_id[predicate.predicate_id] = result

        results = tuple(
            results_by_id[predicate.predicate_id] for predicate in program.predicates
        )
        for result in results:
            program_state.update(
                result.predicate_id,
                result.state,
                rationale=result.evidence_summary,
                evidence=dict(result.evidence),
            )

        cache_summary = cache.summary()
        cache_summary["verifier_mode"] = "programmatic"
        cache_summary["action_verifier"] = {
            "mode": "late_only",
            "requested": requested_action_tool,
            "used_on_this_pass": bool(session.action_vllm_enabled),
            "late_gate_blocking_predicate_ids": list(
                session.action_vllm_gate_blocking_predicate_ids
            ),
            "action_predicate_ids": [
                predicate.predicate_id for predicate in action_predicates
            ],
        }

        return VisualProgramVerificationReport(
            program_id=program.program_id,
            results=results,
            program_state=program_state,
            cache_summary=cache_summary,
        )

    def _get_evaluator(self, predicate: Predicate) -> PredicateEvaluator:
        for predicate_type, evaluator in self.evaluators.items():
            if isinstance(predicate, predicate_type):
                return evaluator
        return UnsupportedPredicateEvaluator()


__all__ = [
    "ActionScoringConfig",
    "ObjectGroupEvidence",
    "RelationScoringConfig",
    "PredicateVerificationResult",
    "VerifierThresholds",
    "VisualProgramVerificationReport",
    "VisualProgramVerifier",
]
