"""Tool abstractions and reusable perception caches for visual-program verification.

This module is intentionally model-agnostic. It defines the typed interfaces that
the visual-program verifier can depend on without knowing whether the concrete
implementation uses SAM, SigLIP, MiDaS, OCR, or precomputed fixtures.

The key design goal is reuse: object proposals, depth maps, and region-text
scores are cached so multiple predicates can evaluate against the same evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable

import numpy as np
from PIL import Image

from graph.visual_program import ObjectSpec, VisualProgram


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            item.name: _serialize(getattr(value, item.name))
            for item in fields(value)
            if item.name != "mask"
        }
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned box in xyxy format."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        if self.x_max < self.x_min or self.y_max < self.y_min:
            raise ValueError(
                "BoundingBox coordinates must satisfy x_max >= x_min and y_max >= y_min."
            )

    @property
    def width(self) -> float:
        return max(0.0, self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return max(0.0, self.y_max - self.y_min)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0)

    def intersection_area(self, other: BoundingBox) -> float:
        inter_x_min = max(self.x_min, other.x_min)
        inter_y_min = max(self.y_min, other.y_min)
        inter_x_max = min(self.x_max, other.x_max)
        inter_y_max = min(self.y_max, other.y_max)
        inter_width = max(0.0, inter_x_max - inter_x_min)
        inter_height = max(0.0, inter_y_max - inter_y_min)
        return inter_width * inter_height

    def contains(self, other: BoundingBox) -> bool:
        return (
            self.x_min <= other.x_min
            and self.y_min <= other.y_min
            and self.x_max >= other.x_max
            and self.y_max >= other.y_max
        )

    @classmethod
    def union(cls, boxes: Sequence[BoundingBox]) -> BoundingBox:
        """Build a minimal enclosing box for a non-empty list of boxes."""
        if not boxes:
            raise ValueError("boxes must be non-empty.")
        return cls(
            x_min=min(box.x_min for box in boxes),
            y_min=min(box.y_min for box in boxes),
            x_max=max(box.x_max for box in boxes),
            y_max=max(box.y_max for box in boxes),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the box into plain Python types."""
        return _serialize(self)


@dataclass(frozen=True)
class DetectedRegion:
    """One proposed image region that may correspond to an object instance."""

    region_id: str
    bbox: BoundingBox
    label: Optional[str] = None
    confidence: Optional[float] = None
    average_depth: Optional[float] = None
    mask: Optional[np.ndarray] = None
    attributes: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.region_id.strip():
            raise ValueError("region_id must be non-empty.")
        if self.confidence is not None:
            object.__setattr__(self, "confidence", _clamp01(self.confidence))

    @property
    def center(self) -> tuple[float, float]:
        return self.bbox.center

    @property
    def area(self) -> float:
        if self.mask is not None:
            return float(np.asarray(self.mask, dtype=bool).sum())
        return self.bbox.area

    def evidence_dict(self) -> dict[str, Any]:
        """Serialize a compact evidence view, omitting the raw mask."""
        return {
            "region_id": self.region_id,
            "label": self.label,
            "confidence": self.confidence,
            "average_depth": self.average_depth,
            "bbox": self.bbox.to_dict(),
            "attributes": dict(self.attributes),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class DepthEstimate:
    """Depth map output from a deterministic estimator such as MiDaS."""

    depth_map: np.ndarray
    nearer_is_smaller: bool = False

    def average_depth(self, region: DetectedRegion) -> Optional[float]:
        """Compute the average depth for a region mask or bbox crop."""
        depth = np.asarray(self.depth_map, dtype=float)
        if depth.ndim != 2:
            raise ValueError("depth_map must have shape (H, W).")

        if region.mask is not None:
            mask = np.asarray(region.mask, dtype=bool)
            if mask.shape != depth.shape:
                raise ValueError("Region mask shape must match depth map shape.")
            if not mask.any():
                return None
            return float(depth[mask].mean())

        x_min = int(max(0, np.floor(region.bbox.x_min)))
        y_min = int(max(0, np.floor(region.bbox.y_min)))
        x_max = int(min(depth.shape[1], np.ceil(region.bbox.x_max)))
        y_max = int(min(depth.shape[0], np.ceil(region.bbox.y_max)))
        if x_max <= x_min or y_max <= y_min:
            return None
        return float(depth[y_min:y_max, x_min:x_max].mean())


@runtime_checkable
class RegionProposalTool(Protocol):
    """Provide candidate regions for a declared object."""

    def propose_regions(
        self,
        image: Image.Image,
        object_spec: ObjectSpec,
    ) -> Sequence[DetectedRegion]:
        """Return candidate regions for the target object."""


@runtime_checkable
class RegionTextScoringTool(Protocol):
    """Score whether a text description applies to a proposed region."""

    def score_region_text(
        self,
        image: Image.Image,
        region: DetectedRegion,
        text: str,
    ) -> Optional[float]:
        """Return a score in [0, 1] when available."""


@dataclass(frozen=True)
class TextVerificationOutcome:
    """Result from a visible-text verifier."""

    ok: Optional[bool]
    visible_text: str = ""
    confidence: Optional[float] = None
    raw_response: Mapping[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the outcome into plain Python types."""
        return _serialize(self)


@dataclass(frozen=True)
class ActionVerificationOutcome:
    """Result from an action verifier."""

    ok: Optional[bool]
    confidence: Optional[float] = None
    reason: str = ""
    raw_response: Mapping[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the outcome into plain Python types."""
        return _serialize(self)


@runtime_checkable
class TextVerificationTool(Protocol):
    """Verify literal visible text in an image."""

    def verify_text(
        self,
        image: Image.Image,
        *,
        expected_text: str,
        match_type: str,
        target_label: Optional[str] = None,
        case_sensitive: bool = False,
    ) -> TextVerificationOutcome:
        """Return whether the image visibly satisfies the requested text match."""


@runtime_checkable
class ActionVerificationTool(Protocol):
    """Verify whether an image crop clearly shows one requested action."""

    def verify_action(
        self,
        image: Image.Image,
        *,
        subject_label: str,
        action_text: str,
        target_label: Optional[str] = None,
        artifact_name: Optional[str] = None,
    ) -> ActionVerificationOutcome:
        """Return whether the crop visibly satisfies the requested action."""


@runtime_checkable
class DepthEstimationTool(Protocol):
    """Estimate a dense depth map for the input image."""

    def estimate_depth(self, image: Image.Image) -> DepthEstimate:
        """Return a depth estimate for the image."""


@dataclass
class PerceptionToolbox:
    """Bundle the concrete tools used by the predicate verifier."""

    region_proposal_tool: RegionProposalTool
    region_text_scoring_tool: Optional[RegionTextScoringTool] = None
    depth_estimation_tool: Optional[DepthEstimationTool] = None
    text_verification_tool: Optional[TextVerificationTool] = None
    action_verification_tool: Optional[ActionVerificationTool] = None


@dataclass
class PerceptionCache:
    """Cache reusable tool outputs during one verification pass."""

    image: Image.Image
    program: VisualProgram
    tools: PerceptionToolbox
    _object_regions: dict[str, tuple[DetectedRegion, ...]] = field(default_factory=dict)
    _text_scores: dict[tuple[str, str], Optional[float]] = field(default_factory=dict)
    _depth_estimate: Optional[DepthEstimate] = None
    region_cache_hits: int = 0
    region_cache_misses: int = 0
    text_cache_hits: int = 0
    text_cache_misses: int = 0
    depth_cache_hits: int = 0
    depth_cache_misses: int = 0

    def get_object_regions(self, object_id: str) -> tuple[DetectedRegion, ...]:
        """Fetch and cache candidate regions for an object."""
        if object_id in self._object_regions:
            self.region_cache_hits += 1
            return self._object_regions[object_id]

        object_spec = self.program.get_object(object_id)
        regions = tuple(
            self.tools.region_proposal_tool.propose_regions(self.image, object_spec)
        )
        self._object_regions[object_id] = regions
        self.region_cache_misses += 1
        return regions

    def score_region_text(self, region: DetectedRegion, text: str) -> Optional[float]:
        """Fetch and cache region-text scores when a scoring tool is available."""
        key = (region.region_id, text.strip().lower())
        if key in self._text_scores:
            self.text_cache_hits += 1
            return self._text_scores[key]

        if self.tools.region_text_scoring_tool is None:
            self.text_cache_misses += 1
            self._text_scores[key] = None
            return None

        score = self.tools.region_text_scoring_tool.score_region_text(
            self.image,
            region,
            text,
        )
        self._text_scores[key] = None if score is None else _clamp01(score)
        self.text_cache_misses += 1
        return self._text_scores[key]

    def get_depth_estimate(self) -> Optional[DepthEstimate]:
        """Fetch and cache a depth estimate when a tool is available."""
        if self._depth_estimate is not None:
            self.depth_cache_hits += 1
            return self._depth_estimate

        if self.tools.depth_estimation_tool is None:
            self.depth_cache_misses += 1
            return None

        self._depth_estimate = self.tools.depth_estimation_tool.estimate_depth(
            self.image
        )
        self.depth_cache_misses += 1
        return self._depth_estimate

    def get_region_average_depth(self, region: DetectedRegion) -> Optional[float]:
        """Return a region depth, reusing region metadata when available."""
        if region.average_depth is not None:
            return float(region.average_depth)

        depth_estimate = self.get_depth_estimate()
        if depth_estimate is None:
            return None
        return depth_estimate.average_depth(region)

    def summary(self) -> dict[str, Any]:
        """Expose cache statistics for debugging and tests."""
        return {
            "region_cache_hits": self.region_cache_hits,
            "region_cache_misses": self.region_cache_misses,
            "text_cache_hits": self.text_cache_hits,
            "text_cache_misses": self.text_cache_misses,
            "depth_cache_hits": self.depth_cache_hits,
            "depth_cache_misses": self.depth_cache_misses,
            "cached_object_ids": sorted(self._object_regions),
            "cached_text_queries": len(self._text_scores),
            "depth_cached": self._depth_estimate is not None,
        }


@dataclass
class PrecomputedRegionProposalTool:
    """Region proposal tool for tests or precomputed detector outputs."""

    regions_by_object_id: Mapping[str, Sequence[DetectedRegion]]
    calls: list[str] = field(default_factory=list)

    def propose_regions(
        self,
        image: Image.Image,
        object_spec: ObjectSpec,
    ) -> Sequence[DetectedRegion]:
        self.calls.append(object_spec.object_id)
        return tuple(
            self.regions_by_object_id.get(
                object_spec.object_id,
                self.regions_by_object_id.get(
                    object_spec.proposal_text or object_spec.label,
                    self.regions_by_object_id.get(object_spec.label, ()),
                ),
            )
        )


@dataclass
class PrecomputedRegionTextScoringTool:
    """Region-text scoring tool for tests or cached SigLIP outputs."""

    scores: Mapping[tuple[str, str], float]
    calls: list[tuple[str, str]] = field(default_factory=list)

    def score_region_text(
        self,
        image: Image.Image,
        region: DetectedRegion,
        text: str,
    ) -> Optional[float]:
        normalized_key = (region.region_id, text.strip().lower())
        self.calls.append(normalized_key)
        if normalized_key not in self.scores:
            return None
        return _clamp01(self.scores[normalized_key])


@dataclass
class StaticDepthEstimationTool:
    """Depth tool backed by one precomputed depth map."""

    depth_estimate: DepthEstimate
    calls: int = 0

    def estimate_depth(self, image: Image.Image) -> DepthEstimate:
        self.calls += 1
        return self.depth_estimate


__all__ = [
    "ActionVerificationOutcome",
    "ActionVerificationTool",
    "BoundingBox",
    "DepthEstimate",
    "DepthEstimationTool",
    "DetectedRegion",
    "PerceptionCache",
    "PerceptionToolbox",
    "PrecomputedRegionProposalTool",
    "PrecomputedRegionTextScoringTool",
    "RegionProposalTool",
    "RegionTextScoringTool",
    "StaticDepthEstimationTool",
    "TextVerificationOutcome",
    "TextVerificationTool",
]
