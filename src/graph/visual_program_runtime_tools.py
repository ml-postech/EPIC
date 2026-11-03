"""Runtime perception tools for VP verification.

These tools are used inside the VP refinement loop. Most are deterministic
perception components such as SAM3, SigLIP, and MiDaS; the optional text
verifier calls the already-running VLM server instead of loading local OCR
weights.

All heavyweight imports are kept lazy so the module remains importable in test
environments that do not have every vision dependency installed.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import io
from importlib import resources
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Optional, Sequence
import urllib.error
import urllib.request

import numpy as np
import torch
from PIL import Image, ImageDraw

from graph.timing import ensure_timing_log, record_duration
from graph.visual_program import ObjectSpec
from graph.visual_program_tools import (
    ActionVerificationOutcome,
    BoundingBox,
    DepthEstimate,
    DepthEstimationTool,
    DetectedRegion,
    PerceptionToolbox,
    RegionProposalTool,
    RegionTextScoringTool,
    TextVerificationOutcome,
)


DEFAULT_SIGLIP_MODEL_ID = "google/siglip2-base-patch16-224"
DEFAULT_SAM3_MODEL_ID = "facebook/sam3"
SAM3_BPE_FILENAME = "bpe_simple_vocab_16e6.txt.gz"

_SIGLIP_MODEL = None
_SIGLIP_PROCESSOR = None
_SIGLIP_MODEL_ID = None
_SAM3_MODEL = None
_SAM3_PROCESSOR = None
_MIDAS_ESTIMATOR = None
_DEBUG_COLORS = (
    (239, 83, 80),
    (66, 165, 245),
    (102, 187, 106),
    (255, 202, 40),
    (171, 71, 188),
    (255, 112, 67),
)


def _prepare_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _enable_tf32(device: str) -> None:
    if device != "cuda":
        return
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def _normalize_sam3_text(object_spec: ObjectSpec) -> str:
    candidates = [object_spec.proposal_text, object_spec.label, *object_spec.aliases]
    for candidate in candidates:
        if candidate is None:
            continue
        phrase = re.sub(r"[_-]+", " ", str(candidate).strip()).strip().lower()
        if phrase:
            return phrase
    fallback = re.sub(r"[_-]+", " ", str(object_spec.object_id).strip()).strip().lower()
    return fallback or "object"


def _bbox_crop(image: Image.Image, bbox: BoundingBox) -> Image.Image:
    left = int(max(0, np.floor(bbox.x_min)))
    upper = int(max(0, np.floor(bbox.y_min)))
    right = int(min(image.width, np.ceil(bbox.x_max)))
    lower = int(min(image.height, np.ceil(bbox.y_max)))
    if right <= left or lower <= upper:
        return image.crop((0, 0, max(1, image.width), max(1, image.height)))
    return image.crop((left, upper, right, lower))


def _square_mask_crop(
    image: Image.Image,
    region: DetectedRegion,
    *,
    pad: int = 20,
) -> Image.Image:
    """Crop one region for SigLIP, using mask bounds but preserving local context."""
    rgb_image = image.convert("RGB")
    if region.mask is None:
        return _bbox_crop(rgb_image, region.bbox)

    mask = np.asarray(region.mask, dtype=bool)
    if mask.shape != (rgb_image.height, rgb_image.width) or not mask.any():
        return _bbox_crop(rgb_image, region.bbox)

    coords = np.argwhere(mask)
    y_min = int(max(0, coords[:, 0].min() - pad))
    x_min = int(max(0, coords[:, 1].min() - pad))
    y_max = int(min(rgb_image.height, coords[:, 0].max() + pad))
    x_max = int(min(rgb_image.width, coords[:, 1].max() + pad))
    if x_max <= x_min or y_max <= y_min:
        return _bbox_crop(rgb_image, region.bbox)

    # A fully cut-out mask often removes the surface/background cues that help
    # SigLIP recognize attributes such as color or material. Keep the original
    # pixels inside the padded mask bounds instead of zeroing everything else.
    cropped = rgb_image.crop((x_min, y_min, x_max, y_max))
    width, height = cropped.size
    square_size = max(width, height)
    square = Image.new("RGB", (square_size, square_size), (0, 0, 0))
    square.paste(cropped, ((square_size - width) // 2, (square_size - height) // 2))
    return square


def _siglip_templates(text: str) -> tuple[str, ...]:
    """Return a small prompt ensemble for one SigLIP text query."""
    normalized = str(text).strip()
    if not normalized:
        return ("object",)
    return (
        f"a photo of {normalized}",
        f"a close-up photo of {normalized}",
        normalized,
    )


def _sanitize_filename_component(value: str, *, fallback: str = "artifact") -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("._")
    return normalized[:80] or fallback


def _normalize_openai_base_url(base_url: str) -> str:
    normalized = str(base_url).strip()
    if "://" not in normalized:
        normalized = f"http://{normalized}"
    if not normalized.rstrip("/").endswith("/v1"):
        normalized = f"{normalized.rstrip('/')}/v1"
    return normalized.rstrip("/")


def _openai_model_name(model: str) -> str:
    normalized = str(model).strip()
    for provider_prefix in ("openai/",):
        if normalized.startswith(provider_prefix):
            return normalized[len(provider_prefix) :]
    return normalized


def _image_to_png_data_url(image: Image.Image, *, max_resolution: int) -> str:
    rgb_image = image.convert("RGB")
    if max_resolution > 0:
        scale = min(
            float(max_resolution) / max(1, rgb_image.width),
            float(max_resolution) / max(1, rgb_image.height),
            1.0,
        )
        if scale < 1.0:
            rgb_image = rgb_image.resize(
                (
                    max(1, int(round(rgb_image.width * scale))),
                    max(1, int(round(rgb_image.height * scale))),
                ),
                Image.LANCZOS,
            )
    buffer = io.BytesIO()
    rgb_image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _extract_json_object(text: str) -> dict[str, Any]:
    raw_text = str(text).strip()
    try:
        parsed = json.loads(raw_text)
        return parsed if isinstance(parsed, dict) else {"raw": parsed}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if match is None:
        return {"parse_error": "no_json_object", "raw_text": raw_text}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {"raw": parsed}
    except json.JSONDecodeError as exc:
        return {"parse_error": str(exc), "raw_text": raw_text}


def _parse_optional_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"true", "yes", "pass", "passed", "match", "matched"}:
        return True
    if normalized in {"false", "no", "fail", "failed", "mismatch"}:
        return False
    return None


def _parse_confidence(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


def _vllm_no_thinking_kwargs() -> dict[str, Any]:
    return {"chat_template_kwargs": {"enable_thinking": False}}


def _chat_completion_message(response_payload: dict[str, Any]) -> dict[str, Any]:
    message = response_payload["choices"][0]["message"]
    return message if isinstance(message, dict) else {"content": message}


def _chat_completion_content_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts = [
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return "".join(text_parts).strip()
    return ""


def _mask_intersection_ratio(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    left = np.asarray(mask_a, dtype=bool)
    right = np.asarray(mask_b, dtype=bool)
    if left.shape != right.shape:
        return 0.0
    left_area = int(left.sum())
    right_area = int(right.sum())
    if left_area <= 0 or right_area <= 0:
        return 0.0
    intersection = int(np.logical_and(left, right).sum())
    return float(intersection / max(1, min(left_area, right_area)))


def _bbox_intersection_ratio(left: BoundingBox, right: BoundingBox) -> float:
    intersection = left.intersection_area(right)
    if intersection <= 0:
        return 0.0
    min_area = min(left.area, right.area)
    if min_area <= 0:
        return 0.0
    return float(intersection / min_area)


def region_intersection_ratio(left: DetectedRegion, right: DetectedRegion) -> float:
    """Return a duplicate-style overlap ratio for two regions.

    Masks take priority when both are available, because they are typically a
    better estimate of whether two detections refer to the same object instance.
    Otherwise the score falls back to bbox intersection over the smaller area.
    """

    if left.mask is not None and right.mask is not None:
        return _mask_intersection_ratio(left.mask, right.mask)
    return _bbox_intersection_ratio(left.bbox, right.bbox)


def _should_merge_regions_for_dedup(
    left: DetectedRegion,
    right: DetectedRegion,
    *,
    intersection_threshold: float,
) -> bool:
    return region_intersection_ratio(left, right) >= intersection_threshold


def _merge_region_masks(regions: Sequence[DetectedRegion]) -> Optional[np.ndarray]:
    masks = [
        np.asarray(region.mask, dtype=bool)
        for region in regions
        if region.mask is not None
    ]
    if not masks:
        return None
    reference_shape = masks[0].shape
    if any(mask.shape != reference_shape for mask in masks):
        return masks[0]
    merged = np.zeros(reference_shape, dtype=bool)
    for mask in masks:
        merged |= mask
    return merged


def _region_sort_key(region: DetectedRegion) -> tuple[float, float, str]:
    return (
        float(region.confidence or 0.0),
        float(region.area),
        region.region_id,
    )


def _merge_region_cluster(regions: Sequence[DetectedRegion]) -> DetectedRegion:
    anchor = max(regions, key=_region_sort_key)
    merged_ids = tuple(region.region_id for region in regions)
    metadata = dict(anchor.metadata)
    metadata["dedup_merged_region_ids"] = list(merged_ids)
    metadata["dedup_source_count"] = len(merged_ids)
    metadata["dedup_intersection_strategy"] = (
        "mask_intersection_over_min_area_when_masks_exist_else_same_query_bbox_fallback"
    )
    return DetectedRegion(
        region_id=anchor.region_id,
        bbox=BoundingBox.union([region.bbox for region in regions]),
        label=anchor.label,
        confidence=max(
            (
                float(region.confidence)
                for region in regions
                if region.confidence is not None
            ),
            default=anchor.confidence,
        ),
        average_depth=None,
        mask=_merge_region_masks(regions),
        attributes=dict(anchor.attributes),
        metadata=metadata,
    )


def deduplicate_regions_by_intersection(
    regions: Sequence[DetectedRegion],
    *,
    intersection_threshold: float = 0.85,
) -> tuple[DetectedRegion, ...]:
    """Merge overlapping detections, preferring masks and falling back to boxes."""

    if not regions:
        return ()

    sorted_regions = sorted(regions, key=_region_sort_key, reverse=True)
    parents = list(range(len(sorted_regions)))

    def _find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def _union(left_index: int, right_index: int) -> None:
        left_root = _find(left_index)
        right_root = _find(right_index)
        if left_root == right_root:
            return
        # Keep the earlier sorted region as the component root so ties remain
        # deterministic and aligned with the confidence ordering.
        if left_root > right_root:
            left_root, right_root = right_root, left_root
        parents[right_root] = left_root

    # Dedup should be transitive: if A matches bridge B and bridge B matches C,
    # all three detections belong to one component even when A and C do not
    # overlap directly.
    for left_index, left_region in enumerate(sorted_regions):
        for right_index in range(left_index + 1, len(sorted_regions)):
            if _should_merge_regions_for_dedup(
                left_region,
                sorted_regions[right_index],
                intersection_threshold=intersection_threshold,
            ):
                _union(left_index, right_index)

    clusters_by_root: dict[int, list[DetectedRegion]] = {}
    for index, region in enumerate(sorted_regions):
        root = _find(index)
        clusters_by_root.setdefault(root, []).append(region)

    merged_regions = [
        _merge_region_cluster(cluster)
        for _, cluster in sorted(clusters_by_root.items())
    ]
    return tuple(sorted(merged_regions, key=_region_sort_key, reverse=True))


def _save_debug_json(
    save_path: str,
    payload,
    *,
    timing_log=None,
    timing_component: str,
    timing_metadata=None,
) -> None:
    from graph.utils import ensure_directory, json_dumps_timed, write_text_file

    ensure_directory(
        os.path.dirname(save_path),
        timing_log=timing_log,
        timing_component=f"{timing_component}:mkdir",
        timing_metadata=timing_metadata,
    )
    serialized = json_dumps_timed(
        payload,
        indent=2,
        ensure_ascii=False,
        timing_log=timing_log,
        timing_component=f"{timing_component}:serialize",
        timing_metadata=timing_metadata,
    )
    write_text_file(
        save_path,
        serialized,
        timing_log=timing_log,
        timing_component=f"{timing_component}:write",
        timing_metadata=timing_metadata,
    )


def _save_debug_image(
    save_path: str,
    image: Image.Image,
    *,
    timing_log=None,
    timing_component: str,
    timing_metadata=None,
) -> None:
    from graph.utils import ensure_directory, save_image_file

    ensure_directory(
        os.path.dirname(save_path),
        timing_log=timing_log,
        timing_component=f"{timing_component}:mkdir",
        timing_metadata=timing_metadata,
    )
    save_image_file(
        image,
        save_path,
        timing_log=timing_log,
        timing_component=f"{timing_component}:write",
        timing_metadata=timing_metadata,
    )


def _render_region_overlay(
    image: Image.Image,
    regions: Sequence[DetectedRegion],
) -> Image.Image:
    canvas = image.convert("RGBA").copy()

    for index, region in enumerate(regions):
        color = _DEBUG_COLORS[index % len(_DEBUG_COLORS)]
        if region.mask is not None:
            mask = np.asarray(region.mask, dtype=bool)
            if mask.shape == (canvas.height, canvas.width) and mask.any():
                alpha = np.zeros((canvas.height, canvas.width), dtype=np.uint8)
                alpha[mask] = 80
                tint = Image.new("RGBA", canvas.size, color + (0,))
                tint.putalpha(Image.fromarray(alpha, mode="L"))
                canvas = Image.alpha_composite(canvas, tint)

        draw = ImageDraw.Draw(canvas)
        draw.rectangle(
            (
                int(region.bbox.x_min),
                int(region.bbox.y_min),
                int(region.bbox.x_max),
                int(region.bbox.y_max),
            ),
            outline=color + (255,),
            width=3,
        )
        confidence = (
            "?" if region.confidence is None else f"{float(region.confidence):.2f}"
        )
        label = f"{index + 1}:{region.region_id} ({confidence})"
        draw.text(
            (int(region.bbox.x_min) + 4, max(0, int(region.bbox.y_min) - 14)),
            label,
            fill=color + (255,),
        )

    return canvas.convert("RGB")


def _mask_preview(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L")


def _normalized_map_preview(map_array: np.ndarray) -> Image.Image:
    finite_values = np.asarray(map_array, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return Image.fromarray(np.zeros_like(map_array, dtype=np.uint8), mode="L")

    lower = float(np.percentile(finite_values, 2.0))
    upper = float(np.percentile(finite_values, 98.0))
    if upper <= lower:
        upper = lower + 1.0
    normalized = np.clip(
        (np.asarray(map_array, dtype=float) - lower) / (upper - lower), 0.0, 1.0
    )
    return Image.fromarray((normalized * 255).astype(np.uint8), mode="L")


def _ensure_sam3_repo_on_path(repo_root: Optional[str]) -> None:
    if not repo_root:
        return
    normalized_repo_root = str(Path(repo_root).expanduser().resolve())
    if not (Path(normalized_repo_root) / "sam3").is_dir():
        raise FileNotFoundError(
            "EPIC_SAM3_REPO_ROOT must point to a SAM3 checkout containing sam3/: "
            f"{normalized_repo_root}"
        )
    package_root = Path(normalized_repo_root) / "sam3"
    for name, module in tuple(sys.modules.items()):
        if name != "sam3" and not name.startswith("sam3."):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file and not Path(module_file).resolve().is_relative_to(package_root):
            raise RuntimeError(
                "SAM3 is already imported from a different location. Set "
                "EPIC_SAM3_REPO_ROOT before importing SAM3 and restart the process: "
                f"{module_file}"
            )
    while normalized_repo_root in sys.path:
        sys.path.remove(normalized_repo_root)
    sys.path.insert(0, normalized_repo_root)


def _resolve_sam3_bpe_path(
    *, repo_root: Optional[str], explicit_bpe_path: Optional[str]
) -> Optional[str]:
    """Find SAM3's tokenizer asset in an override, checkout, or installed package."""
    if explicit_bpe_path:
        path = Path(explicit_bpe_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"SAM3 tokenizer asset not found: {path}")
        return str(path)
    if repo_root:
        path = Path(repo_root).expanduser() / "sam3" / "assets" / SAM3_BPE_FILENAME
        if path.is_file():
            return str(path.resolve())
        raise FileNotFoundError(
            f"SAM3 tokenizer asset not found in the selected checkout: {path}. "
            "Set EPIC_SAM3_BPE_PATH to its tokenizer asset."
        )
    asset = resources.files("sam3").joinpath("assets", SAM3_BPE_FILENAME)
    if asset.is_file():
        return str(asset)
    raise FileNotFoundError(
        "SAM3 tokenizer asset is missing. Install SAM3 from source or set "
        "EPIC_SAM3_BPE_PATH to bpe_simple_vocab_16e6.txt.gz."
    )


def _load_siglip(
    *,
    model_id: str,
    timing_log=None,
):
    global _SIGLIP_MODEL, _SIGLIP_PROCESSOR, _SIGLIP_MODEL_ID
    if (
        _SIGLIP_MODEL is not None
        and _SIGLIP_PROCESSOR is not None
        and _SIGLIP_MODEL_ID == model_id
    ):
        return _SIGLIP_MODEL, _SIGLIP_PROCESSOR

    if _SIGLIP_MODEL is not None and _SIGLIP_MODEL_ID != model_id:
        try:
            _SIGLIP_MODEL.to("cpu")
        except Exception:
            pass
        _SIGLIP_MODEL = None
        _SIGLIP_PROCESSOR = None
        _SIGLIP_MODEL_ID = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    from transformers import SiglipModel, SiglipProcessor

    device = _prepare_device()
    _enable_tf32(device)
    load_start = time.perf_counter()
    _SIGLIP_PROCESSOR = SiglipProcessor.from_pretrained(model_id)
    _SIGLIP_MODEL = SiglipModel.from_pretrained(model_id).to(device)
    _SIGLIP_MODEL.eval()
    _SIGLIP_MODEL_ID = model_id
    record_duration(
        timing_log,
        "model_loading_seconds",
        time.perf_counter() - load_start,
        component=f"SigLIP:{model_id}",
        count_key="model_loads",
    )
    return _SIGLIP_MODEL, _SIGLIP_PROCESSOR


def _resolve_sam3_checkpoint_path(
    *,
    model_id: str,
    explicit_checkpoint_path: Optional[str],
) -> Optional[str]:
    if explicit_checkpoint_path:
        path = Path(explicit_checkpoint_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"SAM3 checkpoint not found: {path}")
        return str(path)

    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(
            repo_id=model_id,
            filename="sam3.pt",
            local_files_only=True,
        )
    except Exception:
        return hf_hub_download(repo_id=model_id, filename="sam3.pt")


def _load_sam3_processor(
    *,
    repo_root: Optional[str],
    model_id: str,
    confidence_threshold: float,
    checkpoint_path: Optional[str],
    bpe_path: Optional[str],
    timing_log=None,
):
    global _SAM3_MODEL, _SAM3_PROCESSOR
    if repo_root:
        _ensure_sam3_repo_on_path(repo_root)
    if _SAM3_MODEL is not None and _SAM3_PROCESSOR is not None:
        if (
            abs(
                float(_SAM3_PROCESSOR.confidence_threshold)
                - float(confidence_threshold)
            )
            > 1e-8
        ):
            _SAM3_PROCESSOR.confidence_threshold = confidence_threshold
        return _SAM3_MODEL, _SAM3_PROCESSOR

    try:
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
    except Exception as exc:
        raise RuntimeError(
            "SAM3 could not be imported. Install the SAM3 runtime dependencies "
            "or set EPIC_SAM3_REPO_ROOT to a SAM3 checkout."
        ) from exc

    device = _prepare_device()
    _enable_tf32(device)

    resolved_bpe_path = _resolve_sam3_bpe_path(
        repo_root=repo_root, explicit_bpe_path=bpe_path
    )
    resolved_checkpoint_path = _resolve_sam3_checkpoint_path(
        model_id=model_id,
        explicit_checkpoint_path=checkpoint_path,
    )

    load_start = time.perf_counter()
    _SAM3_MODEL = build_sam3_image_model(
        bpe_path=resolved_bpe_path,
        device=device,
        checkpoint_path=resolved_checkpoint_path,
        load_from_HF=resolved_checkpoint_path is None,
        enable_segmentation=True,
        enable_inst_interactivity=False,
    )
    _SAM3_PROCESSOR = Sam3Processor(
        model=_SAM3_MODEL,
        device=device,
        confidence_threshold=confidence_threshold,
    )
    record_duration(
        timing_log,
        "model_loading_seconds",
        time.perf_counter() - load_start,
        component=f"SAM3:{model_id}",
        count_key="model_loads",
    )
    return _SAM3_MODEL, _SAM3_PROCESSOR


def _load_midas_estimator(timing_log=None):
    global _MIDAS_ESTIMATOR
    if _MIDAS_ESTIMATOR is not None:
        return _MIDAS_ESTIMATOR

    from graph.depth import MidasDepthEstimator

    device = _prepare_device()
    load_start = time.perf_counter()
    _MIDAS_ESTIMATOR = MidasDepthEstimator(device=device)
    record_duration(
        timing_log,
        "model_loading_seconds",
        time.perf_counter() - load_start,
        component="MiDaS-DPT_Large",
        count_key="model_loads",
    )
    return _MIDAS_ESTIMATOR


@dataclass
class SAM3TextRegionProposalTool(RegionProposalTool):
    """Propose object regions with SAM3 text prompting.

    The tool caches image features for the current image so multiple predicate
    evaluations can reuse the same SAM3 image encoding across existence, count,
    relation, and attribute checks.
    """

    timing_log: Optional[dict] = None
    sam3_model_id: str = field(
        default_factory=lambda: os.environ.get(
            "EPIC_SAM3_MODEL_ID", DEFAULT_SAM3_MODEL_ID
        )
    )
    sam3_repo_root: Optional[str] = field(
        default_factory=lambda: os.environ.get("EPIC_SAM3_REPO_ROOT")
    )
    sam3_checkpoint_path: Optional[str] = field(
        default_factory=lambda: os.environ.get("EPIC_SAM3_CHECKPOINT_PATH")
    )
    sam3_bpe_path: Optional[str] = field(
        default_factory=lambda: os.environ.get("EPIC_SAM3_BPE_PATH")
    )
    confidence_threshold: float = 0.30
    dedup_intersection_threshold: float = 0.85
    debug_artifact_dir: Optional[str] = None
    _cached_image_obj: Optional[Image.Image] = field(default=None, init=False)
    _cached_image_state: Optional[dict] = field(default=None, init=False)

    def propose_regions(
        self,
        image: Image.Image,
        object_spec: ObjectSpec,
    ) -> Sequence[DetectedRegion]:
        timing_log = (
            ensure_timing_log(self.timing_log) if self.timing_log is not None else None
        )
        _, sam3_processor = _load_sam3_processor(
            repo_root=self.sam3_repo_root,
            model_id=self.sam3_model_id,
            confidence_threshold=self.confidence_threshold,
            checkpoint_path=self.sam3_checkpoint_path,
            bpe_path=self.sam3_bpe_path,
            timing_log=timing_log,
        )
        text_query = _normalize_sam3_text(object_spec)

        device = _prepare_device()
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"
            ),
        ):
            # Keep a direct reference to the current PIL image object instead of
            # keying by id(image). Python can reuse object ids across loop
            # iterations, which risks reusing stale SAM3 state for a different
            # image that happens to share the same dimensions.
            if self._cached_image_obj is not image or self._cached_image_state is None:
                image_setup_start = time.perf_counter()
                self._cached_image_state = sam3_processor.set_image(
                    image.convert("RGB")
                )
                self._cached_image_obj = image
                record_duration(
                    timing_log,
                    "grounding_seconds",
                    time.perf_counter() - image_setup_start,
                    component="SAM3:image_encoding",
                    count_key="grounding_calls",
                    metadata={"object_id": object_spec.object_id, "query": text_query},
                )

            inference_state = self._cached_image_state
            sam3_processor.reset_all_prompts(inference_state)
            detection_start = time.perf_counter()
            inference_state = sam3_processor.set_text_prompt(
                prompt=text_query, state=inference_state
            )
            record_duration(
                timing_log,
                "grounding_seconds",
                time.perf_counter() - detection_start,
                component="SAM3:region_proposal",
                count_key="grounding_calls",
                metadata={"object_id": object_spec.object_id, "query": text_query},
            )

        boxes_tensor = inference_state.get("boxes")
        scores_tensor = inference_state.get("scores")
        masks_tensor = inference_state.get("masks")

        boxes = (
            boxes_tensor.detach().float().cpu().numpy()
            if boxes_tensor is not None and len(boxes_tensor)
            else np.zeros((0, 4), dtype=float)
        )
        scores = (
            scores_tensor.detach().float().cpu().numpy().tolist()
            if scores_tensor is not None and len(scores_tensor)
            else []
        )
        if masks_tensor is not None and len(masks_tensor):
            masks = masks_tensor.detach().cpu().numpy()
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks[:, 0]
        else:
            masks = None

        raw_regions = []
        for index, box in enumerate(boxes.tolist()):
            mask = None if masks is None else np.asarray(masks[index], dtype=bool)
            raw_regions.append(
                DetectedRegion(
                    region_id=f"{object_spec.object_id}:{index}",
                    bbox=BoundingBox(*[float(value) for value in box]),
                    label=object_spec.label,
                    confidence=float(scores[index]) if index < len(scores) else None,
                    mask=mask,
                    metadata={"query": text_query, "region_source": "sam3"},
                )
            )
        deduped_regions = deduplicate_regions_by_intersection(
            raw_regions,
            intersection_threshold=self.dedup_intersection_threshold,
        )
        self._save_region_debug_artifacts(
            image=image,
            object_spec=object_spec,
            text_query=text_query,
            raw_regions=tuple(raw_regions),
            deduped_regions=deduped_regions,
        )
        return deduped_regions

    def _save_region_debug_artifacts(
        self,
        *,
        image: Image.Image,
        object_spec: ObjectSpec,
        text_query: str,
        raw_regions: tuple[DetectedRegion, ...],
        deduped_regions: tuple[DetectedRegion, ...],
    ) -> None:
        if not self.debug_artifact_dir:
            return

        object_dir = os.path.join(
            self.debug_artifact_dir,
            "regions",
            _sanitize_filename_component(object_spec.object_id, fallback="object"),
        )
        timing_metadata = {"object_id": object_spec.object_id, "query": text_query}
        self._save_region_set_artifacts(
            root_dir=os.path.join(object_dir, "raw"),
            image=image,
            object_id=object_spec.object_id,
            text_query=text_query,
            regions=raw_regions,
            timing_metadata={**timing_metadata, "stage": "raw_regions"},
            extra_payload={"deduped_candidate_count": len(deduped_regions)},
        )
        self._save_region_set_artifacts(
            root_dir=os.path.join(object_dir, "deduped"),
            image=image,
            object_id=object_spec.object_id,
            text_query=text_query,
            regions=deduped_regions,
            timing_metadata={**timing_metadata, "stage": "deduped_regions"},
            extra_payload={"raw_candidate_count": len(raw_regions)},
        )

    def _save_region_set_artifacts(
        self,
        *,
        root_dir: str,
        image: Image.Image,
        object_id: str,
        text_query: str,
        regions: Sequence[DetectedRegion],
        timing_metadata: dict[str, object],
        extra_payload: dict[str, object],
    ) -> None:
        overlay = _render_region_overlay(image, regions)
        _save_debug_image(
            os.path.join(root_dir, "overlay.jpg"),
            overlay,
            timing_log=self.timing_log,
            timing_component="vp_debug:regions_overlay",
            timing_metadata=timing_metadata,
        )
        _save_debug_json(
            os.path.join(root_dir, "summary.json"),
            {
                "object_id": object_id,
                "query": text_query,
                "region_count": len(regions),
                **extra_payload,
                "regions": [region.evidence_dict() for region in regions],
            },
            timing_log=self.timing_log,
            timing_component="vp_debug:regions_summary",
            timing_metadata=timing_metadata,
        )

        for index, region in enumerate(regions, start=1):
            basename = (
                f"region_{index:02d}_{_sanitize_filename_component(region.region_id)}"
            )
            _save_debug_image(
                os.path.join(root_dir, f"{basename}_crop.png"),
                _bbox_crop(image.convert("RGB"), region.bbox),
                timing_log=self.timing_log,
                timing_component="vp_debug:regions_crop",
                timing_metadata={**timing_metadata, "region_id": region.region_id},
            )
            if region.mask is not None:
                _save_debug_image(
                    os.path.join(root_dir, f"{basename}_mask.png"),
                    _mask_preview(region.mask),
                    timing_log=self.timing_log,
                    timing_component="vp_debug:regions_mask",
                    timing_metadata={**timing_metadata, "region_id": region.region_id},
                )


@dataclass
class SiglipRegionTextScoringRuntimeTool(RegionTextScoringTool):
    """Score region-text compatibility with SigLIP."""

    timing_log: Optional[dict] = None
    model_id: str = DEFAULT_SIGLIP_MODEL_ID
    debug_artifact_dir: Optional[str] = None
    _debug_call_index: int = field(default=0, init=False)

    def score_region_text(
        self,
        image: Image.Image,
        region: DetectedRegion,
        text: str,
    ) -> Optional[float]:
        timing_log = (
            ensure_timing_log(self.timing_log) if self.timing_log is not None else None
        )
        model, processor = _load_siglip(model_id=self.model_id, timing_log=timing_log)
        crop = _square_mask_crop(image, region)
        templates = _siglip_templates(text)
        device = model.device

        score_start = time.perf_counter()
        inputs = processor(
            images=[crop],
            text=list(templates),
            return_tensors="pt",
            padding="max_length",
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits_per_image[0]
        score = float(torch.sigmoid(logits).max().item())
        record_duration(
            timing_log,
            "region_text_scoring_seconds",
            time.perf_counter() - score_start,
            component="SigLIP:region_text",
            count_key="region_text_scores",
            metadata={"region_id": region.region_id, "text": text},
        )
        self._save_siglip_debug_artifact(
            crop=crop,
            region=region,
            text=text,
            score=score,
        )
        return score

    def _save_siglip_debug_artifact(
        self,
        *,
        crop: Image.Image,
        region: DetectedRegion,
        text: str,
        score: float,
    ) -> None:
        if not self.debug_artifact_dir:
            return

        self._debug_call_index += 1
        basename = (
            f"{self._debug_call_index:04d}_"
            f"{_sanitize_filename_component(region.region_id, fallback='region')}_"
            f"{_sanitize_filename_component(text, fallback='query')}"
        )
        timing_metadata = {"region_id": region.region_id, "text": text}
        _save_debug_image(
            os.path.join(self.debug_artifact_dir, "siglip", f"{basename}.png"),
            crop,
            timing_log=self.timing_log,
            timing_component="vp_debug:siglip_crop",
            timing_metadata=timing_metadata,
        )
        _save_debug_json(
            os.path.join(self.debug_artifact_dir, "siglip", f"{basename}.json"),
            {
                "region_id": region.region_id,
                "text": text,
                "score": score,
                "bbox": region.bbox.to_dict(),
                "confidence": region.confidence,
                "metadata": dict(region.metadata),
            },
            timing_log=self.timing_log,
            timing_component="vp_debug:siglip_summary",
            timing_metadata=timing_metadata,
        )


@dataclass
class MidasDepthEstimationRuntimeTool(DepthEstimationTool):
    """Estimate depth with MiDaS."""

    timing_log: Optional[dict] = None
    debug_artifact_dir: Optional[str] = None

    def estimate_depth(self, image: Image.Image) -> DepthEstimate:
        timing_log = (
            ensure_timing_log(self.timing_log) if self.timing_log is not None else None
        )
        estimator = _load_midas_estimator(timing_log=timing_log)
        depth_start = time.perf_counter()
        depth_image = estimator(image)
        depth_map = np.asarray(depth_image, dtype=float)
        record_duration(
            timing_log,
            "depth_estimation_seconds",
            time.perf_counter() - depth_start,
            component="MiDaS-DPT_Large",
            count_key="depth_calls",
        )
        self._save_depth_debug_artifacts(depth_map)
        # The runtime consumes the viewable 8-bit MiDaS map returned by
        # `MidasDepthEstimator`, where smaller values correspond to regions that
        # are closer to the camera.
        return DepthEstimate(depth_map=depth_map, nearer_is_smaller=True)

    def _save_depth_debug_artifacts(self, depth_map: np.ndarray) -> None:
        if not self.debug_artifact_dir:
            return

        finite_values = depth_map[np.isfinite(depth_map)]
        summary = {
            "shape": list(depth_map.shape),
            "nearer_is_smaller": True,
            "min": None if finite_values.size == 0 else float(np.min(finite_values)),
            "max": None if finite_values.size == 0 else float(np.max(finite_values)),
            "mean": None if finite_values.size == 0 else float(np.mean(finite_values)),
        }
        _save_debug_image(
            os.path.join(self.debug_artifact_dir, "depth", "depth_preview.png"),
            _normalized_map_preview(depth_map),
            timing_log=self.timing_log,
            timing_component="vp_debug:depth_preview",
            timing_metadata={"stage": "depth"},
        )
        _save_debug_json(
            os.path.join(self.debug_artifact_dir, "depth", "depth_summary.json"),
            summary,
            timing_log=self.timing_log,
            timing_component="vp_debug:depth_summary",
            timing_metadata={"stage": "depth"},
        )


@dataclass
class VLLMTextVerificationRuntimeTool:
    """Visible-text verifier backed by the configured OpenAI-compatible VLM."""

    base_url: str = "http://127.0.0.1:8000"
    model: str = "Qwen/Qwen3-VL-32B-Instruct"
    temperature: float = 0.0
    seed: int = 42
    timeout_seconds: int = 180
    max_resolution: int = 1024
    timing_log: Optional[dict] = None

    system_prompt: str = (
        "Verify visible text in images. Return JSON only: "
        '{"ok":true|false|null,"text":"...","conf":0-1}.'
    )

    def verify_text(
        self,
        image: Image.Image,
        *,
        expected_text: str,
        match_type: str,
        target_label: Optional[str] = None,
        case_sensitive: bool = False,
    ) -> TextVerificationOutcome:
        timing_log = None
        if self.timing_log is not None:
            timing_log = ensure_timing_log(self.timing_log)
            if timing_log is not self.timing_log:
                self.timing_log.update(timing_log)
                timing_log = self.timing_log

        encode_start = time.perf_counter()
        image_data_url = _image_to_png_data_url(
            image,
            max_resolution=int(self.max_resolution),
        )
        if timing_log is not None:
            record_duration(
                timing_log,
                "image_conversion_seconds",
                time.perf_counter() - encode_start,
                component="vp_text_verifier:image_encode",
                count_key="image_conversions",
            )

        user_text = (
            f"target: {expected_text!r}\n"
            f"match: {match_type}\n"
            f"case_sensitive: {bool(case_sensitive)}\n"
            f"where: {target_label or 'image'}\n"
            "Inspect pixels only. ok=true iff the visible readable text satisfies "
            "the target. ok=false for a clear mismatch. ok=null if unreadable "
            "or ambiguous."
        )
        payload = {
            "model": _openai_model_name(self.model),
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                },
            ],
            "temperature": float(self.temperature),
            "seed": int(self.seed),
            "max_tokens": 96,
            **_vllm_no_thinking_kwargs(),
        }
        request = urllib.request.Request(
            f"{_normalize_openai_base_url(self.base_url)}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + os.environ.get("OPENAI_API_KEY", "EMPTY"),
            },
            method="POST",
        )

        request_start = time.perf_counter()
        try:
            with urllib.request.urlopen(
                request,
                timeout=int(self.timeout_seconds),
            ) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
            request_duration = time.perf_counter() - request_start
            if timing_log is not None:
                record_duration(
                    timing_log,
                    "vp_text_verification_seconds",
                    request_duration,
                    component="vp_text_verifier",
                    count_key="text_verifier_requests",
                )
            message = _chat_completion_message(response_payload)
            raw_text = _chat_completion_content_text(message)
            parsed = _extract_json_object(raw_text)
            ok = _parse_optional_bool(parsed.get("ok"))
            confidence = _parse_confidence(parsed.get("conf"))
            visible_text = str(parsed.get("text", "")).strip()
            error = None
            if parsed.get("parse_error"):
                error = str(parsed["parse_error"])
                ok = None
            return TextVerificationOutcome(
                ok=ok,
                visible_text=visible_text,
                confidence=confidence,
                raw_response={
                    "parsed": parsed,
                    "raw_text": raw_text,
                    "model": payload["model"],
                    "message": message,
                },
                error=error,
            )
        except (OSError, KeyError, json.JSONDecodeError, urllib.error.HTTPError) as exc:
            if timing_log is not None:
                record_duration(
                    timing_log,
                    "vp_text_verification_seconds",
                    time.perf_counter() - request_start,
                    component="vp_text_verifier:error",
                    count_key="text_verifier_requests",
                )
            return TextVerificationOutcome(
                ok=None,
                visible_text="",
                confidence=None,
                raw_response={"model": payload["model"]},
                error=str(exc),
            )


@dataclass
class VLLMActionVerificationRuntimeTool:
    """Action verifier backed by the configured OpenAI-compatible VLM."""

    base_url: str = "http://127.0.0.1:8000"
    model: str = "Qwen/Qwen3-VL-32B-Instruct"
    temperature: float = 0.0
    seed: int = 42
    timeout_seconds: int = 180
    max_resolution: int = 1024
    timing_log: Optional[dict] = None
    debug_artifact_dir: Optional[str] = None

    system_prompt: str = (
        "Verify action faithfulness in image crops. Return JSON only: "
        '{"ok":true|false|null,"reason":"...","conf":0-1}.'
    )

    def verify_action(
        self,
        image: Image.Image,
        *,
        subject_label: str,
        action_text: str,
        target_label: Optional[str] = None,
        artifact_name: Optional[str] = None,
    ) -> ActionVerificationOutcome:
        timing_log = None
        if self.timing_log is not None:
            timing_log = ensure_timing_log(self.timing_log)
            if timing_log is not self.timing_log:
                self.timing_log.update(timing_log)
                timing_log = self.timing_log

        crop = image.convert("RGB")
        encode_start = time.perf_counter()
        image_data_url = _image_to_png_data_url(
            crop,
            max_resolution=int(self.max_resolution),
        )
        if timing_log is not None:
            record_duration(
                timing_log,
                "image_conversion_seconds",
                time.perf_counter() - encode_start,
                component="vp_action_verifier:image_encode",
                count_key="image_conversions",
            )

        target_line = (
            "target: none\n"
            if target_label is None or not str(target_label).strip()
            else f"target: {str(target_label).strip()!r}\n"
        )
        user_text = (
            f"subject: {str(subject_label).strip()!r}\n"
            f"action: {str(action_text).strip()!r}\n"
            f"{target_line}"
            "Inspect pixels only. ok=true only if the named subject clearly performs "
            "the named action. If a target is given, the action must be directed "
            "toward or involve the named target in the correct roles. "
            "Mere co-presence, similar motion, or an ambiguous pose is not enough. "
            "ok=false for a clear mismatch or reversed roles. ok=null if unclear."
        )
        payload = {
            "model": _openai_model_name(self.model),
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                },
            ],
            "temperature": float(self.temperature),
            "seed": int(self.seed),
            "max_tokens": 128,
            **_vllm_no_thinking_kwargs(),
        }
        request = urllib.request.Request(
            f"{_normalize_openai_base_url(self.base_url)}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + os.environ.get("OPENAI_API_KEY", "EMPTY"),
            },
            method="POST",
        )

        request_start = time.perf_counter()
        try:
            with urllib.request.urlopen(
                request,
                timeout=int(self.timeout_seconds),
            ) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
            request_duration = time.perf_counter() - request_start
            if timing_log is not None:
                record_duration(
                    timing_log,
                    "vp_action_verification_seconds",
                    request_duration,
                    component="vp_action_verifier",
                    count_key="action_verifier_requests",
                )
            message = _chat_completion_message(response_payload)
            raw_text = _chat_completion_content_text(message)
            parsed = _extract_json_object(raw_text)
            ok = _parse_optional_bool(parsed.get("ok"))
            confidence = _parse_confidence(parsed.get("conf"))
            reason = str(parsed.get("reason", "")).strip()
            error = None
            if parsed.get("parse_error"):
                error = str(parsed["parse_error"])
                ok = None
            outcome = ActionVerificationOutcome(
                ok=ok,
                confidence=confidence,
                reason=reason,
                raw_response={
                    "parsed": parsed,
                    "raw_text": raw_text,
                    "model": payload["model"],
                    "message": message,
                    "subject_label": str(subject_label).strip(),
                    "action_text": str(action_text).strip(),
                    "target_label": (
                        None if target_label is None else str(target_label).strip()
                    ),
                },
                error=error,
            )
            self._save_action_debug_artifact(
                crop=crop,
                outcome=outcome,
                artifact_name=artifact_name,
                timing_log=timing_log,
            )
            return outcome
        except (OSError, KeyError, json.JSONDecodeError, urllib.error.HTTPError) as exc:
            if timing_log is not None:
                record_duration(
                    timing_log,
                    "vp_action_verification_seconds",
                    time.perf_counter() - request_start,
                    component="vp_action_verifier:error",
                    count_key="action_verifier_requests",
                )
            outcome = ActionVerificationOutcome(
                ok=None,
                confidence=None,
                reason="",
                raw_response={"model": payload["model"]},
                error=str(exc),
            )
            self._save_action_debug_artifact(
                crop=crop,
                outcome=outcome,
                artifact_name=artifact_name,
                timing_log=timing_log,
            )
            return outcome

    def _save_action_debug_artifact(
        self,
        *,
        crop: Image.Image,
        outcome: ActionVerificationOutcome,
        artifact_name: Optional[str],
        timing_log=None,
    ) -> None:
        if not self.debug_artifact_dir:
            return
        basename = _sanitize_filename_component(
            artifact_name or "action_verifier",
            fallback="action_verifier",
        )
        root_dir = os.path.join(self.debug_artifact_dir, "action_verifier")
        _save_debug_image(
            os.path.join(root_dir, f"{basename}_crop.png"),
            crop,
            timing_log=timing_log,
            timing_component="vp_debug:action_verifier_crop",
            timing_metadata={"stage": "action_verifier"},
        )
        _save_debug_json(
            os.path.join(root_dir, f"{basename}.json"),
            outcome.to_dict(),
            timing_log=timing_log,
            timing_component="vp_debug:action_verifier_summary",
            timing_metadata={"stage": "action_verifier"},
        )


def get_visual_program_perception_toolbox(
    *,
    timing_log=None,
    siglip_model_id: Optional[str] = None,
    sam3_confidence_threshold: float = 0.30,
    text_verifier_base_url: Optional[str] = None,
    text_verifier_model: Optional[str] = None,
    text_verifier_temperature: float = 0.0,
    text_verifier_seed: int = 42,
    text_verifier_timeout_seconds: int = 180,
    text_verifier_max_resolution: int = 1024,
    action_verifier_base_url: Optional[str] = None,
    action_verifier_model: Optional[str] = None,
    action_verifier_temperature: float = 0.0,
    action_verifier_seed: int = 42,
    action_verifier_timeout_seconds: int = 180,
    action_verifier_max_resolution: int = 1024,
    action_verifier_system_prompt: Optional[str] = None,
    debug_artifact_dir: Optional[str] = None,
) -> PerceptionToolbox:
    """Build the main EPIC toolbox; model weights load lazily when needed."""
    return PerceptionToolbox(
        region_proposal_tool=SAM3TextRegionProposalTool(
            timing_log=timing_log,
            confidence_threshold=float(sam3_confidence_threshold),
            debug_artifact_dir=debug_artifact_dir,
        ),
        region_text_scoring_tool=SiglipRegionTextScoringRuntimeTool(
            timing_log=timing_log,
            model_id=str(siglip_model_id or DEFAULT_SIGLIP_MODEL_ID),
            debug_artifact_dir=debug_artifact_dir,
        ),
        depth_estimation_tool=MidasDepthEstimationRuntimeTool(
            timing_log=timing_log,
            debug_artifact_dir=debug_artifact_dir,
        ),
        text_verification_tool=VLLMTextVerificationRuntimeTool(
            base_url=str(text_verifier_base_url or "http://127.0.0.1:8000"),
            model=str(text_verifier_model or "Qwen/Qwen3-VL-32B-Instruct"),
            temperature=float(text_verifier_temperature),
            seed=int(text_verifier_seed),
            timeout_seconds=int(text_verifier_timeout_seconds),
            max_resolution=int(text_verifier_max_resolution),
            timing_log=timing_log,
        ),
        action_verification_tool=VLLMActionVerificationRuntimeTool(
            base_url=str(action_verifier_base_url or "http://127.0.0.1:8000"),
            model=str(action_verifier_model or "Qwen/Qwen3-VL-32B-Instruct"),
            temperature=float(action_verifier_temperature),
            seed=int(action_verifier_seed),
            timeout_seconds=int(action_verifier_timeout_seconds),
            max_resolution=int(action_verifier_max_resolution),
            timing_log=timing_log,
            debug_artifact_dir=debug_artifact_dir,
            system_prompt=str(action_verifier_system_prompt or "").strip()
            or (
                "Verify action faithfulness in image crops. Return JSON only: "
                '{"ok":true|false|null,"reason":"...","conf":0-1}.'
            ),
        ),
    )


__all__ = [
    "SAM3TextRegionProposalTool",
    "MidasDepthEstimationRuntimeTool",
    "SiglipRegionTextScoringRuntimeTool",
    "VLLMTextVerificationRuntimeTool",
    "VLLMActionVerificationRuntimeTool",
    "deduplicate_regions_by_intersection",
    "get_visual_program_perception_toolbox",
    "region_intersection_ratio",
]
