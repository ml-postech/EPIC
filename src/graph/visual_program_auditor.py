"""In-loop semantic review of predicates that stay stuck across rounds."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, Field

from graph.timing import ensure_timing_log, record_duration

if TYPE_CHECKING:
    from graph.context import Context
    from graph.state import State
    from langgraph.runtime import Runtime
else:
    Runtime = Any
    State = Any


class VisualCheckReviewOutput(BaseModel):
    """Structured VLM output for the in-loop visual check review."""

    all_checks_passed: bool = Field(
        ...,
        description="Whether every listed visual check is satisfied in the image.",
    )
    short_reason: str = Field(
        ...,
        description="A short explanation for the visual-check decision.",
    )
    check_reasoning: str = Field(
        ...,
        description=(
            "A concise reasoning trace that states whether each listed check passes "
            "or fails and why."
        ),
    )


@dataclass(frozen=True)
class InLoopAuditResult:
    """Lightweight in-loop override auditor result returned to the pipeline."""

    called: bool
    override_allowed: bool
    short_reason: str
    auditor_reasoning: str = ""
    skip_reason: str | None = None
    blocking_predicates: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result into plain Python types."""
        return {
            "called": self.called,
            "override_allowed": self.override_allowed,
            "short_reason": self.short_reason,
            "auditor_reasoning": self.auditor_reasoning,
            "skip_reason": self.skip_reason,
            "blocking_predicates": [
                _blocking for _blocking in self.blocking_predicates
            ],
        }


def skipped_in_loop_audit_result(
    reason: str,
    *,
    override_allowed: bool = False,
    blocking_predicates: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
) -> InLoopAuditResult:
    """Return a sentinel result when the in-loop auditor is skipped."""
    return InLoopAuditResult(
        called=False,
        override_allowed=override_allowed,
        short_reason=reason,
        auditor_reasoning=reason,
        skip_reason=reason,
        blocking_predicates=tuple(dict(item) for item in blocking_predicates),
    )


def _build_visual_checks_to_verify(
    blocking_predicates: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> list[dict[str, Any]]:
    """Convert internal predicate payloads into LLM-facing visual checks."""
    checks: list[dict[str, Any]] = []
    for raw_item in blocking_predicates:
        item = dict(raw_item)
        check: dict[str, Any] = {
            "check_id": item.get("predicate_id"),
            "check_type": item.get("predicate_kind", item.get("category")),
            "requirement": item.get("description"),
        }

        if item.get("predicate_kind") == "attribute":
            subject = {
                "object_id": item.get("object_id"),
                "label": item.get("object_label"),
            }
            check["subject"] = subject
            check["attribute"] = item.get("attribute_name")
            check["required_value"] = item.get("expected_value")
            if item.get("action_target_object_id") or item.get("action_target_label"):
                check["target"] = {
                    "object_id": item.get("action_target_object_id"),
                    "label": item.get("action_target_label"),
                }
        elif item.get("predicate_kind") == "spatial_relation":
            check["subject"] = {
                "object_id": item.get("subject_id"),
                "label": item.get("subject_label"),
            }
            check["relation"] = item.get("relation")
            check["target"] = {
                "object_id": item.get("reference_id"),
                "label": item.get("reference_label"),
            }
        elif item.get("predicate_kind") == "global_scene":
            check["scene_attribute"] = item.get("attribute_name")
            check["required_value"] = item.get("expected_value")

        if item.get("stuck_streak") is not None:
            check["stuck_rounds"] = item.get("stuck_streak")

        checks.append(check)
    return checks


def build_in_loop_auditor_input(
    *,
    original_prompt: str,
    image_data_url: str,
    system_prompt: str,
    blocking_predicates: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> list[Any]:
    """Build the in-loop visual check review model input."""
    checks_payload = json.dumps(
        _build_visual_checks_to_verify(blocking_predicates),
        ensure_ascii=False,
        indent=2,
    )
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Inspect this candidate image and decide whether every listed "
                        "visual check passes."
                    ),
                },
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        },
        {"role": "user", "content": f"user_prompt:\n{original_prompt}"},
        {"role": "user", "content": f"checks_to_verify:\n{checks_payload}"},
    ]


async def run_in_loop_auditor(
    *,
    state: State,
    runtime: Runtime[Context],
    original_prompt: str,
    candidate_image,
    blocking_predicates: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    component_name: str = "in_loop_auditor",
) -> InLoopAuditResult:
    """Run the in-loop semantic override auditor once on a stuck candidate image."""
    from graph.models import ainvoke_with_timeout_retry
    from graph.utils import image_to_data_url

    blocking_payload = tuple(dict(item) for item in blocking_predicates)
    node_start = time.perf_counter()
    state.timing_log = ensure_timing_log(state.timing_log)

    prompt_prep_start = time.perf_counter()
    image_data_url = image_to_data_url(
        candidate_image,
        timing_log=state.timing_log,
        timing_component="in_loop_auditor:image_encode",
    )
    auditor_input = build_in_loop_auditor_input(
        original_prompt=original_prompt,
        image_data_url=image_data_url,
        system_prompt=runtime.context.system_prompt_in_loop_auditor,
        blocking_predicates=blocking_payload,
    )
    record_duration(
        state.timing_log,
        "prompt_preparation_seconds",
        time.perf_counter() - prompt_prep_start,
        component=component_name,
        count_key="prompt_preparations",
    )

    model_output = cast(
        VisualCheckReviewOutput,
        await ainvoke_with_timeout_retry(
            state=state,
            runtime=runtime,
            inp=auditor_input,
            output_class=VisualCheckReviewOutput,
            timeout=180,
            component_name=component_name,
        ),
    )
    record_duration(
        state.timing_log,
        "vp_in_loop_auditor_seconds",
        time.perf_counter() - node_start,
        component=component_name,
        count_key="in_loop_audits",
    )
    return InLoopAuditResult(
        called=True,
        override_allowed=model_output.all_checks_passed,
        short_reason=model_output.short_reason,
        auditor_reasoning=model_output.check_reasoning,
        skip_reason=None,
        blocking_predicates=blocking_payload,
    )


__all__ = [
    "InLoopAuditResult",
    "VisualCheckReviewOutput",
    "build_in_loop_auditor_input",
    "run_in_loop_auditor",
    "skipped_in_loop_audit_result",
]
