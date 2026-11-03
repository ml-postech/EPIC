"""Generate detailed rewritten prompts for VP pipeline generations."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, Field

from graph.prompts import SYSTEM_PROMPT_DETAILED_REWRITER
from graph.timing import ensure_timing_log, record_duration

if TYPE_CHECKING:
    from graph.context import Context
    from graph.state import State
    from langgraph.runtime import Runtime
else:
    Runtime = Any
    State = Any


class RewrittenPromptGeneratorOutput(BaseModel):
    """Structured output for the detailed rewritten prompt generator."""

    rewritten_prompts: list[str] = Field(
        default_factory=list,
        description="Detailed rewritten prompts that preserve the original meaning.",
    )


@dataclass(frozen=True)
class RewrittenPromptGenerationResult:
    """Persisted rewritten prompt generation result returned to the pipeline."""

    called: bool
    component_name: str
    source_prompt: str
    requested_count: int
    rewritten_prompts: tuple[str, ...]
    raw_output: RewrittenPromptGeneratorOutput | None = None
    normalized_output: RewrittenPromptGeneratorOutput | None = None
    skip_reason: str | None = None
    fallback_used: bool = False
    error_type: str | None = None
    error_message: str | None = None

    def to_report(self) -> dict[str, Any]:
        """Serialize the result into plain Python types."""
        return {
            "called": self.called,
            "component_name": self.component_name,
            "source_prompt": self.source_prompt,
            "requested_count": self.requested_count,
            "generated_count": len(self.rewritten_prompts),
            "rewritten_prompts": list(self.rewritten_prompts),
            "skip_reason": self.skip_reason,
            "fallback_used": self.fallback_used,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


def _normalize_rewritten_prompts(
    source_prompt: str,
    rewritten_prompts: list[str] | tuple[str, ...] | None,
) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    normalized_source = " ".join(str(source_prompt).strip().split()).casefold()
    for rewrite in rewritten_prompts or ():
        cleaned = " ".join(str(rewrite).strip().split())
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key == normalized_source or key in seen:
            continue
        normalized.append(cleaned)
        seen.add(key)
    return normalized


def build_rewritten_prompt_generator_input(
    prompt: str,
    *,
    requested_count: int,
    system_prompt: str = SYSTEM_PROMPT_DETAILED_REWRITER,
) -> list[dict[str, str]]:
    """Build the chat input payload for detailed rewrite generation."""
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Original prompt:\n"
                f"{prompt}\n\n"
                f"Return exactly {requested_count} rewritten prompts as JSON with "
                "one key named rewritten_prompts."
            ),
        },
    ]


def failed_rewritten_prompt_generation_result(
    prompt: str,
    *,
    requested_count: int,
    component_name: str,
    exc: Exception,
) -> RewrittenPromptGenerationResult:
    """Return a fallback result when rewrite generation fails."""
    return RewrittenPromptGenerationResult(
        called=True,
        component_name=component_name,
        source_prompt=prompt,
        requested_count=int(requested_count),
        rewritten_prompts=(),
        raw_output=None,
        normalized_output=RewrittenPromptGeneratorOutput(rewritten_prompts=[]),
        skip_reason="rewritten_prompt_generator_failed",
        fallback_used=True,
        error_type=type(exc).__name__,
        error_message=str(exc),
    )


async def generate_rewritten_prompts(
    prompt: str,
    *,
    state: State,
    runtime: Runtime[Context],
    requested_count: int = 8,
    timeout: int = 180,
    component_name: str = "rewritten_prompt_generator",
) -> RewrittenPromptGenerationResult:
    """Generate detailed rewritten prompts from the original prompt only."""
    state.timing_log = ensure_timing_log(state.timing_log)
    from graph.models import ainvoke_with_timeout_retry

    node_start = time.perf_counter()
    generator_input = build_rewritten_prompt_generator_input(
        prompt,
        requested_count=int(requested_count),
        system_prompt=runtime.context.system_prompt_rewritten_prompt_generator,
    )
    raw_output = cast(
        RewrittenPromptGeneratorOutput,
        await ainvoke_with_timeout_retry(
            state=state,
            runtime=runtime,
            inp=generator_input,
            output_class=RewrittenPromptGeneratorOutput,
            timeout=timeout,
            component_name=component_name,
        ),
    )
    normalized_output = RewrittenPromptGeneratorOutput(
        rewritten_prompts=_normalize_rewritten_prompts(
            prompt,
            raw_output.rewritten_prompts,
        )
    )
    record_duration(
        state.timing_log,
        "vp_rewritten_prompt_generator_seconds",
        time.perf_counter() - node_start,
        component=component_name,
        count_key="rewritten_prompt_generations",
        metadata={
            "requested_count": int(requested_count),
            "generated_count": len(normalized_output.rewritten_prompts),
        },
    )
    return RewrittenPromptGenerationResult(
        called=True,
        component_name=component_name,
        source_prompt=prompt,
        requested_count=int(requested_count),
        rewritten_prompts=tuple(normalized_output.rewritten_prompts),
        raw_output=raw_output,
        normalized_output=normalized_output,
        skip_reason=None,
        fallback_used=False,
        error_type=None,
        error_message=None,
    )


__all__ = [
    "RewrittenPromptGenerationResult",
    "RewrittenPromptGeneratorOutput",
    "build_rewritten_prompt_generator_input",
    "failed_rewritten_prompt_generation_result",
    "generate_rewritten_prompts",
]
