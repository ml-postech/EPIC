"""VLM connection and active prompt templates for the EPIC main method."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

from graph import prompts
from graph.cli import DEFAULT_BASE_URL


@dataclass(kw_only=True)
class Context:
    """Explicit arguments take precedence over typed environment defaults."""

    model: str = field(
        default_factory=lambda: os.environ.get(
            "MODEL", "openai/Qwen/Qwen3-VL-32B-Instruct"
        )
    )
    base_url: str = field(
        default_factory=lambda: os.environ.get("BASE_URL", DEFAULT_BASE_URL)
    )
    seed: int = field(default_factory=lambda: int(os.environ.get("SEED", "42")))
    llm_temperature: float = field(
        default_factory=lambda: float(os.environ.get("LLM_TEMPERATURE", "0"))
    )
    system_prompt_visual_program_parser: str = (
        prompts.SYSTEM_PROMPT_VISUAL_PROGRAM_PARSER
    )
    system_prompt_visual_program_reviewer: str = (
        prompts.SYSTEM_PROMPT_VISUAL_PROGRAM_REVIEWER
    )
    system_prompt_rewritten_prompt_generator: str = (
        prompts.SYSTEM_PROMPT_DETAILED_REWRITER
    )
    system_prompt_action_verifier: str = prompts.SYSTEM_PROMPT_ACTION_VERIFIER
    system_prompt_in_loop_auditor: str = prompts.SYSTEM_PROMPT_IN_LOOP_AUDITOR

    def __post_init__(self) -> None:
        if not self.model or "/" not in self.model:
            raise ValueError("MODEL must use provider/model-name format.")
        if not str(self.base_url).strip():
            raise ValueError("The VLM base_url must not be empty.")
        if self.seed < 0:
            raise ValueError("The VLM seed must be non-negative.")
        if not math.isfinite(self.llm_temperature) or self.llm_temperature < 0:
            raise ValueError("The VLM temperature must be finite and non-negative.")
