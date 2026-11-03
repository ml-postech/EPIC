"""Shared command-line settings for the EPIC main method."""

from __future__ import annotations

import argparse
import math
import os


DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return result


def nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return result


def nonnegative_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return result


def add_epic_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the same main-method controls to every public entry point."""
    group = parser.add_argument_group("EPIC main method")
    group.add_argument(
        "--seed",
        type=nonnegative_int,
        default=42,
        help="Base image and VLM seed (default: 42).",
    )
    group.add_argument(
        "--base_url",
        default=os.environ.get("BASE_URL", DEFAULT_BASE_URL),
        help="OpenAI-compatible VLM endpoint (or BASE_URL).",
    )
    group.add_argument(
        "--llm_temperature",
        type=nonnegative_float,
        default=0.0,
        help="Parser, reviewer, rewriter, and auditor temperature (default: 0).",
    )
    group.add_argument(
        "--max_round",
        type=positive_int,
        default=32,
        help="Total generation/refinement rounds, including the initial image (default: 32).",
    )
    group.add_argument(
        "--rewritten_prompt_count",
        type=positive_int,
        default=8,
        help="Prompt rewrites generated per visual program (default: 8).",
    )
    group.add_argument(
        "--persistent_stuck_in_loop_auditor_streak",
        type=positive_int,
        default=2,
        help="Consecutive eligible blocked rounds before semantic auditing (default: 2).",
    )
    group.add_argument(
        "--llm_max_retries",
        type=nonnegative_int,
        default=os.environ.get("EPIC_LLM_MAX_RETRIES", "20"),
        help="Retries after an LLM timeout or parsing failure (default: 20; EPIC_LLM_MAX_RETRIES).",
    )
    group.add_argument(
        "--disable_saving_files",
        action="store_true",
        help="Skip detailed per-round debug artifacts; final images and result traces are still saved.",
    )


def validate_epic_args(opt: argparse.Namespace) -> None:
    """Validate programmatic namespaces too and configure LLM retries."""
    for name in (
        "max_round",
        "rewritten_prompt_count",
        "persistent_stuck_in_loop_auditor_streak",
    ):
        if getattr(opt, name) < 1:
            raise ValueError(f"--{name} must be at least 1.")
    if opt.seed < 0:
        raise ValueError("--seed must be non-negative.")
    if not math.isfinite(opt.llm_temperature) or opt.llm_temperature < 0:
        raise ValueError("--llm_temperature must be finite and non-negative.")
    if opt.llm_max_retries < 0:
        raise ValueError("--llm_max_retries must be non-negative.")
    if not str(opt.base_url).strip():
        raise ValueError("--base_url must not be empty.")
    os.environ["EPIC_LLM_MAX_RETRIES"] = str(opt.llm_max_retries)


def epic_kwargs(opt: argparse.Namespace) -> dict:
    """Map public options to the graph API, excluding per-sample identifiers."""
    return {
        "base_url": opt.base_url,
        "llm_temperature": opt.llm_temperature,
        "max_rounds": opt.max_round,
        "rewritten_prompt_count": opt.rewritten_prompt_count,
        "persistent_stuck_in_loop_auditor_streak": opt.persistent_stuck_in_loop_auditor_streak,
        "saving_files": not opt.disable_saving_files,
    }
