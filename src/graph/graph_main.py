"""Run EPIC prompt parsing, verification, and image refinement."""

from __future__ import annotations

import argparse
import asyncio
import math
from pathlib import Path

from langgraph.graph import StateGraph
from PIL import Image
from tqdm import tqdm

from graph.cli import (
    DEFAULT_BASE_URL,
    add_epic_arguments,
    epic_kwargs,
    validate_epic_args,
)
from graph.context import Context
from graph.node_vp_pipeline import vp_pipeline
from graph.state import State
from graph.utils import data_url_to_image, save_result


builder = StateGraph(State, input_schema=State, context_schema=Context)
builder.add_node("vp_pipeline", vp_pipeline)
builder.add_edge("__start__", "vp_pipeline")
graph_main = builder.compile(name="EPIC Image Generation")


def build_graph_input_payload(
    *,
    prompt: str,
    image_seed: int,
    saving_files: bool,
    saving_path: str,
    max_rounds: int = 32,
    rewritten_prompt_count: int = 8,
    persistent_stuck_in_loop_auditor_streak: int = 2,
) -> dict:
    """Build the main-method state without alternate pipelines or fallback modes."""
    if not prompt.strip():
        raise ValueError("prompt must not be empty.")
    if image_seed < 0:
        raise ValueError("seed must be non-negative.")
    for name, value in (
        ("max_rounds", max_rounds),
        ("rewritten_prompt_count", rewritten_prompt_count),
        (
            "persistent_stuck_in_loop_auditor_streak",
            persistent_stuck_in_loop_auditor_streak,
        ),
    ):
        if value < 1:
            raise ValueError(f"{name} must be at least 1.")
    return {
        "original_prompt": prompt,
        "image_seed": image_seed,
        "saving_files": saving_files,
        "saving_path": saving_path,
        "max_rounds": max_rounds,
        "vp_rewritten_prompt_count": rewritten_prompt_count,
        "vp_persistent_stuck_in_loop_auditor_streak": persistent_stuck_in_loop_auditor_streak,
    }


def resolve_final_output_image_data_url(result: dict) -> str:
    images = result.get("output_images") or []
    if not images:
        raise RuntimeError("EPIC finished without returning a final output image.")
    return images[-1]


async def async_generate_with_prompt(
    prompt: str,
    seed: int = 42,
    save_idx: int = 0,
    saving_path: str = "outputs/default",
    saving_files: bool = True,
    llm_temperature: float = 0.0,
    max_rounds: int = 32,
    base_url: str = DEFAULT_BASE_URL,
    rewritten_prompt_count: int = 8,
    persistent_stuck_in_loop_auditor_streak: int = 2,
) -> Image.Image:
    """Generate and save one EPIC result; failures propagate to the caller."""
    if not math.isfinite(llm_temperature) or llm_temperature < 0:
        raise ValueError("llm_temperature must be finite and non-negative.")
    payload = build_graph_input_payload(
        prompt=prompt,
        image_seed=seed,
        saving_files=saving_files,
        saving_path=str(saving_path),
        max_rounds=max_rounds,
        rewritten_prompt_count=rewritten_prompt_count,
        persistent_stuck_in_loop_auditor_streak=persistent_stuck_in_loop_auditor_streak,
    )
    Path(saving_path).mkdir(parents=True, exist_ok=True)
    result = await graph_main.ainvoke(
        payload,
        context=Context(seed=seed, base_url=base_url, llm_temperature=llm_temperature),
        config={"recursion_limit": 1000},
    )
    output_image = data_url_to_image(
        resolve_final_output_image_data_url(result),
        timing_log=result.get("timing_log"),
        timing_component="graph_main:return_output_decode",
    )
    save_result(result, str(Path(saving_path) / f"{save_idx:06d}"))
    return output_image


async def test_on_generation_benchmark_txt(
    benchmark_txt_path: str,
    saving_path: str,
    seed: int = 42,
    **kwargs,
) -> None:
    """Run a prompt file sequentially, retaining completed result artifacts."""
    prompts = [
        line.strip()
        for line in Path(benchmark_txt_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for index, prompt in enumerate(tqdm(prompts, desc="Processing Prompts")):
        prefix = Path(saving_path) / f"{index:06d}_0"
        if (
            Path(f"{prefix}_result.json").is_file()
            and Path(f"{prefix}_output.jpg").is_file()
        ):
            continue
        await async_generate_with_prompt(
            prompt, seed=seed, save_idx=index, saving_path=saving_path, **kwargs
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--prompts", nargs="+", help="One or more prompts to generate.")
    inputs.add_argument(
        "--benchmark_txt_path", help="Text file containing one prompt per line."
    )
    parser.add_argument(
        "--saving_path",
        default="outputs/default",
        help="Final images, traces, and debug artifacts.",
    )
    add_epic_arguments(parser)
    args = parser.parse_args()
    validate_epic_args(args)
    if args.benchmark_txt_path:
        asyncio.run(
            test_on_generation_benchmark_txt(
                args.benchmark_txt_path,
                args.saving_path,
                seed=args.seed,
                **epic_kwargs(args),
            )
        )
    else:
        with asyncio.Runner() as runner:
            for index, prompt in enumerate(
                tqdm(args.prompts, desc="Processing Prompts")
            ):
                runner.run(
                    async_generate_with_prompt(
                        prompt,
                        seed=args.seed,
                        save_idx=index,
                        saving_path=args.saving_path,
                        **epic_kwargs(args),
                    )
                )


if __name__ == "__main__":
    main()
