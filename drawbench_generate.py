"""Batch runner for DrawBench-style prompt TXT files using EPIC."""

import argparse
import asyncio
import json
import os
import sys
import traceback
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from einops import rearrange
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from tqdm import tqdm, trange

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from graph.cli import add_epic_arguments, epic_kwargs, validate_epic_args
from graph.randomness import seed_everything

torch.set_grad_enabled(False)


def parse_args():
    parser = argparse.ArgumentParser()
    add_epic_arguments(parser)
    parser.add_argument(
        "--benchmark_txt_path",
        type=str,
        default="benchmarks/drawbench.txt",
        help="TXT file containing one DrawBench prompt per line.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        default="outputs/drawbench_results/",
        help="Directory to write prompt folders and agent results to.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="Number of samples to generate for each prompt.",
    )
    parser.add_argument(
        "--skip_grid", action="store_true", help="Skip saving a grid image per prompt."
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Inclusive prompt index to start from.",
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=None,
        help="Exclusive prompt index to stop at. Omit to run through the end.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=1,
        help="How many times to retry a sample when generation fails.",
    )
    return parser.parse_args()


def load_main_graph_runner():
    try:
        from graph.graph_main import async_generate_with_prompt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Failed to import graph.graph_main. Make sure the main graph runtime dependencies are installed in the active environment."
        ) from exc
    return async_generate_with_prompt


def load_prompts(benchmark_txt_path: str) -> list[str]:
    with open(benchmark_txt_path, encoding="utf-8") as fp:
        return [line.strip() for line in fp if line.strip()]


def get_prompt_dir(outdir: str, prompt_index: int) -> str:
    return os.path.join(outdir, f"{prompt_index:0>5}")


def get_sample_path(outdir: str, prompt_index: int, sample_index: int) -> str:
    return os.path.join(
        get_prompt_dir(outdir, prompt_index), "samples", f"{sample_index:05}.png"
    )


def get_grid_path(outdir: str, prompt_index: int) -> str:
    return os.path.join(get_prompt_dir(outdir, prompt_index), "grid.png")


def get_metadata_path(outdir: str, prompt_index: int) -> str:
    return os.path.join(get_prompt_dir(outdir, prompt_index), "metadata.jsonl")


def get_agent_result_path(outdir: str, sample_seed: int, prompt_index: int) -> str:
    return os.path.join(
        outdir,
        "agent_results",
        f"seed_{sample_seed}",
        f"{prompt_index:06d}_0_result.json",
    )


def prompt_already_processed(opt, prompt_index: int) -> bool:
    agent_results_exist = all(
        (
            os.path.exists(
                get_agent_result_path(opt.outdir, opt.seed + sample_index, prompt_index)
            )
            for sample_index in range(opt.n_samples)
        )
    )
    samples_exist = all(
        (
            os.path.exists(get_sample_path(opt.outdir, prompt_index, sample_index))
            for sample_index in range(opt.n_samples)
        )
    )
    metadata_exists = os.path.exists(get_metadata_path(opt.outdir, prompt_index))
    grid_exists = opt.skip_grid or os.path.exists(
        get_grid_path(opt.outdir, prompt_index)
    )
    return agent_results_exist and samples_exist and metadata_exists and grid_exists


def save_prompt_metadata(
    *, outdir: str, prompt_index: int, prompt: str, benchmark_txt_path: str
):
    metadata = {
        "prompt": prompt,
        "prompt_index": prompt_index,
        "benchmark_txt_path": benchmark_txt_path,
    }
    with open(get_metadata_path(outdir, prompt_index), "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, ensure_ascii=False, indent=2)


def save_failure_report(
    *,
    outdir: str,
    prompt_index: int,
    prompt: str,
    sample_seed: int,
    attempt: int,
    error: Exception,
):
    failure_payload = {
        "prompt": prompt,
        "prompt_index": prompt_index,
        "sample_seed": sample_seed,
        "attempt": attempt,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
    }
    failure_path = os.path.join(get_prompt_dir(outdir, prompt_index), "failure.json")
    with open(failure_path, "w", encoding="utf-8") as fp:
        json.dump(failure_payload, fp, ensure_ascii=False, indent=2)


def validate_args(opt):
    validate_epic_args(opt)
    if opt.n_samples < 1:
        raise ValueError("--n_samples must be at least 1.")
    if opt.start_index < 0:
        raise ValueError("--start_index must be >= 0.")
    if opt.end_index is not None and opt.end_index < opt.start_index:
        raise ValueError("--end_index must be >= --start_index.")
    if opt.max_retries < 1:
        raise ValueError("--max_retries must be at least 1.")


def main(opt):
    validate_args(opt)
    async_generate_with_prompt = load_main_graph_runner()
    prompts = load_prompts(opt.benchmark_txt_path)
    selected_entries = [
        (prompt_index, prompt)
        for (prompt_index, prompt) in enumerate(prompts)
        if prompt_index >= opt.start_index
        and (opt.end_index is None or prompt_index < opt.end_index)
    ]
    with asyncio.Runner() as runner:
        for loop_pos, (prompt_index, prompt) in enumerate(
            tqdm(
                selected_entries, total=len(selected_entries), desc="Prompts Progress"
            ),
            start=1,
        ):
            if prompt_already_processed(opt, prompt_index):
                continue
            outpath = get_prompt_dir(opt.outdir, prompt_index)
            os.makedirs(outpath, exist_ok=True)
            sample_path = os.path.join(outpath, "samples")
            os.makedirs(sample_path, exist_ok=True)
            save_prompt_metadata(
                outdir=opt.outdir,
                prompt_index=prompt_index,
                prompt=prompt,
                benchmark_txt_path=opt.benchmark_txt_path,
            )
            sample_count = 0
            print(
                f"Prompt ({loop_pos: >3}/{len(selected_entries)}, benchmark #{prompt_index: >3}): '{prompt}'"
            )
            with torch.no_grad():
                all_samples = []
                for n in trange(opt.n_samples, desc="Sampling"):
                    current_sample_seed = opt.seed + n
                    seed_everything(current_sample_seed)
                    last_error = None
                    for attempt in range(1, opt.max_retries + 1):
                        try:
                            agent_saving_path = os.path.join(
                                opt.outdir,
                                "agent_results",
                                f"seed_{current_sample_seed}",
                            )
                            sample = runner.run(
                                async_generate_with_prompt(
                                    prompt,
                                    seed=current_sample_seed,
                                    save_idx=prompt_index,
                                    saving_path=agent_saving_path,
                                    **epic_kwargs(opt),
                                )
                            )
                            break
                        except Exception as exc:
                            last_error = exc
                            save_failure_report(
                                outdir=opt.outdir,
                                prompt_index=prompt_index,
                                prompt=prompt,
                                sample_seed=current_sample_seed,
                                attempt=attempt,
                                error=exc,
                            )
                            print(
                                f"  Sample seed {current_sample_seed} failed on attempt {attempt}/{opt.max_retries}: {exc}"
                            )
                    else:
                        raise RuntimeError(
                            f"Failed to generate prompt index {prompt_index} after {opt.max_retries} attempts."
                        ) from last_error
                    samples = [sample]
                    for current_sample in samples:
                        current_sample.save(
                            os.path.join(sample_path, f"{sample_count:05}.png")
                        )
                        sample_count += 1
                    if not opt.skip_grid:
                        all_samples.append(
                            torch.stack(
                                [
                                    ToTensor()(current_sample)
                                    for current_sample in samples
                                ],
                                0,
                            )
                        )
                if not opt.skip_grid:
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, "n b c h w -> (n b) c h w")
                    grid = make_grid(grid, nrow=1)
                    grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
                    grid = Image.fromarray(grid.astype(np.uint8))
                    grid.save(get_grid_path(opt.outdir, prompt_index))
                    del grid
            del all_samples
    print("Done.")


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
