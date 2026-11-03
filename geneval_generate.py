"""Adapted from GenEval benchmark: https://github.com/djghosh13/geneval/tree/main"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from einops import rearrange
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from tqdm import tqdm

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
        "--metadata_file",
        type=str,
        help="JSONL file containing lines of metadata for each prompt",
        default="benchmarks/geneval_evaluation_metadata.jsonl",
    )
    parser.add_argument(
        "--tags",
        type=str,
        default=None,
        help="Optional comma-separated GenEval tags to run, for example 'position,color_attr'. If omitted, run all metadata entries.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/geneval_results/",
    )
    parser.add_argument("--n_samples", type=int, default=4, help="number of samples")
    parser.add_argument("--skip_grid", action="store_true", help="skip saving grid")
    parser.add_argument(
        "--n_shard",
        type=int,
        default=1,
        help="Number of sample shards to split each prompt into. Samples are assigned by sample_index %% n_shard.",
    )
    parser.add_argument(
        "--shard_idx", type=int, default=0, help="Zero-based sample shard index to run."
    )
    opt = parser.parse_args()
    return opt


def get_agent_result_path(outdir: str, sample_seed: int, prompt_index: int) -> str:
    return os.path.join(
        outdir,
        "agent_results",
        f"seed_{sample_seed}",
        f"{prompt_index:06d}_0_result.json",
    )


def get_sample_image_path(outdir: str, prompt_index: int, sample_index: int) -> str:
    return os.path.join(
        outdir, f"{prompt_index:0>5}", "samples", f"{sample_index:05}.png"
    )


def sample_already_processed(opt, prompt_index: int, sample_index: int) -> bool:
    sample_seed = opt.seed + sample_index
    return os.path.exists(
        get_agent_result_path(opt.outdir, sample_seed, prompt_index)
    ) and os.path.exists(get_sample_image_path(opt.outdir, prompt_index, sample_index))


def get_assigned_sample_indices(opt) -> list[int]:
    return [
        sample_index
        for sample_index in range(opt.n_samples)
        if sample_index % opt.n_shard == opt.shard_idx
    ]


def all_samples_ready(opt, prompt_index: int) -> bool:
    return all(
        (
            sample_already_processed(opt, prompt_index, sample_index)
            for sample_index in range(opt.n_samples)
        )
    )


def prompt_already_processed(opt, prompt_index: int) -> bool:
    samples_ready = all_samples_ready(opt, prompt_index)
    if not samples_ready:
        return False
    if opt.skip_grid:
        return True
    return os.path.exists(os.path.join(opt.outdir, f"{prompt_index:0>5}", "grid.png"))


def shard_already_processed(opt, prompt_index: int) -> bool:
    return all(
        (
            sample_already_processed(opt, prompt_index, sample_index)
            for sample_index in get_assigned_sample_indices(opt)
        )
    )


def validate_args(opt):
    validate_epic_args(opt)
    if opt.n_samples < 1:
        raise ValueError("--n_samples must be at least 1.")
    if opt.n_shard < 1:
        raise ValueError("--n_shard must be at least 1.")
    if opt.shard_idx < 0:
        raise ValueError("--shard_idx must be >= 0.")
    if opt.shard_idx >= opt.n_shard:
        raise ValueError("--shard_idx must be smaller than --n_shard.")


def maybe_save_grid(opt, prompt_index: int):
    if opt.skip_grid or not all_samples_ready(opt, prompt_index):
        return
    outpath = os.path.join(opt.outdir, f"{prompt_index:0>5}")
    grid_path = os.path.join(outpath, "grid.png")
    if os.path.exists(grid_path):
        return
    all_samples = []
    for sample_index in range(opt.n_samples):
        sample_image_path = get_sample_image_path(
            opt.outdir, prompt_index, sample_index
        )
        with Image.open(sample_image_path) as sample_image:
            all_samples.append(torch.stack([ToTensor()(sample_image.copy())], 0))
    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, "n b c h w -> (n b) c h w")
    grid = make_grid(grid, nrow=1)
    grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
    grid = Image.fromarray(grid.astype(np.uint8))
    grid.save(grid_path)


def parse_requested_tags(raw_tags: str | None) -> list[str]:
    if not raw_tags:
        return []
    return [tag.strip() for tag in raw_tags.split(",") if tag.strip()]


def load_filtered_metadata_entries(
    metadata_file: str, requested_tags: list[str]
) -> list[tuple[int, dict]]:
    with open(metadata_file) as fp:
        metadata_entries = [
            (index, json.loads(line)) for (index, line) in enumerate(fp)
        ]
    if not requested_tags:
        return metadata_entries
    available_tags = {metadata.get("tag") for (_, metadata) in metadata_entries}
    unknown_tags = [tag for tag in requested_tags if tag not in available_tags]
    if unknown_tags:
        raise ValueError(
            f"Unknown tag(s) requested: {', '.join(unknown_tags)}. Available tags: {', '.join(sorted((tag for tag in available_tags if tag)))}"
        )
    requested_tag_set = set(requested_tags)
    filtered_entries = [
        (index, metadata)
        for (index, metadata) in metadata_entries
        if metadata.get("tag") in requested_tag_set
    ]
    if not filtered_entries:
        raise ValueError(
            f"No metadata entries matched the requested tag filter: {', '.join(requested_tags)}"
        )
    return filtered_entries


def load_main_graph_runner():
    try:
        from graph.graph_main import async_generate_with_prompt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Failed to import graph.graph_main. Make sure the main graph runtime dependencies are installed in the active environment."
        ) from exc
    return async_generate_with_prompt


def main(opt):
    validate_args(opt)
    async_generate_with_prompt = load_main_graph_runner()
    with asyncio.Runner() as runner:
        requested_tags = parse_requested_tags(opt.tags)
        metadata_entries = load_filtered_metadata_entries(
            opt.metadata_file, requested_tags
        )
        if requested_tags:
            print(
                f"Filtering GenEval prompts by tag(s): {', '.join(requested_tags)} ({len(metadata_entries)} matched prompts)"
            )
        assigned_sample_indices = get_assigned_sample_indices(opt)
        assigned_sample_seeds = [
            opt.seed + sample_index for sample_index in assigned_sample_indices
        ]
        print(
            f"Running sample shard {opt.shard_idx}/{opt.n_shard} with {len(assigned_sample_indices)}/{opt.n_samples} samples per prompt: indices {assigned_sample_indices}, seeds {assigned_sample_seeds}"
        )
        for filtered_pos, (prompt_index, metadata) in enumerate(
            tqdm(
                metadata_entries, total=len(metadata_entries), desc="Prompts Progress"
            ),
            start=1,
        ):
            if prompt_already_processed(opt, prompt_index):
                continue
            outpath = os.path.join(opt.outdir, f"{prompt_index:0>5}")
            os.makedirs(outpath, exist_ok=True)
            prompt = metadata["prompt"]
            print(
                f"Prompt ({filtered_pos: >3}/{len(metadata_entries)}, benchmark #{prompt_index: >3}): '{prompt}'"
            )
            sample_path = os.path.join(outpath, "samples")
            os.makedirs(sample_path, exist_ok=True)
            with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
                json.dump(metadata, fp)
            with torch.no_grad():
                for sample_index in tqdm(assigned_sample_indices, desc="Sampling"):
                    current_sample_seed = opt.seed + sample_index
                    sample_image_path = get_sample_image_path(
                        opt.outdir, prompt_index, sample_index
                    )
                    if not sample_already_processed(opt, prompt_index, sample_index):
                        seed_everything(current_sample_seed)
                        sample = runner.run(
                            async_generate_with_prompt(
                                prompt,
                                seed=current_sample_seed,
                                save_idx=prompt_index,
                                saving_path=os.path.join(
                                    opt.outdir,
                                    "agent_results",
                                    f"seed_{current_sample_seed}",
                                ),
                                **epic_kwargs(opt),
                            )
                        )
                        sample.save(sample_image_path)
            if shard_already_processed(opt, prompt_index):
                maybe_save_grid(opt, prompt_index)
        print("Done.")


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
