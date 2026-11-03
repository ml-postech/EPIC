"""Batch runner for GenEval 2 prompts using the EPIC pipeline."""

import argparse
import asyncio
import fcntl
import json
import os
import random
import sys
import tempfile
import traceback
from contextlib import contextmanager
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
        "--benchmark_data",
        type=str,
        default="GenEval2/geneval2_data.jsonl",
        help="JSONL file containing the GenEval 2 benchmark data.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        default="outputs/geneval2_results/",
        help="Directory to write prompt folders, image maps, and agent results to.",
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
        "--n_shard",
        type=int,
        default=1,
        help="Number of round-robin shards to split prompts into. Prompts are assigned by prompt_index %% n_shard.",
    )
    parser.add_argument(
        "--shard_idx",
        type=int,
        default=0,
        help="Zero-based round-robin shard index to run.",
    )
    parser.add_argument(
        "--shuffle_prompts",
        action="store_true",
        help="Shuffle the selected prompt run order after start/end/shard filtering. Original benchmark prompt indices, output folders, and evaluation mappings stay unchanged.",
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=None,
        help="Seed for --shuffle_prompts. Defaults to --seed so the shuffled order is reproducible across reruns.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=1,
        help="How many times to retry a sample when generation fails.",
    )
    parser.add_argument(
        "--default_eval_sample_index",
        type=int,
        default=0,
        help="Which sample index should be exposed as outdir/image_paths.json for GenEval2 evaluation.",
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


def load_benchmark_entries(benchmark_data: str) -> list[tuple[int, dict]]:
    entries = []
    with open(benchmark_data, encoding="utf-8") as fp:
        for prompt_index, line in enumerate(fp):
            stripped = line.strip()
            if not stripped:
                continue
            entry = json.loads(stripped)
            if "prompt" not in entry:
                raise ValueError(
                    f"Missing 'prompt' key in benchmark row {prompt_index}."
                )
            entries.append((prompt_index, entry))
    return entries


def validate_unique_prompts(entries: list[tuple[int, dict]]):
    seen_prompts = {}
    duplicates = []
    for prompt_index, entry in entries:
        prompt = entry["prompt"]
        if prompt in seen_prompts:
            duplicates.append((prompt, seen_prompts[prompt], prompt_index))
        else:
            seen_prompts[prompt] = prompt_index
    if duplicates:
        duplicate_descriptions = ", ".join(
            (
                f"'{prompt}' at rows {first_idx} and {second_idx}"
                for (prompt, first_idx, second_idx) in duplicates[:5]
            )
        )
        raise ValueError(
            f"GenEval 2 prompt strings must be unique because evaluation expects a prompt->image mapping. Found duplicates: {duplicate_descriptions}"
        )


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


def get_image_paths_map_path(outdir: str, sample_index: int | None = None) -> str:
    if sample_index is None:
        return os.path.join(outdir, "image_paths.json")
    return os.path.join(outdir, f"image_paths.sample_{sample_index:05}.json")


def get_image_paths_lock_path(outdir: str) -> str:
    return os.path.join(outdir, ".image_paths.lock")


def atomic_write_json(path: str, payload: dict):
    output_dir = os.path.dirname(path) or "."
    os.makedirs(output_dir, exist_ok=True)
    (tmp_fd, tmp_path) = tempfile.mkstemp(
        prefix=".tmp_json_", suffix=".json", dir=output_dir
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@contextmanager
def locked_image_paths_update(outdir: str):
    os.makedirs(outdir, exist_ok=True)
    lock_path = get_image_paths_lock_path(outdir)
    with open(lock_path, "a+", encoding="utf-8") as lock_fp:
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)


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


def prompt_in_requested_slice(opt, prompt_index: int) -> bool:
    if prompt_index < opt.start_index:
        return False
    if opt.end_index is not None and prompt_index >= opt.end_index:
        return False
    return prompt_index % opt.n_shard == opt.shard_idx


def get_prompt_run_order_path(
    outdir: str, shard_idx: int | None = None, n_shard: int | None = None
) -> str:
    if shard_idx is None or n_shard is None or n_shard <= 1:
        return os.path.join(outdir, "prompt_run_order.json")
    return os.path.join(
        outdir, f"prompt_run_order.shard_{shard_idx:05}_of_{n_shard:05}.json"
    )


def get_prompt_shuffle_seed(opt) -> int:
    if opt.shuffle_seed is not None:
        return opt.shuffle_seed
    return opt.seed


def maybe_shuffle_prompt_entries(
    opt, selected_entries: list[tuple[int, dict]]
) -> tuple[list[tuple[int, dict]], int | None]:
    if not opt.shuffle_prompts:
        return (selected_entries, None)
    shuffle_seed = get_prompt_shuffle_seed(opt)
    shuffled_entries = list(selected_entries)
    random.Random(shuffle_seed).shuffle(shuffled_entries)
    return (shuffled_entries, shuffle_seed)


def write_prompt_run_order_manifest(
    *,
    outdir: str,
    selected_entries: list[tuple[int, dict]],
    shuffle_prompts: bool,
    shuffle_seed: int | None,
    shard_idx: int,
    n_shard: int,
):
    os.makedirs(outdir, exist_ok=True)
    manifest = {
        "shuffle_prompts": shuffle_prompts,
        "shuffle_seed": shuffle_seed,
        "shard_idx": shard_idx,
        "n_shard": n_shard,
        "num_selected_prompts": len(selected_entries),
        "run_order": [
            {
                "run_position": run_position,
                "prompt_index": prompt_index,
                "prompt": entry["prompt"],
            }
            for (run_position, (prompt_index, entry)) in enumerate(
                selected_entries, start=1
            )
        ],
    }
    atomic_write_json(
        get_prompt_run_order_path(outdir, shard_idx=shard_idx, n_shard=n_shard),
        manifest,
    )


def save_prompt_metadata(
    *, outdir: str, prompt_index: int, benchmark_data: str, entry: dict
):
    metadata = {**entry, "prompt_index": prompt_index, "benchmark_data": benchmark_data}
    atomic_write_json(get_metadata_path(outdir, prompt_index), metadata)


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
    atomic_write_json(failure_path, failure_payload)


def write_image_path_mappings(
    *,
    outdir: str,
    benchmark_entries: list[tuple[int, dict]],
    n_samples: int,
    default_eval_sample_index: int,
):
    with locked_image_paths_update(outdir):
        sample_counts = {}
        default_payload = {}
        for sample_index in range(n_samples):
            image_paths = {}
            for prompt_index, entry in benchmark_entries:
                sample_path = get_sample_path(outdir, prompt_index, sample_index)
                if os.path.exists(sample_path):
                    image_paths[entry["prompt"]] = os.path.abspath(sample_path)
            if sample_index == default_eval_sample_index:
                default_payload = dict(image_paths)
            map_path = get_image_paths_map_path(outdir, sample_index)
            atomic_write_json(map_path, image_paths)
            sample_counts[sample_index] = len(image_paths)
        atomic_write_json(get_image_paths_map_path(outdir), default_payload)
    return {
        "default_eval_sample_index": default_eval_sample_index,
        "num_prompts_in_benchmark": len(benchmark_entries),
        "num_prompts_in_default_map": sample_counts.get(default_eval_sample_index, 0),
        "sample_counts": sample_counts,
    }


def validate_args(opt):
    validate_epic_args(opt)
    if opt.n_samples < 1:
        raise ValueError("--n_samples must be at least 1.")
    if opt.start_index < 0:
        raise ValueError("--start_index must be >= 0.")
    if opt.end_index is not None and opt.end_index < opt.start_index:
        raise ValueError("--end_index must be >= --start_index.")
    if opt.n_shard < 1:
        raise ValueError("--n_shard must be at least 1.")
    if opt.shard_idx < 0:
        raise ValueError("--shard_idx must be >= 0.")
    if opt.shard_idx >= opt.n_shard:
        raise ValueError("--shard_idx must be smaller than --n_shard.")
    if opt.max_retries < 1:
        raise ValueError("--max_retries must be at least 1.")
    if opt.default_eval_sample_index < 0:
        raise ValueError("--default_eval_sample_index must be >= 0.")
    if opt.default_eval_sample_index >= opt.n_samples:
        raise ValueError(
            "--default_eval_sample_index must be smaller than --n_samples."
        )


def main(opt):
    validate_args(opt)
    async_generate_with_prompt = load_main_graph_runner()
    benchmark_entries = load_benchmark_entries(opt.benchmark_data)
    validate_unique_prompts(benchmark_entries)
    selected_entries = [
        (prompt_index, entry)
        for (prompt_index, entry) in benchmark_entries
        if prompt_in_requested_slice(opt, prompt_index)
    ]
    (selected_entries, shuffle_seed) = maybe_shuffle_prompt_entries(
        opt, selected_entries
    )
    print(
        f"Running shard {opt.shard_idx}/{opt.n_shard} with {len(selected_entries)}/{len(benchmark_entries)} prompts (round-robin by benchmark prompt index)."
    )
    if opt.shuffle_prompts:
        print(
            f"Shuffled selected prompt run order with shuffle seed {shuffle_seed}; benchmark prompt indices are unchanged."
        )
        write_prompt_run_order_manifest(
            outdir=opt.outdir,
            selected_entries=selected_entries,
            shuffle_prompts=opt.shuffle_prompts,
            shuffle_seed=shuffle_seed,
            shard_idx=opt.shard_idx,
            n_shard=opt.n_shard,
        )
    mapping_summary = write_image_path_mappings(
        outdir=opt.outdir,
        benchmark_entries=benchmark_entries,
        n_samples=opt.n_samples,
        default_eval_sample_index=opt.default_eval_sample_index,
    )
    with asyncio.Runner() as runner:
        for loop_pos, (prompt_index, entry) in enumerate(
            tqdm(
                selected_entries, total=len(selected_entries), desc="Prompts Progress"
            ),
            start=1,
        ):
            if prompt_already_processed(opt, prompt_index):
                continue
            outpath = get_prompt_dir(opt.outdir, prompt_index)
            os.makedirs(outpath, exist_ok=True)
            sample_dir = os.path.join(outpath, "samples")
            os.makedirs(sample_dir, exist_ok=True)
            save_prompt_metadata(
                outdir=opt.outdir,
                prompt_index=prompt_index,
                benchmark_data=opt.benchmark_data,
                entry=entry,
            )
            prompt = entry["prompt"]
            print(
                f"Prompt ({loop_pos: >3}/{len(selected_entries)}, benchmark #{prompt_index: >3}): '{prompt}'"
            )
            sample_count = 0
            with torch.no_grad():
                all_samples = []
                for sample_index in trange(opt.n_samples, desc="Sampling"):
                    current_sample_seed = opt.seed + sample_index
                    seed_everything(current_sample_seed)
                    last_error = None
                    for attempt in range(1, opt.max_retries + 1):
                        try:
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
                    sample.save(get_sample_path(opt.outdir, prompt_index, sample_count))
                    sample_count += 1
                    if not opt.skip_grid:
                        all_samples.append(torch.stack([ToTensor()(sample)], 0))
                if not opt.skip_grid:
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, "n b c h w -> (n b) c h w")
                    grid = make_grid(grid, nrow=1)
                    grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
                    grid = Image.fromarray(grid.astype(np.uint8))
                    grid.save(get_grid_path(opt.outdir, prompt_index))
                    del grid
            del all_samples
            mapping_summary = write_image_path_mappings(
                outdir=opt.outdir,
                benchmark_entries=benchmark_entries,
                n_samples=opt.n_samples,
                default_eval_sample_index=opt.default_eval_sample_index,
            )
    print(
        f"Done. Wrote GenEval2 image path mappings with {mapping_summary['num_prompts_in_default_map']}/{mapping_summary['num_prompts_in_benchmark']} prompts in image_paths.json (sample index {mapping_summary['default_eval_sample_index']})."
    )


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
