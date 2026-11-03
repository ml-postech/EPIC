"""Batch runner for GenAI-Bench-1600 prompt manifests using EPIC."""

from __future__ import annotations
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_epic_arguments(parser)
    parser.add_argument(
        "--benchmark_data",
        type=str,
        default="benchmarks/genai_bench/genai_bench_1600.jsonl",
        help="JSONL prompt manifest produced by scripts/prepare_genai_bench.py.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        default="outputs/genai_bench_results/ours_full1600_n1",
        help="Directory to write raw EPIC artifacts and eval-ready samples.",
    )
    parser.add_argument(
        "--n_samples", type=int, default=1, help="Images per prompt. Default: 1."
    )
    parser.add_argument(
        "--skip_grid",
        action="store_true",
        help="Skip saving a grid image per prompt under outdir/raw.",
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
        help="Shuffle the selected prompt run order after filtering.",
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=None,
        help="Seed for --shuffle_prompts. Defaults to --seed.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=1,
        help="How many times to retry a sample when generation fails.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Load data and write manifests, but do not import/run the generation graph.",
    )
    return parser.parse_args()


def load_main_graph_runner():
    try:
        from graph.graph_main import async_generate_with_prompt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Failed to import graph.graph_main. Make sure runtime dependencies are installed in the active environment."
        ) from exc
    return async_generate_with_prompt


def atomic_write_json(path: str | Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    (tmp_fd, tmp_path) = tempfile.mkstemp(
        prefix=".tmp_json_", suffix=".json", dir=str(path.parent)
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
def locked_manifest_update(outdir: str):
    os.makedirs(outdir, exist_ok=True)
    lock_path = os.path.join(outdir, ".genai_bench_manifest.lock")
    with open(lock_path, "a+", encoding="utf-8") as lock_fp:
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)


def load_benchmark_entries(benchmark_data: str) -> list[tuple[int, dict]]:
    entries = []
    with open(benchmark_data, encoding="utf-8") as fp:
        for row_index, line in enumerate(fp):
            stripped = line.strip()
            if not stripped:
                continue
            entry = json.loads(stripped)
            if "prompt" not in entry:
                raise ValueError(f"Missing 'prompt' key in benchmark row {row_index}.")
            prompt_index = int(entry.get("prompt_index", row_index))
            entry = dict(entry)
            entry["prompt_index"] = prompt_index
            entry["id"] = str(entry.get("id", f"{prompt_index:05d}")).zfill(5)
            entries.append((prompt_index, entry))
    return entries


def prompt_in_requested_slice(opt: argparse.Namespace, prompt_index: int) -> bool:
    if prompt_index < opt.start_index:
        return False
    if opt.end_index is not None and prompt_index >= opt.end_index:
        return False
    return prompt_index % opt.n_shard == opt.shard_idx


def maybe_shuffle_prompt_entries(
    opt: argparse.Namespace, selected_entries: list[tuple[int, dict]]
) -> tuple[list[tuple[int, dict]], int | None]:
    if not opt.shuffle_prompts:
        return (selected_entries, None)
    shuffle_seed = opt.shuffle_seed if opt.shuffle_seed is not None else opt.seed
    shuffled_entries = list(selected_entries)
    random.Random(shuffle_seed).shuffle(shuffled_entries)
    return (shuffled_entries, shuffle_seed)


def get_raw_outdir(outdir: str) -> str:
    return os.path.join(outdir, "raw")


def get_raw_prompt_dir(outdir: str, prompt_index: int) -> str:
    return os.path.join(get_raw_outdir(outdir), f"{prompt_index:0>5}")


def get_raw_sample_path(outdir: str, prompt_index: int, sample_index: int) -> str:
    return os.path.join(
        get_raw_prompt_dir(outdir, prompt_index), "samples", f"{sample_index:05}.png"
    )


def get_raw_grid_path(outdir: str, prompt_index: int) -> str:
    return os.path.join(get_raw_prompt_dir(outdir, prompt_index), "grid.png")


def get_raw_metadata_path(outdir: str, prompt_index: int) -> str:
    return os.path.join(get_raw_prompt_dir(outdir, prompt_index), "metadata.json")


def get_agent_result_path(outdir: str, sample_seed: int, prompt_index: int) -> str:
    return os.path.join(
        get_raw_outdir(outdir),
        "agent_results",
        f"seed_{sample_seed}",
        f"{prompt_index:06d}_0_result.json",
    )


def get_eval_outdir(outdir: str) -> str:
    return os.path.join(outdir, "eval")


def get_eval_sample_dir(outdir: str, sample_index: int) -> str:
    return os.path.join(get_eval_outdir(outdir), f"sample_{sample_index:05d}")


def get_eval_sample_path(outdir: str, prompt_index: int, sample_index: int) -> str:
    return os.path.join(
        get_eval_sample_dir(outdir, sample_index), f"{prompt_index:05d}.png"
    )


def get_eval_sample_manifest_path(outdir: str, sample_index: int) -> str:
    return os.path.join(get_eval_sample_dir(outdir, sample_index), "manifest.json")


def get_image_paths_map_path(outdir: str, sample_index: int) -> str:
    return os.path.join(
        get_eval_outdir(outdir), f"image_paths.sample_{sample_index:05d}.json"
    )


def prompt_already_processed(opt: argparse.Namespace, prompt_index: int) -> bool:
    agent_results_exist = all(
        (
            os.path.exists(
                get_agent_result_path(opt.outdir, opt.seed + sample_index, prompt_index)
            )
            for sample_index in range(opt.n_samples)
        )
    )
    raw_samples_exist = all(
        (
            os.path.exists(get_raw_sample_path(opt.outdir, prompt_index, sample_index))
            for sample_index in range(opt.n_samples)
        )
    )
    eval_samples_exist = all(
        (
            os.path.exists(get_eval_sample_path(opt.outdir, prompt_index, sample_index))
            for sample_index in range(opt.n_samples)
        )
    )
    metadata_exists = os.path.exists(get_raw_metadata_path(opt.outdir, prompt_index))
    grid_exists = opt.skip_grid or os.path.exists(
        get_raw_grid_path(opt.outdir, prompt_index)
    )
    return (
        agent_results_exist
        and raw_samples_exist
        and eval_samples_exist
        and metadata_exists
        and grid_exists
    )


def save_prompt_metadata(
    *, outdir: str, prompt_index: int, benchmark_data: str, entry: dict, n_samples: int
) -> None:
    metadata = {
        **entry,
        "prompt_index": prompt_index,
        "benchmark_data": benchmark_data,
        "n_samples": n_samples,
        "eval_samples": [
            {
                "sample_index": sample_index,
                "path": os.path.abspath(
                    get_eval_sample_path(outdir, prompt_index, sample_index)
                ),
            }
            for sample_index in range(n_samples)
        ],
    }
    atomic_write_json(get_raw_metadata_path(outdir, prompt_index), metadata)


def save_failure_report(
    *,
    outdir: str,
    prompt_index: int,
    entry: dict,
    sample_seed: int,
    attempt: int,
    error: Exception,
) -> None:
    failure_payload = {
        "prompt": entry["prompt"],
        "id": entry.get("id"),
        "prompt_index": prompt_index,
        "sample_seed": sample_seed,
        "attempt": attempt,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
    }
    failure_path = os.path.join(
        get_raw_prompt_dir(outdir, prompt_index), "failure.json"
    )
    atomic_write_json(failure_path, failure_payload)


def write_run_manifest(
    *,
    opt: argparse.Namespace,
    benchmark_entries: list[tuple[int, dict]],
    selected_entries: list[tuple[int, dict]],
    shuffle_seed: int | None,
) -> None:
    payload = {
        "benchmark_data": opt.benchmark_data,
        "outdir": opt.outdir,
        "n_samples": opt.n_samples,
        "seed": opt.seed,
        "n_shard": opt.n_shard,
        "shard_idx": opt.shard_idx,
        "start_index": opt.start_index,
        "end_index": opt.end_index,
        "shuffle_prompts": opt.shuffle_prompts,
        "shuffle_seed": shuffle_seed,
        "num_benchmark_prompts": len(benchmark_entries),
        "num_selected_prompts": len(selected_entries),
        "eval_root": os.path.abspath(get_eval_outdir(opt.outdir)),
        "raw_root": os.path.abspath(get_raw_outdir(opt.outdir)),
        "run_order": [
            {
                "run_position": run_position,
                "prompt_index": prompt_index,
                "id": entry.get("id", f"{prompt_index:05d}"),
                "prompt": entry["prompt"],
            }
            for (run_position, (prompt_index, entry)) in enumerate(
                selected_entries, start=1
            )
        ],
    }
    if opt.n_shard <= 1:
        manifest_name = "run_manifest.json"
    else:
        manifest_name = (
            f"run_manifest.shard_{opt.shard_idx:05d}_of_{opt.n_shard:05d}.json"
        )
    atomic_write_json(os.path.join(opt.outdir, manifest_name), payload)


def write_eval_manifests(
    *, outdir: str, benchmark_entries: list[tuple[int, dict]], n_samples: int
) -> dict:
    with locked_manifest_update(outdir):
        summary = {"n_samples": n_samples, "samples": {}}
        for sample_index in range(n_samples):
            sample_dir = get_eval_sample_dir(outdir, sample_index)
            prompt_records = []
            image_paths = {}
            for prompt_index, entry in benchmark_entries:
                sample_path = get_eval_sample_path(outdir, prompt_index, sample_index)
                if os.path.exists(sample_path):
                    prompt_id = entry.get("id", f"{prompt_index:05d}")
                    abs_path = os.path.abspath(sample_path)
                    image_paths[prompt_id] = abs_path
                    prompt_records.append(
                        {
                            "id": prompt_id,
                            "prompt_index": prompt_index,
                            "prompt": entry["prompt"],
                            "path": abs_path,
                        }
                    )
            payload = {
                "sample_index": sample_index,
                "samples_dir": os.path.abspath(sample_dir),
                "num_completed_prompts": len(prompt_records),
                "prompts": prompt_records,
            }
            atomic_write_json(
                get_eval_sample_manifest_path(outdir, sample_index), payload
            )
            atomic_write_json(
                get_image_paths_map_path(outdir, sample_index), image_paths
            )
            summary["samples"][f"{sample_index:05d}"] = {
                "num_completed_prompts": len(prompt_records),
                "samples_dir": os.path.abspath(sample_dir),
                "manifest": os.path.abspath(
                    get_eval_sample_manifest_path(outdir, sample_index)
                ),
                "image_paths": os.path.abspath(
                    get_image_paths_map_path(outdir, sample_index)
                ),
            }
        atomic_write_json(os.path.join(outdir, "eval_manifest.json"), summary)
    return summary


def validate_args(opt: argparse.Namespace) -> None:
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


def main(opt: argparse.Namespace) -> None:
    validate_args(opt)
    benchmark_entries = load_benchmark_entries(opt.benchmark_data)
    selected_entries = [
        (prompt_index, entry)
        for (prompt_index, entry) in benchmark_entries
        if prompt_in_requested_slice(opt, prompt_index)
    ]
    (selected_entries, shuffle_seed) = maybe_shuffle_prompt_entries(
        opt, selected_entries
    )
    write_run_manifest(
        opt=opt,
        benchmark_entries=benchmark_entries,
        selected_entries=selected_entries,
        shuffle_seed=shuffle_seed,
    )
    eval_summary = write_eval_manifests(
        outdir=opt.outdir, benchmark_entries=benchmark_entries, n_samples=opt.n_samples
    )
    print(
        f"Running shard {opt.shard_idx}/{opt.n_shard} with {len(selected_entries)}/{len(benchmark_entries)} prompts."
    )
    if opt.shuffle_prompts:
        print(f"Shuffled selected prompt run order with seed {shuffle_seed}.")
    print(f"Eval samples will be written under {opt.outdir}/eval/sample_<index>.")
    if opt.dry_run:
        completed = sum(
            (
                sample_summary["num_completed_prompts"]
                for sample_summary in eval_summary["samples"].values()
            )
        )
        print(
            f"Dry run complete. Existing eval samples visible in manifests: {completed}."
        )
        return
    async_generate_with_prompt = load_main_graph_runner()
    with asyncio.Runner() as runner:
        for loop_pos, (prompt_index, entry) in enumerate(
            tqdm(
                selected_entries, total=len(selected_entries), desc="Prompts Progress"
            ),
            start=1,
        ):
            if prompt_already_processed(opt, prompt_index):
                continue
            raw_prompt_dir = get_raw_prompt_dir(opt.outdir, prompt_index)
            raw_sample_dir = os.path.join(raw_prompt_dir, "samples")
            os.makedirs(raw_sample_dir, exist_ok=True)
            for sample_index in range(opt.n_samples):
                os.makedirs(
                    get_eval_sample_dir(opt.outdir, sample_index), exist_ok=True
                )
            save_prompt_metadata(
                outdir=opt.outdir,
                prompt_index=prompt_index,
                benchmark_data=opt.benchmark_data,
                entry=entry,
                n_samples=opt.n_samples,
            )
            prompt = entry["prompt"]
            prompt_id = entry.get("id", f"{prompt_index:05d}")
            print(
                f"Prompt ({loop_pos: >4}/{len(selected_entries)}, index #{prompt_index: >4}, id={prompt_id}): '{prompt}'"
            )
            with torch.no_grad():
                all_samples = []
                for sample_index in trange(opt.n_samples, desc="Sampling"):
                    current_sample_seed = opt.seed + sample_index
                    seed_everything(current_sample_seed)
                    last_error = None
                    for attempt in range(1, opt.max_retries + 1):
                        try:
                            agent_saving_path = os.path.join(
                                get_raw_outdir(opt.outdir),
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
                                entry=entry,
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
                    raw_sample_path = get_raw_sample_path(
                        opt.outdir, prompt_index, sample_index
                    )
                    eval_sample_path = get_eval_sample_path(
                        opt.outdir, prompt_index, sample_index
                    )
                    sample.save(raw_sample_path)
                    sample.save(eval_sample_path)
                    if not opt.skip_grid:
                        all_samples.append(torch.stack([ToTensor()(sample)], 0))
                if not opt.skip_grid:
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, "n b c h w -> (n b) c h w")
                    grid = make_grid(grid, nrow=1)
                    grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
                    grid = Image.fromarray(grid.astype(np.uint8))
                    grid.save(get_raw_grid_path(opt.outdir, prompt_index))
                    del grid
            del all_samples
            eval_summary = write_eval_manifests(
                outdir=opt.outdir,
                benchmark_entries=benchmark_entries,
                n_samples=opt.n_samples,
            )
    completed = sum(
        (
            sample_summary["num_completed_prompts"]
            for sample_summary in eval_summary["samples"].values()
        )
    )
    print(f"Done. Completed eval sample files visible in manifests: {completed}.")


if __name__ == "__main__":
    main(parse_args())
