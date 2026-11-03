"""Batch runner for T2I-CompBench prompt manifests using EPIC."""

from __future__ import annotations
import argparse
import asyncio
import fcntl
import json
import os
import random
import re
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
T2I_COMPBENCH_CATEGORIES = (
    "color",
    "shape",
    "texture",
    "spatial",
    "non_spatial",
    "complex",
)
T2I_COMPBENCHPP_CATEGORIES = (
    "color",
    "shape",
    "texture",
    "spatial",
    "non_spatial",
    "3d_spatial",
    "numeracy",
    "complex",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_epic_arguments(parser)
    parser.add_argument(
        "--benchmark_data",
        type=str,
        default="benchmarks/t2i_compbench/t2i_compbench_val.jsonl",
        help="JSONL prompt manifest produced by scripts/prepare_t2i_compbench.py. Use t2i_compbenchpp_val.jsonl for the expanded eight-category suite.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        default="outputs/t2i_compbench_results/ours",
        help="Directory to write raw EPIC artifacts and official eval samples.",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Optional comma-separated category filter, for example 'color,shape,texture'. If omitted, use all rows in --benchmark_data.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="Images per prompt. The official T2I-CompBench metric protocol uses 10 images per prompt; set to 1 for cheaper pilot generation.",
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
        help="Inclusive global prompt index to start from.",
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=None,
        help="Exclusive global prompt index to stop at. Omit to run through the end.",
    )
    parser.add_argument(
        "--n_shard",
        type=int,
        default=1,
        help="Number of round-robin shards to split prompts into. Prompts are assigned by global_prompt_index %% n_shard.",
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
        help="Shuffle the selected prompt run order after filtering. Prompt indices and official question ids stay unchanged.",
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
    lock_path = os.path.join(outdir, ".t2i_compbench_manifest.lock")
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
            for required_key in ("prompt", "category", "category_prompt_index"):
                if required_key not in entry:
                    raise ValueError(
                        f"Missing '{required_key}' key in benchmark row {row_index}."
                    )
            prompt_index = int(entry.get("global_prompt_index", row_index))
            entry = dict(entry)
            entry["global_prompt_index"] = prompt_index
            entry["category_prompt_index"] = int(entry["category_prompt_index"])
            entries.append((prompt_index, entry))
    return entries


def parse_requested_categories(raw_categories: str | None) -> set[str]:
    if not raw_categories:
        return set()
    categories = {
        category.strip() for category in raw_categories.split(",") if category.strip()
    }
    known_categories = set(T2I_COMPBENCHPP_CATEGORIES)
    unknown_categories = sorted(categories - known_categories)
    if unknown_categories:
        raise ValueError(
            f"Unknown T2I-CompBench category filter(s): {', '.join(unknown_categories)}. Known categories: {', '.join(T2I_COMPBENCHPP_CATEGORIES)}"
        )
    return categories


def prompt_in_requested_slice(
    opt: argparse.Namespace,
    prompt_index: int,
    entry: dict,
    requested_categories: set[str],
) -> bool:
    if requested_categories and entry["category"] not in requested_categories:
        return False
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


def safe_prompt_filename_component(prompt: str) -> str:
    component = prompt.strip().replace("_", " ")
    component = component.replace("/", " ").replace("\\", " ")
    component = re.sub("\\s+", " ", component)
    return component or "prompt"


def get_question_id(entry: dict, sample_index: int, n_samples: int) -> int:
    return int(entry["category_prompt_index"]) * n_samples + sample_index


def get_eval_category_dir(outdir: str, category: str) -> str:
    return os.path.join(outdir, "eval", category)


def get_eval_samples_dir(outdir: str, category: str) -> str:
    return os.path.join(get_eval_category_dir(outdir, category), "samples")


def get_eval_sample_path(
    outdir: str, entry: dict, sample_index: int, n_samples: int
) -> str:
    category = entry["category"]
    question_id = get_question_id(entry, sample_index, n_samples)
    filename_prompt = safe_prompt_filename_component(entry["prompt"])
    return os.path.join(
        get_eval_samples_dir(outdir, category),
        f"{filename_prompt}_{question_id:06d}.png",
    )


def prompt_already_processed(
    opt: argparse.Namespace, prompt_index: int, entry: dict
) -> bool:
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
            os.path.exists(
                get_eval_sample_path(opt.outdir, entry, sample_index, opt.n_samples)
            )
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
                "question_id": get_question_id(entry, sample_index, n_samples),
                "path": os.path.abspath(
                    get_eval_sample_path(outdir, entry, sample_index, n_samples)
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
        "category": entry["category"],
        "prompt_index": prompt_index,
        "category_prompt_index": entry["category_prompt_index"],
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
    category_counts = {}
    selected_category_counts = {}
    for _, entry in benchmark_entries:
        category_counts[entry["category"]] = (
            category_counts.get(entry["category"], 0) + 1
        )
    for _, entry in selected_entries:
        selected_category_counts[entry["category"]] = (
            selected_category_counts.get(entry["category"], 0) + 1
        )
    payload = {
        "benchmark_data": opt.benchmark_data,
        "outdir": opt.outdir,
        "n_samples": opt.n_samples,
        "seed": opt.seed,
        "n_shard": opt.n_shard,
        "shard_idx": opt.shard_idx,
        "start_index": opt.start_index,
        "end_index": opt.end_index,
        "categories": opt.categories,
        "shuffle_prompts": opt.shuffle_prompts,
        "shuffle_seed": shuffle_seed,
        "num_benchmark_prompts": len(benchmark_entries),
        "num_selected_prompts": len(selected_entries),
        "category_counts": category_counts,
        "selected_category_counts": selected_category_counts,
        "eval_root": os.path.abspath(os.path.join(opt.outdir, "eval")),
        "raw_root": os.path.abspath(get_raw_outdir(opt.outdir)),
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
        by_category = {}
        for prompt_index, entry in benchmark_entries:
            category = entry["category"]
            category_payload = by_category.setdefault(
                category,
                {
                    "category": category,
                    "n_samples": n_samples,
                    "samples_dir": os.path.abspath(
                        get_eval_samples_dir(outdir, category)
                    ),
                    "prompts": [],
                },
            )
            sample_records = []
            for sample_index in range(n_samples):
                sample_path = get_eval_sample_path(
                    outdir, entry, sample_index, n_samples
                )
                if os.path.exists(sample_path):
                    sample_records.append(
                        {
                            "sample_index": sample_index,
                            "question_id": get_question_id(
                                entry, sample_index, n_samples
                            ),
                            "path": os.path.abspath(sample_path),
                        }
                    )
            if sample_records:
                category_payload["prompts"].append(
                    {
                        "prompt_index": prompt_index,
                        "category_prompt_index": entry["category_prompt_index"],
                        "prompt": entry["prompt"],
                        "samples": sample_records,
                    }
                )
        summary = {"n_samples": n_samples, "categories": {}}
        for category, payload in sorted(by_category.items()):
            prompt_count = len(payload["prompts"])
            sample_count = sum((len(row["samples"]) for row in payload["prompts"]))
            payload["num_completed_prompts"] = prompt_count
            payload["num_completed_samples"] = sample_count
            category_dir = get_eval_category_dir(outdir, category)
            atomic_write_json(os.path.join(category_dir, "manifest.json"), payload)
            summary["categories"][category] = {
                "num_completed_prompts": prompt_count,
                "num_completed_samples": sample_count,
                "samples_dir": payload["samples_dir"],
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
    async_generate_with_prompt = load_main_graph_runner()
    benchmark_entries = load_benchmark_entries(opt.benchmark_data)
    requested_categories = parse_requested_categories(opt.categories)
    selected_entries = [
        (prompt_index, entry)
        for (prompt_index, entry) in benchmark_entries
        if prompt_in_requested_slice(opt, prompt_index, entry, requested_categories)
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
    print(
        f"Official eval samples will be written under {opt.outdir}/eval/<category>/samples."
    )
    with asyncio.Runner() as runner:
        for loop_pos, (prompt_index, entry) in enumerate(
            tqdm(
                selected_entries, total=len(selected_entries), desc="Prompts Progress"
            ),
            start=1,
        ):
            if prompt_already_processed(opt, prompt_index, entry):
                continue
            raw_prompt_dir = get_raw_prompt_dir(opt.outdir, prompt_index)
            raw_sample_dir = os.path.join(raw_prompt_dir, "samples")
            eval_sample_dir = get_eval_samples_dir(opt.outdir, entry["category"])
            os.makedirs(raw_sample_dir, exist_ok=True)
            os.makedirs(eval_sample_dir, exist_ok=True)
            save_prompt_metadata(
                outdir=opt.outdir,
                prompt_index=prompt_index,
                benchmark_data=opt.benchmark_data,
                entry=entry,
                n_samples=opt.n_samples,
            )
            prompt = entry["prompt"]
            print(
                f"Prompt ({loop_pos: >4}/{len(selected_entries)}, global #{prompt_index: >4}, {entry['category']} #{entry['category_prompt_index']: >3}): '{prompt}'"
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
                        opt.outdir, entry, sample_index, opt.n_samples
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
            category_summary["num_completed_samples"]
            for category_summary in eval_summary["categories"].values()
        )
    )
    print(
        f"Done. Completed official eval sample files visible in manifests: {completed}."
    )


if __name__ == "__main__":
    main(parse_args())
