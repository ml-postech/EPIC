"""Generate UniGenBench English-short images with the EPIC main method.

Writes raw artifacts and evaluator images named <prompt_id>_<sample_id>.png.
Use --dry_run to inspect the selected prompt range without loading models."""

from __future__ import annotations
import argparse
import asyncio
import csv
import fcntl
import json
import os
import random
import sys
import tempfile
import traceback
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from graph.cli import add_epic_arguments, epic_kwargs, validate_epic_args
from graph.randomness import seed_everything as seed_runtime

DEFAULT_BENCHMARK_DATA = REPO_ROOT / "UniGenBench/data/test_prompts_en.csv"
DEFAULT_OUTDIR = REPO_ROOT / "outputs/unigenbench_results/EPIC_en_short"
DEFAULT_EVAL_DATA_DIR = REPO_ROOT / "UniGenBench/eval_data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate UniGenBench English-short images with EPIC (Ours).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\nThe generated evaluation files can be scored from the UniGenBench checkout:\n\n  cd UniGenBench\n  bash eval/eval_gemini.sh --model EPIC --categories en --images_per_prompt 1\n  bash eval/eval_vllm.sh --model EPIC --categories en --images_per_prompt 1\n",
    )
    add_epic_arguments(parser)
    parser.add_argument(
        "--benchmark_data",
        type=str,
        default=str(DEFAULT_BENCHMARK_DATA),
        help="UniGenBench English-short CSV.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(DEFAULT_OUTDIR),
        help="Directory for raw images, agent results, and run manifests.",
    )
    parser.add_argument(
        "--eval_data_dir",
        type=str,
        default=str(DEFAULT_EVAL_DATA_DIR),
        help="Base eval_data directory consumed by UniGenBench evaluation.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="EPIC",
        help="Model directory name under eval_data/en.",
    )
    parser.add_argument(
        "--n_samples",
        "--images_per_prompt",
        dest="n_samples",
        type=int,
        choices=(1,),
        default=1,
        help="Final images per prompt. This runner intentionally supports only 1.",
    )
    parser.add_argument(
        "--skip_grid",
        action="store_true",
        help="Do not save a contact sheet for each prompt.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Inclusive benchmark prompt ID to start from.",
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=None,
        help="Exclusive benchmark prompt ID to stop at.",
    )
    parser.add_argument(
        "--n_shard", type=int, default=1, help="Number of round-robin prompt shards."
    )
    parser.add_argument(
        "--shard_idx", type=int, default=0, help="Zero-based round-robin shard index."
    )
    parser.add_argument(
        "--shuffle_prompts",
        action="store_true",
        help="Shuffle selected prompt order without changing prompt IDs.",
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=None,
        help="Seed for --shuffle_prompts; defaults to --seed.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=1,
        help="Generation attempts per image before aborting.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate the CSV and write manifests without loading generation models.",
    )
    return parser.parse_args()


def load_main_graph_runner():
    try:
        from graph.graph_main import async_generate_with_prompt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Could not import graph.graph_main. Install the project runtime dependencies before generation."
        ) from exc
    return async_generate_with_prompt


def load_benchmark_entries(benchmark_data: str) -> list[tuple[int, dict]]:
    entries: list[tuple[int, dict]] = []
    seen_ids: set[int] = set()
    with open(benchmark_data, encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        required = {"index", "prompt_en", "sub_dims_en"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Missing required CSV columns: {', '.join(sorted(missing))}."
            )
        for row_number, row in enumerate(reader, start=2):
            try:
                prompt_index = int(row["index"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid integer index on CSV row {row_number}: {row['index']!r}."
                ) from exc
            if prompt_index in seen_ids:
                raise ValueError(f"Duplicate prompt index {prompt_index}.")
            prompt = (row["prompt_en"] or "").strip()
            if not prompt:
                raise ValueError(f"Empty prompt_en on CSV row {row_number}.")
            try:
                sub_dims = json.loads(row["sub_dims_en"])
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid sub_dims_en JSON on CSV row {row_number}."
                ) from exc
            seen_ids.add(prompt_index)
            entries.append(
                (
                    prompt_index,
                    {"index": prompt_index, "prompt": prompt, "sub_dims_en": sub_dims},
                )
            )
    entries.sort(key=lambda item: item[0])
    return entries


def validate_args(opt: argparse.Namespace) -> None:
    validate_epic_args(opt)
    if opt.n_samples != 1:
        raise ValueError("--n_samples/--images_per_prompt must be exactly 1.")
    if opt.start_index < 0:
        raise ValueError("--start_index must be >= 0.")
    if opt.end_index is not None and opt.end_index < opt.start_index:
        raise ValueError("--end_index must be >= --start_index.")
    if opt.n_shard < 1:
        raise ValueError("--n_shard must be at least 1.")
    if not 0 <= opt.shard_idx < opt.n_shard:
        raise ValueError("--shard_idx must satisfy 0 <= shard_idx < n_shard.")
    if opt.max_retries < 1:
        raise ValueError("--max_retries must be at least 1.")
    if (
        not opt.model_name
        or opt.model_name in {".", ".."}
        or Path(opt.model_name).name != opt.model_name
    ):
        raise ValueError("--model_name must be one non-empty directory name.")


def prompt_in_requested_slice(opt: argparse.Namespace, prompt_index: int) -> bool:
    if prompt_index < opt.start_index:
        return False
    if opt.end_index is not None and prompt_index >= opt.end_index:
        return False
    return prompt_index % opt.n_shard == opt.shard_idx


def maybe_shuffle_entries(
    opt: argparse.Namespace, selected_entries: list[tuple[int, dict]]
) -> tuple[list[tuple[int, dict]], int | None]:
    if not opt.shuffle_prompts:
        return (selected_entries, None)
    shuffle_seed = opt.shuffle_seed if opt.shuffle_seed is not None else opt.seed
    shuffled = list(selected_entries)
    random.Random(shuffle_seed).shuffle(shuffled)
    return (shuffled, shuffle_seed)


def get_raw_dir(outdir: str) -> Path:
    return Path(outdir) / "raw"


def get_prompt_dir(outdir: str, prompt_index: int) -> Path:
    return get_raw_dir(outdir) / f"{prompt_index:05d}"


def get_raw_sample_path(outdir: str, prompt_index: int, sample_index: int) -> Path:
    return get_prompt_dir(outdir, prompt_index) / "samples" / f"{sample_index:05d}.png"


def get_grid_path(outdir: str, prompt_index: int) -> Path:
    return get_prompt_dir(outdir, prompt_index) / "grid.png"


def get_metadata_path(outdir: str, prompt_index: int) -> Path:
    return get_prompt_dir(outdir, prompt_index) / "metadata.json"


def get_agent_result_path(outdir: str, sample_seed: int, prompt_index: int) -> Path:
    return (
        get_raw_dir(outdir)
        / "agent_results"
        / f"seed_{sample_seed}"
        / f"{prompt_index:06d}_0_result.json"
    )


def get_eval_model_dir(eval_data_dir: str, model_name: str) -> Path:
    return Path(eval_data_dir) / "en" / model_name


def get_eval_sample_path(
    eval_data_dir: str, model_name: str, prompt_index: int, sample_index: int
) -> Path:
    return (
        get_eval_model_dir(eval_data_dir, model_name)
        / f"{prompt_index}_{sample_index}.png"
    )


def atomic_write_json(path: str | Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    (fd, tmp_path) = tempfile.mkstemp(
        prefix=".tmp_json_", suffix=".json", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@contextmanager
def locked_manifest_update(outdir: str):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    lock_path = Path(outdir) / ".unigenbench_manifest.lock"
    with open(lock_path, "a+", encoding="utf-8") as lock_fp:
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)


@contextmanager
def event_loop_runner():
    """Yield a coroutine runner compatible with Python 3.9 and newer."""
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        yield loop.run_until_complete
    finally:
        loop.close()
        asyncio.set_event_loop(None)


def prompt_already_processed(opt: argparse.Namespace, prompt_index: int) -> bool:
    agent_results_exist = all(
        (
            get_agent_result_path(
                opt.outdir, opt.seed + sample_index, prompt_index
            ).exists()
            for sample_index in range(opt.n_samples)
        )
    )
    raw_samples_exist = all(
        (
            get_raw_sample_path(opt.outdir, prompt_index, sample_index).exists()
            for sample_index in range(opt.n_samples)
        )
    )
    eval_samples_exist = all(
        (
            get_eval_sample_path(
                opt.eval_data_dir, opt.model_name, prompt_index, sample_index
            ).exists()
            for sample_index in range(opt.n_samples)
        )
    )
    metadata_exists = get_metadata_path(opt.outdir, prompt_index).exists()
    grid_exists = opt.skip_grid or get_grid_path(opt.outdir, prompt_index).exists()
    return (
        agent_results_exist
        and raw_samples_exist
        and eval_samples_exist
        and metadata_exists
        and grid_exists
    )


def save_prompt_metadata(
    *, opt: argparse.Namespace, prompt_index: int, entry: dict
) -> None:
    metadata = {
        **entry,
        "benchmark_data": str(Path(opt.benchmark_data).resolve()),
        "n_samples": opt.n_samples,
        "seeds": [opt.seed + i for i in range(opt.n_samples)],
        "eval_images": [
            str(
                get_eval_sample_path(
                    opt.eval_data_dir, opt.model_name, prompt_index, sample_index
                ).resolve()
            )
            for sample_index in range(opt.n_samples)
        ],
    }
    atomic_write_json(get_metadata_path(opt.outdir, prompt_index), metadata)


def save_failure_report(
    *,
    opt: argparse.Namespace,
    prompt_index: int,
    sample_index: int,
    prompt: str,
    sample_seed: int,
    attempt: int,
    error: Exception,
) -> None:
    payload = {
        "prompt_index": prompt_index,
        "sample_index": sample_index,
        "prompt": prompt,
        "sample_seed": sample_seed,
        "attempt": attempt,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
    }
    atomic_write_json(
        get_prompt_dir(opt.outdir, prompt_index)
        / f"failure.sample_{sample_index:05d}.json",
        payload,
    )


def get_run_manifest_path(opt: argparse.Namespace) -> Path:
    if opt.n_shard == 1:
        name = "run_manifest"
    else:
        name = f"run_manifest.shard_{opt.shard_idx:05d}_of_{opt.n_shard:05d}"
    if opt.start_index != 0 or opt.end_index is not None:
        end_label = "end" if opt.end_index is None else f"{opt.end_index:05d}"
        name += f".range_{opt.start_index:05d}_{end_label}"
    return Path(opt.outdir) / f"{name}.json"


def write_run_manifest(
    *,
    opt: argparse.Namespace,
    benchmark_entries: list[tuple[int, dict]],
    selected_entries: list[tuple[int, dict]],
    shuffle_seed: int | None,
) -> None:
    eval_model_dir = get_eval_model_dir(opt.eval_data_dir, opt.model_name).resolve()
    payload = {
        "benchmark": "UniGenBench English Short",
        "method": "EPIC (Ours)",
        "benchmark_data": str(Path(opt.benchmark_data).resolve()),
        "outdir": str(Path(opt.outdir).resolve()),
        "eval_model_dir": str(eval_model_dir),
        "model_name": opt.model_name,
        "n_samples": opt.n_samples,
        "seed": opt.seed,
        "n_shard": opt.n_shard,
        "shard_idx": opt.shard_idx,
        "start_index": opt.start_index,
        "end_index": opt.end_index,
        "shuffle_prompts": opt.shuffle_prompts,
        "shuffle_seed": shuffle_seed,
        "llm_max_retries": opt.llm_max_retries,
        "num_benchmark_prompts": len(benchmark_entries),
        "num_selected_prompts": len(selected_entries),
        "expected_total_images": len(benchmark_entries) * opt.n_samples,
        "filename_pattern": "{promptID}_{imageID}.png",
        "run_order": [
            {
                "run_position": position,
                "prompt_index": prompt_index,
                "prompt": entry["prompt"],
            }
            for (position, (prompt_index, entry)) in enumerate(
                selected_entries, start=1
            )
        ],
    }
    atomic_write_json(get_run_manifest_path(opt), payload)


def write_eval_manifest(
    *, opt: argparse.Namespace, benchmark_entries: list[tuple[int, dict]]
) -> dict:
    with locked_manifest_update(opt.outdir):
        completed_images = []
        completed_prompts = 0
        for prompt_index, entry in benchmark_entries:
            prompt_images = []
            for sample_index in range(opt.n_samples):
                image_path = get_eval_sample_path(
                    opt.eval_data_dir, opt.model_name, prompt_index, sample_index
                )
                if image_path.exists():
                    prompt_images.append(
                        {
                            "sample_index": sample_index,
                            "path": str(image_path.resolve()),
                        }
                    )
            if len(prompt_images) == opt.n_samples:
                completed_prompts += 1
            if prompt_images:
                completed_images.append(
                    {
                        "prompt_index": prompt_index,
                        "prompt": entry["prompt"],
                        "images": prompt_images,
                    }
                )
        num_completed_images = sum(
            (len(record["images"]) for record in completed_images)
        )
        expected_images = len(benchmark_entries) * opt.n_samples
        payload = {
            "benchmark": "UniGenBench English Short",
            "method": "EPIC (Ours)",
            "benchmark_data": str(Path(opt.benchmark_data).resolve()),
            "model_name": opt.model_name,
            "eval_model_dir": str(
                get_eval_model_dir(opt.eval_data_dir, opt.model_name).resolve()
            ),
            "filename_pattern": "{promptID}_{imageID}.png",
            "num_prompts": len(benchmark_entries),
            "images_per_prompt": opt.n_samples,
            "expected_images": expected_images,
            "completed_images": num_completed_images,
            "completed_prompts": completed_prompts,
            "missing_images": expected_images - num_completed_images,
            "prompts_with_images": completed_images,
        }
        atomic_write_json(Path(opt.outdir) / "eval_manifest.json", payload)
    return payload


def save_grid(images: list, path: Path) -> None:
    from PIL import Image

    if not images:
        return
    normalized = [image.convert("RGB") for image in images]
    width = max((image.width for image in normalized))
    height = max((image.height for image in normalized))
    columns = min(4, len(normalized))
    rows = (len(normalized) + columns - 1) // columns
    grid = Image.new("RGB", (columns * width, rows * height), "white")
    for image_index, image in enumerate(normalized):
        x = image_index % columns * width
        y = image_index // columns * height
        grid.paste(image, (x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)


def build_epic_kwargs(*, opt, prompt, prompt_index, sample_seed):
    return {
        "prompt": prompt,
        "seed": sample_seed,
        "save_idx": prompt_index,
        "saving_path": str(
            get_raw_dir(opt.outdir) / "agent_results" / f"seed_{sample_seed}"
        ),
        **epic_kwargs(opt),
    }


def main(opt: argparse.Namespace) -> None:
    validate_args(opt)
    benchmark_entries = load_benchmark_entries(opt.benchmark_data)
    if not benchmark_entries:
        raise ValueError("The benchmark CSV contains no prompts.")
    selected_entries = [
        (prompt_index, entry)
        for (prompt_index, entry) in benchmark_entries
        if prompt_in_requested_slice(opt, prompt_index)
    ]
    (selected_entries, shuffle_seed) = maybe_shuffle_entries(opt, selected_entries)
    write_run_manifest(
        opt=opt,
        benchmark_entries=benchmark_entries,
        selected_entries=selected_entries,
        shuffle_seed=shuffle_seed,
    )
    eval_summary = write_eval_manifest(opt=opt, benchmark_entries=benchmark_entries)
    print(
        f"Loaded {len(benchmark_entries)} UniGenBench English-short prompts; selected {len(selected_entries)} for shard {opt.shard_idx}/{opt.n_shard}."
    )
    print(
        f"Eval-ready images: {eval_summary['completed_images']}/{eval_summary['expected_images']} under {eval_summary['eval_model_dir']}."
    )
    if opt.shuffle_prompts:
        print(f"Shuffled selected prompt order with seed {shuffle_seed}.")
    if opt.dry_run:
        print("Dry run complete; no generation model was loaded.")
        return
    import torch
    from tqdm import tqdm, trange

    async_generate_with_prompt = load_main_graph_runner()
    eval_model_dir = get_eval_model_dir(opt.eval_data_dir, opt.model_name)
    eval_model_dir.mkdir(parents=True, exist_ok=True)
    with event_loop_runner() as run_async:
        for run_position, (prompt_index, entry) in enumerate(
            tqdm(
                selected_entries, total=len(selected_entries), desc="Prompts Progress"
            ),
            start=1,
        ):
            if prompt_already_processed(opt, prompt_index):
                continue
            raw_sample_dir = get_prompt_dir(opt.outdir, prompt_index) / "samples"
            raw_sample_dir.mkdir(parents=True, exist_ok=True)
            save_prompt_metadata(opt=opt, prompt_index=prompt_index, entry=entry)
            prompt = entry["prompt"]
            print(
                f"Prompt ({run_position: >3}/{len(selected_entries)}, benchmark #{prompt_index: >3}): '{prompt}'"
            )
            generated_images = []
            with torch.no_grad():
                for sample_index in trange(opt.n_samples, desc="Sampling"):
                    sample_seed = opt.seed + sample_index
                    seed_runtime(sample_seed)
                    last_error = None
                    for attempt in range(1, opt.max_retries + 1):
                        try:
                            sample = run_async(
                                async_generate_with_prompt(
                                    **build_epic_kwargs(
                                        opt=opt,
                                        prompt=prompt,
                                        prompt_index=prompt_index,
                                        sample_seed=sample_seed,
                                    )
                                )
                            )
                            break
                        except Exception as exc:
                            last_error = exc
                            save_failure_report(
                                opt=opt,
                                prompt_index=prompt_index,
                                sample_index=sample_index,
                                prompt=prompt,
                                sample_seed=sample_seed,
                                attempt=attempt,
                                error=exc,
                            )
                            print(
                                f"  Prompt {prompt_index}, sample {sample_index} failed on attempt {attempt}/{opt.max_retries}: {exc}"
                            )
                    else:
                        raise RuntimeError(
                            f"Failed prompt {prompt_index}, sample {sample_index} after {opt.max_retries} attempts."
                        ) from last_error
                    raw_sample_path = get_raw_sample_path(
                        opt.outdir, prompt_index, sample_index
                    )
                    eval_sample_path = get_eval_sample_path(
                        opt.eval_data_dir, opt.model_name, prompt_index, sample_index
                    )
                    raw_sample_path.parent.mkdir(parents=True, exist_ok=True)
                    eval_sample_path.parent.mkdir(parents=True, exist_ok=True)
                    sample.save(raw_sample_path)
                    sample.save(eval_sample_path)
                    if not opt.skip_grid:
                        generated_images.append(sample.copy())
            if not opt.skip_grid:
                save_grid(generated_images, get_grid_path(opt.outdir, prompt_index))
            eval_summary = write_eval_manifest(
                opt=opt, benchmark_entries=benchmark_entries
            )
    print(
        f"Done. Eval-ready images: {eval_summary['completed_images']}/{eval_summary['expected_images']} ({eval_summary['completed_prompts']}/{eval_summary['num_prompts']} complete prompts)."
    )


if __name__ == "__main__":
    main(parse_args())
