"""Prepare local T2I-CompBench prompt manifests from the official repo checkout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = REPO_ROOT / "benchmarks" / "T2I-CompBench" / "examples" / "dataset"
DEFAULT_OUTDIR = REPO_ROOT / "benchmarks" / "t2i_compbench"

ORIGINAL_CATEGORIES = (
    "color",
    "shape",
    "texture",
    "spatial",
    "non_spatial",
    "complex",
)
PLUSPLUS_CATEGORIES = (
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
    parser.add_argument(
        "--source_dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Official T2I-CompBench examples/dataset directory.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Directory to write local JSONL/TXT prompt manifests.",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=("val",),
        help="Split to prepare. Only val is used for benchmark evaluation.",
    )
    return parser.parse_args()


def read_prompt_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing T2I-CompBench prompt file: {path}")
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_txt(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{row['prompt']}\n" for row in rows),
        encoding="utf-8",
    )


def build_category_rows(
    *,
    source_dir: Path,
    category: str,
    split: str,
) -> list[dict]:
    source_file = source_dir / f"{category}_{split}.txt"
    prompts = read_prompt_file(source_file)
    return [
        {
            "prompt": prompt,
            "category": category,
            "split": split,
            "category_prompt_index": category_prompt_index,
            "source_file": str(source_file.resolve().relative_to(REPO_ROOT))
            if source_file.resolve().is_relative_to(REPO_ROOT)
            else str(source_file.resolve()),
        }
        for category_prompt_index, prompt in enumerate(prompts)
    ]


def assign_suite_indices(rows: list[dict], suite: str) -> list[dict]:
    output_rows = []
    for global_prompt_index, row in enumerate(rows):
        row = dict(row)
        row["suite"] = suite
        row["global_prompt_index"] = global_prompt_index
        output_rows.append(row)
    return output_rows


def write_readme(outdir: Path, category_counts: dict[str, int]) -> None:
    lines = [
        "# T2I-CompBench Local Manifests",
        "",
        "Generated from `benchmarks/T2I-CompBench/examples/dataset`.",
        "",
        "Recommended runs:",
        "",
        "- `t2i_compbench_val.jsonl`: original NeurIPS 2023-style six-category evaluation.",
        "- `t2i_compbenchpp_val.jsonl`: expanded T2I-CompBench++ eight-category evaluation.",
        "",
        "The official evaluator reads images from category-specific `samples/` folders",
        "and infers `question_id` from the last six digits in each image filename.",
        "Rows keep `category_prompt_index` so generation can produce deterministic",
        "`question_id = category_prompt_index * n_samples + sample_index` filenames.",
        "",
        "Category counts:",
        "",
    ]
    for category in PLUSPLUS_CATEGORIES:
        lines.append(f"- `{category}`: {category_counts[category]} prompts")
    lines.append("")
    outdir.joinpath("README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    outdir = args.outdir.resolve()

    category_rows = {
        category: build_category_rows(
            source_dir=source_dir,
            category=category,
            split=args.split,
        )
        for category in PLUSPLUS_CATEGORIES
    }

    for category, rows in category_rows.items():
        category_manifest_rows = assign_suite_indices(
            rows, suite=f"{category}_{args.split}"
        )
        write_jsonl(outdir / f"{category}_{args.split}.jsonl", category_manifest_rows)
        write_txt(outdir / f"{category}_{args.split}.txt", category_manifest_rows)

    original_rows = [
        row for category in ORIGINAL_CATEGORIES for row in category_rows[category]
    ]
    original_rows = assign_suite_indices(original_rows, suite="t2i_compbench_val")
    write_jsonl(outdir / "t2i_compbench_val.jsonl", original_rows)
    write_txt(outdir / "t2i_compbench_val.txt", original_rows)

    plusplus_rows = [
        row for category in PLUSPLUS_CATEGORIES for row in category_rows[category]
    ]
    plusplus_rows = assign_suite_indices(plusplus_rows, suite="t2i_compbenchpp_val")
    write_jsonl(outdir / "t2i_compbenchpp_val.jsonl", plusplus_rows)
    write_txt(outdir / "t2i_compbenchpp_val.txt", plusplus_rows)

    write_readme(
        outdir,
        {category: len(rows) for category, rows in category_rows.items()},
    )
    print(f"Wrote T2I-CompBench manifests to {outdir}")
    print(f"Original val prompts: {len(original_rows)}")
    print(f"T2I-CompBench++ val prompts: {len(plusplus_rows)}")


if __name__ == "__main__":
    main()
