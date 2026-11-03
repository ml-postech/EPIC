#!/usr/bin/env python3
"""Prepare a JSONL prompt manifest for GenAI-Bench-1600.

The official Hugging Face dataset stores the prompt metadata and skill tags as
two small JSON files. This script downloads those files and converts them into
the same line-oriented manifest style used by the local benchmark runners.
"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_REPO_ID = "BaiqiL/GenAI-Bench-1600"
DEFAULT_OUTPUT = Path("benchmarks/genai_bench/genai_bench_1600.jsonl")
DEFAULT_METADATA_DIR = Path("benchmarks/genai_bench/source")

BASIC_SKILLS = (
    "attribute",
    "scene",
    "spatial relation",
    "action relation",
    "part relation",
)
ADVANCED_SKILLS = (
    "counting",
    "comparison",
    "differentiation",
    "negation",
    "universal",
)
REFERENCE_IMAGE_ARCHIVES = (
    "DALLE_3.zip",
    "DeepFloyd_I_XL_v1.zip",
    "Midjourney_6.zip",
    "SDXL_2_1.zip",
    "SDXL_Base.zip",
    "SDXL_Turbo.zip",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download GenAI-Bench-1600 metadata and build a JSONL manifest."
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face dataset repo. Default: {DEFAULT_REPO_ID}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL manifest. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=DEFAULT_METADATA_DIR,
        help=f"Directory for downloaded source JSON files. Default: {DEFAULT_METADATA_DIR}",
    )
    parser.add_argument(
        "--download-reference-images",
        action="store_true",
        help=(
            "Also download the official generated-image zip archives. This is not "
            "needed for generating new EPIC outputs."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing source JSON and manifest files.",
    )
    return parser.parse_args()


def download_with_hf_hub(repo_id: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
        )
    )


def download_with_urllib(repo_id: str, filename: str, destination: Path) -> Path:
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output)
    return destination


def download_file(
    *,
    repo_id: str,
    filename: str,
    destination: Path,
    force: bool,
) -> Path:
    if destination.exists() and not force:
        return destination

    try:
        cached_path = download_with_hf_hub(repo_id, filename)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached_path, destination)
    except Exception:
        download_with_urllib(repo_id, filename, destination)
    return destination


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def invert_skill_tags(skills: dict[str, list[int]]) -> dict[int, list[str]]:
    prompt_to_skills: dict[int, list[str]] = {}
    for skill, prompt_indices in skills.items():
        for prompt_index in prompt_indices:
            prompt_to_skills.setdefault(int(prompt_index), []).append(skill)
    for prompt_index in prompt_to_skills:
        prompt_to_skills[prompt_index].sort()
    return prompt_to_skills


def build_manifest_rows(
    *,
    image_metadata: dict[str, dict[str, Any]],
    skills: dict[str, list[int]],
    source_dataset: str,
) -> list[dict[str, Any]]:
    prompt_to_skills = invert_skill_tags(skills)
    rows: list[dict[str, Any]] = []

    for key in sorted(image_metadata, key=lambda value: int(value)):
        source_entry = image_metadata[key]
        prompt_index = int(source_entry.get("id", key))
        all_skills = prompt_to_skills.get(prompt_index, [])
        skill_groups = [group for group in ("basic", "advanced") if group in all_skills]
        basic_skills = [skill for skill in BASIC_SKILLS if skill in all_skills]
        advanced_skills = [skill for skill in ADVANCED_SKILLS if skill in all_skills]
        fine_grained_skills = basic_skills + advanced_skills

        rows.append(
            {
                "id": f"{prompt_index:05d}",
                "prompt_index": prompt_index,
                "prompt": source_entry["prompt"],
                "prompt_zh": source_entry.get("prompt in Chinese"),
                "skills": fine_grained_skills,
                "skill_groups": skill_groups,
                "basic_skills": basic_skills,
                "advanced_skills": advanced_skills,
                "human_ratings": source_entry.get("models", {}),
                "source_dataset": source_dataset,
            }
        )

    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            file.write("\n")


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    summary = {
        "all_1600": len(rows),
        "tagged_union": 0,
        "basic": 0,
        "advanced": 0,
    }
    for skill in BASIC_SKILLS + ADVANCED_SKILLS:
        summary[skill] = 0

    for row in rows:
        if row["skills"] or row["skill_groups"]:
            summary["tagged_union"] += 1
        if "basic" in row["skill_groups"]:
            summary["basic"] += 1
        if "advanced" in row["skill_groups"]:
            summary["advanced"] += 1
        for skill in row["skills"]:
            summary[skill] += 1
    return summary


def main() -> None:
    args = parse_args()
    args.metadata_dir.mkdir(parents=True, exist_ok=True)

    image_json_path = download_file(
        repo_id=args.repo_id,
        filename="genai_image.json",
        destination=args.metadata_dir / "genai_image.json",
        force=args.force,
    )
    skills_json_path = download_file(
        repo_id=args.repo_id,
        filename="genai_skills.json",
        destination=args.metadata_dir / "genai_skills.json",
        force=args.force,
    )

    if args.download_reference_images:
        archive_dir = args.metadata_dir / "reference_images"
        for archive_name in REFERENCE_IMAGE_ARCHIVES:
            path = download_file(
                repo_id=args.repo_id,
                filename=archive_name,
                destination=archive_dir / archive_name,
                force=args.force,
            )
            print(f"Downloaded reference archive: {path}")

    image_metadata = load_json(image_json_path)
    skills = load_json(skills_json_path)
    rows = build_manifest_rows(
        image_metadata=image_metadata,
        skills=skills,
        source_dataset=args.repo_id,
    )
    write_jsonl(args.output, rows)

    summary = summarize_rows(rows)
    print(f"Wrote {len(rows)} GenAI-Bench prompt rows to {args.output}")
    print("Skill counts:")
    for key in (
        "all_1600",
        "tagged_union",
        "basic",
        "advanced",
        *BASIC_SKILLS,
        *ADVANCED_SKILLS,
    ):
        print(f"  {key}: {summary[key]}")


if __name__ == "__main__":
    main()
