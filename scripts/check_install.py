#!/usr/bin/env python3
"""Check EPIC imports, CUDA, and optionally the VLM endpoint without loading weights."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url", help="Also check this VLM server's /v1/models endpoint."
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow import-only checks without a CUDA GPU.",
    )
    args = parser.parse_args()
    failures = []
    checks = {
        "EPIC graph and packaged prompts": ("graph.graph_main", "graph_main"),
        "FLUX.2-klein": ("diffusers", "Flux2KleinPipeline"),
        "SigLIP": ("transformers", "SiglipModel"),
        "SAM3 builder": ("sam3.model_builder", "build_sam3_image_model"),
        "SAM3 processor": ("sam3.model.sam3_image_processor", "Sam3Processor"),
        "MiDaS wrapper": ("graph.depth", "MidasDepthEstimator"),
        "MiDaS image transforms": ("cv2", "resize"),
        "MiDaS backbone dependency": ("timm", "create_model"),
    }
    print(f"Python {sys.version.split()[0]}")
    for label, (module, attribute) in checks.items():
        try:
            getattr(importlib.import_module(module), attribute)
            print(f"OK: {label}")
        except Exception as exc:
            failures.append(label)
            print(f"FAIL: {label}: {type(exc).__name__}: {exc}")
    try:
        import torch

        print(f"PyTorch {torch.__version__}; CUDA runtime {torch.version.cuda}")
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                print(
                    f"GPU {index}: {properties.name}, {properties.total_memory / 2**30:.1f} GiB"
                )
        elif not args.allow_cpu:
            failures.append("CUDA")
            print("FAIL: CUDA is unavailable; image generation requires an NVIDIA GPU.")
    except Exception as exc:
        failures.append("PyTorch")
        print(f"FAIL: PyTorch: {exc}")
    if args.base_url:
        endpoint = args.base_url.strip().rstrip("/")
        if "://" not in endpoint:
            endpoint = "http://" + endpoint
        if not endpoint.endswith("/v1"):
            endpoint += "/v1"
        request = urllib.request.Request(endpoint + "/models")
        request.add_header(
            "Authorization", "Bearer " + os.environ.get("OPENAI_API_KEY", "EMPTY")
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                models = [item["id"] for item in json.load(response)["data"]]
            expected = os.environ.get(
                "MODEL", "openai/Qwen/Qwen3-VL-32B-Instruct"
            ).split("/", 1)[-1]
            if expected not in models:
                raise ValueError(f"Expected {expected!r}; server exposes {models!r}")
            print(f"OK: VLM server exposes {expected}")
        except Exception as exc:
            failures.append("VLM server")
            print(f"FAIL: VLM server: {exc}")
    print(
        "Checks failed: " + ", ".join(failures)
        if failures
        else "All requested checks passed (no model weights loaded)."
    )
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
