"""Shared FLUX.2-klein-9B pipeline for EPIC generation and editing."""

import time

import torch

from graph.timing import record_duration


MODEL_ID = "black-forest-labs/FLUX.2-klein-9B"
MODEL_NAME = "FLUX.2-klein-9B"
_KLEIN_PIPELINE = None


def get_klein_pipeline(timing_log=None):
    """Load the shared pipeline once and reuse it for generation and editing."""
    global _KLEIN_PIPELINE
    if _KLEIN_PIPELINE is None:
        from diffusers import Flux2KleinPipeline

        load_start = time.perf_counter()
        _KLEIN_PIPELINE = Flux2KleinPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        record_duration(
            timing_log,
            "model_loading_seconds",
            time.perf_counter() - load_start,
            component=MODEL_NAME,
            count_key="model_loads",
        )
        print("Model loaded:", MODEL_NAME)
    return _KLEIN_PIPELINE
