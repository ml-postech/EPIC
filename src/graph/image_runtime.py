"""Run EPIC image generation and editing with FLUX.2-klein-9B."""

from __future__ import annotations

import random
import time

import torch

from graph.models_dm import MODEL_NAME, get_klein_pipeline
from graph.timing import record_duration


MAX_SEED = 2**31 - 1


def get_generation_executor(state):
    """Return the EPIC generation callable and model name."""
    return generate_image, MODEL_NAME


def get_edit_model_name(state):
    """Return the model shared by generation and editing."""
    return MODEL_NAME


def get_edit_sampling_settings(state):
    """Reuse the generation sampling settings for Klein editing."""
    return state.gen_num_inference_steps, state.gen_guidance_scale


def generate_image(seed_offset, state, prompt, timing_metadata=None):
    """Generate one image with the experiment's deterministic seed schedule."""
    seed = random.Random(state.image_seed + seed_offset).randint(0, MAX_SEED)
    pipe = get_klein_pipeline(timing_log=state.timing_log)
    generation_start = time.perf_counter()
    generated_image = pipe(
        prompt=prompt,
        height=state.gen_height,
        width=state.gen_width,
        guidance_scale=state.gen_guidance_scale,
        num_inference_steps=state.gen_num_inference_steps,
        generator=torch.Generator("cpu").manual_seed(seed),
    ).images[0]
    record_duration(
        state.timing_log,
        "image_generation_seconds",
        time.perf_counter() - generation_start,
        component=MODEL_NAME,
        count_key="image_generations",
        metadata=timing_metadata,
    )
    return generated_image, seed


def import_and_execute_edit_pipe(
    seed_offset,
    state,
    prompt,
    input_image,
    timing_metadata=None,
):
    """Edit one RGB image using the shared Klein pipeline."""
    seed = random.Random(state.image_seed + seed_offset).randint(0, MAX_SEED)
    edit_input_image = input_image.convert("RGB")
    width, height = edit_input_image.size
    edit_num_inference_steps, edit_guidance_scale = get_edit_sampling_settings(state)
    generator_device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = get_klein_pipeline(timing_log=state.timing_log)
    edit_start = time.perf_counter()
    edited_image = pipe(
        image=edit_input_image,
        prompt=prompt,
        guidance_scale=edit_guidance_scale,
        width=width,
        height=height,
        num_inference_steps=edit_num_inference_steps,
        generator=torch.Generator(device=generator_device).manual_seed(seed),
        output_type="pil",
    ).images[0]
    record_duration(
        state.timing_log,
        "editing_seconds",
        time.perf_counter() - edit_start,
        component=MODEL_NAME,
        count_key="edit_calls",
        metadata=timing_metadata,
    )
    return edited_image, seed
