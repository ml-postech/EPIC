from __future__ import annotations

"""Utility & helper functions."""

import io
import re
import base64
import json
import os
import time
from copy import deepcopy
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from PIL import Image

from graph.timing import ensure_timing_log, finalize_timing_log, record_duration


def _normalize_base_url(base_url: str, *, append_v1: bool = False) -> str:
    normalized = base_url.strip()
    if "://" not in normalized:
        normalized = f"http://{normalized}"
    if append_v1 and not normalized.rstrip("/").endswith("/v1"):
        normalized = f"{normalized.rstrip('/')}/v1"
    return normalized


def load_chat_model(
    fully_specified_name: str,
    seed=42,
    base_url="http://127.0.0.1:8000",
    llm_temperature: float = 0.15,
) -> BaseChatModel:
    """Create the client for EPIC's OpenAI-compatible Qwen3-VL server."""
    from langchain_openai import ChatOpenAI

    provider, model = fully_specified_name.split("/", maxsplit=1)
    if provider != "openai":
        raise ValueError("EPIC requires an OpenAI-compatible VLM (openai/<model>).")
    return ChatOpenAI(
        base_url=_normalize_base_url(base_url, append_v1=True),
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        model=model,
        seed=seed,
        temperature=llm_temperature,
        max_retries=0,
    )


def decode_data_url_to_base64(data_url: str) -> str:
    """
    Decode a base64 data URL (e.g., 'data:image/jpeg;base64,...')
    into a base64 string
    """
    # Extract mime type and base64 content
    match = re.match(r"^data:(.*?);base64,(.*)$", data_url, re.DOTALL)
    if not match:
        raise ValueError("Invalid data URL format")

    mime_type, b64_data = match.group(1).lower(), match.group(2).strip()

    # Fix padding if necessary
    missing_padding = len(b64_data) % 4
    if missing_padding:
        b64_data += "=" * (4 - missing_padding)

    # Convert based on MIME type
    if mime_type.startswith("image/"):
        return b64_data
    else:
        raise ValueError(f"Unsupported MIME type: {mime_type}")


def data_url_to_image(
    data_url: str,
    timing_log=None,
    timing_component="data_url_to_image",
    timing_metadata=None,
) -> Image.Image:
    conversion_start = time.perf_counter()
    b64_data = decode_data_url_to_base64(data_url)
    img_bytes = base64.b64decode(b64_data)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    if timing_log is not None:
        record_duration(
            timing_log,
            "image_conversion_seconds",
            time.perf_counter() - conversion_start,
            component=timing_component,
            count_key="image_conversions",
            metadata=timing_metadata,
        )
    return img


def image_to_data_url(
    generated_image: Image.Image,
    timing_log=None,
    timing_component="image_to_data_url",
    timing_metadata=None,
) -> str:
    conversion_start = time.perf_counter()
    buf = io.BytesIO()
    generated_image.save(buf, format="PNG", optimize=True)
    mime = "image/png"
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    data_url = f"data:{mime};base64,{encoded}"
    if timing_log is not None:
        record_duration(
            timing_log,
            "image_conversion_seconds",
            time.perf_counter() - conversion_start,
            component=timing_component,
            count_key="image_conversions",
            metadata=timing_metadata,
        )
    return data_url


def ensure_directory(
    dir_path: str | os.PathLike,
    timing_log=None,
    timing_component="ensure_directory",
    timing_metadata=None,
) -> str:
    io_start = time.perf_counter()
    os.makedirs(dir_path, exist_ok=True)
    if timing_log is not None:
        record_duration(
            timing_log,
            "file_io_seconds",
            time.perf_counter() - io_start,
            component=timing_component,
            metadata=timing_metadata,
        )
    return str(dir_path)


def save_image_file(
    image: Image.Image,
    save_path: str | os.PathLike,
    *,
    timing_log=None,
    timing_component="save_image_file",
    timing_metadata=None,
    **save_kwargs,
) -> None:
    io_start = time.perf_counter()
    image.save(save_path, **save_kwargs)
    if timing_log is not None:
        record_duration(
            timing_log,
            "file_io_seconds",
            time.perf_counter() - io_start,
            component=timing_component,
            count_key="file_writes",
            metadata=timing_metadata,
        )


def write_text_file(
    save_path: str | os.PathLike,
    text: str,
    *,
    timing_log=None,
    timing_component="write_text_file",
    timing_metadata=None,
    encoding="utf-8",
) -> None:
    io_start = time.perf_counter()
    Path(save_path).write_text(text, encoding=encoding)
    if timing_log is not None:
        record_duration(
            timing_log,
            "file_io_seconds",
            time.perf_counter() - io_start,
            component=timing_component,
            count_key="file_writes",
            metadata=timing_metadata,
        )


def json_dumps_timed(
    value,
    *,
    timing_log=None,
    timing_component="json.dumps",
    timing_metadata=None,
    **json_kwargs,
) -> str:
    serialization_start = time.perf_counter()
    serialized = json.dumps(value, **json_kwargs)
    if timing_log is not None:
        record_duration(
            timing_log,
            "result_serialization_seconds",
            time.perf_counter() - serialization_start,
            component=timing_component,
            count_key="result_serializations",
            metadata=timing_metadata,
        )
    return serialized


def save_result(result, current_path_prefix: str, ident=4):
    """Saves the result dict to a JSON file, save images separately if any."""
    raw_timing_log = result.get("timing_log")
    timing_log = ensure_timing_log(raw_timing_log) if raw_timing_log else None

    executor_outputs = result.get("executor_outputs") or []

    # Save the output image if any
    output_images = result.pop("output_images", None)
    round_best_images = result.get("round_best_images") or []
    fallback_output_image_data_url = (
        round_best_images[-1] if round_best_images else None
    )
    decoded_output_images = []
    if output_images:
        for output_index, output_image_data_url in enumerate(output_images, start=1):
            output_image = data_url_to_image(
                output_image_data_url,
                timing_log=timing_log,
                timing_component="save_result:output_image_decode",
                timing_metadata={"output_index": output_index},
            )
            decoded_output_images.append(output_image)
        output_image_data_url = output_images[-1]
    else:
        output_image_data_url = fallback_output_image_data_url
    if decoded_output_images:
        output_image = decoded_output_images[-1]
        save_image_file(
            output_image,
            f"{current_path_prefix}_0_output.jpg",
            timing_log=timing_log,
            timing_component="save_result:output_image_write",
            timing_metadata={"output_index": len(decoded_output_images)},
        )
    elif output_image_data_url:
        output_image = data_url_to_image(
            output_image_data_url,
            timing_log=timing_log,
            timing_component="save_result:output_image_decode",
        )
        save_image_file(
            output_image,
            f"{current_path_prefix}_0_output.jpg",
            timing_log=timing_log,
            timing_component="save_result:output_image_write",
        )

    # Keep round-best images out of the JSON payload, but do not save them as
    # duplicate JPEG artifacts because each round already writes its sample image.
    result.pop("round_best_images", None)

    for current_executor_output_list in executor_outputs:
        for current_executor_output in current_executor_output_list:
            current_executor_output.pop("input_image_data_url", None)
            current_executor_output.pop("output_image_data_url", None)

    # Save the original prompt image if any
    save_original_prompt_image = result.pop("save_original_prompt_image", False)
    if save_original_prompt_image:
        original_prompt_image_data_url = result.pop("original_prompt_image", None)
        if original_prompt_image_data_url:
            original_prompt_image = data_url_to_image(
                original_prompt_image_data_url,
                timing_log=timing_log,
                timing_component="save_result:original_prompt_decode",
            )
            save_image_file(
                original_prompt_image,
                f"{current_path_prefix}_0_original_prompt.jpg",
                timing_log=timing_log,
                timing_component="save_result:original_prompt_write",
            )

    timing_log = finalize_timing_log(timing_log)
    llm_log = None
    if timing_log is not None:
        llm_log = deepcopy(timing_log.get("llm_logs", []))
        compact_timing_log = deepcopy(timing_log)
        compact_timing_log.pop("by_component", None)
        compact_timing_log.pop("events", None)
        compact_timing_log.pop("llm_logs", None)
        result["timing_log"] = compact_timing_log

    # Save the main result JSON
    result.pop("messages", None)

    output_file_path = f"{current_path_prefix}_0_result.json"
    result_json = json_dumps_timed(
        result,
        timing_log=None,
        timing_component="save_result:result_json",
        indent=ident,
    )
    write_text_file(output_file_path, result_json, timing_log=None)

    timing_file_path = None
    if timing_log is not None:
        timing_file_path = f"{current_path_prefix}_0_timing_log.json"
        timing_json = json_dumps_timed(
            timing_log,
            timing_log=None,
            timing_component="save_result:timing_json",
            indent=ident,
        )
        write_text_file(timing_file_path, timing_json, timing_log=None)

    llm_log_file_path = None
    if llm_log:
        llm_log_file_path = f"{current_path_prefix}_0_llm_log.json"
        llm_log_json = json_dumps_timed(
            llm_log,
            timing_log=None,
            timing_component="save_result:llm_log_json",
            indent=ident,
        )
        write_text_file(llm_log_file_path, llm_log_json, timing_log=None)

    print(f"Result saved to {output_file_path}")
    if timing_file_path is not None:
        print(f"Timing log saved to {timing_file_path}")
    if llm_log_file_path is not None:
        print(f"LLM log saved to {llm_log_file_path}")
