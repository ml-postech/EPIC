import time
import asyncio
import json
import os
from contextlib import suppress

import langchain_core
from graph.utils import load_chat_model
from graph.timing import (
    append_llm_log,
    record_duration,
    ensure_timing_log,
    increment_count,
)

DEFAULT_LLM_TIMEOUT_SECONDS = 180
PARSER_RETRY_SLEEP_SECONDS = 3.0
MAX_TIMEOUT_RETRY_SLEEP_SECONDS = 60.0
DEFAULT_LLM_MAX_RETRIES = 20

CHAT_MODEL = None
CHAT_MODEL_KEY = None


def get_llm_max_retries() -> int:
    raw_value = os.environ.get(
        "EPIC_LLM_MAX_RETRIES",
        str(DEFAULT_LLM_MAX_RETRIES),
    )
    try:
        retries = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"EPIC_LLM_MAX_RETRIES must be a non-negative integer, got {raw_value!r}."
        ) from exc
    if retries < 0:
        raise ValueError(
            f"EPIC_LLM_MAX_RETRIES must be a non-negative integer, got {retries}."
        )
    return retries


async def _sleep_with_timing(
    timing_log,
    duration_seconds,
    *,
    component,
    metadata=None,
):
    sleep_start = time.perf_counter()
    await asyncio.sleep(duration_seconds)
    if timing_log is not None:
        record_duration(
            timing_log,
            "retry_backoff_seconds",
            time.perf_counter() - sleep_start,
            component=component,
            count_key="retry_backoffs",
            metadata=metadata,
        )


def _timeout_retry_sleep_seconds(backoff: float, attempt: int) -> float:
    return min(backoff * (2**attempt), MAX_TIMEOUT_RETRY_SLEEP_SECONDS)


def _truncate_text(text, max_chars=2000):
    if text is None:
        return None
    text = str(text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _serialize_for_log(value, max_chars=4000):
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        serialized = json.dumps(
            value.model_dump(), ensure_ascii=False, indent=2, default=str
        )
    elif isinstance(value, dict):
        serialized = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    elif isinstance(value, (list, tuple)):
        serialized = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    else:
        serialized = str(value)
    return _truncate_text(serialized, max_chars=max_chars)


def _extract_tool_calls_text(message):
    tool_calls = getattr(message, "tool_calls", None)
    if not tool_calls:
        tool_calls = getattr(message, "additional_kwargs", {}).get("tool_calls", [])

    extracted = []
    for tool_call in tool_calls or []:
        function_data = tool_call.get("function", tool_call)
        function_name = function_data.get("name", tool_call.get("name", "unknown"))
        function_args = function_data.get("arguments", tool_call.get("args", ""))
        extracted.append(f"[tool_call:{function_name}] {function_args}")
    return "\n".join(extracted).strip()


def _extract_message_text(message):
    if message is None:
        return ""

    content = getattr(message, "content", "")
    content_pieces = []

    if isinstance(content, str):
        if content.strip():
            content_pieces.append(content.strip())
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, str):
                if item.strip():
                    content_pieces.append(item.strip())
            elif isinstance(item, dict):
                text_value = item.get("text")
                if text_value:
                    content_pieces.append(str(text_value).strip())

    tool_calls_text = _extract_tool_calls_text(message)
    if tool_calls_text:
        content_pieces.append(tool_calls_text)

    return "\n".join(piece for piece in content_pieces if piece).strip()


def _extract_input_preview(inp, max_chars=1500):
    preview_parts = []
    for message in inp:
        if isinstance(message, dict):
            role = message.get("role", "unknown")
            content = _format_content_preview(message.get("content", ""))
            preview_parts.append(f"[{role}] {content}")
        else:
            role = getattr(message, "type", message.__class__.__name__)
            content = _format_content_preview(getattr(message, "content", ""))
            preview_parts.append(f"[{role}] {content}")
    return _truncate_text("\n\n".join(preview_parts), max_chars=max_chars)


def _coerce_mapping(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        with suppress(Exception):
            return value.model_dump()
    if hasattr(value, "dict"):
        with suppress(Exception):
            return value.dict()
    return {}


def _coerce_int(value):
    if value is None:
        return None
    with suppress(TypeError, ValueError):
        return int(value)
    return None


def _normalize_token_usage(mapping):
    mapping = _coerce_mapping(mapping)
    if not mapping:
        return None
    input_tokens = None
    for key in (
        "input_tokens",
        "prompt_tokens",
        "input_token_count",
        "prompt_token_count",
        "prompt_eval_count",
    ):
        input_tokens = _coerce_int(mapping.get(key))
        if input_tokens is not None:
            break
    output_tokens = None
    for key in (
        "output_tokens",
        "completion_tokens",
        "output_token_count",
        "completion_token_count",
        "eval_count",
    ):
        output_tokens = _coerce_int(mapping.get(key))
        if output_tokens is not None:
            break
    total_tokens = None
    for key in ("total_tokens", "total_token_count"):
        total_tokens = _coerce_int(mapping.get(key))
        if total_tokens is not None:
            break
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _extract_token_usage(message):
    if message is None:
        return {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "token_usage_source": None,
        }

    message_mapping = _coerce_mapping(message)
    response_metadata = _coerce_mapping(getattr(message, "response_metadata", None))
    additional_kwargs = _coerce_mapping(getattr(message, "additional_kwargs", None))
    candidate_sources = (
        ("message.usage_metadata", message_mapping.get("usage_metadata")),
        ("message.token_usage", message_mapping.get("token_usage")),
        ("message.usage", message_mapping.get("usage")),
        ("message", message_mapping),
        ("usage_metadata", getattr(message, "usage_metadata", None)),
        ("response_metadata.token_usage", response_metadata.get("token_usage")),
        ("response_metadata.usage", response_metadata.get("usage")),
        ("response_metadata", response_metadata),
        ("additional_kwargs.usage", additional_kwargs.get("usage")),
        ("additional_kwargs.token_usage", additional_kwargs.get("token_usage")),
        ("additional_kwargs", additional_kwargs),
    )
    for source, candidate in candidate_sources:
        normalized = _normalize_token_usage(candidate)
        if normalized is not None:
            return {**normalized, "token_usage_source": source}
    return {
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "token_usage_source": None,
    }


def _count_image_inputs_in_content(content):
    if isinstance(content, list):
        count = 0
        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type == "image_url" or "image_url" in item:
                    count += 1
                elif item_type in {"image", "input_image"} or "image" in item:
                    count += 1
        return count
    if isinstance(content, dict):
        item_type = content.get("type")
        if item_type == "image_url" or "image_url" in content:
            return 1
        if item_type in {"image", "input_image"} or "image" in content:
            return 1
    return 0


def _count_image_inputs(inp):
    image_inputs = 0
    for message in inp:
        if isinstance(message, dict):
            image_inputs += _count_image_inputs_in_content(message.get("content", ""))
        else:
            image_inputs += _count_image_inputs_in_content(
                getattr(message, "content", "")
            )
    return image_inputs


def _format_content_preview(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        blocks = []
        for item in content:
            if isinstance(item, str):
                if item.strip():
                    blocks.append(item.strip())
                continue
            if not isinstance(item, dict):
                blocks.append(str(item))
                continue
            item_type = item.get("type")
            if item_type == "text":
                text_value = str(item.get("text", "")).strip()
                if text_value:
                    blocks.append(text_value)
            elif item_type == "image_url" or "image_url" in item:
                blocks.append("<image_url>")
            elif item_type in {"image", "input_image"} or "image" in item:
                blocks.append("<image>")
            else:
                blocks.append(f"<{item_type or 'content_block'}>")
        return "\n".join(block for block in blocks if block)
    if isinstance(content, dict):
        return _format_content_preview([content])
    return str(content)


def _log_llm_response(
    state,
    *,
    component_name,
    attempt,
    seed,
    timeout,
    request_duration,
    raw_text,
    parsed_output=None,
    parsing_error=None,
    input_preview=None,
    image_inputs=0,
    model_name=None,
    output_schema=None,
    token_usage=None,
):
    raw_preview = _truncate_text(raw_text, max_chars=600)
    parsed_preview = _serialize_for_log(parsed_output, max_chars=1200)
    parsing_error_text = (
        None
        if parsing_error is None
        else _truncate_text(
            f"{type(parsing_error).__name__}: {parsing_error}", max_chars=1200
        )
    )

    append_llm_log(
        state.timing_log,
        {
            "component": component_name,
            "attempt": attempt,
            "seed": seed,
            "timeout_seconds": timeout,
            "response_seconds": round(request_duration, 6),
            "status": "parsed" if parsing_error is None else "parser_error",
            "model_name": model_name,
            "output_schema": output_schema,
            "image_inputs": int(image_inputs),
            "input_tokens": None
            if token_usage is None
            else token_usage.get("input_tokens"),
            "output_tokens": None
            if token_usage is None
            else token_usage.get("output_tokens"),
            "total_tokens": None
            if token_usage is None
            else token_usage.get("total_tokens"),
            "token_usage_source": (
                None if token_usage is None else token_usage.get("token_usage_source")
            ),
            "input_preview": input_preview,
            "raw_text": _truncate_text(raw_text, max_chars=12000),
            "parsed_output_preview": parsed_preview,
            "parsing_error": parsing_error_text,
        },
    )

    if raw_preview:
        print(
            f"{component_name} | attempt {attempt} | raw response preview: {raw_preview}"
        )
    if parsed_preview:
        print(
            f"{component_name} | attempt {attempt} | parsed output preview: {parsed_preview}"
        )
    if parsing_error_text:
        print(
            f"{component_name} | attempt {attempt} | parsing error: {parsing_error_text}"
        )


def get_chat_model(
    model,
    seed,
    base_url,
    llm_temperature,
    timing_log=None,
):
    global CHAT_MODEL, CHAT_MODEL_KEY

    cache_key = (model, seed, base_url, float(llm_temperature))

    start_time = time.time()
    if cache_key == CHAT_MODEL_KEY and CHAT_MODEL is not None:
        print_load = False
    else:
        timing_log = ensure_timing_log(timing_log)

        client_init_start = time.perf_counter()
        CHAT_MODEL = load_chat_model(
            fully_specified_name=model,
            seed=seed,
            base_url=base_url,
            llm_temperature=llm_temperature,
        )
        client_init_duration = time.perf_counter() - client_init_start
        record_duration(
            timing_log,
            "model_loading_seconds",
            client_init_duration,
            component=f"{model}:client_init",
            count_key="model_loads",
        )

        CHAT_MODEL_KEY = cache_key
        print_load = True
    end_time = time.time()

    duration = "%.4f" % (end_time - start_time)
    if print_load:
        print(
            "Model loaded:",
            model,
            "| with seed:",
            seed,
            "| temperature:",
            llm_temperature,
            "| Time taken:",
            duration,
            "seconds",
        )

    return CHAT_MODEL


async def ainvoke_with_timeout_retry(
    state,
    runtime,
    inp,
    output_class=None,
    timeout=DEFAULT_LLM_TIMEOUT_SECONDS,
    retries=None,
    backoff=0.6,
    shield=False,
    component_name="llm",
):
    if retries is None:
        retries = get_llm_max_retries()
    output_schema = None if output_class is None else output_class.__name__
    for attempt in range(retries + 1):
        # Get the model. For attempt > 0, recreate the model with a different seed to avoid infinite output
        seed = runtime.context.seed + attempt
        state.timing_log = ensure_timing_log(state.timing_log)

        model = get_chat_model(
            model=runtime.context.model,
            seed=seed,
            base_url=runtime.context.base_url,
            llm_temperature=runtime.context.llm_temperature,
            timing_log=state.timing_log,
        )
        if output_class:
            provider, _ = runtime.context.model.split("/", maxsplit=1)
            structured_output_kwargs = {"include_raw": True}
            if provider == "openai":
                structured_output_kwargs["method"] = "function_calling"
            model = model.with_structured_output(
                output_class, **structured_output_kwargs
            )

        # Create the task
        if attempt > 0:
            print(f"Retrying... attempt {attempt} of {retries}")
            inp[0]["content"] += (
                f"\n\nRetry attempt: {attempt} out of {retries}. Your previous response was repeating and infinite. Please be concise, do not output infinite repeated content or a long list. Follow the output format"
            )
        input_preview_start = time.perf_counter()
        input_preview = _extract_input_preview(inp)
        image_inputs = _count_image_inputs(inp)
        record_duration(
            state.timing_log,
            "llm_bookkeeping_seconds",
            time.perf_counter() - input_preview_start,
            component=f"{component_name}:input_preview",
            metadata={"attempt": attempt},
        )
        task = asyncio.create_task(model.ainvoke(inp))

        # Run the task
        request_start = time.perf_counter()
        try:
            coro = asyncio.wait_for(task, timeout=timeout)
            response = await (asyncio.shield(coro) if shield else coro)
            request_duration = time.perf_counter() - request_start
            if output_class:
                raw_message = response.get("raw")
                parsed_output = response.get("parsed")
                parsing_error = response.get("parsing_error")
                postprocess_start = time.perf_counter()
                raw_text = _extract_message_text(raw_message)
                token_usage = _extract_token_usage(raw_message)
                _log_llm_response(
                    state,
                    component_name=component_name,
                    attempt=attempt,
                    seed=seed,
                    timeout=timeout,
                    request_duration=request_duration,
                    raw_text=raw_text,
                    parsed_output=parsed_output,
                    parsing_error=parsing_error,
                    input_preview=input_preview,
                    image_inputs=image_inputs,
                    model_name=runtime.context.model,
                    output_schema=output_schema,
                    token_usage=token_usage,
                )
                record_duration(
                    state.timing_log,
                    "llm_bookkeeping_seconds",
                    time.perf_counter() - postprocess_start,
                    component=f"{component_name}:response_logging",
                    metadata={"attempt": attempt},
                )
                if parsing_error is not None:
                    record_duration(
                        state.timing_log,
                        "llm_communication_seconds",
                        request_duration,
                        component=f"{component_name}:parser_retry",
                        count_key="llm_requests",
                        metadata={"attempt": attempt, "timeout_seconds": timeout},
                    )
                    if attempt == retries:
                        raise parsing_error
                    await _sleep_with_timing(
                        state.timing_log,
                        PARSER_RETRY_SLEEP_SECONDS,
                        component=component_name,
                        metadata={"attempt": attempt, "reason": "parser_retry"},
                    )
                    continue

                record_duration(
                    state.timing_log,
                    "llm_communication_seconds",
                    request_duration,
                    component=component_name,
                    count_key="llm_requests",
                    metadata={"attempt": attempt, "timeout_seconds": timeout},
                )
                return parsed_output

            postprocess_start = time.perf_counter()
            raw_text = _extract_message_text(response)
            token_usage = _extract_token_usage(response)
            _log_llm_response(
                state,
                component_name=component_name,
                attempt=attempt,
                seed=seed,
                timeout=timeout,
                request_duration=request_duration,
                raw_text=raw_text,
                parsed_output=None,
                parsing_error=None,
                input_preview=input_preview,
                image_inputs=image_inputs,
                model_name=runtime.context.model,
                output_schema=output_schema,
                token_usage=token_usage,
            )
            record_duration(
                state.timing_log,
                "llm_bookkeeping_seconds",
                time.perf_counter() - postprocess_start,
                component=f"{component_name}:response_logging",
                metadata={"attempt": attempt},
            )
            record_duration(
                state.timing_log,
                "llm_communication_seconds",
                request_duration,
                component=component_name,
                count_key="llm_requests",
                metadata={"attempt": attempt, "timeout_seconds": timeout},
            )
            return response

        except asyncio.TimeoutError:
            request_duration = time.perf_counter() - request_start
            append_llm_log(
                state.timing_log,
                {
                    "component": component_name,
                    "attempt": attempt,
                    "seed": seed,
                    "timeout_seconds": timeout,
                    "response_seconds": round(request_duration, 6),
                    "status": "timeout",
                    "model_name": runtime.context.model,
                    "output_schema": output_schema,
                    "image_inputs": int(image_inputs),
                    "input_tokens": None,
                    "output_tokens": None,
                    "total_tokens": None,
                    "token_usage_source": None,
                    "input_preview": input_preview,
                    "raw_text": None,
                    "parsed_output_preview": None,
                    "parsing_error": None,
                },
            )
            print(
                f"{component_name} | attempt {attempt} | timeout after {request_duration:.2f}s"
            )
            record_duration(
                state.timing_log,
                "llm_communication_seconds",
                request_duration,
                component=f"{component_name}:timeout",
                count_key="llm_requests",
                metadata={"attempt": attempt, "timeout_seconds": timeout},
            )
            increment_count(state.timing_log, "llm_timeouts")
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task  # ensure proper cleanup
            if attempt == retries:
                raise
            await _sleep_with_timing(
                state.timing_log,
                _timeout_retry_sleep_seconds(backoff, attempt),
                component=component_name,
                metadata={"attempt": attempt, "reason": "timeout"},
            )

        except langchain_core.exceptions.OutputParserException:
            request_duration = time.perf_counter() - request_start
            append_llm_log(
                state.timing_log,
                {
                    "component": component_name,
                    "attempt": attempt,
                    "seed": seed,
                    "timeout_seconds": timeout,
                    "response_seconds": round(request_duration, 6),
                    "status": "parser_exception",
                    "model_name": runtime.context.model,
                    "output_schema": output_schema,
                    "image_inputs": int(image_inputs),
                    "input_tokens": None,
                    "output_tokens": None,
                    "total_tokens": None,
                    "token_usage_source": None,
                    "input_preview": input_preview,
                    "raw_text": None,
                    "parsed_output_preview": None,
                    "parsing_error": "OutputParserException",
                },
            )
            record_duration(
                state.timing_log,
                "llm_communication_seconds",
                request_duration,
                component=f"{component_name}:parser_retry",
                count_key="llm_requests",
                metadata={"attempt": attempt, "timeout_seconds": timeout},
            )
            if attempt == retries:
                raise
            await _sleep_with_timing(
                state.timing_log,
                PARSER_RETRY_SLEEP_SECONDS,
                component=component_name,
                metadata={"attempt": attempt, "reason": "output_parser_exception"},
            )

        except Exception as e:
            request_duration = time.perf_counter() - request_start
            append_llm_log(
                state.timing_log,
                {
                    "component": component_name,
                    "attempt": attempt,
                    "seed": seed,
                    "timeout_seconds": timeout,
                    "response_seconds": round(request_duration, 6),
                    "status": "error",
                    "model_name": runtime.context.model,
                    "output_schema": output_schema,
                    "image_inputs": int(image_inputs),
                    "input_tokens": None,
                    "output_tokens": None,
                    "total_tokens": None,
                    "token_usage_source": None,
                    "input_preview": input_preview,
                    "raw_text": None,
                    "parsed_output_preview": None,
                    "parsing_error": _truncate_text(
                        f"{type(e).__name__}: {e}", max_chars=1200
                    ),
                },
            )
            print(
                f"{component_name} | attempt {attempt} | error: {type(e).__name__}: {e}"
            )
            record_duration(
                state.timing_log,
                "llm_communication_seconds",
                request_duration,
                component=f"{component_name}:error",
                count_key="llm_requests",
                metadata={"attempt": attempt, "error": type(e).__name__},
            )
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            raise
