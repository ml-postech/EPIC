"""Timing utilities for graph execution profiling."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import time


DEFAULT_TOTALS = {
    "total_runtime_seconds": 0.0,
    "model_loading_seconds": 0.0,
    "model_transfer_seconds": 0.0,
    "llm_bookkeeping_seconds": 0.0,
    "llm_communication_seconds": 0.0,
    "prompt_preparation_seconds": 0.0,
    "output_postprocessing_seconds": 0.0,
    "image_generation_seconds": 0.0,
    "editing_seconds": 0.0,
    "grounding_seconds": 0.0,
    "depth_estimation_seconds": 0.0,
    "region_text_scoring_seconds": 0.0,
    "image_conversion_seconds": 0.0,
    "file_io_seconds": 0.0,
    "result_serialization_seconds": 0.0,
    "retry_backoff_seconds": 0.0,
    "state_snapshot_seconds": 0.0,
    "vp_visual_program_parser_seconds": 0.0,
    "vp_predicate_verification_seconds": 0.0,
    "vp_text_verification_seconds": 0.0,
    "vp_action_verification_seconds": 0.0,
    "vp_action_selection_seconds": 0.0,
    "vp_in_loop_auditor_seconds": 0.0,
    "node_vp_pipeline_seconds": 0.0,
}

DEFAULT_COUNTS = {
    "model_loads": 0,
    "model_transfers": 0,
    "llm_requests": 0,
    "llm_timeouts": 0,
    "prompt_preparations": 0,
    "output_postprocesses": 0,
    "image_generations": 0,
    "edit_calls": 0,
    "grounding_calls": 0,
    "depth_calls": 0,
    "region_text_scores": 0,
    "image_conversions": 0,
    "file_writes": 0,
    "result_serializations": 0,
    "retry_backoffs": 0,
    "state_snapshots": 0,
    "visual_program_parses": 0,
    "predicate_verifications": 0,
    "text_verifier_requests": 0,
    "action_verifier_requests": 0,
    "action_selections": 0,
    "in_loop_audits": 0,
}


def _empty_llm_usage_bucket() -> dict:
    return {
        "calls": 0,
        "successful_calls": 0,
        "timeout_calls": 0,
        "error_calls": 0,
        "parser_error_calls": 0,
        "parser_exception_calls": 0,
        "response_seconds": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "image_inputs": 0,
        "calls_with_usage": 0,
        "calls_with_images": 0,
        "averages": {
            "response_seconds_per_call": 0.0,
            "input_tokens_per_call_with_usage": 0.0,
            "output_tokens_per_call_with_usage": 0.0,
            "total_tokens_per_call_with_usage": 0.0,
            "image_inputs_per_call": 0.0,
        },
    }


def _finalize_llm_usage_bucket(bucket: dict) -> dict:
    calls = int(bucket.get("calls", 0))
    calls_with_usage = int(bucket.get("calls_with_usage", 0))
    finalized = {key: value for key, value in bucket.items() if key != "averages"}
    finalized["response_seconds"] = round(
        float(finalized.get("response_seconds", 0.0)), 6
    )
    finalized["averages"] = {
        "response_seconds_per_call": round(
            float(finalized["response_seconds"]) / calls if calls else 0.0,
            6,
        ),
        "input_tokens_per_call_with_usage": round(
            float(finalized.get("input_tokens", 0)) / calls_with_usage
            if calls_with_usage
            else 0.0,
            6,
        ),
        "output_tokens_per_call_with_usage": round(
            float(finalized.get("output_tokens", 0)) / calls_with_usage
            if calls_with_usage
            else 0.0,
            6,
        ),
        "total_tokens_per_call_with_usage": round(
            float(finalized.get("total_tokens", 0)) / calls_with_usage
            if calls_with_usage
            else 0.0,
            6,
        ),
        "image_inputs_per_call": round(
            float(finalized.get("image_inputs", 0)) / calls if calls else 0.0,
            6,
        ),
    }
    return finalized


def summarize_llm_logs(llm_logs: list[dict] | None) -> dict:
    """Aggregate per-call LLM/VLM logs into totals and per-component summaries."""
    total_bucket = _empty_llm_usage_bucket()
    by_component: dict[str, dict] = {}
    for entry in llm_logs or []:
        component = str(entry.get("component") or "unknown")
        bucket = by_component.setdefault(component, _empty_llm_usage_bucket())
        status = str(entry.get("status") or "unknown")
        image_inputs = int(entry.get("image_inputs") or 0)
        input_tokens = entry.get("input_tokens")
        output_tokens = entry.get("output_tokens")
        total_tokens = entry.get("total_tokens")
        response_seconds = float(entry.get("response_seconds") or 0.0)

        for target in (total_bucket, bucket):
            target["calls"] += 1
            target["response_seconds"] += response_seconds
            target["image_inputs"] += image_inputs
            if image_inputs > 0:
                target["calls_with_images"] += 1
            if status in {"parsed", "success"}:
                target["successful_calls"] += 1
            elif status == "timeout":
                target["timeout_calls"] += 1
            elif status == "parser_error":
                target["parser_error_calls"] += 1
            elif status == "parser_exception":
                target["parser_exception_calls"] += 1
            else:
                target["error_calls"] += 1

            if any(
                value is not None
                for value in (input_tokens, output_tokens, total_tokens)
            ):
                target["calls_with_usage"] += 1
                if input_tokens is not None:
                    target["input_tokens"] += int(input_tokens)
                if output_tokens is not None:
                    target["output_tokens"] += int(output_tokens)
                if total_tokens is not None:
                    target["total_tokens"] += int(total_tokens)

    return {
        "totals": _finalize_llm_usage_bucket(total_bucket),
        "by_component": {
            component: _finalize_llm_usage_bucket(bucket)
            for component, bucket in sorted(by_component.items())
        },
    }


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def create_timing_log() -> dict:
    return {
        "schema_version": 1,
        "run": {
            "started_at_utc": _utc_now_iso(),
            "started_perf_counter": time.perf_counter(),
            "finished_at_utc": None,
            "total_runtime_seconds": None,
        },
        "totals": dict(DEFAULT_TOTALS),
        "counts": dict(DEFAULT_COUNTS),
        "by_component": {key: {} for key in DEFAULT_TOTALS},
        "events": [],
        "llm_logs": [],
        "llm_usage_summary": summarize_llm_logs([]),
    }


def ensure_timing_log(timing_log: dict | None) -> dict:
    if not timing_log:
        return create_timing_log()

    timing_log.setdefault("schema_version", 1)

    run = timing_log.setdefault("run", {})
    run.setdefault("started_at_utc", _utc_now_iso())
    run.setdefault("started_perf_counter", time.perf_counter())
    run.setdefault("finished_at_utc", None)
    run.setdefault("total_runtime_seconds", None)

    totals = timing_log.setdefault("totals", {})
    for key, value in DEFAULT_TOTALS.items():
        totals.setdefault(key, value)

    counts = timing_log.setdefault("counts", {})
    for key, value in DEFAULT_COUNTS.items():
        counts.setdefault(key, value)

    by_component = timing_log.setdefault("by_component", {})
    for key in DEFAULT_TOTALS:
        by_component.setdefault(key, {})

    timing_log.setdefault("events", [])
    timing_log.setdefault("llm_logs", [])
    timing_log.setdefault(
        "llm_usage_summary", summarize_llm_logs(timing_log["llm_logs"])
    )
    return timing_log


def increment_count(timing_log: dict | None, count_key: str, amount: int = 1) -> dict:
    timing_log = ensure_timing_log(timing_log)
    timing_log["counts"][count_key] = timing_log["counts"].get(count_key, 0) + amount
    return timing_log


def record_duration(
    timing_log: dict | None,
    category: str,
    seconds: float,
    *,
    component: str | None = None,
    count_key: str | None = None,
    metadata: dict | None = None,
) -> dict:
    timing_log = ensure_timing_log(timing_log)
    seconds = max(float(seconds), 0.0)

    totals = timing_log["totals"]
    totals[category] = round(totals.get(category, 0.0) + seconds, 6)

    if component is not None:
        by_component = timing_log["by_component"].setdefault(category, {})
        by_component[component] = round(by_component.get(component, 0.0) + seconds, 6)

    if count_key is not None:
        increment_count(timing_log, count_key)

    event = {
        "timestamp_utc": _utc_now_iso(),
        "category": category,
        "seconds": round(seconds, 6),
    }
    if component is not None:
        event["component"] = component
    if metadata:
        event["metadata"] = metadata
    timing_log["events"].append(event)
    return timing_log


def append_llm_log(timing_log: dict | None, entry: dict) -> dict:
    timing_log = ensure_timing_log(timing_log)
    log_entry = {"timestamp_utc": _utc_now_iso(), **entry}
    timing_log["llm_logs"].append(log_entry)
    timing_log["llm_usage_summary"] = summarize_llm_logs(timing_log["llm_logs"])
    return timing_log


def finalize_timing_log(timing_log: dict | None) -> dict | None:
    if not timing_log:
        return None

    timing_log = ensure_timing_log(timing_log)
    run = timing_log["run"]
    if run.get("finished_at_utc") is None:
        total_runtime = time.perf_counter() - run["started_perf_counter"]
        run["finished_at_utc"] = _utc_now_iso()
        run["total_runtime_seconds"] = round(total_runtime, 6)
        timing_log["totals"]["total_runtime_seconds"] = round(total_runtime, 6)
    timing_log["llm_usage_summary"] = summarize_llm_logs(timing_log.get("llm_logs", []))
    return timing_log


def snapshot_timing_log(timing_log: dict | None) -> dict:
    timing_log = ensure_timing_log(timing_log)
    snapshot_start = time.perf_counter()
    snapshot = deepcopy(timing_log)
    snapshot_duration = time.perf_counter() - snapshot_start

    record_duration(
        timing_log,
        "state_snapshot_seconds",
        snapshot_duration,
        component="timing_log",
        count_key="state_snapshots",
    )
    record_duration(
        snapshot,
        "state_snapshot_seconds",
        snapshot_duration,
        component="timing_log",
        count_key="state_snapshots",
    )
    return snapshot


def build_vp_stage_timing_breakdown(timing_log: dict | None) -> dict:
    """Return a compact stage-level timing summary for the VP pipeline."""
    timing_log = ensure_timing_log(timing_log)
    totals = timing_log["totals"]
    counts = timing_log["counts"]
    run = timing_log["run"]

    total_runtime_seconds = run.get("total_runtime_seconds")
    if total_runtime_seconds is None:
        total_runtime_seconds = max(
            0.0, time.perf_counter() - run["started_perf_counter"]
        )

    return {
        "visual_program_parser_seconds": round(
            float(totals.get("vp_visual_program_parser_seconds", 0.0)), 6
        ),
        "image_generation_seconds": round(
            float(totals.get("image_generation_seconds", 0.0)), 6
        ),
        "editing_seconds": round(float(totals.get("editing_seconds", 0.0)), 6),
        "predicate_verification_seconds": round(
            float(totals.get("vp_predicate_verification_seconds", 0.0)), 6
        ),
        "action_selection_seconds": round(
            float(totals.get("vp_action_selection_seconds", 0.0)), 6
        ),
        "llm_bookkeeping_seconds": round(
            float(totals.get("llm_bookkeeping_seconds", 0.0)),
            6,
        ),
        "in_loop_auditor_seconds": round(
            float(
                totals.get(
                    "vp_in_loop_auditor_seconds",
                    0.0,
                )
            ),
            6,
        ),
        "node_vp_pipeline_seconds": round(
            float(totals.get("node_vp_pipeline_seconds", 0.0)), 6
        ),
        "total_runtime_seconds": round(float(total_runtime_seconds), 6),
        "llm_usage_summary": summarize_llm_logs(timing_log.get("llm_logs", [])),
        "counts": {
            "visual_program_parses": int(counts.get("visual_program_parses", 0)),
            "image_generations": int(counts.get("image_generations", 0)),
            "edit_calls": int(counts.get("edit_calls", 0)),
            "predicate_verifications": int(counts.get("predicate_verifications", 0)),
            "action_selections": int(counts.get("action_selections", 0)),
            "in_loop_audits": int(counts.get("in_loop_audits", 0)),
        },
    }
