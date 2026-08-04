# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible Kareserve routing gateway."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from kareserve.policy import (
    CostModel,
    GpuPrefixLoadPolicy,
    KareserveBasePolicy,
    LeastLoadPolicy,
    PrefixHashPolicy,
    RoundRobinPolicy,
    TieredCompletionTimePolicy,
)
from kareserve.lmcache_client import CacheDomainConfig, LMCacheLookupClient
from kareserve.observability import RequestObservation, SSEOutputDetector
from kareserve.state import NodeRoutingState, NodeState, SchedulerRequest
from kareserve.tokenizer import LocalRequestTokenizer
from kareserve.tracker import KareserveTracker
from kareserve.routing import AssignmentResult, RoutePlanner

logger = logging.getLogger("kareserve.server")
logging.basicConfig(level=logging.INFO)


@dataclass(slots=True)
class RouterConfig:
    nodes: list[dict[str, Any]]
    tokenizer_path: str
    tokenizer_max_model_len: int
    tokenizer_revision: str | None = None
    tokenizer_trust_remote_code: bool = False
    chat_template_path: str | None = None
    allow_request_chat_template: bool = False
    lmcache_lookup_enabled: bool = False
    cache_domains: dict[str, CacheDomainConfig] = field(default_factory=dict)
    lmcache_lookup_timeout_seconds: float = 1.0
    policy: str = "tiered_completion_time"
    metrics_interval_seconds: float = 0.5
    expected_output_tokens: int = 16
    prefix_tokens_per_load_unit: float = 256.0
    prefix_block_size: int = 16
    capacity_penalty: float = 2.0
    kv_cache_high_watermark: float = 0.80
    kv_cache_hard_limit: float = 0.95
    queue_history_size: int = 128
    queue_local_min_samples: int = 8
    prefix_hash_tokens: int = 256
    hardware_profile: dict[str, Any] = field(default_factory=dict)


def _log_route_result(
    observation: RequestObservation,
    *,
    outcome: str,
    error: str | None = None,
) -> None:
    logger.info(
        "route_result %s",
        json.dumps(
            observation.result(outcome, error),
            separators=(",", ":"),
        ),
    )


def _resolve_optional_path(value: str | None, config_path: Path) -> str | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    return str(path)


def _resolve_model_reference(value: str, config_path: Path) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    local_path = (config_path.parent / path).resolve()
    return str(local_path) if local_path.exists() else value


def load_config(config_path: str) -> RouterConfig:
    path = Path(config_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Router configuration does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))

    nodes = data.get("nodes", [])
    if not nodes:
        raise ValueError("Router configuration must contain at least one node")
    node_ids = {node["node_id"] for node in nodes}
    if len(node_ids) != len(nodes):
        raise ValueError("Router node_id values must be unique")

    tokenizer = data.get("tokenizer", {})
    tokenizer_path = (
        os.environ.get("KARESERVE_TOKENIZER_PATH")
        or os.environ.get("KARESERVE_MODEL")
        or tokenizer.get("path")
    )
    if not tokenizer_path:
        raise ValueError(
            "Tokenizer path must be set by tokenizer.path, "
            "KARESERVE_TOKENIZER_PATH, or KARESERVE_MODEL"
        )
    tokenizer_path = _resolve_model_reference(tokenizer_path, path)

    configured_template = os.environ.get("KARESERVE_CHAT_TEMPLATE") or tokenizer.get(
        "chat_template_path"
    )
    routing = data.get("routing", {})
    cache_domains = {
        domain_id: CacheDomainConfig(
            domain_id=domain_id,
            http_url=str(domain["http_url"]),
            world_size=max(1, int(domain.get("world_size", 1))),
            cache_salt=str(domain.get("cache_salt", "")),
            model_id=str(domain["model_id"]),
        )
        for domain_id, domain in data.get("cache_domains", {}).items()
    }
    node_domains = {
        node.get("cache_domain_id", node["node_id"]) for node in nodes
    }
    missing_domains = node_domains - cache_domains.keys()
    lmcache_data_plane_enabled = (
        os.environ.get("KARESERVE_ENABLE_LMCACHE", "0") == "1"
    )
    lookup_override = os.environ.get("KARESERVE_ENABLE_LMCACHE_LOOKUP")
    lmcache_lookup_enabled = (
        lmcache_data_plane_enabled
        if lookup_override is None
        else lookup_override == "1"
    )
    if lmcache_lookup_enabled and missing_domains:
        raise ValueError(
            "LMCache is enabled but cache_domains lacks: "
            + ", ".join(sorted(missing_domains))
        )
    policy_override = os.environ.get("KARESERVE_POLICY_OVERRIDE")
    return RouterConfig(
        nodes=nodes,
        tokenizer_path=tokenizer_path,
        tokenizer_max_model_len=max(1, int(tokenizer.get("max_model_len", 2048))),
        tokenizer_revision=tokenizer.get("revision"),
        tokenizer_trust_remote_code=bool(tokenizer.get("trust_remote_code", False)),
        chat_template_path=_resolve_optional_path(configured_template, path),
        allow_request_chat_template=bool(
            tokenizer.get("allow_request_chat_template", False)
        ),
        lmcache_lookup_enabled=lmcache_lookup_enabled,
        cache_domains=cache_domains,
        lmcache_lookup_timeout_seconds=max(
            0.01, float(routing.get("lmcache_lookup_timeout_seconds", 1.0))
        ),
        policy=policy_override or routing.get("policy", "tiered_completion_time"),
        metrics_interval_seconds=float(routing.get("metrics_interval_seconds", 0.5)),
        expected_output_tokens=max(0, int(routing.get("expected_output_tokens", 16))),
        prefix_tokens_per_load_unit=float(
            routing.get("prefix_tokens_per_load_unit", 256.0)
        ),
        prefix_block_size=int(routing.get("prefix_block_size", 16)),
        capacity_penalty=float(routing.get("capacity_penalty", 2.0)),
        kv_cache_high_watermark=float(routing.get("kv_cache_high_watermark", 0.80)),
        kv_cache_hard_limit=float(routing.get("kv_cache_hard_limit", 0.95)),
        queue_history_size=int(routing.get("queue_history_size", 128)),
        queue_local_min_samples=int(
            routing.get("queue_local_min_samples", 8)
        ),
        prefix_hash_tokens=int(routing.get("prefix_hash_tokens", 256)),
        hardware_profile=data.get("hardware_profile", {}),
    )


def build_policy(config: RouterConfig) -> KareserveBasePolicy:
    cost_model = CostModel.from_hardware_profile(
        config.hardware_profile,
        tokens_per_work_unit=config.prefix_tokens_per_load_unit,
    )
    policy_name = config.policy.lower()
    policy_options = {
        "cost_model": cost_model,
        "prefix_block_size": config.prefix_block_size,
        "capacity_penalty": config.capacity_penalty,
        "capacity_high_watermark": config.kv_cache_high_watermark,
        "capacity_hard_limit": config.kv_cache_hard_limit,
        "queue_history_size": config.queue_history_size,
        "queue_local_min_samples": config.queue_local_min_samples,
    }
    if policy_name == "tiered_completion_time":
        return TieredCompletionTimePolicy(
            **policy_options,
        )
    if policy_name == "gpu_prefix_load":
        return GpuPrefixLoadPolicy(**policy_options)
    if policy_name == "round_robin":
        return RoundRobinPolicy(cost_model)
    if policy_name == "prefix_hash":
        return PrefixHashPolicy(config.prefix_hash_tokens, cost_model)
    if policy_name == "least_load":
        return LeastLoadPolicy(cost_model)
    raise ValueError(f"Unsupported routing policy: {policy_name}")


async def poll_metrics(app_state: Any) -> None:
    tracker: KareserveTracker = app_state.tracker
    interval = app_state.config.metrics_interval_seconds
    timeout = aiohttp.ClientTimeout(total=2.0)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            for node in tracker.get_routing_states().values():
                try:
                    async with session.get(f"{node.endpoint_url}/metrics") as response:
                        if response.status == 200:
                            tracker.update_metrics_text(
                                node.node_id, await response.text()
                            )
                        else:
                            tracker.mark_metrics_unavailable(node.node_id)
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    tracker.mark_metrics_unavailable(node.node_id)
            await asyncio.sleep(interval)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config_path = os.environ.get("KARESERVE_CONFIG", "configs/config.json")
    config = load_config(config_path)
    tokenizer = LocalRequestTokenizer(
        config.tokenizer_path,
        max_model_len=config.tokenizer_max_model_len,
        revision=config.tokenizer_revision,
        trust_remote_code=config.tokenizer_trust_remote_code,
        chat_template_path=config.chat_template_path,
        allow_request_chat_template=config.allow_request_chat_template,
    )
    initial_nodes = [
        NodeState(
            node_id=node["node_id"],
            host=node["host"],
            port=int(node["port"]),
            cache_domain_id=node.get("cache_domain_id", node["node_id"]),
        )
        for node in config.nodes
    ]
    tracker = KareserveTracker(initial_nodes)
    policy = build_policy(config)
    lmcache_lookup = None
    if config.lmcache_lookup_enabled:
        lmcache_lookup = LMCacheLookupClient(
            config.cache_domains,
            timeout_seconds=config.lmcache_lookup_timeout_seconds,
        )
        await lmcache_lookup.start()
    route_planner = RoutePlanner(
        tracker=tracker,
        policy=policy,
        prefix_block_size=config.prefix_block_size,
        lmcache_lookup=lmcache_lookup,
    )

    app.state.config = config
    app.state.tokenizer = tokenizer
    app.state.tracker = tracker
    app.state.route_planner = route_planner
    app.state.policy = policy
    app.state.lmcache_lookup = lmcache_lookup
    for node in config.nodes:
        endpoint = node.get("kv_events_endpoint")
        if endpoint:
            tracker.start_zmq_listener(
                node["node_id"],
                endpoint,
                node.get("kv_replay_endpoint"),
            )
    metrics_task = asyncio.create_task(
        poll_metrics(app.state), name="kareserve-metrics"
    )
    try:
        yield
    finally:
        metrics_task.cancel()
        with suppress(asyncio.CancelledError):
            await metrics_task
        await tracker.stop()
        if lmcache_lookup is not None:
            await lmcache_lookup.close()


app = FastAPI(title="Kareserve Cluster Scheduler Gateway", lifespan=lifespan)

HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def upstream_request_headers(raw_request: Request) -> dict[str, str]:
    return {
        name: value
        for name, value in raw_request.headers.items()
        if name.lower() not in HOP_BY_HOP_HEADERS and name.lower() != "content-type"
    }


async def open_upstream(
    node: NodeRoutingState,
    endpoint: str,
    body: dict[str, Any],
    headers: dict[str, str],
) -> tuple[aiohttp.ClientSession, aiohttp.ClientResponse]:
    session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600))
    try:
        response = await session.post(
            f"{node.endpoint_url}{endpoint}", json=body, headers=headers
        )
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        await session.close()
        raise HTTPException(
            status_code=502,
            detail=f"Upstream request failed for {node.node_id}: {exc}",
        ) from exc
    return session, response


async def stream_upstream(
    session: aiohttp.ClientSession,
    response: aiohttp.ClientResponse,
    tracker: KareserveTracker,
    policy: KareserveBasePolicy,
    assignment: AssignmentResult,
    observation: RequestObservation,
) -> AsyncGenerator[bytes, None]:
    output_detector = SSEOutputDetector()
    outcome = "completed"
    error = None
    try:
        async for chunk in response.content.iter_any():
            observation.response_bytes += len(chunk)
            if observation.first_output_at is None and output_detector.feed(chunk):
                observation.first_output_at = time.perf_counter()
            yield chunk
    except asyncio.CancelledError:
        outcome = "cancelled"
        raise
    except Exception as exc:
        outcome = "error"
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        response.release()
        await session.close()
        if observation.first_output_at is not None:
            policy.observe_first_output(
                assignment.assignment,
                (observation.first_output_at - observation.upstream_opened_at)
                * 1000.0,
            )
        tracker.release_route(assignment.node.node_id, assignment.inflight_work)
        _log_route_result(observation, outcome=outcome, error=error)


async def read_upstream(
    session: aiohttp.ClientSession,
    response: aiohttp.ClientResponse,
    tracker: KareserveTracker,
    assignment: AssignmentResult,
    observation: RequestObservation,
) -> bytes:
    outcome = "completed"
    error = None
    try:
        content = await response.read()
        observation.response_bytes = len(content)
        return content
    except asyncio.CancelledError:
        outcome = "cancelled"
        raise
    except Exception as exc:
        outcome = "error"
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        response.release()
        await session.close()
        tracker.release_route(assignment.node.node_id, assignment.inflight_work)
        _log_route_result(observation, outcome=outcome, error=error)


def route_headers(
    request_id: str,
    assignment: AssignmentResult,
    policy_name: str,
    estimated_cost_unit: str,
) -> dict[str, str]:
    node = assignment.node
    match = assignment.prefix_match
    usage = "unknown" if node.kv_cache_usage is None else f"{node.kv_cache_usage:.6f}"
    return {
        "X-Request-Id": request_id,
        "X-Kareserve-Worker-Id": node.node_id,
        "X-Kareserve-Policy": policy_name,
        "X-Kareserve-Planning-Ms": f"{assignment.planning_total_ms:.3f}",
        "X-Kareserve-GPU-Prefix-Tokens": str(match.gpu_prefix_tokens),
        "X-Kareserve-CPU-Prefix-Tokens": str(match.cpu_prefix_tokens),
        "X-Kareserve-FS-Prefix-Tokens": str(match.fs_prefix_tokens),
        "X-Kareserve-OBJ-Prefix-Tokens": str(match.obj_prefix_tokens),
        "X-Kareserve-Estimated-Cost": f"{assignment.estimated_cost:.6f}",
        "X-Kareserve-Estimated-Cost-Unit": estimated_cost_unit,
        "X-Kareserve-KV-Cache-Usage": usage,
    }


async def parse_json_request(raw_request: Request) -> dict[str, Any]:
    try:
        body = await raw_request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON payload") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON payload must be an object")
    return body


@app.post("/tokenize")
async def tokenize(raw_request: Request) -> Response:
    try:
        body = await parse_json_request(raw_request)
        tokens = await asyncio.to_thread(
            raw_request.app.state.tokenizer.encode_request, body
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    payload: dict[str, Any] = {
        "count": len(tokens),
        "max_model_len": raw_request.app.state.tokenizer.max_model_len,
        "tokens": tokens,
        "token_strs": None,
    }
    if body.get("return_token_strs"):
        payload["token_strs"] = await asyncio.to_thread(
            raw_request.app.state.tokenizer.token_strings, tokens
        )
    return Response(
        content=json.dumps(payload),
        media_type="application/json",
    )


@app.post("/detokenize")
async def detokenize(raw_request: Request) -> Response:
    body = await parse_json_request(raw_request)
    tokens = body.get("tokens")
    if not isinstance(tokens, list) or not all(
        isinstance(token, int) and token >= 0 for token in tokens
    ):
        raise HTTPException(
            status_code=400,
            detail="tokens must be non-negative integers",
        )
    prompt = await asyncio.to_thread(
        raw_request.app.state.tokenizer.decode_tokens, tokens
    )
    return Response(
        content=json.dumps({"prompt": prompt}),
        media_type="application/json",
    )


async def handle_completion(raw_request: Request, endpoint: str) -> Response:
    request_started_at = time.perf_counter()
    body = await parse_json_request(raw_request)
    tracker: KareserveTracker = raw_request.app.state.tracker
    config: RouterConfig = raw_request.app.state.config
    tokenizer: LocalRequestTokenizer = raw_request.app.state.tokenizer
    route_planner: RoutePlanner = raw_request.app.state.route_planner
    try:
        prompt_tokens = await asyncio.to_thread(tokenizer.encode_request, body)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    tokenization_completed_at = time.perf_counter()

    request_id = str(
        body.get("request_id")
        or raw_request.headers.get("x-request-id")
        or f"kareserve-{uuid.uuid4().hex}"
    )
    configured_output = body.get("max_tokens")
    if configured_output is None:
        configured_output = body.get(
            "max_completion_tokens", config.expected_output_tokens
        )
    scheduler_request = SchedulerRequest(
        request_id=f"route-{uuid.uuid4().hex}",
        prompt_tokens=prompt_tokens,
        max_tokens=max(0, int(configured_output)),
        raw_body=body,
    )
    try:
        assignment = await route_planner.assign(scheduler_request)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    node = assignment.node
    assignment_completed_at = time.perf_counter()
    estimated_cost_unit = raw_request.app.state.policy.cost_model.unit
    response_headers = route_headers(
        request_id, assignment, config.policy, estimated_cost_unit
    )
    match = assignment.prefix_match
    cost_breakdown = assignment.cost_breakdown
    logger.info(
        "route_decision %s",
        json.dumps(
            {
                "request_id": request_id,
                "prefix_id": raw_request.headers.get("x-prefix-id"),
                "trace_id": raw_request.headers.get("x-trace-id"),
                "policy": config.policy,
                "node_id": node.node_id,
                "cache_domain_id": node.cache_domain_id,
                "tokenization_ms": round(
                    (tokenization_completed_at - request_started_at) * 1000.0,
                    3,
                ),
                "planning_total_ms": round(assignment.planning_total_ms, 3),
                "cache_lookup_ms": round(assignment.cache_lookup_ms, 3),
                "candidate_build_ms": round(assignment.candidate_build_ms, 3),
                "policy_ms": round(assignment.policy_ms, 3),
                "prompt_tokens": len(prompt_tokens),
                "gpu_prefix_tokens": match.gpu_prefix_tokens,
                "cpu_prefix_tokens": match.cpu_prefix_tokens,
                "fs_prefix_tokens": match.fs_prefix_tokens,
                "obj_prefix_tokens": match.obj_prefix_tokens,
                "missing_tokens": match.missing_tokens,
                "router_inflight_work": node.router_inflight_work,
                "estimated_cost": assignment.estimated_cost,
                "estimated_cost_unit": estimated_cost_unit,
                "estimated_prompt_path_cost": (
                    cost_breakdown.prompt_path_cost if cost_breakdown else None
                ),
                "estimated_queue_cost": (
                    cost_breakdown.queue_cost if cost_breakdown else None
                ),
                "estimated_capacity_cost": (
                    cost_breakdown.capacity_cost if cost_breakdown else None
                ),
                "kv_cache_usage": node.kv_cache_usage,
                "metrics_status": node.metrics_status.value,
                "gpu_catalog_status": node.gpu_catalog_status.value,
                "lmcache_catalog_status": node.lmcache_catalog_status.value,
            },
            separators=(",", ":"),
        ),
    )

    try:
        session, response = await open_upstream(
            node,
            endpoint,
            body,
            upstream_request_headers(raw_request),
        )
    except HTTPException:
        tracker.release_route(node.node_id, assignment.inflight_work)
        raise

    observation = RequestObservation(
        request_id=request_id,
        prefix_id=raw_request.headers.get("x-prefix-id"),
        trace_id=raw_request.headers.get("x-trace-id"),
        node_id=node.node_id,
        request_started_at=request_started_at,
        assignment_completed_at=assignment_completed_at,
        upstream_opened_at=time.perf_counter(),
        upstream_status=response.status,
    )

    content_type = response.headers.get("Content-Type")
    if content_type:
        response_headers["Content-Type"] = content_type
    if body.get("stream", False):
        return StreamingResponse(
            stream_upstream(
                session,
                response,
                tracker,
                raw_request.app.state.policy,
                assignment,
                observation,
            ),
            status_code=response.status,
            headers=response_headers,
        )
    content = await read_upstream(
        session,
        response,
        tracker,
        assignment,
        observation,
    )
    return Response(
        content=content,
        status_code=response.status,
        headers=response_headers,
    )


@app.post("/v1/chat/completions")
async def chat_completions(raw_request: Request):
    return await handle_completion(raw_request, "/v1/chat/completions")


@app.post("/v1/completions")
async def completions(raw_request: Request):
    return await handle_completion(raw_request, "/v1/completions")


@app.get("/health")
async def health_check(request: Request):
    tracker: KareserveTracker = request.app.state.tracker
    config: RouterConfig = request.app.state.config
    route_planner: RoutePlanner = request.app.state.route_planner
    return {
        "status": "ok",
        "nodes": list(tracker.get_routing_states()),
        "policy": config.policy,
        "route_planner": route_planner.stats(),
    }


@app.get("/routing/state")
async def routing_state(request: Request):
    tracker: KareserveTracker = request.app.state.tracker
    config: RouterConfig = request.app.state.config
    route_planner: RoutePlanner = request.app.state.route_planner
    states = tracker.get_routing_states()
    return {
        "policy": config.policy,
        "nodes": {
            node_id: {
                "endpoint": node.endpoint_url,
                "cache_domain_id": node.cache_domain_id,
                "router_active_requests": node.router_active_requests,
                "router_inflight_work": node.router_inflight_work,
                "running_requests": node.running_requests,
                "waiting_requests": node.waiting_requests,
                "kv_cache_usage": node.kv_cache_usage,
                "gpu_total_blocks": node.gpu_total_blocks,
                "estimated_gpu_free_blocks": node.estimated_gpu_free_blocks,
                "gpu_block_size": node.gpu_block_size,
                "metrics_status": node.metrics_status.value,
                "metrics_updated_at": node.metrics_updated_at,
                "process_start_time_seconds": node.process_start_time_seconds,
                "gpu_catalog_status": node.gpu_catalog_status.value,
                "lmcache_catalog_status": node.lmcache_catalog_status.value,
            }
            for node_id, node in states.items()
        },
        "gpu_catalog": {
            "blocks": tracker.gpu_cached_block_count,
        },
        "lmcache": {
            "source": (
                "lmcache_authoritative_lookup"
                if config.lmcache_lookup_enabled
                else "disabled"
            ),
            "lookup": (
                request.app.state.lmcache_lookup.stats()
                if request.app.state.lmcache_lookup is not None
                else None
            ),
        },
        "monitoring": {
            node_id: {
                "lmcache_queries": tracker.nodes[node_id].lmcache_queries,
                "lmcache_hits": tracker.nodes[node_id].lmcache_hits,
            }
            for node_id in states
        },
        "route_planner": route_planner.stats(),
    }
