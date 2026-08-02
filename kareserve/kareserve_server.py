# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible Kareserve routing gateway."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from kareserve.kareserve_policy import (
    CostModel,
    KareserveBasePolicy,
    LeastLoadPolicy,
    PrefixHashPolicy,
    RoundRobinPolicy,
    WindowedPrefixAffinityPolicy,
)
from kareserve.kareserve_state import NodeRoutingState, NodeState, SchedulerRequest
from kareserve.kareserve_tokenizer import LocalRequestTokenizer
from kareserve.kareserve_tracker import KareserveTracker
from kareserve.request_pool import AssignmentResult, RequestPool

logger = logging.getLogger("kareserve.server")
logging.basicConfig(level=logging.INFO)


@dataclass(slots=True)
class RouterConfig:
    nodes: list[dict[str, Any]]
    tokenizer_path: str
    tokenizer_revision: str | None = None
    tokenizer_trust_remote_code: bool = False
    chat_template_path: str | None = None
    allow_request_chat_template: bool = False
    external_cache_enabled: bool = False
    external_cache_chunk_size: int = 256
    policy: str = "windowed_prefix"
    window_ms: float = 2.0
    max_batch_size: int = 64
    metrics_interval_seconds: float = 0.5
    expected_output_tokens: int = 16
    prefix_tokens_per_load_unit: float = 256.0
    queue_weight: float = 1.0
    group_block_size: int = 16
    kv_cache_weight: float = 2.0
    kv_cache_high_watermark: float = 0.80
    kv_cache_hard_limit: float = 0.95
    decode_token_weight: float = 4.0
    prefix_hash_tokens: int = 256
    hardware_profile: dict[str, Any] = field(default_factory=dict)


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
    policy_override = os.environ.get("KARESERVE_POLICY_OVERRIDE")
    window_override = os.environ.get("KARESERVE_WINDOW_MS_OVERRIDE")
    return RouterConfig(
        nodes=nodes,
        tokenizer_path=tokenizer_path,
        tokenizer_revision=tokenizer.get("revision"),
        tokenizer_trust_remote_code=bool(tokenizer.get("trust_remote_code", False)),
        chat_template_path=_resolve_optional_path(configured_template, path),
        allow_request_chat_template=bool(
            tokenizer.get("allow_request_chat_template", False)
        ),
        external_cache_enabled=(os.environ.get("KARESERVE_ENABLE_LMCACHE", "0") == "1"),
        external_cache_chunk_size=max(
            1, int(routing.get("external_cache_chunk_size", 256))
        ),
        policy=policy_override or routing.get("policy", "windowed_prefix"),
        window_ms=(
            float(window_override)
            if window_override is not None
            else float(routing.get("window_ms", 2.0))
        ),
        max_batch_size=int(routing.get("max_batch_size", 64)),
        metrics_interval_seconds=float(routing.get("metrics_interval_seconds", 0.5)),
        expected_output_tokens=max(0, int(routing.get("expected_output_tokens", 16))),
        prefix_tokens_per_load_unit=float(
            routing.get("prefix_tokens_per_load_unit", 256.0)
        ),
        queue_weight=float(routing.get("queue_weight", 1.0)),
        group_block_size=int(routing.get("group_block_size", 16)),
        kv_cache_weight=float(routing.get("kv_cache_weight", 2.0)),
        kv_cache_high_watermark=float(routing.get("kv_cache_high_watermark", 0.80)),
        kv_cache_hard_limit=float(routing.get("kv_cache_hard_limit", 0.95)),
        decode_token_weight=float(routing.get("decode_token_weight", 4.0)),
        prefix_hash_tokens=int(routing.get("prefix_hash_tokens", 256)),
        hardware_profile=data.get("hardware_profile", {}),
    )


def build_policy(config: RouterConfig) -> KareserveBasePolicy:
    cost_model = CostModel.from_hardware_profile(
        config.hardware_profile,
        tokens_per_work_unit=config.prefix_tokens_per_load_unit,
        decode_token_weight=config.decode_token_weight,
    )
    policy_name = config.policy.lower()
    if policy_name in {"windowed_prefix", "medium_aware"}:
        return WindowedPrefixAffinityPolicy(
            cost_model=cost_model,
            queue_weight=config.queue_weight,
            group_block_size=config.group_block_size,
            kv_cache_weight=config.kv_cache_weight,
            kv_cache_high_watermark=config.kv_cache_high_watermark,
            kv_cache_hard_limit=config.kv_cache_hard_limit,
        )
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
    config_path = os.environ.get("KARESERVE_CONFIG", "configs/router.single-node.json")
    config = load_config(config_path)
    tokenizer = LocalRequestTokenizer(
        config.tokenizer_path,
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
    request_pool = RequestPool(
        tracker=tracker,
        policy=policy,
        window_ms=config.window_ms,
        max_batch_size=config.max_batch_size,
        group_block_size=config.group_block_size,
    )

    app.state.config = config
    app.state.tokenizer = tokenizer
    app.state.tracker = tracker
    app.state.request_pool = request_pool
    app.state.policy = policy
    for node in config.nodes:
        endpoint = node.get("kv_events_endpoint")
        if endpoint:
            tracker.start_zmq_listener(
                node["node_id"],
                endpoint,
                node.get("kv_replay_endpoint"),
            )
    request_pool.start()
    metrics_task = asyncio.create_task(
        poll_metrics(app.state), name="kareserve-metrics"
    )
    try:
        yield
    finally:
        metrics_task.cancel()
        with suppress(asyncio.CancelledError):
            await metrics_task
        await request_pool.stop()
        await tracker.stop()


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
    node_id: str,
    inflight_work: float,
    prompt_tokens: list[int],
    external_cache_chunk_size: int | None,
) -> AsyncGenerator[bytes, None]:
    completed = False
    try:
        async for chunk in response.content.iter_any():
            yield chunk
        completed = True
    finally:
        response.release()
        await session.close()
        if completed and external_cache_chunk_size is not None:
            tracker.record_external_prefix(
                node_id, prompt_tokens, external_cache_chunk_size
            )
        tracker.release_route(node_id, inflight_work)


async def read_upstream(
    session: aiohttp.ClientSession,
    response: aiohttp.ClientResponse,
    tracker: KareserveTracker,
    node_id: str,
    inflight_work: float,
    prompt_tokens: list[int],
    external_cache_chunk_size: int | None,
) -> bytes:
    try:
        content = await response.read()
        if external_cache_chunk_size is not None:
            tracker.record_external_prefix(
                node_id, prompt_tokens, external_cache_chunk_size
            )
        return content
    finally:
        response.release()
        await session.close()
        tracker.release_route(node_id, inflight_work)


def route_headers(
    request_id: str,
    assignment: AssignmentResult,
    policy_name: str,
) -> dict[str, str]:
    node = assignment.node
    match = assignment.prefix_match
    usage = "unknown" if node.kv_cache_usage is None else f"{node.kv_cache_usage:.6f}"
    return {
        "X-Request-Id": request_id,
        "X-Kareserve-Worker-Id": node.node_id,
        "X-Kareserve-Policy": policy_name,
        "X-Kareserve-Route-Batch-Size": str(assignment.route_batch_size),
        "X-Kareserve-Queue-Wait-Ms": f"{assignment.queue_wait_ms:.3f}",
        "X-Kareserve-GPU-Prefix-Tokens": str(match.gpu_prefix_tokens),
        "X-Kareserve-CPU-Prefix-Tokens": str(match.cpu_prefix_tokens),
        "X-Kareserve-Estimated-Cost": f"{assignment.estimated_cost:.6f}",
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


async def handle_completion(raw_request: Request, endpoint: str) -> Response:
    body = await parse_json_request(raw_request)
    tracker: KareserveTracker = raw_request.app.state.tracker
    config: RouterConfig = raw_request.app.state.config
    tokenizer: LocalRequestTokenizer = raw_request.app.state.tokenizer
    request_pool: RequestPool = raw_request.app.state.request_pool
    try:
        prompt_tokens = tokenizer.encode_request(body)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

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
        request_id=request_id,
        prompt_tokens=prompt_tokens,
        max_tokens=max(0, int(configured_output)),
        raw_body=body,
    )
    try:
        assignment = await request_pool.assign(scheduler_request)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    node = assignment.node
    response_headers = route_headers(request_id, assignment, config.policy)
    match = assignment.prefix_match
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
                "route_batch_size": assignment.route_batch_size,
                "queue_wait_ms": round(assignment.queue_wait_ms, 3),
                "prompt_tokens": len(prompt_tokens),
                "gpu_prefix_tokens": match.gpu_prefix_tokens,
                "cpu_prefix_tokens": match.cpu_prefix_tokens,
                "missing_tokens": match.missing_tokens,
                "router_inflight_work": node.router_inflight_work,
                "estimated_cost": assignment.estimated_cost,
                "kv_cache_usage": node.kv_cache_usage,
                "metrics_status": node.metrics_status.value,
                "catalog_status": node.catalog_status.value,
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

    content_type = response.headers.get("Content-Type")
    if content_type:
        response_headers["Content-Type"] = content_type
    if body.get("stream", False):
        return StreamingResponse(
            stream_upstream(
                session,
                response,
                tracker,
                node.node_id,
                assignment.inflight_work,
                prompt_tokens,
                (
                    config.external_cache_chunk_size
                    if config.external_cache_enabled and response.status < 400
                    else None
                ),
            ),
            status_code=response.status,
            headers=response_headers,
        )
    content = await read_upstream(
        session,
        response,
        tracker,
        node.node_id,
        assignment.inflight_work,
        prompt_tokens,
        (
            config.external_cache_chunk_size
            if config.external_cache_enabled and response.status < 400
            else None
        ),
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
    request_pool: RequestPool = request.app.state.request_pool
    return {
        "status": "ok",
        "nodes": list(tracker.get_routing_states()),
        "policy": config.policy,
        "request_pool": request_pool.stats(),
    }


@app.get("/routing/state")
async def routing_state(request: Request):
    tracker: KareserveTracker = request.app.state.tracker
    config: RouterConfig = request.app.state.config
    request_pool: RequestPool = request.app.state.request_pool
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
                "catalog_status": node.catalog_status.value,
            }
            for node_id, node in states.items()
        },
        "cache_catalog": {
            "blocks": len(tracker.cached_blocks),
            "placements": sum(
                len(block.placements) for block in tracker.cached_blocks.values()
            ),
            "external_source": (
                "completed_request_admission"
                if config.external_cache_enabled
                else "disabled"
            ),
            "external_chunk_size": config.external_cache_chunk_size,
        },
        "monitoring": {
            node_id: {
                "external_cache_queries": tracker.nodes[node_id].external_cache_queries,
                "external_cache_hits": tracker.nodes[node_id].external_cache_hits,
            }
            for node_id in states
        },
        "request_pool": request_pool.stats(),
    }
