# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible Kareserve routing gateway."""

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator

import aiohttp
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from kareserve.kareserve_policy import (
    KareserveBasePolicy,
    LeastLoadPolicy,
    NodeState,
    PrefixHashPolicy,
    RoundRobinPolicy,
    SchedulerRequest,
    WindowedPrefixAffinityPolicy,
    NodeRoutingState,
)
from kareserve.kareserve_tracker import KareserveTracker
from kareserve.request_pool import RequestPool

logger = logging.getLogger("kareserve.server")
logging.basicConfig(level=logging.INFO)

@dataclass
class RouterConfig:
    tokenizer_node_id: str
    nodes: list[dict[str, Any]]
    policy: str = "windowed_prefix"
    window_ms: float = 2.0
    max_batch_size: int = 64
    metrics_interval_seconds: float = 0.5
    prefix_tokens_per_load_unit: float = 256.0
    queue_weight: float = 1.0
    group_block_size: int = 16
    kv_cache_weight: float = 2.0
    kv_cache_high_watermark: float = 0.80
    kv_cache_hard_limit: float = 0.95
    decode_token_weight: float = 4.0
    prefix_hash_tokens: int = 256
    hardware_profile: dict[str, Any] = field(default_factory=dict)

def build_policy(config: RouterConfig) -> KareserveBasePolicy:
    policy_name = config.policy.lower()
    if policy_name == "windowed_prefix":
        return WindowedPrefixAffinityPolicy(
            prefix_tokens_per_load_unit=config.prefix_tokens_per_load_unit,
            queue_weight=config.queue_weight,
            group_block_size=config.group_block_size,
            kv_cache_weight=config.kv_cache_weight,
            kv_cache_high_watermark=config.kv_cache_high_watermark,
            kv_cache_hard_limit=config.kv_cache_hard_limit,
            decode_token_weight=config.decode_token_weight,
        )
    if policy_name == "round_robin":
        return RoundRobinPolicy()
    if policy_name == "prefix_hash":
        return PrefixHashPolicy(prefix_hash_tokens=config.prefix_hash_tokens)
    if policy_name == "least_load":
        return LeastLoadPolicy()
    raise ValueError(f"Unsupported routing policy: {policy_name}")

def load_config(config_path: str) -> RouterConfig:
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Router configuration does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    nodes = data.get("nodes", [])
    if not nodes:
        raise ValueError("Router configuration must contain at least one node")

    node_ids = {node["node_id"] for node in nodes}
    if len(node_ids) != len(nodes):
        raise ValueError("Router node_id values must be unique")

    tokenizer_node_id = data.get("tokenizer_node_id")
    if tokenizer_node_id not in node_ids:
        raise ValueError(
            "tokenizer_node_id must identify one configured Router node"
        )

    routing = data.get("routing", {})
    policy_override = os.environ.get("KARESERVE_POLICY_OVERRIDE")
    window_override = os.environ.get("KARESERVE_WINDOW_MS_OVERRIDE")

    return RouterConfig(
        tokenizer_node_id=tokenizer_node_id,
        nodes=nodes,
        policy=policy_override or routing.get("policy", "windowed_prefix"),
        window_ms=float(window_override) if window_override is not None else float(routing.get("window_ms", 2.0)),
        max_batch_size=int(routing.get("max_batch_size", 64)),
        metrics_interval_seconds=float(routing.get("metrics_interval_seconds", 0.5)),
        prefix_tokens_per_load_unit=float(routing.get("prefix_tokens_per_load_unit", 256.0)),
        queue_weight=float(routing.get("queue_weight", 1.0)),
        group_block_size=int(routing.get("group_block_size", 16)),
        kv_cache_weight=float(routing.get("kv_cache_weight", 2.0)),
        kv_cache_high_watermark=float(routing.get("kv_cache_high_watermark", 0.80)),
        kv_cache_hard_limit=float(routing.get("kv_cache_hard_limit", 0.95)),
        decode_token_weight=float(routing.get("decode_token_weight", 4.0)),
        prefix_hash_tokens=int(routing.get("prefix_hash_tokens", 256)),
        hardware_profile=data.get("hardware_profile", {}),
    )

def get_tokenizer_node(states: dict[str, NodeRoutingState], tokenizer_node_id: str) -> NodeRoutingState:
    node = states.get(tokenizer_node_id)
    if node is None:
        raise HTTPException(
            status_code=503,
            detail=f"Tokenizer node is unavailable: {tokenizer_node_id}",
        )
    return node

async def poll_metrics(app_state: Any) -> None:
    tracker = app_state.tracker
    interval = app_state.config.metrics_interval_seconds
    timeout = aiohttp.ClientTimeout(total=2.0)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            states = tracker.get_routing_states([])
            for node in states.values():
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
                    logger.debug("Metrics unavailable for %s", node.node_id)
            await asyncio.sleep(interval)

@asynccontextmanager
async def lifespan(app: FastAPI):
    config_path = os.environ.get("KARESERVE_CONFIG", "configs/router.single-node.json")
    config = load_config(config_path)

    initial_nodes = [
        NodeState(node_id=node["node_id"], host=node["host"], port=node["port"])
        for node in config.nodes
    ]
    policy = build_policy(config)
    tracker = KareserveTracker(initial_nodes)
    request_pool = RequestPool(
        tracker=tracker,
        policy=policy,
        window_ms=config.window_ms,
        max_batch_size=config.max_batch_size,
        group_block_size=config.group_block_size,
    )

    app.state.config = config
    app.state.tracker = tracker
    app.state.request_pool = request_pool
    app.state.policy = policy

    for node_config in config.nodes:
        endpoint = node_config.get("kv_events_endpoint")
        if endpoint:
            tracker.start_zmq_listener(node_config["node_id"], endpoint)

    request_pool.start()
    metrics_task = asyncio.create_task(poll_metrics(app.state))

    yield

    metrics_task.cancel()
    with suppress(asyncio.CancelledError):
        await metrics_task
    await request_pool.stop()
    tracker.stop()

app = FastAPI(title="Kareserve Cluster Scheduler Gateway", lifespan=lifespan)

def tokenize_payload(body: dict[str, Any]) -> dict[str, Any]:
    """Build the documented vLLM /tokenize request for chat or completion."""
    allowed = {
        "model",
        "messages",
        "prompt",
        "tools",
        "chat_template",
        "chat_template_kwargs",
        "add_generation_prompt",
        "continue_final_message",
        "add_special_tokens",
        "media_io_kwargs",
        "mm_processor_kwargs",
    }
    return {key: value for key, value in body.items() if key in allowed}

async def tokenize_request(node: NodeRoutingState, body: dict[str, Any]) -> list[int]:
    timeout = aiohttp.ClientTimeout(total=30.0)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            f"{node.endpoint_url}/tokenize", json=tokenize_payload(body)
        ) as response:
            response_body = await response.json()
            if response.status != 200:
                raise HTTPException(
                    status_code=502,
                    detail={"message": "vLLM tokenization failed", "upstream": response_body},
                )
            tokens = response_body.get("tokens")
            if not isinstance(tokens, list) or not all(
                isinstance(token, int) for token in tokens
            ):
                raise HTTPException(
                    status_code=502, detail="Invalid vLLM /tokenize response"
                )
            return tokens

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
        if name.lower() not in HOP_BY_HOP_HEADERS
        and name.lower() != "content-type"
    }

async def open_upstream(
    node: NodeRoutingState,
    endpoint: str,
    body: dict[str, Any],
    headers: dict[str, str],
) -> tuple[aiohttp.ClientSession, aiohttp.ClientResponse]:
    timeout = aiohttp.ClientTimeout(total=3600)
    upstream_session = aiohttp.ClientSession(timeout=timeout)
    try:
        upstream_response = await upstream_session.post(
            f"{node.endpoint_url}{endpoint}",
            json=body,
            headers=headers,
        )
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        await upstream_session.close()
        raise HTTPException(
            status_code=502,
            detail=f"Upstream request failed for {node.node_id}: {exc}",
        ) from exc
    return upstream_session, upstream_response

async def stream_upstream(
    upstream_session: aiohttp.ClientSession,
    upstream_response: aiohttp.ClientResponse,
    tracker: KareserveTracker | None = None,
    node_id: str | None = None,
) -> AsyncGenerator[bytes, None]:
    try:
        async for chunk in upstream_response.content.iter_any():
            yield chunk
    finally:
        upstream_response.release()
        await upstream_session.close()
        if tracker and node_id:
            tracker.update_active_requests(node_id, -1)

async def read_upstream(
    upstream_session: aiohttp.ClientSession,
    upstream_response: aiohttp.ClientResponse,
    tracker: KareserveTracker | None = None,
    node_id: str | None = None,
) -> bytes:
    try:
        return await upstream_response.read()
    finally:
        upstream_response.release()
        await upstream_session.close()
        if tracker and node_id:
            tracker.update_active_requests(node_id, -1)

def route_headers(
    request_id: str,
    node: NodeRoutingState,
    route_batch_size: int,
    queue_wait_ms: float,
    prefix_hit_blocks: int,
    policy_name: str,
) -> dict[str, str]:
    return {
        "X-Request-Id": request_id,
        "X-Kareserve-Worker-Id": node.node_id,
        "X-Kareserve-Policy": policy_name,
        "X-Kareserve-Route-Batch-Size": str(route_batch_size),
        "X-Kareserve-Queue-Wait-Ms": f"{queue_wait_ms:.3f}",
        "X-Kareserve-Prefix-Hit-Blocks": str(prefix_hit_blocks),
        "X-Kareserve-KV-Cache-Usage": f"{node.kv_cache_usage:.6f}",
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
    request_pool: RequestPool = raw_request.app.state.request_pool

    states = tracker.get_routing_states([])
    if not states:
        raise HTTPException(status_code=503, detail="No available vLLM servers")
    tokenizer_node = get_tokenizer_node(states, config.tokenizer_node_id)
    prompt_tokens = await tokenize_request(tokenizer_node, body)
    request_id = str(
        body.get("request_id")
        or raw_request.headers.get("x-request-id")
        or f"kareserve-{uuid.uuid4().hex}"
    )
    scheduler_request = SchedulerRequest(
        request_id=request_id,
        prompt_tokens=prompt_tokens,
        max_tokens=int(body.get("max_tokens", 16)),
        raw_body=body,
    )
    try:
        assignment = await request_pool.assign(scheduler_request)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    selected_node = assignment.node
    prefix_hit_blocks = assignment.prefix_hit_blocks
    response_headers = route_headers(
        request_id=request_id,
        node=selected_node,
        route_batch_size=assignment.route_batch_size,
        queue_wait_ms=assignment.queue_wait_ms,
        prefix_hit_blocks=prefix_hit_blocks,
        policy_name=config.policy,
    )
    logger.info(
        "route_decision %s",
        json.dumps(
            {
                "request_id": request_id,
                "prefix_id": raw_request.headers.get("x-prefix-id"),
                "trace_id": raw_request.headers.get("x-trace-id"),
                "policy": config.policy,
                "node_id": selected_node.node_id,
                "route_batch_size": assignment.route_batch_size,
                "queue_wait_ms": round(assignment.queue_wait_ms, 3),
                "prompt_tokens": len(prompt_tokens),
                "prefix_hit_blocks": prefix_hit_blocks,
                "node_observed_load": selected_node.observed_load,
                "node_kv_cache_usage": selected_node.kv_cache_usage,
                "node_metrics_available": selected_node.metrics_available,
            },
            separators=(",", ":"),
        ),
    )

    tracker.update_active_requests(selected_node.node_id, 1)
    try:
        upstream_session, upstream_response = await open_upstream(
            selected_node,
            endpoint,
            body,
            upstream_request_headers(raw_request),
        )
    except HTTPException:
        tracker.update_active_requests(selected_node.node_id, -1)
        raise

    content_type = upstream_response.headers.get("Content-Type")
    if content_type:
        response_headers["Content-Type"] = content_type

    if body.get("stream", False):
        return StreamingResponse(
            stream_upstream(
                upstream_session,
                upstream_response,
                tracker,
                selected_node.node_id,
            ),
            status_code=upstream_response.status,
            headers=response_headers,
        )

    content = await read_upstream(
        upstream_session,
        upstream_response,
        tracker,
        selected_node.node_id,
    )

    return Response(
        content=content,
        status_code=upstream_response.status,
        headers=response_headers,
    )

@app.post("/v1/chat/completions")
async def chat_completions(raw_request: Request):
    return await handle_completion(raw_request, "/v1/chat/completions")

@app.post("/v1/completions")
async def completions(raw_request: Request):
    return await handle_completion(raw_request, "/v1/completions")

# @app.post("/tokenize")
# async def tokenize(raw_request: Request):
#     body = await parse_json_request(raw_request)
#     tracker: KareserveTracker = raw_request.app.state.tracker
#     config: RouterConfig = raw_request.app.state.config

#     states = tracker.get_node_states()
#     if not states:
#         raise HTTPException(status_code=503, detail="No available vLLM servers")
#     node = get_tokenizer_node(states, config.tokenizer_node_id)
#     upstream_session, upstream_response = await open_upstream(
#         node,
#         "/tokenize",
#         tokenize_payload(body),
#         upstream_request_headers(raw_request),
#     )
#     content = await read_upstream(
#         upstream_session,
#         upstream_response,
#     )
#     headers: dict[str, str] = {"X-Kareserve-Worker-Id": node.node_id}
#     content_type = upstream_response.headers.get("Content-Type")
#     if content_type:
#         headers["Content-Type"] = content_type
#     return Response(
#         content=content,
#         status_code=upstream_response.status,
#         headers=headers,
#     )

@app.get("/health")
async def health_check(request: Request):
    tracker: KareserveTracker = request.app.state.tracker
    config: RouterConfig = request.app.state.config
    request_pool: RequestPool = request.app.state.request_pool

    return {
        "status": "ok",
        "nodes": [node.node_id for node in tracker.get_routing_states([]).values()],
        "policy": config.policy,
        "request_pool": request_pool.stats(),
    }

# @app.get("/routing/state")
# async def routing_state(request: Request):
#     tracker: KareserveTracker = request.app.state.tracker
#     config: RouterConfig = request.app.state.config
#     request_pool: RequestPool = request.app.state.request_pool

#     states = tracker.get_routing_states([])
#     return {
#         "policy": config.policy,
#         "hardware_profile": config.hardware_profile,
#         "nodes": {
#             node_id: {
#                 "endpoint": node.endpoint_url,
#                 "active_requests": node.active_requests,
#                 "running_requests": node.running_requests,
#                 "waiting_requests": node.waiting_requests,
#                 "kv_cache_usage": node.kv_cache_usage,
#                 "metrics_available": node.metrics_available,
#                 "metrics_updated_at": node.metrics_updated_at,
#                 "external_cache_queries": node.external_cache_queries,
#                 "external_cache_hits": node.external_cache_hits,
#                 "external_cache_hit_rate": node.external_cache_hit_rate,
#             }
#             for node_id, node in states.items()
#         },
#         "request_pool": request_pool.stats(),
#     }
