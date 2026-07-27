# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible Kareserve routing gateway."""

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List

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
)
from kareserve.kareserve_tracker import KareserveTracker
from kareserve.request_pool import RequestPool

logger = logging.getLogger("kareserve.server")
logging.basicConfig(level=logging.INFO)


def build_policy(routing: Dict[str, Any]) -> KareserveBasePolicy:
    policy_name = str(routing.get("policy", "windowed_prefix")).lower()
    if policy_name == "windowed_prefix":
        return WindowedPrefixAffinityPolicy(
            prefix_tokens_per_load_unit=float(
                routing.get("prefix_tokens_per_load_unit", 256.0)
            ),
            queue_weight=float(routing.get("queue_weight", 1.0)),
            group_block_size=int(routing.get("group_block_size", 16)),
            kv_cache_weight=float(routing.get("kv_cache_weight", 2.0)),
            kv_cache_high_watermark=float(
                routing.get("kv_cache_high_watermark", 0.80)
            ),
            kv_cache_hard_limit=float(
                routing.get("kv_cache_hard_limit", 0.95)
            ),
            decode_token_weight=float(
                routing.get("decode_token_weight", 4.0)
            ),
        )
    if policy_name == "round_robin":
        return RoundRobinPolicy()
    if policy_name == "prefix_hash":
        return PrefixHashPolicy(
            prefix_hash_tokens=int(routing.get("prefix_hash_tokens", 256))
        )
    if policy_name == "least_load":
        return LeastLoadPolicy()
    raise ValueError(f"Unsupported routing policy: {policy_name}")


def load_config(
    config_path: str = "configs/router.single-node.json",
) -> tuple[List[NodeState], KareserveBasePolicy, Dict[str, Any]]:
    path = Path(config_path)
    if not path.is_file():
        nodes = [
            NodeState("node-1", "127.0.0.1", 8101),
            NodeState("node-2", "127.0.0.1", 8102),
        ]
        return nodes, WindowedPrefixAffinityPolicy(), {}

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    nodes = [
        NodeState(node_id=node["node_id"], host=node["host"], port=node["port"])
        for node in data.get("nodes", [])
    ]
    routing = dict(data.get("routing", {}))
    policy_override = os.environ.get("KARESERVE_POLICY_OVERRIDE")
    window_override = os.environ.get("KARESERVE_WINDOW_MS_OVERRIDE")
    if policy_override:
        routing["policy"] = policy_override
    if window_override is not None:
        routing["window_ms"] = float(window_override)
    data["routing"] = routing
    policy = build_policy(routing)
    return nodes, policy, data


initial_nodes, policy, config = load_config(
    os.environ.get("KARESERVE_CONFIG", "configs/router.single-node.json")
)
tracker = KareserveTracker(initial_nodes)
routing_config = config.get("routing", {})
request_pool = RequestPool(
    tracker,
    policy,
    window_ms=float(routing_config.get("window_ms", 2.0)),
    max_batch_size=int(routing_config.get("max_batch_size", 64)),
)
_metrics_task: asyncio.Task[None] | None = None


async def poll_metrics() -> None:
    interval = float(routing_config.get("metrics_interval_seconds", 0.5))
    timeout = aiohttp.ClientTimeout(total=2.0)
    while True:
        states = tracker.get_node_states()
        async with aiohttp.ClientSession(timeout=timeout) as session:
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
async def lifespan(_: FastAPI):
    global _metrics_task
    for node_config in config.get("nodes", []):
        endpoint = node_config.get("kv_events_endpoint")
        if endpoint:
            tracker.start_zmq_listener(node_config["node_id"], endpoint)
    request_pool.start()
    _metrics_task = asyncio.create_task(poll_metrics())
    yield
    if _metrics_task is not None:
        _metrics_task.cancel()
        with suppress(asyncio.CancelledError):
            await _metrics_task
    await request_pool.stop()
    tracker.stop()


app = FastAPI(title="Kareserve Cluster Scheduler Gateway", lifespan=lifespan)


def tokenize_payload(body: Dict[str, Any]) -> Dict[str, Any]:
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


async def tokenize_request(node: NodeState, body: Dict[str, Any]) -> List[int]:
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


def upstream_request_headers(raw_request: Request) -> Dict[str, str]:
    return {
        name: value
        for name, value in raw_request.headers.items()
        if name.lower() not in HOP_BY_HOP_HEADERS
        and name.lower() != "content-type"
    }


async def open_upstream(
    node: NodeState,
    endpoint: str,
    body: Dict[str, Any],
    headers: Dict[str, str],
) -> tuple[aiohttp.ClientSession, aiohttp.ClientResponse]:
    timeout = aiohttp.ClientTimeout(total=3600)
    session = aiohttp.ClientSession(timeout=timeout)
    try:
        response = await session.post(
            f"{node.endpoint_url}{endpoint}",
            json=body,
            headers=headers,
        )
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        await session.close()
        raise HTTPException(
            status_code=502,
            detail=f"Upstream request failed for {node.node_id}: {exc}",
        ) from exc
    return session, response


async def stream_upstream(
    response: aiohttp.ClientResponse,
    session: aiohttp.ClientSession,
    node_id: str,
) -> AsyncGenerator[bytes, None]:
    try:
        async for chunk in response.content.iter_any():
            yield chunk
    finally:
        response.release()
        await session.close()
        tracker.update_active_requests(node_id, -1)


def route_headers(
    request_id: str,
    node: NodeState,
    batch_size: int,
    queue_wait_ms: float,
    prefix_hit_tokens: int,
) -> Dict[str, str]:
    return {
        "X-Request-Id": request_id,
        "X-Route-Engine-Id": node.node_id,
        "X-Kareserve-Node-Id": node.node_id,
        "X-Kareserve-Policy": policy.name,
        "X-Kareserve-Batch-Size": str(batch_size),
        "X-Kareserve-Queue-Wait-Ms": f"{queue_wait_ms:.3f}",
        "X-Kareserve-Prefix-Hit-Tokens": str(prefix_hit_tokens),
        "X-Kareserve-KV-Cache-Usage": f"{node.kv_cache_usage:.6f}",
    }


async def parse_json_request(raw_request: Request) -> Dict[str, Any]:
    try:
        body = await raw_request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON payload") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON payload must be an object")
    return body


async def handle_completion(raw_request: Request, endpoint: str) -> Response:
    body = await parse_json_request(raw_request)

    states = tracker.get_node_states()
    if not states:
        raise HTTPException(status_code=503, detail="No available vLLM servers")
    tokenizer_node = next(iter(states.values()))
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
        extra_body=body,
    )
    try:
        assignment = await request_pool.assign(scheduler_request)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    selected_node = assignment.node
    prefix_hit_tokens = selected_node.longest_cached_prefix_tokens(prompt_tokens)
    response_headers = route_headers(
        request_id=request_id,
        node=selected_node,
        batch_size=assignment.batch_size,
        queue_wait_ms=assignment.queue_wait_ms,
        prefix_hit_tokens=prefix_hit_tokens,
    )
    logger.info(
        "route_decision %s",
        json.dumps(
            {
                "request_id": request_id,
                "prefix_id": raw_request.headers.get("x-prefix-id"),
                "trace_id": raw_request.headers.get("x-trace-id"),
                "policy": policy.name,
                "node_id": selected_node.node_id,
                "batch_size": assignment.batch_size,
                "queue_wait_ms": round(assignment.queue_wait_ms, 3),
                "prompt_tokens": len(prompt_tokens),
                "prefix_hit_tokens": prefix_hit_tokens,
                "node_observed_load": selected_node.observed_load,
                "node_kv_cache_usage": selected_node.kv_cache_usage,
                "node_metrics_available": selected_node.metrics_available,
            },
            separators=(",", ":"),
        ),
    )

    tracker.update_active_requests(selected_node.node_id, 1)
    try:
        session, upstream = await open_upstream(
            selected_node,
            endpoint,
            body,
            upstream_request_headers(raw_request),
        )
    except HTTPException:
        tracker.update_active_requests(selected_node.node_id, -1)
        raise

    content_type = upstream.headers.get("Content-Type")
    if content_type:
        response_headers["Content-Type"] = content_type

    if body.get("stream", False):
        return StreamingResponse(
            stream_upstream(
                upstream,
                session,
                selected_node.node_id,
            ),
            status_code=upstream.status,
            headers=response_headers,
        )

    try:
        content = await upstream.read()
        return Response(
            content=content,
            status_code=upstream.status,
            headers=response_headers,
        )
    finally:
        upstream.release()
        await session.close()
        tracker.update_active_requests(selected_node.node_id, -1)


@app.post("/v1/chat/completions")
async def chat_completions(raw_request: Request):
    return await handle_completion(raw_request, "/v1/chat/completions")


@app.post("/v1/completions")
async def completions(raw_request: Request):
    return await handle_completion(raw_request, "/v1/completions")


@app.post("/tokenize")
async def tokenize(raw_request: Request):
    body = await parse_json_request(raw_request)
    states = tracker.get_node_states()
    if not states:
        raise HTTPException(status_code=503, detail="No available vLLM servers")
    node = next(iter(states.values()))
    session, upstream = await open_upstream(
        node,
        "/tokenize",
        tokenize_payload(body),
        upstream_request_headers(raw_request),
    )
    try:
        content = await upstream.read()
        headers: Dict[str, str] = {"X-Kareserve-Node-Id": node.node_id}
        content_type = upstream.headers.get("Content-Type")
        if content_type:
            headers["Content-Type"] = content_type
        return Response(
            content=content,
            status_code=upstream.status,
            headers=headers,
        )
    finally:
        upstream.release()
        await session.close()


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "nodes": [node.node_id for node in tracker.get_node_states().values()],
        "policy": policy.name,
        "request_pool": request_pool.stats(),
    }


@app.get("/routing/state")
async def routing_state():
    states = tracker.get_node_states()
    return {
        "policy": policy.name,
        "hardware_profile": config.get("hardware_profile", {}),
        "nodes": {
            node_id: {
                "endpoint": node.endpoint_url,
                "active_requests": node.active_requests,
                "running_requests": node.running_requests,
                "waiting_requests": node.waiting_requests,
                "kv_cache_usage": node.kv_cache_usage,
                "metrics_available": node.metrics_available,
                "metrics_updated_at": node.metrics_updated_at,
                "external_cache_queries": node.external_cache_queries,
                "external_cache_hits": node.external_cache_hits,
                "external_cache_hit_rate": node.external_cache_hit_rate,
                "cached_blocks": len(node.cached_blocks),
            }
            for node_id, node in states.items()
        },
        "request_pool": request_pool.stats(),
    }
