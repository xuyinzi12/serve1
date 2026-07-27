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
from fastapi.responses import JSONResponse, StreamingResponse

from kareserve.kareserve_policy import (
    KareserveBasePolicy,
    NodeState,
    SchedulerRequest,
    WindowedPrefixAffinityPolicy,
)
from kareserve.kareserve_tracker import KareserveTracker
from kareserve.request_pool import RequestPool

logger = logging.getLogger("kareserve.server")
logging.basicConfig(level=logging.INFO)


def load_config(
    config_path: str = "kareserve/config.json",
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
    routing = data.get("routing", {})
    policy = WindowedPrefixAffinityPolicy(
        prefix_tokens_per_load_unit=float(
            routing.get("prefix_tokens_per_load_unit", 256.0)
        ),
        queue_weight=float(routing.get("queue_weight", 1.0)),
        group_block_size=int(routing.get("group_block_size", 16)),
    )
    return nodes, policy, data


initial_nodes, policy, config = load_config(
    os.environ.get("KARESERVE_CONFIG", "kareserve/config.json")
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
                except (aiohttp.ClientError, asyncio.TimeoutError):
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


async def forward_stream(
    target_url: str, request_json: Dict[str, Any]
) -> AsyncGenerator[bytes, None]:
    timeout = aiohttp.ClientTimeout(total=3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            f"{target_url}/v1/chat/completions", json=request_json
        ) as response:
            if response.status != 200:
                yield await response.read()
                return
            async for chunk in response.content.iter_any():
                yield chunk


@app.post("/v1/chat/completions")
async def chat_completions(raw_request: Request):
    try:
        body = await raw_request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON payload") from exc

    states = tracker.get_node_states()
    if not states:
        raise HTTPException(status_code=503, detail="No available vLLM servers")
    tokenizer_node = next(iter(states.values()))
    prompt_tokens = await tokenize_request(tokenizer_node, body)
    request_id = str(body.get("request_id") or f"kareserve-{uuid.uuid4().hex}")
    scheduler_request = SchedulerRequest(
        request_id=request_id,
        prompt_tokens=prompt_tokens,
        max_tokens=int(body.get("max_tokens", 16)),
        extra_body=body,
    )
    try:
        selected_node = await request_pool.assign(scheduler_request)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    tracker.update_active_requests(selected_node.node_id, 1)

    async def stream_wrapper():
        try:
            async for chunk in forward_stream(selected_node.endpoint_url, body):
                yield chunk
        finally:
            tracker.update_active_requests(selected_node.node_id, -1)

    if body.get("stream", False):
        return StreamingResponse(stream_wrapper(), media_type="text/event-stream")

    full_bytes = b""
    async for chunk in stream_wrapper():
        full_bytes += chunk
    try:
        return JSONResponse(content=json.loads(full_bytes.decode("utf-8")))
    except Exception:
        return JSONResponse(
            content={"raw_response": full_bytes.decode("utf-8", errors="ignore")}
        )


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "nodes": [node.node_id for node in tracker.get_node_states().values()],
        "queued_requests": request_pool.queue.qsize(),
    }


@app.get("/routing/state")
async def routing_state():
    states = tracker.get_node_states()
    return {
        "nodes": {
            node_id: {
                "endpoint": node.endpoint_url,
                "active_requests": node.active_requests,
                "running_requests": node.running_requests,
                "waiting_requests": node.waiting_requests,
                "kv_cache_usage": node.kv_cache_usage,
                "cached_blocks": len(node.cached_blocks),
            }
            for node_id, node in states.items()
        },
        "queued_requests": request_pool.queue.qsize(),
    }
