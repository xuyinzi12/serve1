# SPDX-License-Identifier: Apache-2.0
"""LMCache MP server with authoritative cache lookup APIs."""

from __future__ import annotations

import asyncio
from time import monotonic
from typing import Any

from fastapi import HTTPException, Request
from pydantic import BaseModel, Field


class PrefixQuery(BaseModel):
    request_id: str
    model_name: str
    token_ids: list[int]
    world_size: int = Field(default=1, ge=1)
    cache_salt: str = ""


class BatchLookupRequest(BaseModel):
    queries: list[PrefixQuery]
    timeout_ms: float = Field(default=1000.0, gt=0.0, le=10000.0)


def _prefix_chunks(bits: list[bool], chunk_count: int, world_size: int) -> int:
    prefix = 0
    for chunk_index in range(chunk_count):
        start = chunk_index * world_size
        if not all(bits[start : start + world_size]):
            break
        prefix += 1
    return prefix


def _adapter_medium(type_name: str) -> str:
    local_storage = {"dax", "fs", "fs_native", "raw_block"}
    return "FS" if type_name in local_storage else "OBJ"


async def _lookup_l2_adapter(
    adapter: Any,
    keys: list[Any],
    layout_desc: Any,
    timeout_seconds: float,
) -> list[bool]:
    task_id = adapter.submit_lookup_and_lock_task(keys, layout_desc)
    deadline = monotonic() + timeout_seconds
    bitmap = None
    while bitmap is None and monotonic() < deadline:
        bitmap = adapter.query_lookup_and_lock_result(task_id)
        if bitmap is None:
            await asyncio.sleep(0.001)
    if bitmap is None:
        raise TimeoutError("LMCache L2 lookup timed out")
    bits = [bool(bitmap.test(index)) for index in range(len(keys))]
    hit_keys = [key for key, hit in zip(keys, bits, strict=True) if hit]
    if hit_keys:
        adapter.submit_unlock(hit_keys)
    return bits


async def _lookup_query(
    engine: Any,
    query: PrefixQuery,
    timeout_seconds: float,
) -> dict[str, Any]:
    from lmcache.v1.multiprocess.cache_control.key_resolver import (
        resolve_object_keys,
    )

    context = engine.context
    layout_desc = context.layout_desc_registry.find(
        query.model_name, query.world_size
    )
    if layout_desc is None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"LMCache has no registered layout for model={query.model_name!r} "
                f"world_size={query.world_size}"
            ),
        )
    keys, chunk_count = resolve_object_keys(
        context.token_hasher,
        query.model_name,
        query.world_size,
        query.token_ids,
        query.cache_salt,
    )
    chunk_size = context.token_hasher.chunk_size
    if not keys:
        return {
            "request_id": query.request_id,
            "chunk_size": chunk_size,
            "chunks": 0,
            "l1_prefix_tokens": 0,
            "l2": [],
        }

    storage_manager = engine.storage_manager
    l1_manager = storage_manager._l1_manager
    l1_bits = []
    for key in keys:
        state = l1_manager.get_object_state(key)
        l1_bits.append(state is not None and not state.write_lock.is_locked())

    l2_results = []
    for descriptor, adapter in storage_manager.l2_adapters():
        bits = await _lookup_l2_adapter(
            adapter, keys, layout_desc, timeout_seconds
        )
        prefix_chunks = _prefix_chunks(bits, chunk_count, query.world_size)
        l2_results.append(
            {
                "adapter": descriptor.type_name,
                "medium": _adapter_medium(descriptor.type_name),
                "prefix_tokens": prefix_chunks * chunk_size,
            }
        )

    l1_prefix_chunks = _prefix_chunks(l1_bits, chunk_count, query.world_size)
    return {
        "request_id": query.request_id,
        "chunk_size": chunk_size,
        "chunks": chunk_count,
        "l1_prefix_tokens": l1_prefix_chunks * chunk_size,
        "l2": l2_results,
    }


def _install_bridge() -> Any:
    from lmcache.v1.distributed.config import parse_args_to_config
    from lmcache.v1.mp_observability.config import (
        parse_args_to_observability_config,
    )
    from lmcache.v1.multiprocess import http_server
    from lmcache.v1.multiprocess.config import (
        parse_args_to_coordinator_config,
        parse_args_to_http_frontend_config,
        parse_args_to_mp_server_config,
    )

    app = http_server.app

    @app.post("/kareserve/cache/lookup")
    async def lookup_cache(body: BatchLookupRequest, request: Request):
        engine = getattr(request.app.state, "engine", None)
        if engine is None:
            raise HTTPException(status_code=503, detail="LMCache is not initialized")
        timeout_seconds = body.timeout_ms / 1000.0
        results = await asyncio.gather(
            *(
                _lookup_query(engine, query, timeout_seconds)
                for query in body.queries
            )
        )
        return {"source": "lmcache_authoritative_lookup", "results": results}

    @app.get("/kareserve/cache/status")
    async def cache_status(request: Request):
        engine = getattr(request.app.state, "engine", None)
        if engine is None:
            raise HTTPException(status_code=503, detail="LMCache is not initialized")
        storage_manager = engine.storage_manager
        return {
            "source": "lmcache_authoritative_lookup",
            "chunk_size": engine.context.token_hasher.chunk_size,
            "l1": storage_manager._l1_manager.report_status(),
            "l2": [
                {
                    "adapter": descriptor.type_name,
                    "status": adapter.report_status(),
                }
                for descriptor, adapter in storage_manager.l2_adapters()
            ],
        }

    def run() -> None:
        args = http_server.parse_args()
        http_server.run_http_server(
            http_config=parse_args_to_http_frontend_config(args),
            mp_config=parse_args_to_mp_server_config(args),
            storage_manager_config=parse_args_to_config(args),
            obs_config=parse_args_to_observability_config(args),
            coordinator_config=parse_args_to_coordinator_config(args),
        )

    return run


def main() -> None:
    _install_bridge()()


if __name__ == "__main__":
    main()
