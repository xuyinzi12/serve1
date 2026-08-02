# SPDX-License-Identifier: Apache-2.0
"""Shared state and decision types for Kareserve."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

BlockHash = bytes | int | str
BlockIdentity = tuple[int, BlockHash]


class CacheMedium(str, Enum):
    GPU = "GPU"
    CPU = "CPU"
    FS = "FS"
    OBJ = "OBJ"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def parse(cls, value: Any) -> CacheMedium:
        if value is None:
            return cls.UNKNOWN
        normalized = str(value).upper()
        if normalized.startswith("CUDA"):
            return cls.GPU
        aliases = {"DISK": cls.FS, "REMOTE": cls.OBJ}
        if normalized in aliases:
            return aliases[normalized]
        try:
            return cls(normalized)
        except ValueError:
            return cls.UNKNOWN


class CacheLocality(str, Enum):
    LOCAL = "LOCAL"
    REMOTE = "REMOTE"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def parse(cls, value: Any) -> CacheLocality:
        if value is None:
            return cls.UNKNOWN
        try:
            return cls(str(value).upper())
        except ValueError:
            return cls.UNKNOWN


class MetricsStatus(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class CatalogStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"


@dataclass(frozen=True, slots=True)
class CachePlacement:
    medium: CacheMedium
    owner_id: str
    locality: CacheLocality = CacheLocality.UNKNOWN


@dataclass(slots=True)
class CachedBlock:
    block_hash: BlockHash
    group_idx: int
    parent_block_hash: BlockHash | None
    token_ids: tuple[int, ...]
    placements: set[CachePlacement] = field(default_factory=set)

    @property
    def identity(self) -> BlockIdentity:
        return self.group_idx, self.block_hash


@dataclass(slots=True)
class NodeState:
    """Mutable observations for one independently addressable vLLM server."""

    node_id: str
    host: str
    port: int
    cache_domain_id: str
    router_active_requests: int = 0
    router_inflight_work: float = 0.0
    running_requests: int | None = None
    waiting_requests: int | None = None
    kv_cache_usage: float | None = None
    gpu_total_blocks: int | None = None
    gpu_block_size: int | None = None
    metrics_status: MetricsStatus = MetricsStatus.UNAVAILABLE
    metrics_updated_at: float | None = None
    process_start_time_seconds: float | None = None
    external_cache_queries: float = 0.0
    external_cache_hits: float = 0.0

    @property
    def endpoint_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass(frozen=True, slots=True)
class NodeRoutingState:
    """Immutable node features consumed by routing policies."""

    node_id: str
    host: str
    port: int
    cache_domain_id: str
    router_active_requests: int
    router_inflight_work: float
    running_requests: int | None
    waiting_requests: int | None
    kv_cache_usage: float | None
    gpu_total_blocks: int | None
    gpu_block_size: int | None
    metrics_status: MetricsStatus
    metrics_updated_at: float | None
    process_start_time_seconds: float | None
    catalog_status: CatalogStatus

    @property
    def endpoint_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def queue_depth(self) -> float:
        return float((self.running_requests or 0) + (self.waiting_requests or 0))

    @property
    def estimated_gpu_free_blocks(self) -> int | None:
        if self.gpu_total_blocks is None or self.kv_cache_usage is None:
            return None
        free_fraction = 1.0 - min(max(self.kv_cache_usage, 0.0), 1.0)
        return max(0, int(self.gpu_total_blocks * free_fraction))


@dataclass(frozen=True, slots=True)
class PrefixMatch:
    prompt_tokens: int
    gpu_prefix_tokens: int = 0
    cpu_prefix_tokens: int = 0
    fs_prefix_tokens: int = 0
    obj_prefix_tokens: int = 0
    unknown_prefix_tokens: int = 0

    @property
    def external_prefix_tokens(self) -> int:
        return max(
            self.cpu_prefix_tokens,
            self.fs_prefix_tokens,
            self.obj_prefix_tokens,
            self.unknown_prefix_tokens,
        )

    @property
    def cached_prefix_tokens(self) -> int:
        return max(self.gpu_prefix_tokens, self.external_prefix_tokens)

    @property
    def missing_tokens(self) -> int:
        return max(0, self.prompt_tokens - self.cached_prefix_tokens)


@dataclass(frozen=True, slots=True)
class RouteCandidate:
    request_id: str
    node: NodeRoutingState
    prefix_match: PrefixMatch
    required_new_gpu_blocks: int | None


@dataclass(slots=True)
class SchedulerRequest:
    request_id: str
    prompt_tokens: list[int]
    max_tokens: int
    raw_body: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RouteAssignment:
    candidate: RouteCandidate
    inflight_work: float
    estimated_cost: float
