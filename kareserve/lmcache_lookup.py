# SPDX-License-Identifier: Apache-2.0
"""Authoritative LMCache L1/L2 prefix lookup client."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import aiohttp

from kareserve.kareserve_state import PrefixMatch, SchedulerRequest


@dataclass(frozen=True, slots=True)
class CacheDomainConfig:
    domain_id: str
    http_url: str
    world_size: int = 1
    cache_salt: str = ""
    model_name: str | None = None


class LMCacheLookupClient:
    def __init__(
        self,
        domains: dict[str, CacheDomainConfig],
        timeout_seconds: float = 1.0,
    ) -> None:
        self.domains = domains
        self.timeout_seconds = max(timeout_seconds, 0.01)
        self._session: aiohttp.ClientSession | None = None
        self.lookup_batches = 0
        self.lookup_failures = 0
        self.last_errors: dict[str, str] = {}

    async def start(self) -> None:
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)
            )

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def lookup(
        self, requests: list[SchedulerRequest]
    ) -> tuple[dict[str, dict[str, PrefixMatch]], set[str]]:
        if not self.domains or not requests:
            return {}, set()
        await self.start()
        results: dict[str, dict[str, PrefixMatch]] = {
            request.request_id: {} for request in requests
        }
        prompt_lengths = {
            request.request_id: len(request.prompt_tokens) for request in requests
        }
        failed_domains: set[str] = set()

        async def lookup_domain(domain: CacheDomainConfig) -> None:
            assert self._session is not None
            queries = []
            expected_request_ids = set(prompt_lengths)
            for request in requests:
                request_model = request.raw_body.get("model")
                model_name = domain.model_name or request_model
                if not isinstance(model_name, str) or not model_name:
                    raise ValueError(
                        f"No LMCache model name for request {request.request_id}"
                    )
                queries.append(
                    {
                        "request_id": request.request_id,
                        "model_name": model_name,
                        "world_size": domain.world_size,
                        "token_ids": request.prompt_tokens,
                        "cache_salt": domain.cache_salt,
                    }
                )
            url = f"{domain.http_url.rstrip('/')}/kareserve/cache/lookup"
            async with self._session.post(
                url,
                json={
                    "queries": queries,
                    "timeout_ms": self.timeout_seconds * 1000.0,
                },
            ) as response:
                if response.status != 200:
                    detail = await response.text()
                    raise RuntimeError(
                        f"LMCache lookup returned {response.status}: {detail}"
                    )
                payload: dict[str, Any] = await response.json()
            domain_matches: dict[str, PrefixMatch] = {}
            for item in payload.get("results", []):
                request_id = str(item.get("request_id", ""))
                if request_id not in expected_request_ids:
                    raise RuntimeError(
                        f"LMCache lookup returned unknown request_id={request_id!r}"
                    )
                l2 = item.get("l2", [])
                fs_tokens = max(
                    (
                        int(entry.get("prefix_tokens", 0))
                        for entry in l2
                        if entry.get("medium") == "FS"
                    ),
                    default=0,
                )
                obj_tokens = max(
                    (
                        int(entry.get("prefix_tokens", 0))
                        for entry in l2
                        if entry.get("medium") == "OBJ"
                    ),
                    default=0,
                )
                domain_matches[request_id] = PrefixMatch(
                    prompt_tokens=prompt_lengths[request_id],
                    cpu_prefix_tokens=int(item.get("l1_prefix_tokens", 0)),
                    fs_prefix_tokens=fs_tokens,
                    obj_prefix_tokens=obj_tokens,
                )
            missing_request_ids = expected_request_ids - domain_matches.keys()
            if missing_request_ids:
                raise RuntimeError(
                    "LMCache lookup omitted request IDs: "
                    + ", ".join(sorted(missing_request_ids))
                )
            for request_id, match in domain_matches.items():
                results[request_id][domain.domain_id] = match

        async def guarded(domain: CacheDomainConfig) -> None:
            try:
                await lookup_domain(domain)
                self.last_errors.pop(domain.domain_id, None)
            except (
                aiohttp.ClientError,
                asyncio.TimeoutError,
                ValueError,
                RuntimeError,
            ) as exc:
                failed_domains.add(domain.domain_id)
                self.last_errors[domain.domain_id] = str(exc)

        await asyncio.gather(*(guarded(domain) for domain in self.domains.values()))
        self.lookup_batches += 1
        self.lookup_failures += len(failed_domains)
        return results, failed_domains

    def stats(self) -> dict[str, Any]:
        return {
            "source": "lmcache_authoritative_lookup",
            "domains": sorted(self.domains),
            "lookup_batches": self.lookup_batches,
            "lookup_failures": self.lookup_failures,
            "last_errors": dict(self.last_errors),
        }
