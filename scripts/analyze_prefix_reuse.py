#!/usr/bin/env python3
"""Measure block-aligned prefix reuse in the exact vLLM ShareGPT sample."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field

from transformers import AutoTokenizer
from vllm.benchmarks.datasets import ShareGPTDataset


@dataclass(slots=True)
class TrieNode:
    children: dict[tuple[int, ...], "TrieNode"] = field(default_factory=dict)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--num-requests", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--output-len", type=int, default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    requests = ShareGPTDataset(
        random_seed=args.seed,
        dataset_path=args.dataset,
        disable_shuffle=False,
    ).sample(
        tokenizer=tokenizer,
        num_requests=args.num_requests,
        output_len=args.output_len,
        no_oversample=True,
    )

    root = TrieNode()
    reusable_lengths: list[int] = []
    prompt_lengths: list[int] = []
    exact_prompts: set[tuple[int, ...]] = set()
    exact_duplicates = 0
    block_size = max(1, args.block_size)

    for request in requests:
        token_ids = tokenizer(request.prompt).input_ids
        prompt_lengths.append(len(token_ids))
        prompt_key = tuple(token_ids)
        if prompt_key in exact_prompts:
            exact_duplicates += 1
        exact_prompts.add(prompt_key)

        node = root
        reusable_blocks = 0
        full_blocks = len(token_ids) // block_size
        blocks = [
            tuple(token_ids[index * block_size : (index + 1) * block_size])
            for index in range(full_blocks)
        ]
        for block in blocks:
            child = node.children.get(block)
            if child is None:
                break
            reusable_blocks += 1
            node = child
        reusable_lengths.append(reusable_blocks * block_size)

        node = root
        for block in blocks:
            node = node.children.setdefault(block, TrieNode())

    total_prompt_tokens = sum(prompt_lengths)
    total_reusable_tokens = sum(reusable_lengths)
    result = {
        "dataset": args.dataset,
        "seed": args.seed,
        "sampled_requests": len(requests),
        "block_size": block_size,
        "total_prompt_tokens": total_prompt_tokens,
        "requests_with_reusable_prefix": sum(
            length > 0 for length in reusable_lengths
        ),
        "exact_prompt_duplicates": exact_duplicates,
        "total_reusable_prefix_tokens": total_reusable_tokens,
        "reusable_prefix_token_ratio": (
            total_reusable_tokens / total_prompt_tokens
            if total_prompt_tokens
            else 0.0
        ),
        "mean_reusable_prefix_tokens": (
            total_reusable_tokens / len(requests) if requests else 0.0
        ),
        "max_reusable_prefix_tokens": max(reusable_lengths, default=0),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
