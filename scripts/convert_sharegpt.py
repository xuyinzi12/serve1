#!/usr/bin/env python3
"""Convert the server JSONL conversation corpus to vLLM ShareGPT JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    source = Path(args.input).resolve()
    destination = Path(args.output).resolve()
    if source == destination:
        raise ValueError("Input and output paths must differ")
    destination.parent.mkdir(parents=True, exist_ok=True)

    accepted = 0
    rejected = 0
    with source.open(encoding="utf-8") as input_file, destination.open(
        "w", encoding="utf-8"
    ) as output_file:
        output_file.write("[\n")
        first = True
        for line in input_file:
            if args.limit > 0 and accepted >= args.limit:
                break
            try:
                source_entry = json.loads(line)
                turns = source_entry.get("conversation", [])
                pair = turns[0]
                human = pair.get("human")
                assistant = pair.get("assistant")
                if not isinstance(human, str) or not isinstance(assistant, str):
                    raise ValueError("conversation pair is incomplete")
            except (json.JSONDecodeError, IndexError, TypeError, ValueError):
                rejected += 1
                continue

            converted = {
                "id": source_entry.get("conversation_id"),
                "conversations": [
                    {"from": "human", "value": human},
                    {"from": "gpt", "value": assistant},
                ],
            }
            if not first:
                output_file.write(",\n")
            json.dump(converted, output_file, ensure_ascii=False)
            first = False
            accepted += 1
        output_file.write("\n]\n")

    print(
        json.dumps(
            {
                "input": str(source),
                "output": str(destination),
                "accepted": accepted,
                "rejected": rejected,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
