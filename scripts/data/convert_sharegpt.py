#!/usr/bin/env python3
"""Convert collaborator ShareGPT JSONL into vLLM custom JSONL requests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Iterator


def read_records(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        first = file.read(1)
        file.seek(0)
        if first == "[":
            records = json.load(file)
            if not isinstance(records, list):
                raise ValueError("JSON input must contain an array")
            yield from records
            return
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number} is not a JSON object")
            yield record


def turns_from_record(record: dict[str, Any]) -> Iterable[tuple[str, str]]:
    compact = record.get("conversation")
    if isinstance(compact, list):
        for turn in compact:
            if not isinstance(turn, dict):
                continue
            human = turn.get("human")
            assistant = turn.get("assistant")
            if isinstance(human, str) and isinstance(assistant, str):
                yield human, assistant
        return

    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        return
    for index in range(0, len(conversations) - 1, 2):
        human = conversations[index]
        assistant = conversations[index + 1]
        if not isinstance(human, dict) or not isinstance(assistant, dict):
            continue
        human_text = human.get("value")
        assistant_text = assistant.get("value")
        if isinstance(human_text, str) and isinstance(assistant_text, str):
            yield human_text, assistant_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument(
        "--mode",
        choices=("first-turn", "cumulative"),
        default="cumulative",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with output_path.open("w", encoding="utf-8") as output:
        for record in read_records(input_path):
            history: list[str] = []
            for human, assistant in turns_from_record(record):
                prompt = "".join(history) + f"Human: {human}\nAssistant:"
                output.write(
                    json.dumps(
                        {
                            "prompt": prompt,
                            "output_tokens": args.output_tokens,
                            "source_id": record.get("conversation_id"),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                written += 1
                if args.limit > 0 and written >= args.limit:
                    break
                if args.mode == "first-turn":
                    break
                history.extend(
                    [
                        f"Human: {human}\n",
                        f"Assistant: {assistant}\n",
                    ]
                )
            if args.limit > 0 and written >= args.limit:
                break

    if written == 0:
        raise ValueError("No compatible conversations were found")
    print(
        json.dumps(
            {
                "input": str(input_path),
                "output": str(output_path),
                "requests": written,
                "mode": args.mode,
                "output_tokens": args.output_tokens,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
