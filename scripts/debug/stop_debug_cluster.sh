#!/usr/bin/env bash
set -euo pipefail

ROOT=${KARESERVE_ROOT:-/home/zn/xyz/serve1}
RUNTIME="$ROOT/runtime"

for pid_file in "$RUNTIME"/pids/*.pid; do
  [[ -e "$pid_file" ]] || continue
  pid="$(cat "$pid_file")"
  command_line="$(ps -p "$pid" -o args= 2>/dev/null || true)"
  if [[ "$command_line" == *"$ROOT"* ]]; then
    kill "$pid"
    echo "stopped $(basename "$pid_file" .pid) pid=$pid"
  fi
  rm -f -- "$pid_file"
done
