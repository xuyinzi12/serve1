# SPDX-License-Identifier: Apache-2.0
"""Command-line entry point for Kareserve."""

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the Kareserve router")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the Kareserve JSON configuration",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    os.environ["KARESERVE_CONFIG"] = os.path.abspath(args.config)

    import uvicorn

    uvicorn.run(
        "kareserve.server:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
