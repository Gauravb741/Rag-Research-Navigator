#!/usr/bin/env python3
"""Run the secure local web interface."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.interface.web import serve


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve the RAG Research Onboarding UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    serve(args.host, args.port)
