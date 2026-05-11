#!/usr/bin/env python3
"""Quick test for OpenAICompatibleClient with Qwen3 via Tailscale."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.clients import OpenAICompatibleClient


def main() -> None:
    print("=== Test 1: List models ===")
    client = OpenAICompatibleClient(
        base_url="https://localagent.tail053d0c.ts.net/v1",
        tailscale_host="localagent.tail053d0c.ts.net",
        tailscale_ip="100.105.94.49",
        socks5_proxy="socks5://localhost:1080",
        verify_ssl=False,
        timeout=30,
    )
    print(f"Model: {client.model}")

    print("\n=== Test 2: Simple chat ===")
    response = client.chat(
        messages=[{"role": "user", "content": "Say hi in one word"}],
    )
    print(f"Response: {response!r}")

    print("\n=== Test 3: JSON mode ===")
    json_response = client.chat(
        messages=[{
            "role": "user",
            "content": 'Return JSON: {"celltype": "T cell", "function": "immune response"}',
        }],
        json_mode=True,
    )
    print(f"JSON Response: {json_response}")

    print("\n=== All tests passed ===")


if __name__ == "__main__":
    main()
