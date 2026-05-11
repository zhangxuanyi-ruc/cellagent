#!/usr/bin/env python3
"""Diagnose vLLM service issues on the LLM machine."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


def run(cmd: list[str] | str, timeout: int = 10) -> tuple[int, str, str]:
    """Run a shell command and return (returncode, stdout, stderr)."""
    if isinstance(cmd, str):
        cmd = ["bash", "-c", cmd]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired as exc:
        return -1, exc.stdout or "", f"TIMEOUT after {timeout}s"


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def check_vllm_process() -> None:
    section("1. vLLM Process")
    rc, out, err = run("ps aux | grep 'vllm serve' | grep -v grep")
    if out.strip():
        print(out)
        # Extract PID and check CPU/GPU usage
        pid = out.split()[1]
        rc2, top_out, _ = run(f"top -b -n1 -p {pid} | tail -1", timeout=5)
        if top_out.strip():
            print(f"  CPU/MEM: {top_out.strip()}")
    else:
        print("  [ERROR] No vLLM process found!")


def check_gpu() -> None:
    section("2. GPU Status")
    rc, out, err = run("nvidia-smi")
    if rc == 0:
        lines = out.splitlines()
        for i, line in enumerate(lines):
            if "MiB" in line or "%" in line or "PID" in line or "GPU" in line or " processes" in line:
                print(f"  {line}")
    else:
        print(f"  [ERROR] nvidia-smi failed: {err[:200]}")


def check_api_health() -> None:
    section("3. API Health Check")

    # 3a. List models
    rc, out, err = run([
        "curl", "-s", "--max-time", "10",
        "http://localhost:8000/v1/models"
    ], timeout=15)
    if rc == 0 and out.strip():
        try:
            data = json.loads(out)
            models = [m.get("id", "?") for m in data.get("data", [])]
            print(f"  [OK] Models: {models}")
        except json.JSONDecodeError:
            print(f"  [WARN] Non-JSON response: {out[:200]}")
    else:
        print(f"  [ERROR] List models failed: rc={rc}, err={err[:200]}")

    # 3b. Simple completion
    rc, out, err = run([
        "curl", "-s", "--max-time", "30",
        "-X", "POST", "http://localhost:8000/v1/chat/completions",
        "-H", "Content-Type: application/json",
        "-d", json.dumps({
            "model": "Qwen3.6-27B",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
        })
    ], timeout=35)
    if rc == 0 and out.strip():
        try:
            data = json.loads(out)
            choice = data.get("choices", [{}])[0]
            msg = choice.get("message", {})
            content = msg.get("content")
            reasoning = msg.get("reasoning_content")
            usage = data.get("usage", {})
            print(f"  [OK] Completion response:")
            print(f"    content={content!r}")
            print(f"    reasoning_content={reasoning!r}")
            print(f"    usage={usage}")
            if content is None and reasoning is None:
                print(f"    [WARN] Both content and reasoning_content are None!")
                print(f"    Full response: {json.dumps(data, ensure_ascii=False, indent=2)[:1000]}")
        except json.JSONDecodeError:
            print(f"  [WARN] Non-JSON completion response: {out[:500]}")
    elif rc == -1:
        print(f"  [ERROR] Completion timed out after 30s")
    else:
        print(f"  [ERROR] Completion failed: rc={rc}, err={err[:200]}")


def check_json_mode() -> None:
    section("4. JSON Mode Test")
    rc, out, err = run([
        "curl", "-s", "--max-time", "60",
        "-X", "POST", "http://localhost:8000/v1/chat/completions",
        "-H", "Content-Type: application/json",
        "-d", json.dumps({
            "model": "Qwen3.6-27B",
            "messages": [{"role": "user", "content": 'Return JSON: {"celltype": "T cell", "function": "immune response"}'}],
            "max_tokens": 100,
            "response_format": {"type": "json_object"},
        })
    ], timeout=65)
    if rc == 0 and out.strip():
        try:
            data = json.loads(out)
            content = data.get("choices", [{}])[0].get("message", {}).get("content")
            print(f"  [OK] JSON mode content: {content!r}")
            # Try to parse the returned JSON
            try:
                parsed = json.loads(content)
                print(f"  [OK] Parsed JSON keys: {list(parsed.keys())}")
            except (json.JSONDecodeError, TypeError) as e:
                print(f"  [WARN] Returned content is not valid JSON: {e}")
        except json.JSONDecodeError:
            print(f"  [WARN] Non-JSON response: {out[:500]}")
    elif rc == -1:
        print(f"  [ERROR] JSON mode timed out after 60s")
    else:
        print(f"  [ERROR] JSON mode failed: rc={rc}, err={err[:200]}")


def check_with_extra_body() -> None:
    section("5. Test with enable_thinking=false (extra_body)")
    rc, out, err = run([
        "curl", "-s", "--max-time", "60",
        "-X", "POST", "http://localhost:8000/v1/chat/completions",
        "-H", "Content-Type: application/json",
        "-d", json.dumps({
            "model": "Qwen3.6-27B",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
            "extra_body": {"enable_thinking": False},
        })
    ], timeout=65)
    if rc == 0 and out.strip():
        try:
            data = json.loads(out)
            msg = data.get("choices", [{}])[0].get("message", {})
            print(f"  [OK] content={msg.get('content')!r}, reasoning={msg.get('reasoning_content')!r}")
        except json.JSONDecodeError:
            print(f"  [WARN] Non-JSON: {out[:500]}")
    elif rc == -1:
        print(f"  [ERROR] Timed out after 60s")
    else:
        print(f"  [ERROR] Failed: rc={rc}, err={err[:200]}")


def check_logs() -> None:
    section("6. Recent vLLM Logs (last 20 lines)")
    # Try common log locations
    log_paths = [
        "/var/log/vllm.log",
        "/tmp/vllm.log",
        "/mnt/c20250607/user/wanghaoran/zhr/models/vllm.log",
    ]
    # Also try to find the log via lsof
    rc, out, _ = run("lsof -p $(pgrep -f 'vllm serve' | head -1) 2>/dev/null | grep -E 'log|out|err' | head -5")
    if out.strip():
        print(f"  Open log files:")
        for line in out.strip().splitlines():
            print(f"    {line}")

    for path in log_paths:
        p = Path(path)
        if p.exists():
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                print(f"  [Log: {path}] last 20 lines:")
                for line in lines[-20:]:
                    print(f"    {line.rstrip()}")
            return
    print("  No log files found at common paths.")


def main() -> None:
    print(f"vLLM Diagnostic Script")
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version}")

    check_vllm_process()
    check_gpu()
    check_api_health()
    check_with_extra_body()
    check_json_mode()
    check_logs()

    section("Summary")
    print("  If tests 3/4/5 all time out → vLLM is hanging (restart it)")
    print("  If content is None but reasoning_content exists → Qwen3 format issue")
    print("  If JSON mode returns non-JSON → model doesn't support structured output")
    print(f"\nDone at: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
