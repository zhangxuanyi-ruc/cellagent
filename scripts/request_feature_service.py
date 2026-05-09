#!/usr/bin/env python
"""Request cell encoder feature extraction from the scGPT feature service."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import requests
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit feature extraction to CellAgent scGPT service.")
    parser.add_argument("--input", required=True, help="CellAgent *_preprocessed.h5ad path.")
    parser.add_argument("--feature-dir", default=None, help="Feature output directory.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--timeout", type=int, default=None, help="Max seconds to wait.")
    parser.add_argument("--poll-interval", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    service_cfg = cfg.get("feature_service", {})
    encoder_cfg = cfg.get("cell_encoder", {})
    url = service_cfg.get("url", "http://127.0.0.1:18080").rstrip("/")
    timeout = args.timeout or int(service_cfg.get("timeout", 24 * 3600))
    payload = {
        "input_h5ad": args.input,
        "feature_dir": args.feature_dir,
        "batch_size": encoder_cfg.get("batch_size"),
    }
    payload = {k: v for k, v in payload.items() if v is not None}

    submit = requests.post(f"{url}/extract", json=payload, timeout=30)
    submit.raise_for_status()
    job = submit.json()
    job_id = job["job_id"]
    start = time.time()

    while True:
        status_resp = requests.get(f"{url}/jobs/{job_id}", timeout=30)
        status_resp.raise_for_status()
        status = status_resp.json()
        if status["status"] == "succeeded":
            print(json.dumps(status, ensure_ascii=False, indent=2))
            return
        if status["status"] == "failed":
            raise RuntimeError(json.dumps(status, ensure_ascii=False, indent=2))
        if time.time() - start > timeout:
            raise TimeoutError(f"Feature service job timed out after {timeout}s: {job_id}")
        print(f"[feature-service] job={job_id} status={status['status']}", flush=True)
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
