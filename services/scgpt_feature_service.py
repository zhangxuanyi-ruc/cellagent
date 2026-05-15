#!/usr/bin/env python
"""HTTP service for CellAgent cell encoder feature extraction.

Run this file in the scGPT-compatible environment. The CellAgent environment can
then request features without importing torch/scGPT dependencies directly.

CUDA isolation: each job is executed in an independent Python subprocess via
`scripts/extract_cell_encoder_features.py`. This way a CUDA error in one job
(illegal memory access, OOM, etc.) cannot poison the service's CUDA context
because the subprocess owns its own context and dies with the process.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.extract_cell_encoder_features import DEFAULT_LOCAL_SCGPT_ROOT

EXTRACT_SCRIPT = PROJECT_ROOT / "scripts" / "extract_cell_encoder_features.py"


class ExtractRequest(BaseModel):
    input_h5ad: str = Field(..., description="CellAgent *_preprocessed.h5ad path.")
    feature_dir: Optional[str] = Field(None, description="Output feature directory. Defaults to pipeline.output_root/features.")
    model_path: Optional[str] = None
    vocab_path: Optional[str] = None
    feature_layer: Optional[str] = None
    gene_id_column: Optional[str] = None
    n_genes: Optional[int] = None
    batch_size: Optional[int] = None
    output_format: Optional[str] = None
    device: Optional[str] = None
    scgpt_root: Optional[str] = None


class JobRecord(BaseModel):
    job_id: str
    status: str
    created_at: float
    updated_at: float
    request: dict[str, Any]
    feature_path: Optional[str] = None
    error: Optional[str] = None
    traceback: Optional[str] = None


def load_full_config(config_path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}


def resolve_extract_kwargs(request: ExtractRequest, cfg: dict[str, Any]) -> dict[str, Any]:
    encoder_cfg = cfg.get("cell_encoder", {})
    pipeline_cfg = cfg.get("pipeline", {})
    output_root = Path(pipeline_cfg["output_root"]) if pipeline_cfg.get("output_root") else None
    feature_dir = request.feature_dir or (str(output_root / "features") if output_root else "output/features")
    return {
        "preprocessed_h5ad": request.input_h5ad,
        "output_dir": feature_dir,
        "model_path": request.model_path or encoder_cfg["model_path"],
        "vocab_path": request.vocab_path or encoder_cfg["vocab_path"],
        "feature_layer": request.feature_layer or encoder_cfg.get("feature_layer", "X_binned"),
        "gene_id_column": request.gene_id_column or encoder_cfg.get("gene_id_column", "scgpt_id"),
        "n_genes": int(request.n_genes or encoder_cfg.get("n_genes", 1200)),
        "batch_size": int(request.batch_size or encoder_cfg.get("batch_size", 1024)),
        "output_format": request.output_format or encoder_cfg.get("output_format", "npz"),
        "device": request.device or encoder_cfg.get("device"),
        "scgpt_root": request.scgpt_root or encoder_cfg.get("scgpt_root", DEFAULT_LOCAL_SCGPT_ROOT),
    }


def run_feature_extraction_subprocess(
    kwargs: dict[str, Any],
    config_path: str | Path,
    gpu_id: Any = None,
) -> Path:
    """Run feature extraction in an isolated Python subprocess.

    Each job gets its own Python process with its own CUDA context, so a CUDA
    error in one job cannot corrupt the service's main process state.
    """
    cmd = [
        sys.executable,
        str(EXTRACT_SCRIPT),
        "--input", str(kwargs["preprocessed_h5ad"]),
        "--config", str(config_path),
        "--feature-dir", str(kwargs["output_dir"]),
        "--model-path", str(kwargs["model_path"]),
        "--vocab-path", str(kwargs["vocab_path"]),
        "--batch-size", str(kwargs["batch_size"]),
        "--output-format", str(kwargs["output_format"]),
        "--scgpt-root", str(kwargs["scgpt_root"]),
    ]
    if kwargs.get("device"):
        cmd.extend(["--device", str(kwargs["device"])])

    env = os.environ.copy()
    if gpu_id is not None and str(gpu_id).lower() not in {"", "none", "null"}:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Feature extraction subprocess failed.\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout: {proc.stdout}\n"
            f"stderr: {proc.stderr}"
        )

    # extract_cell_encoder_features.py prints {"feature_path": "..."} as the
    # last JSON object on stdout.
    feature_path: Path | None = None
    for token in proc.stdout.splitlines():
        token = token.strip()
        if not token:
            continue
    # Try to find the JSON object containing feature_path in stdout.
    decoder = json.JSONDecoder()
    text = proc.stdout
    idx = 0
    while idx < len(text):
        try:
            obj, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            idx += 1
            continue
        if isinstance(obj, dict) and obj.get("feature_path"):
            feature_path = Path(obj["feature_path"])
        idx = end
    if feature_path is None:
        raise RuntimeError(
            "Feature extraction subprocess returned no feature_path.\n"
            f"stdout: {proc.stdout}\n"
            f"stderr: {proc.stderr}"
        )
    return feature_path


def create_app(config_path: str | Path) -> FastAPI:
    cfg = load_full_config(config_path)
    service_cfg = cfg.get("feature_service", {})
    gpu_id = service_cfg.get("gpu_id", cfg.get("cell_encoder", {}).get("gpu_id"))
    if gpu_id is not None and str(gpu_id).lower() not in {"", "none", "null"}:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    app = FastAPI(title="CellAgent scGPT Feature Service", version="0.1.0")
    executor = ThreadPoolExecutor(max_workers=int(service_cfg.get("max_workers", 1)))
    jobs: dict[str, JobRecord] = {}

    def run_job(job_id: str, request: ExtractRequest) -> None:
        job = jobs[job_id]
        job.status = "running"
        job.updated_at = time.time()
        try:
            kwargs = resolve_extract_kwargs(request, cfg)
            # Isolate each job in its own Python subprocess: a CUDA error in one
            # job (illegal memory access, OOM, etc.) cannot poison the service's
            # main process state because the subprocess owns its own CUDA context.
            feature_path = run_feature_extraction_subprocess(
                kwargs,
                config_path=config_path,
                gpu_id=gpu_id,
            )
            job.feature_path = str(feature_path)
            job.status = "succeeded"
            job.updated_at = time.time()
        except Exception as exc:  # API boundary must return structured failures.
            job.status = "failed"
            job.error = str(exc)
            job.traceback = traceback.format_exc()
            job.updated_at = time.time()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "config_path": str(config_path),
            "gpu_id": gpu_id,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "jobs": {status: sum(1 for job in jobs.values() if job.status == status) for status in ["queued", "running", "succeeded", "failed"]},
        }

    @app.post("/extract")
    def submit_extract(request: ExtractRequest) -> dict[str, Any]:
        input_path = Path(request.input_h5ad)
        if not input_path.exists():
            raise HTTPException(status_code=404, detail=f"input_h5ad not found: {input_path}")
        job_id = uuid.uuid4().hex
        now = time.time()
        jobs[job_id] = JobRecord(
            job_id=job_id,
            status="queued",
            created_at=now,
            updated_at=now,
            request=request.model_dump(),
        )
        executor.submit(run_job, job_id, request)
        return {"job_id": job_id, "status": "queued", "status_url": f"/jobs/{job_id}"}

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail=f"job not found: {job_id}")
        return json.loads(jobs[job_id].model_dump_json())

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CellAgent scGPT feature extraction service.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config/config.yaml"))
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    import uvicorn

    args = parse_args()
    cfg = load_full_config(args.config)
    service_cfg = cfg.get("feature_service", {})
    host = args.host or service_cfg.get("host", "127.0.0.1")
    port = int(args.port or service_cfg.get("port", 18080))
    app = create_app(args.config)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
