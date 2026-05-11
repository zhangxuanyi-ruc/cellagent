#!/usr/bin/env python
"""Write the CellAgent pipeline artifact manifest."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.manifest import PipelineManifest, default_manifest_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output", default=None, help="Default: <pipeline.output_root>/pipeline_manifest.json")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--preprocessed-h5ad", default=None)
    parser.add_argument("--qc-report-dir", default=None)
    parser.add_argument("--feature-npz", default=None)
    parser.add_argument("--clusters-csv", default=None)
    parser.add_argument("--clustering-metrics-json", default=None)
    parser.add_argument("--de-summary-csv", default=None)
    parser.add_argument("--de-dir", default=None)
    parser.add_argument("--multimodal-prior-json", default=None)
    parser.add_argument("--reasoning-dir", default=None)
    parser.add_argument("--final-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    cfg: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    pipeline = cfg.get("pipeline", {})
    if not pipeline.get("input_path") or not pipeline.get("output_root"):
        raise ValueError("config.pipeline.input_path and config.pipeline.output_root are required.")

    output_root = Path(pipeline["output_root"])
    input_h5ad = Path(pipeline["input_path"])
    run_id = args.run_id or input_h5ad.stem
    manifest = PipelineManifest(
        run_id=run_id,
        input_h5ad=str(input_h5ad),
        output_root=str(output_root),
        config_path=str(config_path),
        preprocessed_h5ad=args.preprocessed_h5ad,
        qc_report_dir=args.qc_report_dir,
        feature_npz=args.feature_npz,
        clusters_csv=args.clusters_csv,
        clustering_metrics_json=args.clustering_metrics_json,
        de_summary_csv=args.de_summary_csv,
        de_dir=args.de_dir,
        multimodal_prior_json=args.multimodal_prior_json,
        reasoning_dir=args.reasoning_dir,
        final_dir=args.final_dir,
        config_snapshot=cfg,
    )
    output_path = Path(args.output) if args.output else default_manifest_path(output_root)
    manifest.write(output_path)
    print(json.dumps({"manifest": str(output_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
