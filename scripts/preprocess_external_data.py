#!/usr/bin/env python
"""Preprocess an external h5ad file for CellAgent.

Example:
    conda run -n cellagent python scripts/preprocess_external_data.py \
        --config config/config.yaml \
        --input data/raw/example.h5ad
"""
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

from src.tools.preprocessor import ExternalDataPreprocessor
from src.tools.qc_report import write_preprocessing_qc_report


def load_config(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess external AnnData for CellAgent.")
    parser.add_argument("--input", required=False, help="Input .h5ad path. Overrides config.pipeline.input_path.")
    parser.add_argument("--output", required=False, help="Explicit output .h5ad path.")
    parser.add_argument("--output-dir", required=False, help="Directory for default *_preprocessed.h5ad output.")
    parser.add_argument("--qc-report-dir", required=False, help="Directory for preprocessing QC report files.")
    parser.add_argument("--config", default="config/config.yaml", help="YAML config path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    full_cfg = load_config(args.config)
    pipeline_cfg = full_cfg.get("pipeline", {})
    cfg = full_cfg.get("preprocessing", full_cfg)

    output_root = Path(pipeline_cfg["output_root"]) if pipeline_cfg.get("output_root") else None
    input_path = args.input or pipeline_cfg.get("input_path")
    if not input_path:
        raise ValueError("Input h5ad is required via --input or pipeline.input_path.")

    output_path = args.output
    output_dir = args.output_dir or (str(output_root / "preprocessed") if output_root else None)
    qc_report_cfg = cfg.get("qc_report", {}) or {}
    qc_report_enabled = bool(qc_report_cfg.get("enabled", True))
    qc_report_dir = args.qc_report_dir or (str(output_root / "qc_report") if output_root else None)
    if qc_report_dir is None and output_dir:
        qc_report_dir = str(Path(output_dir).parent / "qc_report")

    # Path keys are pipeline-level concerns, not PreprocessConfig fields.
    cfg = dict(cfg)
    cfg.pop("qc_report", None)

    result = ExternalDataPreprocessor(cfg).run(
        input_path=input_path,
        output_path=output_path,
        output_dir=output_dir,
    )
    qc_outputs = {}
    if qc_report_enabled and qc_report_dir:
        qc_outputs = {
            key: str(value)
            for key, value in write_preprocessing_qc_report(
                result.adata,
                result.summary,
                qc_report_dir,
            ).items()
        }
    print(
        json.dumps(
            {"output_path": str(result.output_path), "summary": result.summary, "qc_report": qc_outputs},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
