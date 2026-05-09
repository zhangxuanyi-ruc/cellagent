#!/usr/bin/env python
"""Run per-cluster differential expression for CellAgent."""
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

from src.tools.de_analysis import config_from_dict, run_de_analysis


def load_de_config(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DE analysis for each Leiden cluster.")
    parser.add_argument("--preprocessed", required=True, help="CellAgent *_preprocessed.h5ad expression dataset.")
    parser.add_argument("--clusters", required=True, help="Cluster CSV or clustered h5ad containing cell_id/leiden assignments.")
    parser.add_argument("--config", default="config/config.yaml", help="YAML config path.")
    parser.add_argument("--output-dir", default=None, help="Override de_analysis.output_dir.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    full_cfg = load_de_config(args.config)
    pipeline_cfg = full_cfg.get("pipeline", {})
    output_root = Path(pipeline_cfg["output_root"]) if pipeline_cfg.get("output_root") else None
    output_dir = args.output_dir or (str(output_root / "de") if output_root else None)
    cfg = config_from_dict(full_cfg.get("de_analysis", full_cfg))
    outputs = run_de_analysis(
        preprocessed_h5ad=args.preprocessed,
        clusters_path=args.clusters,
        cfg=cfg,
        output_dir=output_dir,
    )
    print(json.dumps({k: str(v) for k, v in outputs.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
