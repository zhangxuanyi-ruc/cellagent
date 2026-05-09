#!/usr/bin/env python
"""Write preprocessing QC report from an existing CellAgent preprocessed h5ad."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.qc_report import write_preprocessing_qc_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write CellAgent preprocessing QC report from *_preprocessed.h5ad.")
    parser.add_argument("--input", required=True, help="CellAgent *_preprocessed.h5ad path.")
    parser.add_argument("--output-dir", required=True, help="QC report output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adata = ad.read_h5ad(args.input)
    summary = dict(adata.uns.get("cellagent_preprocessing", {}))
    outputs = write_preprocessing_qc_report(adata, summary, args.output_dir)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
