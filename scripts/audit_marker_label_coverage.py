#!/usr/bin/env python
"""Audit marker evidence coverage for dataset cell-type labels."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import anndata as ad
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.tools.marker_registry import MarkerRegistry  # noqa: E402
from src.tools.rag import RAGFacade  # noqa: E402


DEFAULT_LABEL_COLUMNS = [
    "cell_type",
    "author_cell_type",
    "author_cell_types",
    "annotation",
    "CellType",
    "ImmuneCell_labels",
    "subtype",
    "Subtype",
    "precisest_label",
    "broad_lineage",
]
INVALID_LABELS = {"", "nan", "none", "na", "n/a", "unknown", "unassigned"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--rag-config", default="config/rag_sources.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-columns", nargs="*", default=None)
    parser.add_argument("--top-n", type=int, default=30, help="Top labels per column to include.")
    parser.add_argument("--min-marker-count", type=int, default=3)
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def is_invalid_label(label: str) -> bool:
    return label.strip().lower() in INVALID_LABELS


def registry_query_candidates(rag: RAGFacade, label: str) -> tuple[str | None, list[str]]:
    candidates: list[str] = [label]
    cl_id = rag.mapper.normalize_cell_type(label)
    if cl_id:
        name = rag.get_cell_type_name(cl_id)
        if name:
            candidates.append(name)
        try:
            candidates.extend(rag.mapper.cell_type_synonyms(cl_id))
        except Exception:
            pass
    deduped = list(dict.fromkeys(str(x) for x in candidates if x))
    return cl_id, deduped


def audit_label(
    label: str,
    count: int,
    label_column: str,
    dataset_name: str,
    rag: RAGFacade,
    registry: MarkerRegistry,
    min_marker_count: int,
) -> dict[str, Any]:
    label = str(label)
    if is_invalid_label(label):
        return {
            "dataset": dataset_name,
            "label_column": label_column,
            "label": label,
            "count": int(count),
            "status": "INVALID_OR_UNKNOWN_LABEL",
            "cell_type_cl_id": None,
            "registry_hit": False,
            "registry_marker_count": 0,
            "registry_matched_cell_type": None,
            "rag_hit": False,
            "rag_marker_count": 0,
            "marker_source_priority": "none",
            "marker_count_priority": 0,
            "marker_count_sufficient": False,
        }

    cl_id, candidates = registry_query_candidates(rag, label)
    registry_hit = registry.query(candidates)
    registry_markers = registry_hit["markers"] if registry_hit else []

    rag_records = rag.query_markers(label, top_k=None, min_markers=5)
    rag_genes = sorted({
        str(r.gene_normalized or r.gene).strip().upper()
        for r in rag_records
        if str(r.gene_normalized or r.gene).strip()
    })

    if registry_markers:
        source = "sctype_registry"
        marker_count = len(registry_markers)
        status = "REGISTRY_HIT"
    elif rag_genes:
        source = "rag_fallback"
        marker_count = len(rag_genes)
        status = "RAG_FALLBACK_HIT"
    elif cl_id:
        source = "none"
        marker_count = 0
        status = "INSUFFICIENT_DATABASE_EVIDENCE"
    else:
        source = "none"
        marker_count = 0
        status = "UNMAPPED_CELL_TYPE"

    return {
        "dataset": dataset_name,
        "label_column": label_column,
        "label": label,
        "count": int(count),
        "status": status,
        "cell_type_cl_id": cl_id,
        "registry_hit": bool(registry_markers),
        "registry_marker_count": len(registry_markers),
        "registry_matched_cell_type": registry_hit.get("matched_cell_type") if registry_hit else None,
        "rag_hit": bool(rag_genes),
        "rag_marker_count": len(rag_genes),
        "marker_source_priority": source,
        "marker_count_priority": marker_count,
        "marker_count_sufficient": marker_count >= int(min_marker_count),
        "registry_markers_preview": ",".join(registry_markers[:20]),
        "rag_markers_preview": ",".join(rag_genes[:20]),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    registry_cfg = cfg.get("marker_registry", {})
    registry = MarkerRegistry(registry_cfg.get("path"))
    rag = RAGFacade(args.rag_config)

    adata = ad.read_h5ad(args.h5ad, backed="r")
    obs = adata.obs
    label_columns = args.label_columns or [c for c in DEFAULT_LABEL_COLUMNS if c in obs.columns]
    rows: list[dict[str, Any]] = []
    for col in label_columns:
        if col not in obs.columns:
            continue
        counts = obs[col].astype(str).value_counts(dropna=False).head(args.top_n)
        for label, count in counts.items():
            rows.append(
                audit_label(
                    label=str(label),
                    count=int(count),
                    label_column=col,
                    dataset_name=args.dataset_name,
                    rag=rag,
                    registry=registry,
                    min_marker_count=args.min_marker_count,
                )
            )
    adata.file.close()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    detail_path = output_dir / f"{args.dataset_name}_marker_label_coverage.csv"
    summary_path = output_dir / f"{args.dataset_name}_marker_label_coverage_summary.json"
    df.to_csv(detail_path, index=False)

    valid = df[df["status"] != "INVALID_OR_UNKNOWN_LABEL"] if not df.empty else df
    summary = {
        "dataset": args.dataset_name,
        "h5ad": str(args.h5ad),
        "label_columns": label_columns,
        "n_rows": int(len(df)),
        "n_valid_rows": int(len(valid)),
        "status_counts": df["status"].value_counts().to_dict() if not df.empty else {},
        "priority_source_counts": df["marker_source_priority"].value_counts().to_dict() if not df.empty else {},
        "weighted_cell_counts_by_source": (
            df.groupby("marker_source_priority")["count"].sum().sort_values(ascending=False).to_dict()
            if not df.empty
            else {}
        ),
        "registry_used_rate_valid": float((valid["marker_source_priority"] == "sctype_registry").mean()) if len(valid) else 0.0,
        "rag_fallback_used_rate_valid": float((valid["marker_source_priority"] == "rag_fallback").mean()) if len(valid) else 0.0,
        "insufficient_or_unmapped_rate_valid": float(valid["marker_source_priority"].eq("none").mean()) if len(valid) else 0.0,
        "detail_csv": str(detail_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not df.empty:
        print("\nTop labels needing attention:")
        needs = df[
            (df["status"].isin(["INSUFFICIENT_DATABASE_EVIDENCE", "UNMAPPED_CELL_TYPE"]))
            | (~df["marker_count_sufficient"])
        ].sort_values("count", ascending=False)
        print(
            needs[
                [
                    "label_column",
                    "label",
                    "count",
                    "status",
                    "cell_type_cl_id",
                    "marker_source_priority",
                    "registry_marker_count",
                    "rag_marker_count",
                ]
            ].head(30).to_string(index=False)
        )


if __name__ == "__main__":
    main()
