#!/usr/bin/env python3
"""Build unified CL-ID keyed marker mapper from CellMarker and PanglaoDB."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.rag import RAGFacade


DEFAULT_OUTPUT = PROJECT_ROOT / "rag" / "mappers" / "marker_celltype_to_genes.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rag-config", default=str(PROJECT_ROOT / "config" / "rag_sources.yaml"))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--audit-csv", default=None)
    parser.add_argument("--summary-json", default=None)
    return parser.parse_args()


def _clean_str(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "na"}:
        return None
    return text


def _add_gene(
    bucket: dict[str, dict[str, set[str]]],
    cl_id: str | None,
    source: str,
    gene: str | None,
) -> None:
    cl_id = _clean_str(cl_id)
    gene = _clean_str(gene)
    if not cl_id or not gene:
        return
    bucket[cl_id][source].add(gene)


def _canonical_cl_id(value: Any) -> str | None:
    text = _clean_str(value)
    if not text:
        return None
    text = text.replace("_", ":")
    if text.upper().startswith("CL:"):
        prefix, _, suffix = text.partition(":")
        if suffix:
            return f"{prefix.upper()}:{suffix}"
    return None


def _row_cellmarker_cl_id(row: pd.Series) -> str | None:
    return _canonical_cl_id(row.get("CellOntologyID")) or _canonical_cl_id(row.get("cell_type_cl_id"))


def _cellmarker_unmapped(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "cellName" not in df.columns:
        return pd.DataFrame()
    work = df.copy()
    work["effective_cl_id"] = work.apply(_row_cellmarker_cl_id, axis=1)
    if "speciesType" in work.columns:
        work = work[work["speciesType"].astype(str).str.lower().str.contains("human", na=False)]
    unmapped = work[work["effective_cl_id"].isna()]
    if unmapped.empty:
        return pd.DataFrame()
    return (
        unmapped.groupby("cellName", dropna=False)
        .agg(n_rows=("cellName", "size"), n_unique_genes=("gene_normalized", "nunique"))
        .reset_index()
        .sort_values(["n_rows", "n_unique_genes"], ascending=False)
    )


def _panglao_unmapped(annot_df: pd.DataFrame) -> pd.DataFrame:
    if annot_df.empty or "cell_type" not in annot_df.columns:
        return pd.DataFrame()
    work = annot_df.copy()
    if "cell_type_cl_id" not in work.columns:
        work["cell_type_cl_id"] = None
    unmapped = work[work["cell_type_cl_id"].isna()]
    if unmapped.empty:
        return pd.DataFrame()
    return (
        unmapped.groupby("cell_type", dropna=False)
        .agg(n_clusters=("cell_type", "size"))
        .reset_index()
        .sort_values("n_clusters", ascending=False)
    )


def _build_from_cellmarker(rag: RAGFacade, bucket: dict[str, dict[str, set[str]]]) -> pd.DataFrame:
    if rag.cellmarker is None:
        return pd.DataFrame()
    df = rag.cellmarker._load()
    if df.empty:
        return df
    if "speciesType" in df.columns:
        df_for_mapper = df[df["speciesType"].astype(str).str.lower().str.contains("human", na=False)]
    else:
        df_for_mapper = df
    for _, row in df_for_mapper.iterrows():
        _add_gene(bucket, _row_cellmarker_cl_id(row), "cellmarker", row.get("gene_normalized") or row.get("geneSymbol"))
    return df


def _build_from_panglao(rag: RAGFacade, bucket: dict[str, dict[str, set[str]]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if rag.panglao is None:
        return pd.DataFrame(), pd.DataFrame()
    annot = rag.panglao._load_annotations()
    markers = rag.panglao._load_markers()
    if annot.empty or markers.empty:
        return annot, markers
    cols = ["sra", "srs", "cluster", "cell_type", "cell_type_cl_id"]
    merged = annot[cols].merge(
        markers[["sra", "srs", "cluster", "gene_symbol", "gene_normalized"]],
        on=["sra", "srs", "cluster"],
        how="inner",
    )
    for _, row in merged.iterrows():
        _add_gene(bucket, row.get("cell_type_cl_id"), "panglaodb", row.get("gene_normalized") or row.get("gene_symbol"))
    return annot, markers


def _mapper_payload(rag: RAGFacade, bucket: dict[str, dict[str, set[str]]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for cl_id in sorted(bucket):
        cellmarker_genes = sorted(bucket[cl_id].get("cellmarker", set()))
        panglao_genes = sorted(bucket[cl_id].get("panglaodb", set()))
        all_genes = sorted(set(cellmarker_genes) | set(panglao_genes))
        sources = []
        if cellmarker_genes:
            sources.append("cellmarker")
        if panglao_genes:
            sources.append("panglaodb")
        payload[cl_id] = {
            "cell_type_name": (rag.mapper.cell_type_synonyms(cl_id) or [cl_id])[0],
            "cellmarker_genes": cellmarker_genes,
            "panglaodb_genes": panglao_genes,
            "all_genes": all_genes,
            "sources": sources,
            "source_counts": {
                "cellmarker": len(cellmarker_genes),
                "panglaodb": len(panglao_genes),
                "all": len(all_genes),
            },
        }
    return payload


def _summary(
    mapper_payload: dict[str, Any],
    cellmarker_df: pd.DataFrame,
    panglao_annot: pd.DataFrame,
    panglao_markers: pd.DataFrame,
    cellmarker_unmapped: pd.DataFrame,
    panglao_unmapped: pd.DataFrame,
) -> dict[str, Any]:
    cellmarker_cls = {
        cl_id
        for cl_id, rec in mapper_payload.items()
        if rec["source_counts"]["cellmarker"] > 0
    }
    panglao_cls = {
        cl_id
        for cl_id, rec in mapper_payload.items()
        if rec["source_counts"]["panglaodb"] > 0
    }
    return {
        "n_cl_ids_with_any_marker": len(mapper_payload),
        "cellmarker": {
            "n_rows_loaded": int(len(cellmarker_df)),
            "n_unique_cell_names": int(cellmarker_df["cellName"].nunique()) if "cellName" in cellmarker_df else 0,
            "n_mapped_cl": len(cellmarker_cls),
            "n_unmapped_cell_names": int(len(cellmarker_unmapped)),
        },
        "panglaodb": {
            "n_annotation_rows": int(len(panglao_annot)),
            "n_marker_rows": int(len(panglao_markers)),
            "n_unique_cell_types": int(panglao_annot["cell_type"].nunique()) if "cell_type" in panglao_annot else 0,
            "n_mapped_cl": len(panglao_cls),
            "n_unmapped_cell_types": int(len(panglao_unmapped)),
        },
        "intersection": {
            "union_mapped_cl": len(cellmarker_cls | panglao_cls),
            "intersection_mapped_cl": len(cellmarker_cls & panglao_cls),
            "cellmarker_only_cl": len(cellmarker_cls - panglao_cls),
            "panglaodb_only_cl": len(panglao_cls - cellmarker_cls),
        },
        "notes": [
            "This mapper only includes marker database cell type names that normalize to CL IDs.",
            "Unmapped database cell type names are reported for mapper improvement, not resolved at runtime.",
        ],
    }


def _write_audit(
    audit_csv: Path,
    mapper_payload: dict[str, Any],
    cellmarker_unmapped: pd.DataFrame,
    panglao_unmapped: pd.DataFrame,
) -> None:
    rows: list[dict[str, Any]] = []
    for cl_id, rec in mapper_payload.items():
        rows.append(
            {
                "record_type": "mapped_cl",
                "source": ",".join(rec["sources"]),
                "cl_id": cl_id,
                "cell_type_name": rec["cell_type_name"],
                "cellmarker_genes": rec["source_counts"]["cellmarker"],
                "panglaodb_genes": rec["source_counts"]["panglaodb"],
                "all_genes": rec["source_counts"]["all"],
                "raw_cell_type": "",
                "n_records": "",
            }
        )
    for _, row in cellmarker_unmapped.iterrows():
        rows.append(
            {
                "record_type": "unmapped_cell_type",
                "source": "cellmarker",
                "cl_id": "",
                "cell_type_name": "",
                "cellmarker_genes": "",
                "panglaodb_genes": "",
                "all_genes": "",
                "raw_cell_type": row.get("cellName"),
                "n_records": int(row.get("n_rows", 0)),
            }
        )
    for _, row in panglao_unmapped.iterrows():
        rows.append(
            {
                "record_type": "unmapped_cell_type",
                "source": "panglaodb",
                "cl_id": "",
                "cell_type_name": "",
                "cellmarker_genes": "",
                "panglaodb_genes": "",
                "all_genes": "",
                "raw_cell_type": row.get("cell_type"),
                "n_records": int(row.get("n_clusters", 0)),
            }
        )
    pd.DataFrame(rows).to_csv(audit_csv, index=False)


def main() -> None:
    args = parse_args()
    rag = RAGFacade(args.rag_config)

    bucket: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    cellmarker_df = _build_from_cellmarker(rag, bucket)
    panglao_annot, panglao_markers = _build_from_panglao(rag, bucket)
    mapper_payload = _mapper_payload(rag, bucket)

    output_json = Path(args.output_json)
    audit_csv = Path(args.audit_csv) if args.audit_csv else output_json.with_name("marker_mapper_audit.csv")
    summary_json = Path(args.summary_json) if args.summary_json else output_json.with_name("marker_mapper_summary.json")
    output_json.parent.mkdir(parents=True, exist_ok=True)

    cellmarker_unmapped = _cellmarker_unmapped(cellmarker_df)
    panglao_unmapped = _panglao_unmapped(panglao_annot)
    summary = _summary(
        mapper_payload=mapper_payload,
        cellmarker_df=cellmarker_df,
        panglao_annot=panglao_annot,
        panglao_markers=panglao_markers,
        cellmarker_unmapped=cellmarker_unmapped,
        panglao_unmapped=panglao_unmapped,
    )

    output_json.write_text(json.dumps(mapper_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_audit(audit_csv, mapper_payload, cellmarker_unmapped, panglao_unmapped)

    print(json.dumps({"output_json": str(output_json), "audit_csv": str(audit_csv), "summary": summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
