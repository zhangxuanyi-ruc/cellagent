#!/usr/bin/env python
"""Audit marker database cell-name alignment against Cell Ontology.

This script is diagnostic only. It does not modify mapper files or marker
loaders. The goal is to identify CL/marker naming mismatches before tuning
marker scoring thresholds.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.rag import RAGFacade  # noqa: E402


TISSUE_HINT_TO_CL = {
    ("alpha cell", "pancreas"): "CL:0000171",
    ("alpha cell", "pancreatic"): "CL:0000171",
    ("alpha cell", "islet"): "CL:0000171",
    ("delta cell", "pancreas"): "CL:0000173",
    ("delta cell", "pancreatic"): "CL:0000173",
    ("delta cell", "islet"): "CL:0000173",
    ("pp cell", "pancreas"): "CL:0002275",
    ("pp cell", "pancreatic"): "CL:0002275",
    ("pancreatic polypeptide cell", "pancreas"): "CL:0002275",
}

AMBIGUOUS_SHORT_NAMES = {
    "alpha cell",
    "delta cell",
    "gamma cell",
    "pp cell",
    "a cell",
    "b cell",
    "d cell",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rag-config", default="config/rag_sources.yaml")
    parser.add_argument("--output-dir", default=None, help="Default: <project>/output/marker_alignment_audit")
    parser.add_argument("--top-unmapped", type=int, default=100)
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def contextual_cl_suggestion(cell_name: str, tissue: str) -> str | None:
    cell_norm = normalize_text(cell_name)
    tissue_norm = normalize_text(tissue)
    for (name, hint), cl_id in TISSUE_HINT_TO_CL.items():
        if cell_norm == name and hint in tissue_norm:
            return cl_id
    return None


def audit_cellmarker(rag: RAGFacade) -> tuple[pd.DataFrame, dict[str, Any]]:
    if rag.cellmarker is None:
        return pd.DataFrame(), {"available": False}
    df = rag.cellmarker._load()
    if df.empty:
        return pd.DataFrame(), {"available": True, "empty": True}

    rows: list[dict[str, Any]] = []
    grouped = df.groupby(["cellName", "tissueType"], dropna=False)
    for (cell_name, tissue), sub in grouped:
        cell_name_s = str(cell_name)
        tissue_s = "" if pd.isna(tissue) else str(tissue)
        current_cl = None
        if "cell_type_cl_id" in sub.columns:
            values = [v for v in sub["cell_type_cl_id"].dropna().astype(str).unique().tolist() if v and v != "None"]
            current_cl = values[0] if values else None
        suggested = contextual_cl_suggestion(cell_name_s, tissue_s)
        rows.append(
            {
                "source": "cellmarker",
                "cell_name": cell_name_s,
                "tissue": tissue_s,
                "current_cl_id": current_cl,
                "current_cl_name": rag.get_cell_type_name(current_cl) if current_cl else None,
                "is_mapped": current_cl is not None,
                "is_ambiguous_short_name": normalize_text(cell_name_s) in AMBIGUOUS_SHORT_NAMES,
                "suggested_contextual_cl_id": suggested,
                "suggested_contextual_cl_name": rag.get_cell_type_name(suggested) if suggested else None,
                "n_marker_rows": int(len(sub)),
                "n_unique_genes": int(sub["gene_normalized"].dropna().nunique()) if "gene_normalized" in sub else int(sub["geneSymbol"].dropna().nunique()),
                "example_genes": ",".join(
                    list(dict.fromkeys(
                        (sub["gene_normalized"] if "gene_normalized" in sub else sub["geneSymbol"])
                        .dropna()
                        .astype(str)
                        .tolist()
                    ))[:20]
                ),
                "warning": build_warning(cell_name_s, tissue_s, current_cl, suggested),
            }
        )
    audit = pd.DataFrame(rows)
    summary = {
        "available": True,
        "n_rows": int(len(df)),
        "n_unique_cell_names": int(df["cellName"].nunique()),
        "n_cell_name_tissue_pairs": int(len(audit)),
        "n_mapped_pairs": int(audit["is_mapped"].sum()),
        "n_unmapped_pairs": int((~audit["is_mapped"]).sum()),
        "n_ambiguous_pairs": int(audit["is_ambiguous_short_name"].sum()),
        "n_contextual_suggestions": int(audit["suggested_contextual_cl_id"].notna().sum()),
    }
    return audit, summary


def audit_panglao(rag: RAGFacade) -> tuple[pd.DataFrame, dict[str, Any]]:
    if rag.panglao is None:
        return pd.DataFrame(), {"available": False}
    annot = rag.panglao._load_annotations()
    markers = rag.panglao._load_markers()
    if annot.empty:
        return pd.DataFrame(), {"available": True, "empty": True}

    rows: list[dict[str, Any]] = []
    marker_counts = Counter()
    if not markers.empty:
        marker_counts = Counter(zip(markers["sra"], markers["srs"], markers["cluster"]))

    for _, row in annot.drop_duplicates(["cell_type", "sra", "srs", "cluster"]).iterrows():
        cell_name = str(row["cell_type"])
        current_cl = row.get("cell_type_cl_id")
        if pd.isna(current_cl):
            current_cl = None
        key = (row["sra"], row["srs"], row["cluster"])
        rows.append(
            {
                "source": "panglaodb",
                "cell_name": cell_name,
                "tissue": "",
                "current_cl_id": current_cl,
                "current_cl_name": rag.get_cell_type_name(current_cl) if current_cl else None,
                "is_mapped": current_cl is not None,
                "is_ambiguous_short_name": normalize_text(cell_name) in AMBIGUOUS_SHORT_NAMES,
                "suggested_contextual_cl_id": None,
                "suggested_contextual_cl_name": None,
                "n_marker_rows": int(marker_counts.get(key, 0)),
                "n_unique_genes": int(marker_counts.get(key, 0)),
                "example_genes": "",
                "warning": build_warning(cell_name, "", current_cl, None),
            }
        )
    audit = pd.DataFrame(rows)
    summary = {
        "available": True,
        "n_annotation_rows": int(len(annot)),
        "n_unique_cell_names": int(annot["cell_type"].nunique()),
        "n_annotation_clusters": int(len(audit)),
        "n_mapped_clusters": int(audit["is_mapped"].sum()),
        "n_unmapped_clusters": int((~audit["is_mapped"]).sum()),
        "n_ambiguous_clusters": int(audit["is_ambiguous_short_name"].sum()),
    }
    return audit, summary


def build_warning(cell_name: str, tissue: str, current_cl: str | None, suggested_cl: str | None) -> str:
    warnings = []
    name_norm = normalize_text(cell_name)
    if not current_cl:
        warnings.append("unmapped")
    if name_norm in AMBIGUOUS_SHORT_NAMES:
        warnings.append("ambiguous_short_name")
    if suggested_cl and current_cl and suggested_cl != current_cl:
        warnings.append(f"context_conflict:{current_cl}->{suggested_cl}")
    elif suggested_cl and not current_cl:
        warnings.append(f"context_suggestion:{suggested_cl}")
    if tissue and any(token in normalize_text(tissue) for token in ("pancreas", "pancreatic", "islet")) and name_norm in {"alpha cell", "delta cell"}:
        warnings.append("pancreatic_context_requires_override")
    return ";".join(warnings)


def top_unmapped(audit: pd.DataFrame, n: int) -> list[dict[str, Any]]:
    if audit.empty:
        return []
    unmapped = audit[~audit["is_mapped"]]
    if unmapped.empty:
        return []
    grouped = (
        unmapped.groupby(["source", "cell_name"], dropna=False)
        .agg(n_pairs=("cell_name", "size"), n_marker_rows=("n_marker_rows", "sum"))
        .reset_index()
        .sort_values(["n_marker_rows", "n_pairs"], ascending=False)
        .head(n)
    )
    return grouped.to_dict(orient="records")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "output" / "marker_alignment_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    rag = RAGFacade(args.rag_config)
    cellmarker_audit, cellmarker_summary = audit_cellmarker(rag)
    panglao_audit, panglao_summary = audit_panglao(rag)

    combined = pd.concat([df for df in [cellmarker_audit, panglao_audit] if not df.empty], ignore_index=True)
    if not cellmarker_audit.empty:
        cellmarker_audit.to_csv(out_dir / "cellmarker_alignment_audit.csv", index=False)
    if not panglao_audit.empty:
        panglao_audit.to_csv(out_dir / "panglaodb_alignment_audit.csv", index=False)
    if not combined.empty:
        combined.to_csv(out_dir / "marker_alignment_audit.csv", index=False)

    summary = {
        "cellmarker": cellmarker_summary,
        "panglaodb": panglao_summary,
        "top_unmapped": top_unmapped(combined, args.top_unmapped),
        "notes": [
            "This audit is diagnostic only.",
            "Contextual suggestions must be used only for marker retrieval, not for overwriting the final CL ID.",
            "Ambiguous short names such as alpha cell require tissue context before CL resolution.",
        ],
    }
    (out_dir / "marker_alignment_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(out_dir), "summary": summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
