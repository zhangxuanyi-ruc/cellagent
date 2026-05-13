#!/usr/bin/env python3
"""Audit effective marker coverage after CellMarker -> PanglaoDB fallback.

This diagnostic answers: for CL IDs known by the CellAgent mapper, how often
does marker retrieval still fail after applying the current fallback policy?
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.rag import RAGFacade


PANCREATIC_HINTS = ("pancreas", "pancreatic", "islet", "langerhans")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rag-config", default="config/rag_sources.yaml")
    parser.add_argument("--output-dir", default="output/marker_fallback_coverage")
    parser.add_argument("--min-markers", type=int, default=5)
    parser.add_argument(
        "--infer-tissue-context",
        action="store_true",
        help="Use a simple pancreatic metadata context when CL synonyms contain pancreatic/islet hints.",
    )
    return parser.parse_args()


def _cell_type_synonyms(rag: RAGFacade) -> dict[str, list[str]]:
    return dict(getattr(rag.mapper.cell_type, "_cl_to_synonyms", {}))


def _cell_type_name(synonyms: list[str]) -> str:
    return synonyms[0] if synonyms else ""


def _infer_metadata(synonyms: list[str], enabled: bool) -> dict[str, str]:
    if not enabled:
        return {"species": "human"}
    text = " ".join(synonyms).lower()
    if any(hint in text for hint in PANCREATIC_HINTS):
        return {"species": "human", "tissue": "islet of Langerhans"}
    return {"species": "human"}


def _marker_db_cl_sets(rag: RAGFacade) -> dict[str, set[str]]:
    cellmarker_cls: set[str] = set()
    panglao_cls: set[str] = set()

    if rag.cellmarker is not None:
        cellmarker_df = rag.cellmarker._load()
        if "cell_type_cl_id" in cellmarker_df.columns:
            cellmarker_cls = {
                str(v)
                for v in cellmarker_df["cell_type_cl_id"].dropna().astype(str)
                if v and v != "None" and v != "nan"
            }
    if rag.panglao is not None:
        panglao_df = rag.panglao._load_annotations()
        if "cell_type_cl_id" in panglao_df.columns:
            panglao_cls = {
                str(v)
                for v in panglao_df["cell_type_cl_id"].dropna().astype(str)
                if v and v != "None" and v != "nan"
            }

    return {
        "cellmarker": cellmarker_cls,
        "panglaodb": panglao_cls,
        "union": cellmarker_cls | panglao_cls,
        "intersection": cellmarker_cls & panglao_cls,
    }


def _marker_db_intersection(db_sets: dict[str, set[str]]) -> dict[str, Any]:
    cellmarker_cls = db_sets["cellmarker"]
    panglao_cls = db_sets["panglaodb"]
    return {
        "cellmarker_unique_cl": len(cellmarker_cls),
        "panglaodb_unique_cl": len(panglao_cls),
        "union_unique_cl": len(cellmarker_cls | panglao_cls),
        "intersection_unique_cl": len(cellmarker_cls & panglao_cls),
        "cellmarker_only_cl": len(cellmarker_cls - panglao_cls),
        "panglaodb_only_cl": len(panglao_cls - cellmarker_cls),
    }


def _classify(cellmarker_count: int, panglao_count: int, min_markers: int) -> str:
    total = cellmarker_count + panglao_count
    if total == 0:
        return "both_failed"
    if cellmarker_count >= min_markers:
        return "cellmarker_sufficient"
    if cellmarker_count == 0 and panglao_count > 0:
        return "cellmarker_empty_panglao_rescued"
    if 0 < cellmarker_count < min_markers and panglao_count > 0:
        return "cellmarker_low_panglao_supplemented"
    return "cellmarker_low_no_panglao"


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rag = RAGFacade(args.rag_config)
    cl_to_synonyms = _cell_type_synonyms(rag)

    rows: list[dict[str, Any]] = []
    for cl_id, synonyms in sorted(cl_to_synonyms.items()):
        metadata = _infer_metadata(synonyms, enabled=args.infer_tissue_context)
        markers = rag.query_markers(
            cl_id,
            species="human",
            top_k=None,
            min_markers=args.min_markers,
            metadata=metadata,
        )
        source_counts = Counter(marker.source for marker in markers)
        cellmarker_count = int(source_counts.get("cellmarker", 0))
        panglao_count = int(source_counts.get("panglao", 0))
        total = int(len(markers))
        rows.append(
            {
                "cl_id": cl_id,
                "cell_type_name": _cell_type_name(synonyms),
                "n_synonyms": len(synonyms),
                "metadata_used": json.dumps(metadata, ensure_ascii=False),
                "cellmarker_markers": cellmarker_count,
                "panglaodb_markers": panglao_count,
                "total_markers": total,
                "coverage_class": _classify(cellmarker_count, panglao_count, args.min_markers),
                "example_markers": ",".join([marker.gene for marker in markers[:20]]),
            }
        )

    df = pd.DataFrame(rows)
    class_counts = df["coverage_class"].value_counts().to_dict()
    n_total = int(len(df))
    n_failed = int(class_counts.get("both_failed", 0))
    n_rescued = int(class_counts.get("cellmarker_empty_panglao_rescued", 0))
    n_supplemented = int(class_counts.get("cellmarker_low_panglao_supplemented", 0))
    db_sets = _marker_db_cl_sets(rag)
    query_nonempty_cls = set(df.loc[df["total_markers"] > 0, "cl_id"].astype(str))
    db_has_but_query_failed = sorted(db_sets["union"] - query_nonempty_cls)

    summary = {
        "n_cl_ids_in_mapper": n_total,
        "min_markers": args.min_markers,
        "infer_tissue_context": bool(args.infer_tissue_context),
        "coverage_class_counts": class_counts,
        "both_failed_fraction": n_failed / n_total if n_total else None,
        "panglaodb_rescued_fraction": n_rescued / n_total if n_total else None,
        "panglaodb_supplemented_fraction": n_supplemented / n_total if n_total else None,
        "marker_db_cl_intersection": _marker_db_intersection(db_sets),
        "marker_db_effective_query_gap": {
            "db_union_cl": len(db_sets["union"]),
            "db_union_covered_by_current_query": len(db_sets["union"] & query_nonempty_cls),
            "db_has_but_current_query_failed": len(db_has_but_query_failed),
            "examples": [
                {
                    "cl_id": cl_id,
                    "cell_type_name": (rag.mapper.cell_type_synonyms(cl_id) or [""])[0],
                }
                for cl_id in db_has_but_query_failed[:100]
            ],
        },
        "notes": [
            "both_failed means CellMarker plus PanglaoDB fallback returned no marker genes.",
            "This is an upper-bound audit over all CL IDs known by the mapper, not only common scRNA-seq annotation labels.",
            "infer_tissue_context currently only adds a pancreatic/islet metadata hint for CL synonyms containing pancreatic/islet terms.",
        ],
    }

    df.to_csv(out_dir / "marker_fallback_coverage.csv", index=False)
    (out_dir / "marker_fallback_coverage_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(out_dir), "summary": summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
