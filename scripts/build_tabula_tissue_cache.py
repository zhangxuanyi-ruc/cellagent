#!/usr/bin/env python3
"""Build the primary Tabula Sapiens cell-type -> tissue evidence mapper.

The primary output is a mapper-style JSON file:
  CL_ID -> multiple standardized tissue-distribution records.

Audit outputs can also be written as a standardized table:
  cell_type, cell_type_cl_id, tissue, tissue_uberon_id, n_cells

This mapper is treated as the primary tissue-distribution source by RAGFacade;
HPA and ImmGen remain auxiliary evidence.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
RAG_MODULE_DIR = PROJECT_ROOT / "src" / "tools" / "rag"
if str(RAG_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_MODULE_DIR))

from mapper import Mapper  # noqa: E402


def load_config(config_path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(config_path).read_text())


def build_tabula_tissue_cache(
    h5ad_path: str | Path,
    mapper: Mapper,
    output_json: str | Path,
    output_parquet: str | Path,
    output_csv: str | Path | None = None,
    cell_type_col: str = "cell_type",
    tissue_col: str = "tissue",
    tissue_id_col: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    import anndata as ad

    h5ad_path = Path(h5ad_path)
    if not h5ad_path.exists():
        raise FileNotFoundError(f"Tabula Sapiens h5ad not found: {h5ad_path}")

    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        if tissue_id_col is None:
            tissue_id_col = detect_tissue_id_col(adata.obs.columns)

        required_cols = [cell_type_col]
        if tissue_id_col:
            required_cols.append(tissue_id_col)
        else:
            required_cols.append(tissue_col)

        missing = [c for c in required_cols if c not in adata.obs.columns]
        if missing:
            raise KeyError(f"Missing obs columns in {h5ad_path}: {missing}")

        obs_cols = list(dict.fromkeys([cell_type_col, tissue_col, tissue_id_col]))
        obs_cols = [c for c in obs_cols if c and c in adata.obs.columns]
        obs = adata.obs[obs_cols].copy()
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()

    obs[cell_type_col] = obs[cell_type_col].astype(str).str.strip()
    if tissue_col in obs.columns:
        obs[tissue_col] = obs[tissue_col].astype(str).str.strip()
    if tissue_id_col and tissue_id_col in obs.columns:
        obs[tissue_id_col] = obs[tissue_id_col].astype(str).str.strip()

    tissue_group_col = tissue_id_col if tissue_id_col and tissue_id_col in obs.columns else tissue_col
    obs = obs[(obs[cell_type_col] != "") & (obs[tissue_group_col] != "")]

    agg = (
        obs.groupby([cell_type_col, tissue_group_col], observed=True)
        .size()
        .reset_index(name="n_cells")
        .rename(columns={cell_type_col: "cell_type", tissue_group_col: "tissue_ontology_id"})
    )
    agg["cell_type_cl_id"] = agg["cell_type"].apply(mapper.normalize_cell_type)
    if tissue_id_col:
        agg["tissue_uberon_id"] = agg["tissue_ontology_id"].apply(normalize_ontology_id)
    else:
        agg["tissue_uberon_id"] = agg["tissue_ontology_id"].apply(mapper.normalize_tissue)
    agg["tissue"] = agg.apply(
        lambda row: resolve_tissue_name(
            row["tissue_uberon_id"],
            mapper,
            fallback=str(row["tissue_ontology_id"]),
        ),
        axis=1,
    )

    # Keep unmapped rows for auditability, but put standardized columns first.
    agg = agg[
        ["cell_type", "cell_type_cl_id", "tissue", "tissue_ontology_id", "tissue_uberon_id", "n_cells"]
    ].sort_values(["cell_type", "tissue"], kind="mergesort")

    mapper_json = build_distribution_mapper(agg, source_path=str(h5ad_path))
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(mapper_json, indent=2, ensure_ascii=False), encoding="utf-8")

    output_parquet = Path(output_parquet)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(output_parquet, index=False)

    if output_csv:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        agg.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    return agg, mapper_json


def build_distribution_mapper(agg: pd.DataFrame, source_path: str) -> dict[str, Any]:
    """Convert the aggregate table into a mapper-like JSON object."""
    by_cl_id: dict[str, dict[str, Any]] = {}
    unmapped: dict[str, dict[str, Any]] = {}

    for _, row in agg.iterrows():
        cell_type = str(row["cell_type"])
        cl_id = row["cell_type_cl_id"]
        tissue = str(row["tissue"])
        tissue_ontology_id = str(row["tissue_ontology_id"])
        uberon_id = row["tissue_uberon_id"]
        n_cells = int(row["n_cells"])

        target = by_cl_id if pd.notnull(cl_id) else unmapped
        key = str(cl_id) if pd.notnull(cl_id) else cell_type.lower()

        entry = target.setdefault(
            key,
            {
                "cell_type_cl_id": str(cl_id) if pd.notnull(cl_id) else None,
                "cell_type_names": [],
                "total_cells": 0,
                "tissues": [],
                "source": "tabula_sapiens",
            },
        )
        if cell_type not in entry["cell_type_names"]:
            entry["cell_type_names"].append(cell_type)
        entry["total_cells"] += n_cells
        entry["tissues"].append(
            {
                "tissue": tissue,
                "tissue_ontology_id": tissue_ontology_id,
                "tissue_uberon_id": str(uberon_id) if pd.notnull(uberon_id) else None,
                "n_cells": n_cells,
                "source": "tabula_sapiens",
            }
        )

    for entries in (by_cl_id, unmapped):
        for entry in entries.values():
            total = max(int(entry["total_cells"]), 1)
            entry["cell_type_names"] = sorted(entry["cell_type_names"])
            entry["tissues"] = sorted(
                entry["tissues"],
                key=lambda x: (-int(x["n_cells"]), x["tissue"]),
            )
            for tissue_record in entry["tissues"]:
                tissue_record["fraction"] = float(tissue_record["n_cells"] / total)

    return {
        "source": "tabula_sapiens",
        "source_path": source_path,
        "schema_version": "tabula_cell_type_tissue_mapper.v1",
        "description": "Primary tissue-distribution mapper built from full Tabula Sapiens obs.",
        "cell_type_to_tissue_distribution": by_cl_id,
        "unmapped_cell_type_to_tissue_distribution": unmapped,
        "n_mapped_cell_types": len(by_cl_id),
        "n_unmapped_cell_types": len(unmapped),
    }


def detect_tissue_id_col(columns: Any) -> str | None:
    """Detect a pre-existing tissue ontology ID column in Tabula Sapiens obs."""
    candidates = [
        "tissue_ontology_term_id",
        "tissue_ontology_id",
        "tissue_uberon_id",
        "uberon_id",
        "tissue_id",
    ]
    available = {str(c) for c in columns}
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def normalize_ontology_id(value: Any) -> str | None:
    """Normalize ontology IDs such as UBERON_0002106 -> UBERON:0002106."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    if text.upper().startswith("UBERON_"):
        return "UBERON:" + text.split("_", 1)[1]
    if text.upper().startswith("UBERON:"):
        return text.upper()
    return text


def resolve_tissue_name(tissue_uberon_id: str | None, mapper: Mapper, fallback: str) -> str:
    """Resolve tissue ID to a readable name for audit output."""
    if tissue_uberon_id:
        synonyms = mapper.tissue_synonyms(tissue_uberon_id)
        if synonyms:
            return synonyms[0]
    return fallback


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "rag_sources.yaml"),
        help="RAG source config YAML.",
    )
    parser.add_argument("--h5ad", default=None, help="Override Tabula Sapiens h5ad path.")
    parser.add_argument("--cell-type-col", default=None, help="Override obs cell type column.")
    parser.add_argument("--tissue-col", default=None, help="Override obs tissue column.")
    parser.add_argument("--tissue-id-col", default=None, help="Override obs tissue ontology ID column.")
    parser.add_argument("--output-json", default=None, help="Override primary mapper JSON path.")
    parser.add_argument("--output-parquet", default=None, help="Override output parquet path.")
    parser.add_argument("--output-csv", default=None, help="Optional audit CSV path.")
    parser.add_argument(
        "--report-json",
        default=None,
        help="Optional JSON summary path.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    tabula_cfg = cfg.get("tabula_sapiens") or cfg.get("tabula") or {}
    mapper = Mapper(cfg.get("mapper", {}))

    h5ad_path = args.h5ad or tabula_cfg.get("path")
    output_parquet = args.output_parquet or tabula_cfg.get(
        "cache_path",
        str(PROJECT_ROOT / "rag" / "cache" / "tabula_celltype_tissue.parquet"),
    )
    output_json = args.output_json or tabula_cfg.get(
        "mapper_json",
        str(PROJECT_ROOT / "rag" / "mappers" / "tabula_cell_type_tissue_mapper.json"),
    )
    obs_cfg = tabula_cfg.get("obs_columns", {})
    cell_type_col = args.cell_type_col or obs_cfg.get("cell_type", "cell_type")
    tissue_col = args.tissue_col or obs_cfg.get("tissue", "tissue")
    tissue_id_col = args.tissue_id_col or obs_cfg.get("tissue_id")

    if not h5ad_path:
        raise ValueError("No Tabula Sapiens h5ad path provided in config or --h5ad.")

    agg, mapper_json = build_tabula_tissue_cache(
        h5ad_path=h5ad_path,
        mapper=mapper,
        output_json=output_json,
        output_parquet=output_parquet,
        output_csv=args.output_csv,
        cell_type_col=cell_type_col,
        tissue_col=tissue_col,
        tissue_id_col=tissue_id_col,
    )

    summary = {
        "h5ad_path": str(h5ad_path),
        "output_json": str(output_json),
        "output_parquet": str(output_parquet),
        "output_csv": str(args.output_csv) if args.output_csv else None,
        "rows": int(len(agg)),
        "unique_cell_types": int(agg["cell_type"].nunique()),
        "unique_tissues": int(agg["tissue"].nunique()),
        "mapped_cell_types": int(mapper_json["n_mapped_cell_types"]),
        "unmapped_cell_types": int(mapper_json["n_unmapped_cell_types"]),
        "unmapped_cell_type_rows": int(agg["cell_type_cl_id"].isna().sum()),
        "unmapped_tissue_rows": int(agg["tissue_uberon_id"].isna().sum()),
    }

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
