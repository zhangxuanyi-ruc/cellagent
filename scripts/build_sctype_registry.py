#!/usr/bin/env python
"""Build a compact ScType marker registry for CellAgent marker scoring."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="resources/marker_sources/sctype/ScTypeDB_full.xlsx",
        help="Path to ScTypeDB_full.xlsx.",
    )
    parser.add_argument(
        "--aliases",
        default="resources/marker_registry/celltype_aliases.yaml",
        help="Cell type alias YAML.",
    )
    parser.add_argument(
        "--output",
        default="resources/marker_registry/sctype_markers.json",
        help="Output compact marker registry JSON.",
    )
    parser.add_argument("--cap", type=int, default=100, help="Max positive markers per cell type.")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def normalize_cell_type(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def split_genes(value: Any) -> list[str]:
    if value is None or pd.isna(value):
        return []
    genes: list[str] = []
    for token in re.split(r"[,;/\n\r\t]+", str(value)):
        gene = token.strip().upper()
        if gene:
            genes.append(gene)
    return list(dict.fromkeys(genes))


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def ordered_union(*gene_lists: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for genes in gene_lists:
        for gene in genes:
            norm = str(gene).strip().upper()
            if norm and norm not in seen:
                seen.add(norm)
                merged.append(norm)
    return merged


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    aliases_path = resolve_path(args.aliases)
    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(input_path)
    required = {"tissueType", "cellName", "geneSymbolmore1", "geneSymbolmore2"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"ScTypeDB missing columns {sorted(missing)}. Columns: {list(df.columns)}")

    aliases_raw = load_yaml(aliases_path)
    registry: dict[str, dict[str, Any]] = {}

    for _, row in df.iterrows():
        cell_type = normalize_cell_type(row["cellName"])
        if not cell_type:
            continue
        record = registry.setdefault(
            cell_type,
            {
                "aliases": [],
                "core_positive": [],
                "negative": [],
                "source": [],
                "tissue_types": [],
            },
        )
        record["core_positive"] = ordered_union(record["core_positive"], split_genes(row["geneSymbolmore1"]))
        record["negative"] = ordered_union(record["negative"], split_genes(row["geneSymbolmore2"]))
        tissue = str(row["tissueType"]).strip()
        if tissue and tissue not in record["tissue_types"]:
            record["tissue_types"].append(tissue)
        if "ScType" not in record["source"]:
            record["source"].append("ScType")

    for canonical, alias_values in aliases_raw.items():
        cell_type = normalize_cell_type(canonical)
        record = registry.setdefault(
            cell_type,
            {
                "aliases": [],
                "core_positive": [],
                "negative": [],
                "source": [],
                "tissue_types": [],
            },
        )
        aliases = [normalize_cell_type(v) for v in (alias_values or [])]
        record["aliases"] = sorted({a for a in [*record["aliases"], *aliases] if a})

    for record in registry.values():
        record["core_positive"] = ordered_union(record["core_positive"])[: int(args.cap)]
        record["negative"] = ordered_union(record["negative"])
        record["source"] = list(dict.fromkeys(record["source"]))

    payload = {
        "version": "sctype_v1_no_manual_override",
        "source_file": str(input_path),
        "cap": int(args.cap),
        "cell_types": dict(sorted(registry.items())),
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    n_with_markers = sum(1 for r in registry.values() if r["core_positive"])
    print(f"wrote={output_path}")
    print(f"cell_types={len(registry)}")
    print(f"cell_types_with_positive_markers={n_with_markers}")


if __name__ == "__main__":
    main()
