#!/usr/bin/env python
"""Diagnose positive marker scoring on selected cluster/cell-type cases."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.manifest import PipelineManifest, default_manifest_path  # noqa: E402
from src.tools.rag import RAGFacade  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--rag-config", default="config/rag_sources.yaml")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--case", action="append", default=[], help="Case as cluster_id:cell_type. Can be repeated.")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def score_top10(n_hits: int) -> int:
    if n_hits >= 5:
        return 40
    if n_hits >= 3:
        return 30
    if n_hits >= 1:
        return 10
    return 0


def score_top30_rescue(n_hits: int) -> int:
    if n_hits >= 10:
        return 30
    if n_hits >= 6:
        return 20
    if n_hits >= 3:
        return 10
    return 0


def marker_status(n_top10_hits: int, n_top30_hits: int, has_markers: bool) -> tuple[str, int, bool, str]:
    if not has_markers:
        return "INSUFFICIENT_DATABASE_EVIDENCE", 0, False, "none"
    top10 = score_top10(n_top10_hits)
    top30 = score_top30_rescue(n_top30_hits)
    score = max(top10, top30)
    branch = "top30_rescue" if top30 > top10 else ("top10" if top10 > 0 else "none")
    if branch == "top30_rescue":
        return "RESCUED_SUPPORT", score, False, branch
    if n_top10_hits >= 5:
        return "STRONG_SUPPORT", score, False, branch
    if n_top10_hits >= 3:
        return "MODERATE_SUPPORT", score, False, branch
    if n_top10_hits >= 1:
        return "WEAK_SUPPORT", score, False, branch
    return "NO_SUPPORT", score, True, branch


def parse_case(raw: str) -> tuple[str, str]:
    if ":" not in raw:
        raise ValueError(f"Invalid --case {raw!r}. Expected cluster_id:cell_type.")
    cluster, cell_type = raw.split(":", 1)
    cluster = cluster.strip()
    cell_type = cell_type.strip()
    if not cluster or not cell_type:
        raise ValueError(f"Invalid --case {raw!r}. Expected cluster_id:cell_type.")
    return cluster, cell_type


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    output_root = Path(cfg.get("pipeline", {}).get("output_root") or "output")
    manifest_path = Path(args.manifest) if args.manifest else default_manifest_path(output_root)
    manifest = PipelineManifest.read(manifest_path)
    if not manifest.de_dir:
        raise ValueError("manifest.de_dir is required.")

    rag = RAGFacade(args.rag_config)
    cases = [parse_case(raw) for raw in args.case]
    if not cases:
        raise ValueError("At least one --case cluster_id:cell_type is required.")

    rows: list[dict[str, Any]] = []
    for cluster, cell_type in cases:
        safe_cluster = cluster.replace("/", "_")
        de_csv = Path(manifest.de_dir) / f"cluster_{safe_cluster}_vs_all.csv"
        if not de_csv.exists():
            raise FileNotFoundError(f"DE csv not found for cluster={cluster}: {de_csv}")
        de = pd.read_csv(de_csv)
        top10 = de["gene"].dropna().astype(str).head(10).tolist()
        top30 = de["gene"].dropna().astype(str).head(30).tolist()
        marker_records = rag.query_markers(cell_type, species="human", top_k=None, min_markers=5)
        marker_genes = {
            rag.mapper.normalize_gene_to_human(record.gene, species="human") or record.gene
            for record in marker_records
        }
        top10_hits = [
            gene for gene in top10
            if (rag.mapper.normalize_gene_to_human(gene, species="human") or gene) in marker_genes
        ]
        top30_hits = [
            gene for gene in top30
            if (rag.mapper.normalize_gene_to_human(gene, species="human") or gene) in marker_genes
        ]
        top10_score = score_top10(len(top10_hits))
        top30_rescue_score = score_top30_rescue(len(top30_hits))
        status, score, veto, branch = marker_status(len(top10_hits), len(top30_hits), bool(marker_genes))
        rows.append(
            {
                "cluster": cluster,
                "prior_cell_type": cell_type,
                "n_marker_genes": len(marker_genes),
                "top10": ";".join(top10),
                "top10_hits": ";".join(top10_hits),
                "n_top10_hits": len(top10_hits),
                "top10_score": top10_score,
                "top30_hits": ";".join(top30_hits),
                "n_top30_hits": len(top30_hits),
                "top30_rescue_score": top30_rescue_score,
                "selected_branch": branch,
                "marker_status": status,
                "marker_score": score,
                "veto": veto,
            }
        )

    result = pd.DataFrame(rows)
    output = Path(args.output) if args.output else output_root / "marker_threshold_diagnostic.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    print(json.dumps({"output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
