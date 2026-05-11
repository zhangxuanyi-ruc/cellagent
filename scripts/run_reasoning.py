#!/usr/bin/env python
"""Build deterministic reasoning evidence JSON for multimodal prior outputs."""
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

from src.core.manifest import PipelineManifest, default_manifest_path  # noqa: E402
from src.core.reasoning import (  # noqa: E402
    DeterministicReasoner,
    cluster_id_from_payload,
    load_de_genes,
    load_de_summary,
    load_prior_payloads,
    prediction_from_payload,
    write_reasoning_result,
)
from src.tools.rag import RAGFacade  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--rag-config", default="config/rag_sources.yaml")
    parser.add_argument("--manifest", default=None, help="Default: <pipeline.output_root>/pipeline_manifest.json")
    parser.add_argument("--prior-json", default=None, help="JSON/JSONL multimodal prior output.")
    parser.add_argument("--de-summary", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--top-de-genes", type=int, default=30)
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def resolve_manifest(args: argparse.Namespace, cfg: dict[str, Any]) -> PipelineManifest | None:
    manifest_path = Path(args.manifest) if args.manifest else None
    if manifest_path is None:
        output_root = cfg.get("pipeline", {}).get("output_root")
        if output_root:
            manifest_path = default_manifest_path(output_root)
    if manifest_path and manifest_path.exists():
        return PipelineManifest.read(manifest_path)
    return None


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    manifest = resolve_manifest(args, cfg)
    output_root = Path(
        (manifest.output_root if manifest else None)
        or cfg.get("pipeline", {}).get("output_root")
        or "output"
    )

    prior_json = args.prior_json or (manifest.multimodal_prior_json if manifest else None)
    if not prior_json:
        raise ValueError("Multimodal prior JSON is required via --prior-json or manifest.multimodal_prior_json.")
    de_summary = args.de_summary or (manifest.de_summary_csv if manifest else None) or str(output_root / "de" / "de_summary.csv")
    output_dir = args.output_dir or (manifest.reasoning_dir if manifest else None) or str(output_root / "reasoning")

    rag = RAGFacade(args.rag_config)
    marker_cfg = cfg.get("rag", {}).get("markers", {})
    reasoner = DeterministicReasoner(
        rag=rag,
        mapper=rag.mapper,
        marker_top_k=int(marker_cfg.get("top_k", 10)),
        min_markers=int(marker_cfg.get("min_markers", 5)),
    )

    de_by_cluster = load_de_summary(de_summary)
    output_paths: list[dict[str, str]] = []
    for payload in load_prior_payloads(prior_json):
        cluster_id = cluster_id_from_payload(payload)
        if cluster_id not in de_by_cluster:
            raise ValueError(
                f"No DE csv for cluster_id={cluster_id!r}. "
                f"Available clusters: {sorted(de_by_cluster)[:20]}"
            )
        metadata = payload.get("metadata") or {}
        prediction = prediction_from_payload(payload)
        de_genes = load_de_genes(de_by_cluster[cluster_id], top_k=args.top_de_genes)
        result = reasoner.run(
            cluster_id=cluster_id,
            prediction=prediction,
            metadata=metadata,
            de_genes=de_genes,
            provenance={
                "prior_json": str(prior_json),
                "de_csv": de_genes.de_csv_path,
                "manifest": str(args.manifest or default_manifest_path(output_root)),
                "rag_config": str(args.rag_config),
            },
        )
        path = write_reasoning_result(result, output_dir)
        output_paths.append({"cluster_id": cluster_id, "path": str(path)})

    summary_path = Path(output_dir) / "reasoning_summary.json"
    summary_path.write_text(json.dumps(output_paths, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"reasoning_dir": str(output_dir), "outputs": output_paths}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
