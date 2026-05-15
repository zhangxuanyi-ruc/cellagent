#!/usr/bin/env python
"""Validate raw LLM marker suggestions into a runtime registry JSON."""
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

from src.agent.tools.marker_registry import normalize_cell_type_key  # noqa: E402
from src.tools.rag import RAGFacade  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="resources/marker_registry/offline_llm_marker_raw.jsonl")
    parser.add_argument("--rag-config", default="config/rag_sources.yaml")
    parser.add_argument("--output", default="resources/marker_registry/offline_llm_curated_markers.json")
    parser.add_argument("--audit-output", default="resources/marker_registry/offline_llm_marker_audit.csv")
    parser.add_argument("--min-valid-markers", type=int, default=3)
    parser.add_argument("--low-marker-warning-threshold", type=int, default=10)
    parser.add_argument("--cap", type=int, default=50)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def normalize_gene(gene: Any, rag: RAGFacade) -> str | None:
    text = str(gene or "").strip()
    if not text:
        return None
    norm = rag.mapper.normalize_gene(text, species="human")
    if not norm or not rag.mapper.gene.is_known(norm, species="human"):
        return None
    return norm.upper()


def extract_response(row: dict[str, Any]) -> dict[str, Any]:
    response = row.get("response")
    return response if isinstance(response, dict) else {}


def csv_escape(value: Any) -> str:
    text = str(value if value is not None else "")
    return '"' + text.replace('"', '""') + '"'


def main() -> None:
    args = parse_args()
    rag = RAGFacade(args.rag_config)
    raw_path = resolve(args.raw)
    output = resolve(args.output)
    audit_output = resolve(args.audit_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.parent.mkdir(parents=True, exist_ok=True)

    cell_types: dict[str, dict[str, Any]] = {}
    audit_rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(raw_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        response = extract_response(row)
        cell_type = str(row.get("cell_type") or response.get("cell_type") or "").strip()
        cl_id = str(row.get("cell_type_cl_id") or response.get("cell_type_cl_id") or "").strip()
        mapped_cl = rag.mapper.normalize_cell_type(cl_id or cell_type)
        raw_markers = response.get("positive_markers_ranked")
        if not isinstance(raw_markers, list):
            raw_markers = []

        valid_genes: list[str] = []
        discarded: list[str] = []
        for gene in raw_markers:
            norm = normalize_gene(gene, rag)
            if norm and norm not in valid_genes:
                valid_genes.append(norm)
            else:
                discarded.append(str(gene))
        valid_genes = valid_genes[: int(args.cap)]

        accepted = bool(mapped_cl) and len(valid_genes) >= int(args.min_valid_markers)
        low_marker_count = accepted and len(valid_genes) < int(args.low_marker_warning_threshold)
        key = normalize_cell_type_key(cell_type)
        if accepted and key:
            name = rag.get_cell_type_name(mapped_cl) if mapped_cl else None
            aliases = [cell_type]
            if name:
                aliases.append(name)
            aliases.extend(rag.mapper.cell_type_synonyms(mapped_cl))
            cell_types[key] = {
                "aliases": sorted({normalize_cell_type_key(a) for a in aliases if str(a).strip()}),
                "core_positive": valid_genes,
                "negative": [],
                "source": ["offline_llm_curated_registry"],
                "cell_type_cl_id": mapped_cl,
            }
        audit_rows.append(
            {
                "line_no": line_no,
                "cell_type": cell_type,
                "cell_type_cl_id": cl_id,
                "mapped_cl_id": mapped_cl or "",
                "accepted": accepted,
                "low_marker_count": low_marker_count,
                "n_raw_markers": len(raw_markers),
                "n_valid_markers": len(valid_genes),
                "cap": int(args.cap),
                "low_marker_warning_threshold": int(args.low_marker_warning_threshold),
                "discarded_genes": ";".join(discarded),
            }
        )

    payload = {
        "version": "offline_llm_curated_registry_v1",
        "source_file": str(raw_path),
        "cap": int(args.cap),
        "min_valid_markers": int(args.min_valid_markers),
        "low_marker_warning_threshold": int(args.low_marker_warning_threshold),
        "cell_types": dict(sorted(cell_types.items())),
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    headers = [
        "line_no",
        "cell_type",
        "cell_type_cl_id",
        "mapped_cl_id",
        "accepted",
        "low_marker_count",
        "n_raw_markers",
        "n_valid_markers",
        "cap",
        "low_marker_warning_threshold",
        "discarded_genes",
    ]
    audit_output.write_text(
        ",".join(headers) + "\n"
        + "\n".join(",".join(csv_escape(row.get(h, "")) for h in headers) for row in audit_rows)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "raw_rows": len(audit_rows),
                "accepted_cell_types": len(cell_types),
                "output": str(output),
                "audit_output": str(audit_output),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
