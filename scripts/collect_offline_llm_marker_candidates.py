#!/usr/bin/env python
"""Collect cell types that need offline LLM-curated marker lists."""
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

from src.agent.tools.marker_registry import MarkerRegistry  # noqa: E402
from src.tools.rag import RAGFacade  # noqa: E402


DEFAULT_LABEL_JSON = "/data/bgi/data/projects/multimodal/RNA_data/cellwtext_data/CellwText/config/cellwtext_celltype2label.json"
DEFAULT_AUDIT_GLOBS = [
    "/home/qijinyin/wanghaoran/zxy/cellagent_output/marker_coverage_audit/*_marker_label_coverage.csv",
    "/home/qijinyin/wanghaoran/zxy/cellagent_output/representative_marker_audit/*_marker_label_coverage.csv",
]
INVALID_CELL_TYPE_LABELS = {
    "",
    "human",
    "mouse",
    "colon",
    "stomach",
    "blood",
    "brain",
    "lung",
    "skin",
    "tumor",
    "normal",
    "disease",
    "control",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-json", default=DEFAULT_LABEL_JSON)
    parser.add_argument("--audit-csv", action="append", default=[])
    parser.add_argument("--audit-glob", action="append", default=DEFAULT_AUDIT_GLOBS)
    parser.add_argument("--extra-cell-types", default="resources/marker_registry/offline_llm_extra_celltypes.txt")
    parser.add_argument(
        "--unstable-label-blacklist",
        default="resources/marker_registry/unstable_celltype_label_blacklist.yaml",
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--rag-config", default="config/rag_sources.yaml")
    parser.add_argument("--output", default="resources/marker_registry/offline_llm_marker_candidates.json")
    parser.add_argument("--min-authoritative-markers", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=0, help="0 means no limit.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def load_training_labels(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    labels = data.keys() if isinstance(data, dict) else data
    return [
        {"cell_type": str(label), "sources": ["cellwtext_label"], "max_dataset_count": 0}
        for label in labels
        if is_valid_cell_type_label(str(label))
    ]


def is_valid_cell_type_label(label: str) -> bool:
    return str(label or "").strip().lower() not in INVALID_CELL_TYPE_LABELS


def normalize_label(label: Any) -> str:
    return " ".join(str(label or "").strip().lower().split())


def load_unstable_blacklist(path: str | Path) -> set[str]:
    p = Path(path)
    p = p if p.is_absolute() else PROJECT_ROOT / p
    if not p.exists():
        return set()
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return {normalize_label(x) for x in data.get("exact_labels", []) if normalize_label(x)}


def load_extra_labels(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    p = p if p.is_absolute() else PROJECT_ROOT / p
    if not p.exists():
        return []
    labels = []
    for line in p.read_text(encoding="utf-8").splitlines():
        label = line.split("#", 1)[0].strip()
        if is_valid_cell_type_label(label):
            labels.append({"cell_type": label, "sources": ["offline_llm_extra_celltypes"], "max_dataset_count": 0})
    return labels


def load_audit_labels(paths: list[Path]) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "label" not in df.columns:
            continue
        for _, row in df.iterrows():
            label = str(row.get("label") or "").strip()
            if not is_valid_cell_type_label(label) or label.lower() in {"nan", "none", "unknown", "na", "n/a"}:
                continue
            status = str(row.get("status") or "")
            marker_count = int(row.get("marker_count_priority") or 0)
            needs_llm = (
                status in {"INSUFFICIENT_DATABASE_EVIDENCE", "UNMAPPED_CELL_TYPE"}
                or marker_count <= 5
            )
            if not needs_llm:
                continue
            item = rows.setdefault(label, {"cell_type": label, "sources": [], "max_dataset_count": 0})
            source = f"audit:{path.name}"
            if source not in item["sources"]:
                item["sources"].append(source)
            item["max_dataset_count"] = max(int(item["max_dataset_count"]), int(row.get("count") or 0))
    return list(rows.values())


def registry_query_candidates(rag: RAGFacade, label: str) -> tuple[str | None, list[str]]:
    candidates = [label]
    cl_id = rag.mapper.normalize_cell_type(label)
    if cl_id:
        name = rag.get_cell_type_name(cl_id)
        if name:
            candidates.append(name)
        candidates.extend(rag.mapper.cell_type_synonyms(cl_id))
    return cl_id, list(dict.fromkeys(c for c in candidates if c))


def authoritative_marker_status(
    label: str,
    rag: RAGFacade,
    registry: MarkerRegistry,
    min_authoritative_markers: int,
) -> dict[str, Any]:
    cl_id, candidates = registry_query_candidates(rag, label)
    registry_hit = registry.query(candidates)
    if registry_hit and len(registry_hit.get("markers") or []) >= min_authoritative_markers:
        return {
            "needs_offline_llm": False,
            "reason": "sctype_registry_sufficient",
            "cell_type_cl_id": cl_id,
            "authoritative_marker_count": len(registry_hit.get("markers") or []),
        }

    rag_records = rag.query_markers(label, top_k=None, min_markers=min_authoritative_markers)
    rag_genes = sorted({
        str(r.gene_normalized or r.gene).strip().upper()
        for r in rag_records
        if str(r.gene_normalized or r.gene).strip()
    })
    if len(rag_genes) > min_authoritative_markers:
        return {
            "needs_offline_llm": False,
            "reason": "rag_fallback_sufficient",
            "cell_type_cl_id": cl_id,
            "authoritative_marker_count": len(rag_genes),
        }
    return {
        "needs_offline_llm": bool(cl_id),
        "reason": "INSUFFICIENT_DATABASE_EVIDENCE" if cl_id else "UNMAPPED_CELL_TYPE",
        "cell_type_cl_id": cl_id,
        "authoritative_marker_count": len(rag_genes),
    }


def choose_canonical_candidate(candidates: list[dict[str, Any]], rag: RAGFacade) -> dict[str, Any]:
    """Select one runtime candidate for a CL ID and preserve merged labels."""
    if len(candidates) == 1:
        chosen = dict(candidates[0])
    else:
        cl_id = candidates[0].get("cell_type_cl_id")
        preferred = normalize_label(rag.get_cell_type_name(cl_id)) if cl_id else ""

        def sort_key(item: dict[str, Any]) -> tuple[int, int]:
            label = normalize_label(item.get("cell_type"))
            exact_preferred_penalty = 0 if preferred and label == preferred else 1
            return exact_preferred_penalty, len(label)

        chosen = dict(sorted(candidates, key=sort_key)[0])
    merged_labels = sorted({str(c.get("cell_type")) for c in candidates if str(c.get("cell_type")).strip()})
    chosen["label"] = chosen["cell_type"]
    chosen["normalized_label"] = normalize_label(chosen["cell_type"])
    chosen["cl_id"] = chosen.get("cell_type_cl_id")
    chosen["merged_labels"] = merged_labels
    chosen["merged_label_count"] = len(merged_labels)
    merged_sources: list[str] = []
    max_count = 0
    for item in candidates:
        merged_sources.extend(item.get("sources", []))
        max_count = max(max_count, int(item.get("max_dataset_count") or 0))
    chosen["sources"] = list(dict.fromkeys(merged_sources))
    chosen["max_dataset_count"] = max_count
    return chosen


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    registry = MarkerRegistry((cfg.get("marker_registry") or {}).get("path"))
    rag = RAGFacade(args.rag_config)
    unstable_blacklist = load_unstable_blacklist(args.unstable_label_blacklist)

    audit_paths = [Path(p) for p in args.audit_csv]
    for pattern in args.audit_glob:
        audit_paths.extend(sorted(Path("/").glob(pattern.lstrip("/"))))

    merged: dict[str, dict[str, Any]] = {}
    for item in [*load_training_labels(args.label_json), *load_audit_labels(audit_paths), *load_extra_labels(args.extra_cell_types)]:
        key = item["cell_type"].strip().lower()
        current = merged.setdefault(
            key,
            {"cell_type": item["cell_type"], "sources": [], "max_dataset_count": 0},
        )
        current["sources"] = list(dict.fromkeys([*current["sources"], *item.get("sources", [])]))
        current["max_dataset_count"] = max(int(current["max_dataset_count"]), int(item.get("max_dataset_count") or 0))

    candidates_raw: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    unmapped: list[dict[str, Any]] = []
    filtered_unstable: list[dict[str, Any]] = []
    for item in merged.values():
        status = authoritative_marker_status(
            item["cell_type"],
            rag=rag,
            registry=registry,
            min_authoritative_markers=args.min_authoritative_markers,
        )
        row = {**item, **status}
        if normalize_label(item["cell_type"]) in unstable_blacklist:
            row["needs_offline_llm"] = False
            row["reason"] = "UNSTABLE_OR_NON_TARGET_CELLTYPE_LABEL"
            filtered_unstable.append(row)
            continue
        if status["needs_offline_llm"]:
            candidates_raw.append(row)
        elif status.get("reason") == "UNMAPPED_CELL_TYPE":
            unmapped.append(row)
        else:
            skipped.append(row)

    by_cl: dict[str, list[dict[str, Any]]] = {}
    for row in candidates_raw:
        by_cl.setdefault(str(row["cell_type_cl_id"]), []).append(row)
    candidates = [choose_canonical_candidate(rows, rag) for _, rows in sorted(by_cl.items())]
    candidates.sort(key=lambda x: (-int(x.get("max_dataset_count") or 0), x["cell_type"].lower()))
    if args.max_candidates and args.max_candidates > 0:
        candidates = candidates[: args.max_candidates]

    output = Path(args.output)
    output = output if output.is_absolute() else PROJECT_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "offline_llm_marker_candidates_v1",
        "label_json": str(args.label_json),
        "audit_csvs": [str(p) for p in audit_paths if p.exists()],
        "min_authoritative_markers": int(args.min_authoritative_markers),
        "n_candidates": len(candidates),
        "n_candidates_before_cl_dedup": len(candidates_raw),
        "n_skipped": len(skipped),
        "n_unmapped_needing_mapper": len(unmapped),
        "n_filtered_unstable": len(filtered_unstable),
        "candidates": candidates,
        "filtered_unstable": filtered_unstable,
        "unmapped_needing_mapper": unmapped,
        "skipped": skipped,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                k: payload[k]
                for k in [
                    "version",
                    "n_candidates",
                    "n_candidates_before_cl_dedup",
                    "n_skipped",
                    "n_unmapped_needing_mapper",
                    "n_filtered_unstable",
                ]
            },
            indent=2,
        )
    )
    print(f"wrote={output}")


if __name__ == "__main__":
    main()
