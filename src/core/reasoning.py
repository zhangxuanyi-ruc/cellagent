"""Deterministic reasoning stage for CellAgent.

This module intentionally does not call an LLM. It unpacks the multimodal prior,
normalizes identifiers, queries prepared RAG sources, and returns a structured
evidence package for downstream judge/reflection stages.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.schemas import (
    DEGenes,
    Prediction,
    ReasoningResult,
    StandardizedReasoningInput,
)


def _model_to_dict(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return dict(obj)


def _safe_cluster_id(value: Any) -> str:
    if value is None or str(value).strip() == "":
        return "unknown"
    return str(value)


def normalize_species_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace("_", " ").replace("-", " ")
    if not text:
        return None
    if text in {"human", "homo sapiens", "h sapiens", "hsapiens"}:
        return "human"
    if text in {"mouse", "mus musculus", "m musculus", "mmusculus"}:
        return "mouse"
    return text


def prediction_from_payload(payload: dict[str, Any]) -> Prediction:
    return Prediction(
        cell_type=str(payload.get("cell_type") or payload.get("celltype") or ""),
        function=str(payload.get("function") or ""),
        confidence=float(payload.get("confidence") or 0.0),
    )


def cluster_id_from_payload(payload: dict[str, Any]) -> str:
    for key in ("cluster_id", "cluster", "target_cluster", "leiden"):
        if key in payload and payload[key] is not None:
            return _safe_cluster_id(payload[key])
    return _safe_cluster_id(payload.get("cell_id"))


def load_prior_payloads(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("predictions"), list):
        return data["predictions"]
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported prior JSON payload: {path}")


def load_de_genes(de_csv_path: str | Path, top_k: int | None = None) -> DEGenes:
    df = pd.read_csv(de_csv_path)
    if "gene" not in df:
        raise ValueError(f"DE csv missing 'gene' column: {de_csv_path}")
    genes = df["gene"].dropna().astype(str).tolist()
    if top_k is not None:
        genes = genes[: int(top_k)]
    return DEGenes(top_genes=genes, de_csv_path=str(de_csv_path))


def load_de_summary(path: str | Path) -> dict[str, str]:
    df = pd.read_csv(path)
    required = {"cluster", "csv"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"DE summary missing columns {sorted(missing)}: {path}")
    return {str(row["cluster"]): str(row["csv"]) for _, row in df.iterrows()}


class DeterministicReasoner:
    """Rule-only evidence builder for one cluster/cell-type hypothesis."""

    def __init__(self, rag: Any, mapper: Any, marker_top_k: int = 10, min_markers: int = 5):
        self.rag = rag
        self.mapper = mapper
        self.marker_top_k = marker_top_k
        self.min_markers = min_markers

    def run(
        self,
        cluster_id: str,
        prediction: Prediction,
        metadata: dict[str, Any],
        de_genes: DEGenes,
        provenance: dict[str, Any] | None = None,
    ) -> ReasoningResult:
        warnings: list[str] = []
        species = normalize_species_label(metadata.get("species") or metadata.get("input_species")) or "human"
        tissue_raw = _first_present(metadata, ["tissue", "organ", "tissue_name", "sample_tissue"])
        tissue_uberon = self.mapper.normalize_tissue(tissue_raw) if tissue_raw else None

        cl_id = self.mapper.normalize_cell_type(prediction.cell_type)
        if not cl_id:
            warnings.append(f"Unmapped cell type: {prediction.cell_type}")

        de_norm: list[str] = []
        unmapped: list[str] = []
        for gene in de_genes.top_genes:
            normalized = self.mapper.normalize_gene_to_human(gene, species=species)
            if normalized:
                de_norm.append(normalized)
            else:
                unmapped.append(gene)
        de_norm = list(dict.fromkeys(de_norm))
        if unmapped:
            warnings.append(f"Unmapped DE genes: {len(unmapped)}")

        marker_records = [
            _model_to_dict(r)
            for r in self.rag.query_markers(
                prediction.cell_type,
                species=species,
                top_k=self.marker_top_k,
                min_markers=self.min_markers,
            )
        ]
        function_records = [_model_to_dict(r) for r in self.rag.query_functions(prediction.cell_type)]
        tissue_records = [
            _model_to_dict(r)
            for r in self.rag.query_tissues(
                prediction.cell_type,
                input_species=species,
                metadata=metadata,
            )
        ]

        standardized = StandardizedReasoningInput(
            cell_type_raw=prediction.cell_type,
            cell_type_cl_id=cl_id,
            function_raw=prediction.function,
            species=species,
            tissue_raw=tissue_raw,
            tissue_uberon_id=tissue_uberon,
            de_genes_raw=de_genes.top_genes,
            de_genes_normalized=de_norm,
            unmapped_de_genes=unmapped,
        )
        return ReasoningResult(
            cluster_id=str(cluster_id),
            prediction=prediction,
            standardized=standardized,
            marker_records=marker_records,
            function_records=function_records,
            tissue_records=tissue_records,
            kg_records=[],
            warnings=warnings,
            provenance=provenance or {},
        )


def write_reasoning_result(result: ReasoningResult, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_cluster = str(result.cluster_id).replace("/", "_")
    path = output_dir / f"cluster_{safe_cluster}_reasoning.json"
    path.write_text(json.dumps(result.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _first_present(data: dict[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        value = data.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


__all__ = [
    "DeterministicReasoner",
    "cluster_id_from_payload",
    "load_de_genes",
    "load_de_summary",
    "load_prior_payloads",
    "normalize_species_label",
    "prediction_from_payload",
    "write_reasoning_result",
]
