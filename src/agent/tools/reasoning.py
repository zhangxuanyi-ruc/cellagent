"""Deterministic reasoning branch for CellAgent.

No LLM is used here. This branch unpacks multimodal prior outputs, normalizes
identifiers, queries prepared RAG evidence, and writes ReasoningResult JSON.
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
    raise ValueError("Prior payload missing cluster_id. Resolve cell_id to cluster_id before reasoning.")


def cell_id_from_payload(payload: dict[str, Any]) -> str | None:
    for key in ("cell_id", "target_cell_id", "obs_name"):
        if key in payload and payload[key] is not None:
            return str(payload[key])
    return None


def case_id_from_payload(payload: dict[str, Any]) -> str | None:
    for key in ("case_id", "prediction_id", "run_id"):
        if key in payload and payload[key] is not None:
            return str(payload[key])
    return None


def load_cluster_assignments(path: str | Path, cluster_key: str = "leiden") -> dict[str, str]:
    df = pd.read_csv(path)
    if "cell_id" not in df:
        raise ValueError(f"Cluster CSV must contain 'cell_id': {path}")
    if cluster_key not in df:
        raise ValueError(f"Cluster CSV missing cluster column '{cluster_key}'. Columns: {list(df.columns)}")
    return {
        str(cell_id): str(cluster_id)
        for cell_id, cluster_id in zip(df["cell_id"], df[cluster_key], strict=False)
    }


def resolve_cluster_id(payload: dict[str, Any], cell_to_cluster: dict[str, str]) -> str:
    try:
        return cluster_id_from_payload(payload)
    except ValueError:
        cell_id = cell_id_from_payload(payload)
        if not cell_id:
            raise
        if cell_id not in cell_to_cluster:
            raise ValueError(f"cell_id={cell_id!r} not found in cluster assignments.")
        return cell_to_cluster[cell_id]


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
                metadata=metadata,
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


def safe_output_stem(cluster_id: str, cell_id: str | None = None, case_id: str | None = None) -> str:
    parts = []
    if case_id:
        parts.append(f"case_{case_id}")
    if cell_id:
        parts.append(f"cell_{cell_id}")
    parts.append(f"cluster_{cluster_id}")
    return "_".join(_safe_filename(part) for part in parts if part)


def write_reasoning_result(
    result: ReasoningResult,
    output_dir: str | Path,
    output_stem: str | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem or safe_output_stem(result.cluster_id)
    path = output_dir / f"{stem}_reasoning.json"
    path.write_text(json.dumps(result.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _safe_filename(value: str) -> str:
    text = str(value).strip()
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)


def _first_present(data: dict[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        value = data.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


__all__ = [
    "DeterministicReasoner",
    "case_id_from_payload",
    "cluster_id_from_payload",
    "cell_id_from_payload",
    "load_de_genes",
    "load_de_summary",
    "load_cluster_assignments",
    "load_prior_payloads",
    "normalize_species_label",
    "prediction_from_payload",
    "resolve_cluster_id",
    "safe_output_stem",
    "write_reasoning_result",
]
