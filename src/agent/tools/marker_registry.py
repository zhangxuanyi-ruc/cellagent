"""Compact marker registry used as the primary marker source for scoring."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def normalize_cell_type_key(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def normalize_gene_symbol(value: Any) -> str:
    return str(value or "").strip().upper()


class MarkerRegistry:
    """Lookup cell type -> compact positive marker list.

    The registry intentionally avoids ontology expansion and dynamic marker
    ranking. It only resolves exact normalized names and curated aliases.
    """

    def __init__(self, path: str | Path | None = None):
        self.path = self._resolve_path(path)
        self.payload: dict[str, Any] = {}
        self.cell_types: dict[str, dict[str, Any]] = {}
        self.alias_to_key: dict[str, str] = {}
        if self.path and self.path.exists():
            self.payload = json.loads(self.path.read_text(encoding="utf-8"))
            self.cell_types = dict(self.payload.get("cell_types") or {})
            self._build_alias_index()

    @staticmethod
    def _resolve_path(path: str | Path | None) -> Path | None:
        if path is None:
            default = PROJECT_ROOT / "resources" / "marker_registry" / "sctype_markers.json"
            return default
        p = Path(path)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def _build_alias_index(self) -> None:
        for key, record in self.cell_types.items():
            norm_key = normalize_cell_type_key(key)
            self.alias_to_key[norm_key] = key
            for alias in record.get("aliases") or []:
                norm_alias = normalize_cell_type_key(alias)
                if norm_alias:
                    self.alias_to_key[norm_alias] = key

    @property
    def enabled(self) -> bool:
        return bool(self.cell_types)

    def query(self, candidates: list[str | None]) -> dict[str, Any] | None:
        for candidate in candidates:
            key = self.alias_to_key.get(normalize_cell_type_key(candidate))
            if not key:
                continue
            record = self.cell_types.get(key) or {}
            markers = [
                gene
                for gene in (normalize_gene_symbol(g) for g in record.get("core_positive") or [])
                if gene
            ]
            if not markers:
                continue
            return {
                "query_key": normalize_cell_type_key(candidate),
                "matched_cell_type": key,
                "markers": list(dict.fromkeys(markers)),
                "aliases": record.get("aliases") or [],
                "source": record.get("source") or [],
                "registry_path": str(self.path) if self.path else None,
                "version": self.payload.get("version"),
            }
        return None


class OfflineLLMMarkerRegistry(MarkerRegistry):
    """Validated offline LLM-curated marker registry.

    This registry is intentionally separate from ScType. It is only used when
    authoritative marker evidence is missing or too weak, and never as a veto
    source.
    """

    @staticmethod
    def _resolve_path(path: str | Path | None) -> Path | None:
        if path is None:
            default = PROJECT_ROOT / "resources" / "marker_registry" / "offline_llm_curated_markers.json"
            return default
        p = Path(path)
        return p if p.is_absolute() else PROJECT_ROOT / p


__all__ = ["MarkerRegistry", "OfflineLLMMarkerRegistry", "normalize_cell_type_key", "normalize_gene_symbol"]
