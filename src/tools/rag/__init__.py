"""统一暴露 marker / function / tissue 三类 RAG 查询的门面.

KG 查询走 src/tools/kg_client.py, 不在此聚合.
loader 返回 list[dict], RAGFacade 在边界统一转 pydantic 类型.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from .cellmarker import CellMarkerLoader
from .immgen import ImmGenLoader
from .mapper import Mapper
from .ontology import OntologyLoader
from .panglao import PanglaoLoader
from .tabula import TabulaLoader


class Marker(BaseModel):
    gene: str
    gene_normalized: str | None = None
    cell_type_cl_id: str | None = None
    cell_type_name: str
    evidence: str | None = None
    source: str


class FunctionRecord(BaseModel):
    cell_type_cl_id: str
    cell_type_name: str
    definition: str | None = None
    go_terms: list[str] = Field(default_factory=list)
    parents: list[str] = Field(default_factory=list)
    source: str


class TissueRecord(BaseModel):
    cell_type_cl_id: str | None = None
    cell_type_name: str
    tissue: str
    tissue_ontology_id: str | None = None
    tissue_uberon_id: str | None = None
    expression_level: float | None = None
    n_cells: int | None = None
    fraction: float | None = None
    source: str


class RAGFacade:
    def __init__(self, config_path: str | Path):
        cfg: dict[str, Any] = yaml.safe_load(Path(config_path).read_text())
        self.mapper = Mapper(cfg.get("mapper", {}))

        # Marker 数据库
        self.cellmarker = CellMarkerLoader(cfg["cellmarker"], self.mapper) if "cellmarker" in cfg else None
        panglao_cfg = cfg.get("panglaodb") or cfg.get("panglao")
        self.panglao = PanglaoLoader(panglao_cfg, self.mapper) if panglao_cfg else None

        # Ontology
        self.ontology = OntologyLoader(cfg["ontology"]) if "ontology" in cfg else None

        # Tissue 数据库: Tabula Sapiens 为主库；ImmGen 仅在 mouse 场景门控启用。
        self.immgen = ImmGenLoader(cfg["immgen"], self.mapper) if "immgen" in cfg else None
        tabula_cfg = cfg.get("tabula_sapiens") or cfg.get("tabula")
        self.tabula = TabulaLoader(tabula_cfg, self.mapper) if tabula_cfg else None

    def query_markers(
        self,
        cell_type: str,
        species: str = "human",
        top_k: int = 10,
        min_markers: int = 5,
    ) -> list[Marker]:
        """查询某细胞类型的 marker 基因.

        CellMarker 是主库。PanglaoDB 仅在以下情况作为补充:
        1. CellMarker 无 marker
        2. CellMarker marker 数量少于 min_markers

        Args:
            cell_type: 细胞类型名或 CL ID
            species: "human" 或 "mouse"
            top_k: 每个被启用数据源最多返回条数
            min_markers: CellMarker 少于该数量时启用 PanglaoDB
        """
        cl_id = self.mapper.normalize_cell_type(cell_type)
        names = self.mapper.cell_type_synonyms(cl_id) if cl_id else [cell_type]
        names_lower = {n.lower() for n in names}

        cellmarker_rows: list[dict] = []
        if self.cellmarker:
            cellmarker_rows.extend(self.cellmarker.query(names_lower, species=species, top_k=top_k))

        rows: list[dict] = list(cellmarker_rows)
        if self.panglao and self._should_query_panglao(
            cellmarker_rows=cellmarker_rows,
            min_markers=min_markers,
        ):
            rows.extend(self.panglao.query(names_lower, species=species, top_k=top_k))

        # 去重。若启用附属库，top_k 按每个来源控制，不在合并后截断，避免补充证据被主库占满。
        seen: set[str] = set()
        unique_rows: list[dict] = []
        for r in rows:
            key = (r.get("cell_type_name", ""), r.get("gene", ""))
            if key not in seen:
                seen.add(key)
                unique_rows.append(r)

        results: list[Marker] = []
        for r in unique_rows:
            # 避免 **r 和显式参数重复
            r_copy = dict(r)
            r_copy.pop("cell_type_cl_id", None)
            r_copy.pop("gene_normalized", None)
            results.append(
                Marker(
                    cell_type_cl_id=cl_id,
                    gene_normalized=self.mapper.normalize_gene(r.get("gene", "")),
                    **r_copy,
                )
            )
        return results

    def _should_query_panglao(
        self,
        cellmarker_rows: list[dict],
        min_markers: int,
    ) -> bool:
        if not cellmarker_rows:
            return True
        return len(cellmarker_rows) < min_markers

    def query_functions(self, cell_type: str) -> list[FunctionRecord]:
        """查询细胞类型的功能描述 (GO terms)."""
        cl_id = self.mapper.normalize_cell_type(cell_type)
        rows: list[dict] = []
        if self.ontology:
            rows.extend(self.ontology.query(cell_type, cl_id=cl_id))
        return [FunctionRecord(**r) for r in rows]

    @staticmethod
    def _normalize_species_label(species: Any) -> str | None:
        if species is None:
            return None
        label = str(species).strip().lower().replace("_", " ").replace("-", " ")
        if not label:
            return None
        if label in {"mouse", "mus musculus", "m musculus", "mmusculus"}:
            return "mouse"
        if label in {"human", "homo sapiens", "h sapiens", "hsapiens"}:
            return "human"
        return label

    @classmethod
    def _should_use_immgen(cls, input_species: Any, metadata: dict[str, Any] | None = None) -> bool:
        """ImmGen 只用于 mouse 输入且 metadata 也明确标注为 mouse 的辅助验证."""
        metadata = metadata or {}
        return (
            cls._normalize_species_label(input_species) == "mouse"
            and cls._normalize_species_label(metadata.get("species")) == "mouse"
        )

    def query_tissues(
        self,
        cell_type: str,
        gene: str | None = None,
        input_species: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[TissueRecord]:
        """查询细胞类型的组织分布.

        Tabula Sapiens 是 tissue 查询主库。ImmGen 仅在 input_species == mouse
        且 metadata.species == mouse 时追加为小鼠场景验证证据。
        """
        cl_id = self.mapper.normalize_cell_type(cell_type)
        names = self.mapper.cell_type_synonyms(cl_id) if cl_id else [cell_type]
        names_lower = {n.lower() for n in names}

        rows: list[dict] = []
        if self.tabula:
            rows.extend(self.tabula.query(names_lower))
        if self.immgen and self._should_use_immgen(input_species, metadata):
            rows.extend(self.immgen.query(names_lower))

        # 若提供了 gene，只过滤带 gene 字段的表达记录；Tabula 等纯组织分布记录保留。
        if gene and gene.strip():
            gene_norm = self.mapper.normalize_gene(gene)
            rows = [
                r for r in rows
                if not (r.get("gene_normalized") or r.get("gene"))
                or r.get("gene_normalized") == gene_norm
                or r.get("gene") == gene
            ]

        results: list[TissueRecord] = []
        for r in rows:
            r_copy = dict(r)
            r_copy.pop("cell_type_cl_id", None)
            r_copy.pop("tissue_uberon_id", None)
            results.append(
                TissueRecord(
                    cell_type_cl_id=cl_id,
                    tissue_uberon_id=r.get("tissue_uberon_id"),
                    **r_copy,
                )
            )
        return results

    def query_cell_types_for_gene(self, gene: str, species: str = "human") -> list[str]:
        """反向查询：某基因在哪些细胞类型中作为 marker.

        用于检测反向 marker 冲突（如 FOXP3 强烈指向 Treg）。
        """
        gene_norm = self.mapper.normalize_gene(gene, species=species)
        cell_types: set[str] = set()

        if self.cellmarker:
            cell_types.update(self.cellmarker.query_cell_types(gene_norm, species=species))
        if self.panglao:
            cell_types.update(self.panglao.query_cell_types(gene_norm, species=species))

        # 返回 CL IDs
        return sorted({
            cid for cid in (self.mapper.normalize_cell_type(ct) for ct in cell_types)
            if cid
        })

    def get_parent_cell_type(self, cl_id: str) -> str | None:
        """获取 CL ID 的父类（用于 fallback）."""
        if self.ontology:
            return self.ontology.get_parent(cl_id)
        return None

    def get_cell_type_name(self, cl_id: str) -> str | None:
        """CL ID -> 人类可读名称."""
        if self.ontology:
            return self.ontology.get_name(cl_id)
        names = self.mapper.cell_type_synonyms(cl_id)
        return names[0] if names else None


__all__ = [
    "RAGFacade",
    "Mapper",
    "Marker",
    "FunctionRecord",
    "TissueRecord",
]
