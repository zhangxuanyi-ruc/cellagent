"""Tabula Sapiens (h5ad) loader.

官方数据来源:
  - https://tabula-sapiens-portal.ds.czbiohub.org/
  - figshare: https://figshare.com/projects/Tabula_Sapiens/100973
  - cellxgene: https://cellxgene.cziscience.com/collections/e5f58829-1a66-40b5-a624-9046778e74f5

格式: h5ad (AnnData)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anndata as ad
import pandas as pd

if TYPE_CHECKING:
    from .mapper import Mapper


class TabulaLoader:
    """Tabula Sapiens h5ad 加载器."""

    def __init__(self, config: dict[str, Any], mapper: "Mapper | None" = None):
        self.config = config
        self.mapper = mapper
        self.h5ad_path = Path(config["path"]) if config.get("path") else None
        self.mapper_json_path = Path(config["mapper_json"]) if config.get("mapper_json") else None
        self.cache_path = Path(
            config.get("cache_path", "rag/cache/tabula_celltype_tissue.parquet")
        )
        self.cell_type_col = config.get("obs_columns", {}).get("cell_type", "cell_type")
        self.tissue_col = config.get("obs_columns", {}).get("tissue", "tissue")
        self._mapper_json: dict[str, Any] | None = None
        self._agg: pd.DataFrame | None = None

    def _load_mapper_json(self) -> dict[str, Any] | None:
        """Load the primary Tabula cell_type -> tissue_distribution mapper."""
        if self._mapper_json is not None:
            return self._mapper_json
        if not self.mapper_json_path or not self.mapper_json_path.exists():
            return None
        self._mapper_json = json.loads(self.mapper_json_path.read_text(encoding="utf-8"))
        return self._mapper_json

    def _aggregate(self) -> pd.DataFrame:
        """预聚合 (cell_type, tissue) -> n_cells."""
        if self._agg is not None:
            return self._agg
        if self.cache_path.exists():
            self._agg = pd.read_parquet(self.cache_path)
            return self._agg
        if not self.h5ad_path or not self.h5ad_path.exists():
            self._agg = pd.DataFrame()
            return self._agg

        try:
            adata = ad.read_h5ad(self.h5ad_path, backed="r")
            obs_cols = [self.cell_type_col, self.tissue_col]
            available_cols = [c for c in obs_cols if c in adata.obs.columns]
            if not available_cols:
                print(f"[WARN] Tabula Sapiens h5ad 中未找到列 {obs_cols}")
                self._agg = pd.DataFrame()
                return self._agg

            obs = adata.obs[available_cols].copy()
            agg = (
                obs.groupby(available_cols)
                .size()
                .reset_index(name="n_cells")
            )

            # 标准化细胞类型名
            if self.mapper is not None and self.cell_type_col in agg.columns:
                agg["cell_type_cl_id"] = agg[self.cell_type_col].astype(str).str.strip().apply(
                    lambda x: self.mapper.normalize_cell_type(x) if pd.notnull(x) else None
                )

            # 标准化组织名
            if self.mapper is not None and self.tissue_col in agg.columns:
                agg["tissue_uberon_id"] = agg[self.tissue_col].astype(str).str.strip().apply(
                    lambda x: self.mapper.normalize_tissue(x) if pd.notnull(x) else None
                )

            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            agg.to_parquet(self.cache_path)
            self._agg = agg
        except Exception as e:
            print(f"[WARN] 读取 Tabula Sapiens 失败: {e}")
            self._agg = pd.DataFrame()

        return self._agg

    def query(self, names_lower: set[str]) -> list[dict]:
        """查询细胞类型的组织分布."""
        mapper_json = self._load_mapper_json()
        if mapper_json:
            rows = self._query_mapper_json(names_lower, mapper_json)
            if rows:
                return rows

        agg = self._aggregate()
        if agg.empty or self.cell_type_col not in agg.columns:
            return []

        mask = agg[self.cell_type_col].astype(str).str.lower().isin(names_lower)
        sub = agg.loc[mask]

        return [
            {
                "cell_type_name": str(r[self.cell_type_col]),
                "cell_type_cl_id": str(r["cell_type_cl_id"]) if "cell_type_cl_id" in agg.columns and pd.notnull(r["cell_type_cl_id"]) else None,
                "tissue": str(r[self.tissue_col]) if self.tissue_col in agg.columns and pd.notnull(r[self.tissue_col]) else "",
                "tissue_uberon_id": str(r["tissue_uberon_id"]) if "tissue_uberon_id" in agg.columns and pd.notnull(r["tissue_uberon_id"]) else None,
                "expression_level": None,
                "n_cells": int(r["n_cells"]),
                "source": "tabula_sapiens",
            }
            for _, r in sub.iterrows()
        ]

    def _query_mapper_json(self, names_lower: set[str], mapper_json: dict[str, Any]) -> list[dict]:
        """Query prebuilt mapper JSON by CL ID or cell type synonyms."""
        cl_ids: set[str] = set()
        raw_names = {n.lower() for n in names_lower}
        if self.mapper is not None:
            for name in raw_names:
                cl_id = self.mapper.normalize_cell_type(name)
                if cl_id:
                    cl_ids.add(cl_id)

        by_cl_id = mapper_json.get("cell_type_to_tissue_distribution", {})
        unmapped = mapper_json.get("unmapped_cell_type_to_tissue_distribution", {})
        matched_entries: list[dict[str, Any]] = []

        for cl_id in sorted(cl_ids):
            if cl_id in by_cl_id:
                matched_entries.append(by_cl_id[cl_id])

        if not matched_entries:
            for key, entry in unmapped.items():
                names = {str(n).lower() for n in entry.get("cell_type_names", [])}
                if key in raw_names or names.intersection(raw_names):
                    matched_entries.append(entry)

        rows: list[dict] = []
        for entry in matched_entries:
            cell_type_name = entry.get("cell_type_names", [""])[0] if entry.get("cell_type_names") else ""
            for tissue in entry.get("tissues", []):
                rows.append(
                    {
                        "cell_type_name": cell_type_name,
                        "cell_type_cl_id": entry.get("cell_type_cl_id"),
                        "tissue": tissue.get("tissue", ""),
                        "tissue_ontology_id": tissue.get("tissue_ontology_id"),
                        "tissue_uberon_id": tissue.get("tissue_uberon_id"),
                        "expression_level": None,
                        "n_cells": int(tissue.get("n_cells", 0)),
                        "fraction": tissue.get("fraction"),
                        "source": "tabula_sapiens",
                    }
                )
        return rows
