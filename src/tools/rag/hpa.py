"""Human Protein Atlas (HPA) loader.

官方数据来源:
  - https://www.proteinatlas.org/about/download
  - 文件: proteinatlas.tsv.zip (解压后为 TSV)

格式: TSV，含表头，列包括 Gene, Cell type, Tissue, nTPM 等.
"""
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .mapper import Mapper


class HPALoader:
    """HPA 蛋白表达数据加载器."""

    CELL_TYPE_COL = "Cell type"
    GENE_COL = "Gene"
    TISSUE_COL = "Tissue"
    EXPRESSION_COL = "nTPM"

    def __init__(self, config: dict, mapper: "Mapper | None" = None):
        self.config = config
        self.mapper = mapper
        self.path = Path(config["path"])
        self._df: pd.DataFrame | None = None

    def _load(self) -> pd.DataFrame:
        """从 zip 文件加载 HPA TSV 数据."""
        if self._df is not None:
            return self._df

        if not self.path.exists():
            self._df = pd.DataFrame()
            return self._df

        try:
            with zipfile.ZipFile(self.path, "r") as z:
                # zip 内通常只有一个 tsv 文件
                tsv_name = [n for n in z.namelist() if n.endswith(".tsv")][0]
                with z.open(tsv_name) as f:
                    df = pd.read_csv(f, sep="\t")

            # 标准化基因名
            if self.mapper is not None and self.GENE_COL in df.columns:
                df[self.GENE_COL] = df[self.GENE_COL].astype(str).str.strip()
                df["gene_normalized"] = df[self.GENE_COL].apply(
                    lambda x: self.mapper.normalize_gene(x) if pd.notnull(x) else None
                )

            # 标准化细胞类型名
            if self.mapper is not None and self.CELL_TYPE_COL in df.columns:
                df[self.CELL_TYPE_COL] = df[self.CELL_TYPE_COL].astype(str).str.strip()
                df["cell_type_cl_id"] = df[self.CELL_TYPE_COL].apply(
                    lambda x: self.mapper.normalize_cell_type(x) if pd.notnull(x) else None
                )

            # 标准化组织名
            if self.mapper is not None and self.TISSUE_COL in df.columns:
                df[self.TISSUE_COL] = df[self.TISSUE_COL].astype(str).str.strip()
                df["tissue_uberon_id"] = df[self.TISSUE_COL].apply(
                    lambda x: self.mapper.normalize_tissue(x) if pd.notnull(x) else None
                )

            self._df = df
        except Exception as e:
            print(f"[WARN] 读取 HPA 文件失败 {self.path}: {e}")
            self._df = pd.DataFrame()

        return self._df

    def query(self, names_lower: set[str]) -> list[dict]:
        """查询细胞类型的组织分布和表达水平."""
        df = self._load()
        if df.empty or self.CELL_TYPE_COL not in df.columns:
            return []

        mask = df[self.CELL_TYPE_COL].astype(str).str.lower().isin(names_lower)
        sub = df.loc[mask]

        return [
            {
                "cell_type_name": str(r[self.CELL_TYPE_COL]),
                "cell_type_cl_id": str(r["cell_type_cl_id"]) if "cell_type_cl_id" in df.columns and pd.notnull(r["cell_type_cl_id"]) else None,
                "tissue": str(r[self.TISSUE_COL]) if self.TISSUE_COL in df.columns and pd.notnull(r[self.TISSUE_COL]) else "",
                "tissue_uberon_id": str(r["tissue_uberon_id"]) if "tissue_uberon_id" in df.columns and pd.notnull(r["tissue_uberon_id"]) else None,
                "expression_level": (
                    float(r[self.EXPRESSION_COL])
                    if self.EXPRESSION_COL in df.columns and pd.notnull(r[self.EXPRESSION_COL])
                    else None
                ),
                "gene": str(r[self.GENE_COL]) if self.GENE_COL in df.columns and pd.notnull(r[self.GENE_COL]) else None,
                "gene_normalized": str(r["gene_normalized"]) if "gene_normalized" in df.columns and pd.notnull(r["gene_normalized"]) else None,
                "n_cells": None,
                "source": "hpa",
            }
            for _, r in sub.iterrows()
        ]
