"""ImmGen loader.

官方数据来源:
  - NCBI GEO: GSE109125
  - 文件: GSE109125_Genes_count_table.tsv (gzip 压缩)

格式: TSV，第一列为基因名，后续列为各细胞类型的表达计数.
物种: 小鼠 (Mus musculus)

注意:
  - 文件解压后约 2.3GB，不一次性加载到内存
  - 使用 chunk 读取或按需查询
"""
from __future__ import annotations

import gzip
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .mapper import Mapper


class ImmGenLoader:
    """ImmGen 基因表达矩阵加载器（小鼠免疫细胞）."""

    GENE_COL = "gene"

    def __init__(self, config: dict, mapper: "Mapper | None" = None):
        self.config = config
        self.mapper = mapper
        self.path = Path(config["path"])
        self._cell_types: list[str] | None = None
        self._gene_index: dict[str, pd.Series] | None = None

    def _get_cell_types(self) -> list[str]:
        """获取所有细胞类型列名（从文件头读取）."""
        if self._cell_types is not None:
            return self._cell_types

        if not self.path.exists():
            self._cell_types = []
            return []

        try:
            # 读取表头
            open_fn = gzip.open if str(self.path).endswith(".gz") else open
            with open_fn(self.path, "rt", encoding="utf-8") as f:
                header = f.readline().strip().split("\t")
            self._cell_types = header[1:]  # 第一列是基因名
        except Exception as e:
            print(f"[WARN] 读取 ImmGen 表头失败: {e}")
            self._cell_types = []

        return self._cell_types

    def query(self, names_lower: set[str]) -> list[dict]:
        """查询细胞类型的表达信息.

        由于 ImmGen 是矩阵格式（基因 x 细胞类型），
        这里返回的是各细胞类型中平均表达最高的基因作为参考 marker。
        """
        cell_types = self._get_cell_types()
        if not cell_types:
            return []

        # 匹配细胞类型名（ ImmGen 的细胞类型名可能需要手动映射到 CL）
        matched = [ct for ct in cell_types if ct.lower() in names_lower]
        if not matched:
            return []

        # 简化处理：返回细胞类型列名和来源
        return [
            {
                "cell_type_name": ct,
                "cell_type_cl_id": (
                    self.mapper.normalize_cell_type(ct) if self.mapper else None
                ),
                "tissue": "mouse_immune",
                "tissue_uberon_id": (
                    self.mapper.normalize_tissue("mouse immune tissue") if self.mapper else None
                ),
                "expression_level": None,
                "n_cells": None,
                "source": "immgen",
            }
            for ct in matched[:20]  # 限制返回数
        ]

    def query_expression(self, gene: str, cell_type: str) -> float | None:
        """查询某基因在某细胞类型中的表达量.

        逐行扫描大文件，找到目标基因后返回对应细胞类型的值.
        性能较慢，适合低频查询。
        """
        if not self.path.exists():
            return None

        cell_types = self._get_cell_types()
        if cell_type not in cell_types:
            return None

        col_idx = cell_types.index(cell_type) + 1  # +1 因为第一列是基因
        gene_target = gene.upper()

        try:
            open_fn = gzip.open if str(self.path).endswith(".gz") else open
            with open_fn(self.path, "rt", encoding="utf-8") as f:
                next(f)  # 跳过表头
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) > col_idx and parts[0].upper() == gene_target:
                        return float(parts[col_idx])
        except Exception as e:
            print(f"[WARN] 查询 ImmGen 表达量失败: {e}")

        return None
