"""CellMarker 2.0 loader.

官方数据来源:
  - 备用服务器: http://xteam.xbio.top/CellMarker/
  - 文件: all_cell_markers.txt, Human_cell_markers.txt, Mouse_cell_markers.txt, Single_cell_markers.txt

格式说明 (来自官方下载页面):
  制表符分隔的 TSV，含表头。
  主要列: cellName, geneSymbol, speciesType, tissueType, cancerType

重要数据处理:
  geneSymbol 列中部分条目包含多个基因，格式为:
    - 逗号分隔: "ITGAM, CD14, CD68"
    - 方括号包裹: "[CD3D, CD3E, CD3G]", "[FCGR3A, FCGR3B]"
  本 loader 在加载时会将复合基因拆分为独立行，确保每个基因单独一条记录。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .mapper import Mapper


# ---- 辅助函数: 拆分复合 geneSymbol ----

def _parse_gene_symbols(raw: str) -> list[str]:
    """将 CellMarker geneSymbol 字符串拆分为独立基因列表.

    输入示例:
      "ALPI"                                    -> ["ALPI"]
      "ITGAM, CD14, CD68"                       -> ["ITGAM", "CD14", "CD68"]
      "[CD3D, CD3E, CD3G]"                      -> ["CD3D", "CD3E", "CD3G"]
      "ITGAM, [FCGR2A, FCGR2B, FCGR2C], CD68"   -> ["ITGAM", "FCGR2A", "FCGR2B", "FCGR2C", "CD68"]
    """
    if not raw or not isinstance(raw, str):
        return []

    # 去掉所有方括号（保留内部内容）
    cleaned = raw.replace("[", "").replace("]", "")
    # 按逗号拆分，去掉空格
    genes = [g.strip() for g in cleaned.split(",")]
    # 过滤空字符串
    return [g for g in genes if g]


def _expand_multi_gene_rows(df: pd.DataFrame, gene_col: str) -> pd.DataFrame:
    """将 DataFrame 中含多个基因的行拆分为多行.

    例如原来一行 geneSymbol="[CD3D, CD3E, CD3G]" 会变成三行，
    每行 gene 分别为 CD3D, CD3E, CD3G，其余列保持不变。
    """
    if gene_col not in df.columns:
        return df

    rows: list[dict] = []
    for _, r in df.iterrows():
        raw_gene = str(r[gene_col]) if pd.notnull(r[gene_col]) else ""
        genes = _parse_gene_symbols(raw_gene)
        if not genes:
            # 无有效基因，保留原行（gene 为空）
            rows.append(dict(r))
            continue
        for g in genes:
            new_row = dict(r)
            new_row[gene_col] = g
            rows.append(new_row)

    return pd.DataFrame(rows)


class CellMarkerLoader:
    """CellMarker 数据加载器，支持多文件加载、复合基因拆分和标准化."""

    CELL_NAME_COL = "cellName"
    GENE_COL = "geneSymbol"
    SPECIES_COL = "speciesType"
    TISSUE_COL = "tissueType"

    def __init__(self, config: dict, mapper: "Mapper | None" = None):
        self.config = config
        self.mapper = mapper
        self.path = Path(config["path"])
        self.files = config.get("files", [])
        self._df: pd.DataFrame | None = None

    def _load(self) -> pd.DataFrame:
        """加载所有 CellMarker 文件到单个 DataFrame，并拆分复合基因."""
        if self._df is not None:
            return self._df

        frames: list[pd.DataFrame] = []
        for fname in self.files:
            fpath = self.path / fname
            if not fpath.exists():
                print(f"[WARN] CellMarker 文件不存在，跳过: {fpath}")
                continue
            try:
                df = pd.read_csv(fpath, sep="\t")
                # 拆分复合基因（关键修复）
                df = _expand_multi_gene_rows(df, self.GENE_COL)
                # 标准化基因名和细胞类型名
                if self.mapper is not None:
                    if self.GENE_COL in df.columns:
                        df[self.GENE_COL] = df[self.GENE_COL].astype(str).str.strip()
                        df["gene_normalized"] = df[self.GENE_COL].apply(
                            lambda x: self.mapper.normalize_gene(x) if pd.notnull(x) and x else None
                        )
                    if self.CELL_NAME_COL in df.columns:
                        df[self.CELL_NAME_COL] = df[self.CELL_NAME_COL].astype(str).str.strip()
                        df["cell_type_cl_id"] = df[self.CELL_NAME_COL].apply(
                            lambda x: self.mapper.normalize_cell_type(x) if pd.notnull(x) and x else None
                        )
                frames.append(df)
            except Exception as e:
                print(f"[WARN] 读取 CellMarker 文件失败 {fpath}: {e}")

        if frames:
            self._df = pd.concat(frames, ignore_index=True)
        else:
            self._df = pd.DataFrame()

        return self._df

    def query(self, names_lower: set[str], species: str = "human", top_k: int = 10) -> list[dict]:
        """查询指定细胞类型的 marker."""
        df = self._load()
        if df.empty or self.CELL_NAME_COL not in df.columns:
            return []

        mask = df[self.CELL_NAME_COL].astype(str).str.lower().isin(names_lower)
        if self.SPECIES_COL in df.columns and species:
            species_mask = df[self.SPECIES_COL].astype(str).str.lower().str.contains(
                species.lower(), na=False
            )
            mask = mask & species_mask

        sub = df.loc[mask].head(top_k)
        return [
            {
                "gene": str(r[self.GENE_COL]) if self.GENE_COL in df.columns and pd.notnull(r[self.GENE_COL]) else "",
                "gene_normalized": str(r["gene_normalized"]) if "gene_normalized" in df.columns and pd.notnull(r["gene_normalized"]) else None,
                "cell_type_name": str(r[self.CELL_NAME_COL]),
                "cell_type_cl_id": str(r["cell_type_cl_id"]) if "cell_type_cl_id" in df.columns and pd.notnull(r["cell_type_cl_id"]) else None,
                "evidence": "cellmarker",
                "source": "cellmarker",
            }
            for _, r in sub.iterrows()
        ]

    def query_cell_types(self, gene: str, species: str = "human") -> set[str]:
        """反向查询：某基因出现在哪些细胞类型中."""
        df = self._load()
        if df.empty or self.GENE_COL not in df.columns:
            return set()

        # 标准化输入基因名
        query_gene = gene.upper()
        if self.mapper is not None:
            norm = self.mapper.normalize_gene(gene)
            if norm:
                query_gene = norm.upper()

        # 精确匹配（已拆分后的单基因）
        mask = df[self.GENE_COL].astype(str).str.upper() == query_gene
        # 同时匹配标准化后的基因名
        if "gene_normalized" in df.columns:
            mask = mask | (df["gene_normalized"].astype(str).str.upper() == query_gene)

        if self.SPECIES_COL in df.columns and species:
            species_mask = df[self.SPECIES_COL].astype(str).str.lower().str.contains(
                species.lower(), na=False
            )
            mask = mask & species_mask

        return set(df.loc[mask, self.CELL_NAME_COL].dropna().astype(str).unique())
