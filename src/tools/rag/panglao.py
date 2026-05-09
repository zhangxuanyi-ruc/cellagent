"""PanglaoDB loader.

官方数据来源:
  - GitHub: https://github.com/oscar-franzen/PanglaoDB
  - data/ 目录下 6 个 txt 文件，逗号分隔，无表头

各文件列说明（来自官方 README）:
  cell_type_annotations_markers.txt: SRA accession, SRS accession, cluster, gene_symbol
  cell_type_annotations.txt: SRA, SRS, cluster, cell_type, p_value, adj_p_value, activity_score
  genes.txt: gene_symbol
  metadata.txt: 样本元数据 (多列)
  cell_type_desc.txt: cell_type, description, synonyms
  cell_type_abbrev.txt: cell_type, abbreviation
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .mapper import Mapper


class PanglaoLoader:
    """PanglaoDB 数据加载器."""

    # cell_type_annotations_markers.txt 列（无表头）
    MARKER_COLS = ["sra", "srs", "cluster", "gene_symbol"]
    # cell_type_annotations.txt 列（无表头）
    ANNOT_COLS = ["sra", "srs", "cluster", "cell_type", "p_value", "adj_p_value", "activity_score"]

    def __init__(self, config: dict, mapper: "Mapper | None" = None):
        self.config = config
        self.mapper = mapper
        self.path = Path(config["path"])
        self.files = config.get("files", {})
        self.delimiter = config.get("delimiter", ",")
        self._markers_df: pd.DataFrame | None = None
        self._annot_df: pd.DataFrame | None = None

    def _load_markers(self) -> pd.DataFrame:
        """加载 marker 文件 (cell_type_annotations_markers.txt)."""
        if self._markers_df is not None:
            return self._markers_df

        fname = self.files.get("cell_type_annotations_markers", "cell_type_annotations_markers.txt")
        fpath = self.path / fname
        if not fpath.exists():
            self._markers_df = pd.DataFrame()
            return self._markers_df

        df = pd.read_csv(fpath, sep=self.delimiter, header=None, names=self.MARKER_COLS)
        if self.mapper is not None:
            df["gene_symbol"] = df["gene_symbol"].astype(str).str.strip()
            df["gene_normalized"] = df["gene_symbol"].apply(
                lambda x: self.mapper.normalize_gene(x) if pd.notnull(x) else None
            )
        self._markers_df = df
        return df

    def _load_annotations(self) -> pd.DataFrame:
        """加载细胞类型注释文件 (cell_type_annotations.txt)."""
        if self._annot_df is not None:
            return self._annot_df

        fname = self.files.get("cell_type_annotations", "cell_type_annotations.txt")
        fpath = self.path / fname
        if not fpath.exists():
            self._annot_df = pd.DataFrame()
            return self._annot_df

        df = pd.read_csv(fpath, sep=self.delimiter, header=None, names=self.ANNOT_COLS)
        if self.mapper is not None:
            df["cell_type_cl_id"] = df["cell_type"].astype(str).str.strip().apply(
                lambda x: self.mapper.normalize_cell_type(x) if pd.notnull(x) else None
            )
        self._annot_df = df
        return df

    def query(self, names_lower: set[str], species: str = "human", top_k: int = 10) -> list[dict]:
        """查询细胞类型的 marker 基因.

        通过 annotations 文件匹配细胞类型名，再通过 markers 文件获取基因。
        """
        annot_df = self._load_annotations()
        marker_df = self._load_markers()

        if annot_df.empty or marker_df.empty:
            return []

        # 匹配细胞类型
        mask = annot_df["cell_type"].astype(str).str.lower().isin(names_lower)
        matched = annot_df.loc[mask]
        if matched.empty:
            return []

        # 获取匹配的 (SRA, SRS, cluster) 组合
        keys = matched[["sra", "srs", "cluster"]].drop_duplicates()

        # 在 marker 文件中找对应基因
        results: list[dict] = []
        for _, key in keys.head(top_k).iterrows():
            mm = (
                (marker_df["sra"] == key["sra"])
                & (marker_df["srs"] == key["srs"])
                & (marker_df["cluster"] == key["cluster"])
            )
            genes = marker_df.loc[mm, "gene_symbol"].dropna().astype(str).tolist()
            for g in genes[:5]:  # 每个 cluster 最多取 5 个基因
                results.append({
                    "gene": g,
                    "gene_normalized": (
                        str(marker_df.loc[mm & (marker_df["gene_symbol"] == g), "gene_normalized"].iloc[0])
                        if "gene_normalized" in marker_df.columns and mm.any() else None
                    ),
                    "cell_type_name": str(matched.loc[matched.index[0], "cell_type"]),
                    "cell_type_cl_id": str(matched.loc[matched.index[0], "cell_type_cl_id"])
                    if "cell_type_cl_id" in matched.columns else None,
                    "evidence": f"PanglaoDB cluster {key['cluster']}",
                    "source": "panglao",
                })
            if len(results) >= top_k:
                break

        return results[:top_k]

    def query_cell_types(self, gene: str, species: str = "human") -> set[str]:
        """反向查询：某基因出现在哪些细胞类型中."""
        marker_df = self._load_markers()
        annot_df = self._load_annotations()

        if marker_df.empty or annot_df.empty:
            return set()

        # 先找含该基因的 marker 记录
        gene_mask = marker_df["gene_symbol"].astype(str).str.upper() == gene.upper()
        if "gene_normalized" in marker_df.columns:
            gene_mask = gene_mask | (marker_df["gene_normalized"] == gene)

        matched_markers = marker_df.loc[gene_mask]
        if matched_markers.empty:
            return set()

        # 关联到 annotations 获取细胞类型
        cell_types: set[str] = set()
        for _, m in matched_markers.iterrows():
            am = (
                (annot_df["sra"] == m["sra"])
                & (annot_df["srs"] == m["srs"])
                & (annot_df["cluster"] == m["cluster"])
            )
            if am.any():
                cell_types.update(annot_df.loc[am, "cell_type"].dropna().astype(str).tolist())

        return cell_types
