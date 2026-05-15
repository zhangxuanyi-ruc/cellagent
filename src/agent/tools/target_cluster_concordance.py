"""Target-cell versus cluster-signature concordance scoring."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


class TargetClusterConcordanceScorer:
    """Rule-only scorer for whether one target cell supports its cluster DE signature."""

    def __init__(
        self,
        h5ad_path: str | Path | None,
        clusters_csv: str | Path | None,
        cluster_key: str = "leiden",
        expression_layer: str | None = None,
    ):
        self.h5ad_path = Path(h5ad_path) if h5ad_path else None
        self.clusters_csv = Path(clusters_csv) if clusters_csv else None
        self.cluster_key = cluster_key
        self.expression_layer = expression_layer
        self._adata: ad.AnnData | None = None
        self._clusters: pd.DataFrame | None = None

    def score(self, target_cell_id: str | None, cluster_id: str, top_de_genes: list[str]) -> dict[str, Any]:
        top10 = [str(g) for g in top_de_genes[:10] if str(g).strip()]
        empty = self._empty_report(top10)
        if not target_cell_id:
            return {**empty, "veto_reason": "target_cell_id is missing"}
        if not top10:
            return {**empty, "veto_reason": "cluster top10 DE genes are missing"}
        if not self.h5ad_path or not self.h5ad_path.exists():
            return {**empty, "veto_reason": f"h5ad path is missing or not found: {self.h5ad_path}"}
        if not self.clusters_csv or not self.clusters_csv.exists():
            return {**empty, "veto_reason": f"clusters_csv is missing or not found: {self.clusters_csv}"}

        adata = self._load_adata()
        clusters = self._load_clusters()
        if target_cell_id not in adata.obs_names:
            return {**empty, "veto_reason": f"target_cell_id not found in h5ad obs_names: {target_cell_id}"}
        if "cell_id" not in clusters or self.cluster_key not in clusters:
            return {
                **empty,
                "veto_reason": f"clusters_csv must contain cell_id and {self.cluster_key!r} columns",
            }

        available_genes = [gene for gene in top10 if gene in adata.var_names]
        missing_genes = [gene for gene in top10 if gene not in adata.var_names]
        if not available_genes:
            return {
                **empty,
                "missing_genes": missing_genes,
                "veto_reason": "none of cluster top10 DE genes were found in h5ad var_names",
            }

        cluster_cells = clusters.loc[
            clusters[self.cluster_key].astype(str) == str(cluster_id),
            "cell_id",
        ].astype(str)
        cluster_cells = [cell for cell in cluster_cells if cell in adata.obs_names]
        if not cluster_cells:
            return {**empty, "missing_genes": missing_genes, "veto_reason": f"no cells found for cluster {cluster_id!r}"}
        if target_cell_id not in cluster_cells:
            cluster_cells.append(target_cell_id)

        target_expr = np.asarray(
            [self._gene_values(adata, [target_cell_id], gene)[0] for gene in available_genes],
            dtype=float,
        )
        detected = [gene for gene, value in zip(available_genes, target_expr, strict=False) if float(value) > 0]
        presence_score = self._presence_score(len(detected))

        cluster_matrix = np.column_stack([
            self._gene_values(adata, cluster_cells, gene) for gene in available_genes
        ])
        cluster_signatures = cluster_matrix.mean(axis=1)
        target_signature = float(np.mean(target_expr)) if len(target_expr) else 0.0
        signature_percentile = self._percentile(target_signature, cluster_signatures)
        signature_score = self._signature_score(signature_percentile)

        veto = presence_score == 0 and signature_score == 0
        return {
            "score": int(presence_score + signature_score),
            "presence_score": int(presence_score),
            "signature_score": int(signature_score),
            "presence_detected_genes": detected,
            "presence_detected_count": len(detected),
            "signature_percentile": signature_percentile,
            "target_signature_mean": target_signature,
            "cluster_signature_n_cells": len(cluster_cells),
            "used_top_genes": available_genes,
            "missing_genes": missing_genes,
            "veto": veto,
            "veto_reason": (
                "target cell has no detected cluster top10 DE genes and signature percentile < 5%"
                if veto
                else ""
            ),
        }

    def _load_adata(self) -> ad.AnnData:
        if self._adata is None:
            self._adata = ad.read_h5ad(self.h5ad_path, backed="r")
        return self._adata

    def _load_clusters(self) -> pd.DataFrame:
        if self._clusters is None:
            self._clusters = pd.read_csv(self.clusters_csv)
        return self._clusters

    def _matrix(self, subset: ad.AnnData):
        if self.expression_layer:
            if self.expression_layer not in subset.layers:
                raise ValueError(f"expression_layer={self.expression_layer!r} not found in h5ad layers.")
            return subset.layers[self.expression_layer]
        return subset.X

    def _gene_values(self, adata: ad.AnnData, cell_ids: list[str], gene: str) -> np.ndarray:
        subset = adata[cell_ids, gene]
        return self._dense_vector(self._matrix(subset)).astype(float)

    @staticmethod
    def _dense_vector(matrix: Any) -> np.ndarray:
        if sparse.issparse(matrix):
            return np.asarray(matrix.toarray()).reshape(-1)
        return np.asarray(matrix).reshape(-1)

    @staticmethod
    def _row_means(matrix: Any) -> np.ndarray:
        if sparse.issparse(matrix):
            return np.asarray(matrix.mean(axis=1)).reshape(-1)
        return np.asarray(matrix).mean(axis=1).reshape(-1)

    @staticmethod
    def _presence_score(n_detected: int) -> int:
        if n_detected >= 5:
            return 15
        if n_detected >= 3:
            return 10
        if n_detected >= 1:
            return 5
        return 0

    @staticmethod
    def _signature_score(percentile: float) -> int:
        if percentile >= 25:
            return 15
        if percentile >= 10:
            return 10
        if percentile >= 5:
            return 5
        return 0

    @staticmethod
    def _percentile(value: float, distribution: np.ndarray) -> float:
        if distribution.size == 0:
            return 0.0
        return float(np.mean(distribution <= value) * 100.0)

    @staticmethod
    def _empty_report(top_genes: list[str]) -> dict[str, Any]:
        return {
            "score": 0,
            "presence_score": 0,
            "signature_score": 0,
            "presence_detected_genes": [],
            "presence_detected_count": 0,
            "signature_percentile": 0.0,
            "target_signature_mean": 0.0,
            "cluster_signature_n_cells": 0,
            "used_top_genes": top_genes,
            "missing_genes": [],
            "veto": False,
            "veto_reason": "",
        }


__all__ = ["TargetClusterConcordanceScorer"]
