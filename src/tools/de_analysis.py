"""Differential expression analysis for CellAgent clusters."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


@dataclass
class DEConfig:
    output_dir: str = "output/de_results"
    cluster_key: str = "leiden"
    method: str = "wilcoxon"
    top_k: int = 10
    adj_pval_threshold: float | None = 0.05
    logfc_threshold: float | None = 0.25
    expression_layer: str | None = None


def config_from_dict(raw: dict[str, Any] | None) -> DEConfig:
    raw = raw or {}
    return DEConfig(
        output_dir=raw.get("output_dir", "output/de_results"),
        cluster_key=raw.get("cluster_key", "leiden"),
        method=raw.get("method", "wilcoxon"),
        top_k=int(raw.get("top_k", 10)),
        adj_pval_threshold=raw.get("adj_pval_threshold", 0.05),
        logfc_threshold=raw.get("logfc_threshold", 0.25),
        expression_layer=raw.get("expression_layer"),
    )


def load_cluster_assignments(path: str | Path, cluster_key: str = "leiden") -> pd.Series:
    path = Path(path)
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        if "cell_id" not in df:
            raise ValueError(f"Cluster CSV must contain 'cell_id': {path}")
        if cluster_key not in df:
            raise ValueError(f"Cluster CSV missing cluster column '{cluster_key}'. Columns: {list(df.columns)}")
        return pd.Series(df[cluster_key].astype(str).values, index=df["cell_id"].astype(str).values, name=cluster_key)
    if path.suffix == ".h5ad":
        clustered = ad.read_h5ad(path)
        if cluster_key not in clustered.obs:
            raise ValueError(f"Cluster h5ad missing obs['{cluster_key}']: {path}")
        return pd.Series(clustered.obs[cluster_key].astype(str).values, index=clustered.obs_names.astype(str), name=cluster_key)
    raise ValueError(f"Unsupported cluster assignment format: {path.suffix}. Expected .csv or .h5ad.")


def attach_clusters(adata: ad.AnnData, clusters: pd.Series, cluster_key: str) -> ad.AnnData:
    obs_names = adata.obs_names.astype(str)
    missing = obs_names.difference(clusters.index)
    if len(missing) > 0:
        # Cells dropped during clustering (e.g. NaN encoder features) are filtered out here.
        adata = adata[~obs_names.isin(missing)].copy()
    adata.obs[cluster_key] = clusters.reindex(adata.obs_names.astype(str)).astype(str).values
    return adata


def run_de_analysis(
    preprocessed_h5ad: str | Path,
    clusters_path: str | Path,
    cfg: DEConfig,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    adata = ad.read_h5ad(preprocessed_h5ad)
    clusters = load_cluster_assignments(clusters_path, cluster_key=cfg.cluster_key)
    adata = attach_clusters(adata, clusters, cfg.cluster_key)

    if cfg.expression_layer:
        if cfg.expression_layer not in adata.layers:
            raise ValueError(f"Missing expression layer '{cfg.expression_layer}' in {preprocessed_h5ad}")
        adata.X = adata.layers[cfg.expression_layer].copy()

    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError(f"Empty AnnData for DE: shape={adata.shape}")
    if adata.obs[cfg.cluster_key].nunique() < 2:
        raise ValueError("DE requires at least two clusters.")

    sc.tl.rank_genes_groups(
        adata,
        groupby=cfg.cluster_key,
        method=cfg.method,
        reference="rest",
        pts=True,
    )

    out_dir = Path(output_dir or cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    outputs: dict[str, Path] = {}

    for cluster in sorted(adata.obs[cfg.cluster_key].astype(str).unique(), key=_cluster_sort_key):
        de_df = _cluster_de_table(adata, cluster, cfg)
        safe_cluster = str(cluster).replace("/", "_")
        csv_path = out_dir / f"cluster_{safe_cluster}_vs_all.csv"
        de_df.to_csv(csv_path, index=False)
        outputs[f"cluster_{safe_cluster}"] = csv_path
        summary_rows.append(
            {
                "cluster": str(cluster),
                "n_cells": int((adata.obs[cfg.cluster_key].astype(str) == str(cluster)).sum()),
                "n_genes_saved": int(len(de_df)),
                "csv": str(csv_path),
            }
        )

    summary_path = out_dir / "de_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    outputs["summary"] = summary_path
    return outputs


def _cluster_de_table(adata: ad.AnnData, cluster: str, cfg: DEConfig) -> pd.DataFrame:
    rg = adata.uns["rank_genes_groups"]
    names = pd.Series(rg["names"][cluster].astype(str), name="gene")
    scores = pd.Series(rg["scores"][cluster], name="score")
    logfc = pd.Series(rg["logfoldchanges"][cluster], name="logFC")
    pvals_adj = pd.Series(rg["pvals_adj"][cluster], name="adjusted_p_value")
    pct_in = pd.Series(rg["pts"][cluster].reindex(names).values, name="pct_in_cluster")
    pct_out = pd.Series(rg["pts_rest"][cluster].reindex(names).values, name="pct_out_cluster")

    table = pd.concat([names, scores, logfc, pvals_adj, pct_in, pct_out], axis=1)
    if cfg.adj_pval_threshold is not None:
        table = table[table["adjusted_p_value"] <= float(cfg.adj_pval_threshold)]
    if cfg.logfc_threshold is not None:
        table = table[table["logFC"] >= float(cfg.logfc_threshold)]
    table = table.head(cfg.top_k).copy()
    table.insert(1, "rank", np.arange(1, len(table) + 1, dtype=int))

    labels = adata.obs[cfg.cluster_key].astype(str).values
    in_mask = labels == str(cluster)
    means_in, means_out = _mean_expr_by_gene(adata, in_mask, table["gene"].tolist())
    table["source_cluster_size"] = int(in_mask.sum())
    table["mean_expr_in"] = means_in
    table["mean_expr_out"] = means_out
    return table[
        [
            "gene",
            "rank",
            "score",
            "logFC",
            "adjusted_p_value",
            "pct_in_cluster",
            "pct_out_cluster",
            "source_cluster_size",
            "mean_expr_in",
            "mean_expr_out",
        ]
    ]


def _mean_expr_by_gene(adata: ad.AnnData, in_mask: np.ndarray, genes: list[str]) -> tuple[list[float], list[float]]:
    if len(genes) == 0:
        return [], []
    gene_indices = [adata.var_names.get_loc(gene) for gene in genes]
    x = adata.X
    x_in = x[in_mask][:, gene_indices]
    x_out = x[~in_mask][:, gene_indices]
    mean_in = _axis_mean(x_in)
    mean_out = _axis_mean(x_out)
    return mean_in.tolist(), mean_out.tolist()


def _axis_mean(x) -> np.ndarray:
    if sparse.issparse(x):
        return np.asarray(x.mean(axis=0)).ravel()
    return np.asarray(x).mean(axis=0)


def _cluster_sort_key(value: str):
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)
