"""Feature-based clustering utilities for CellAgent.

Clustering is performed strictly on extracted cell encoder features, not on raw
or preprocessed expression matrices.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


@dataclass
class ClusteringConfig:
    resolutions: list[float] = field(default_factory=lambda: [0.25, 0.5, 0.75, 1.0])
    n_pcs: int = 30
    n_neighbors: int = 30
    metric: str = "cosine"
    random_state: int = 0
    output_dir: str = "output/clustering"
    feature_key: str = "features"
    choose_by: str = "modularity"
    min_clusters: int = 5
    max_clusters: int = 15


def config_from_dict(raw: dict[str, Any] | None) -> ClusteringConfig:
    raw = raw or {}
    return ClusteringConfig(
        resolutions=[float(x) for x in raw.get("resolutions", [0.25, 0.5, 0.75, 1.0])],
        n_pcs=int(raw.get("n_pcs", 30)),
        n_neighbors=int(raw.get("n_neighbors", 30)),
        metric=raw.get("metric", "cosine"),
        random_state=int(raw.get("random_state", 0)),
        output_dir=raw.get("output_dir", "output/clustering"),
        feature_key=raw.get("feature_key", "features"),
        choose_by=raw.get("choose_by", "modularity"),
        min_clusters=int(raw.get("min_clusters", 5)),
        max_clusters=int(raw.get("max_clusters", 15)),
    )


def load_feature_matrix(feature_path: str | Path, feature_key: str = "features") -> tuple[np.ndarray, pd.Index, dict[str, Any]]:
    path = Path(feature_path)
    metadata: dict[str, Any] = {"feature_path": str(path)}
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        if feature_key not in data:
            raise ValueError(f"Missing feature key '{feature_key}' in {path}. Keys: {list(data.keys())}")
        features = np.asarray(data[feature_key])
        if "obs_names" in data:
            obs_names = pd.Index(data["obs_names"].astype(str))
        else:
            obs_names = pd.Index([str(i) for i in range(features.shape[0])])
        if "metadata" in data:
            raw_meta = data["metadata"].item() if data["metadata"].shape == () else str(data["metadata"])
            try:
                metadata.update(json.loads(raw_meta))
            except Exception:
                metadata["raw_metadata"] = raw_meta
    elif path.suffix == ".npy":
        features = np.asarray(np.load(path))
        meta_path = path.with_suffix(".metadata.json")
        if meta_path.exists():
            metadata.update(json.loads(meta_path.read_text(encoding="utf-8")))
            obs_names = pd.Index([str(x) for x in metadata.get("obs_names", [])])
            if len(obs_names) != features.shape[0]:
                obs_names = pd.Index([str(i) for i in range(features.shape[0])])
        else:
            obs_names = pd.Index([str(i) for i in range(features.shape[0])])
    else:
        raise ValueError(f"Unsupported feature file format: {path.suffix}. Expected .npz or .npy.")

    if features.ndim != 2:
        raise ValueError(f"Feature matrix must be 2D, got shape {features.shape}.")
    if len(obs_names) != features.shape[0]:
        raise ValueError(f"obs_names length {len(obs_names)} != n_cells {features.shape[0]}.")
    return features.astype(np.float32, copy=False), obs_names, metadata


def build_feature_adata(features: np.ndarray, obs_names: pd.Index) -> ad.AnnData:
    obs = pd.DataFrame(index=obs_names.astype(str))
    return ad.AnnData(X=features, obs=obs)


def run_feature_clustering(
    feature_path: str | Path,
    cfg: ClusteringConfig,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    features, obs_names, feature_metadata = load_feature_matrix(feature_path, feature_key=cfg.feature_key)
    adata = build_feature_adata(features, obs_names)
    adata.uns["cellagent_feature_metadata"] = feature_metadata

    n_cells, n_features = adata.shape
    if n_cells < 3:
        raise ValueError(f"At least 3 cells are required for Leiden clustering, got {n_cells}.")

    n_pcs = min(cfg.n_pcs, n_features, n_cells - 1)
    n_neighbors = min(cfg.n_neighbors, n_cells - 1)
    sc.pp.pca(adata, n_comps=n_pcs, random_state=cfg.random_state)
    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        use_rep="X_pca",
        metric=cfg.metric,
        random_state=cfg.random_state,
    )

    metrics: list[dict[str, Any]] = []
    for resolution in cfg.resolutions:
        key = _resolution_key(resolution)
        sc.tl.leiden(adata, resolution=resolution, key_added=key, random_state=cfg.random_state)
        labels = adata.obs[key].astype(str).tolist()
        metrics.append(
            {
                "resolution": resolution,
                "key": key,
                "n_clusters": int(adata.obs[key].nunique()),
                "modularity": compute_modularity(adata.obsp["connectivities"], labels),
            }
        )

    for item in metrics:
        item["cluster_count_pass"] = cfg.min_clusters <= int(item["n_clusters"]) <= cfg.max_clusters

    best = choose_best_resolution(
        metrics,
        choose_by=cfg.choose_by,
        min_clusters=cfg.min_clusters,
        max_clusters=cfg.max_clusters,
    )
    adata.obs["leiden"] = adata.obs[best["key"]].astype(str)
    clustering_summary = {
        "feature_path": str(feature_path),
        "n_cells": int(n_cells),
        "n_features": int(n_features),
        "n_pcs": int(n_pcs),
        "n_neighbors": int(n_neighbors),
        "metric": cfg.metric,
        "resolutions": cfg.resolutions,
        "choose_by": cfg.choose_by,
        "min_clusters": cfg.min_clusters,
        "max_clusters": cfg.max_clusters,
        "best_resolution": best["resolution"],
        "best_key": best["key"],
        "best_n_clusters": best["n_clusters"],
        "best_modularity": best["modularity"],
        "metrics": metrics,
    }
    adata.uns["cellagent_clustering"] = {
        **{k: v for k, v in clustering_summary.items() if k != "metrics"},
        "metrics_json": json.dumps(metrics, ensure_ascii=False),
    }

    out_dir = Path(output_dir or cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(feature_path).name
    stem = stem.replace(".npz", "").replace(".npy", "")
    h5ad_path = out_dir / f"{stem}_clustered.h5ad"
    csv_path = out_dir / f"{stem}_cell_clusters.csv"
    metrics_path = out_dir / f"{stem}_clustering_metrics.json"

    adata.write_h5ad(h5ad_path, compression="gzip")
    cluster_cols = [m["key"] for m in metrics] + ["leiden"]
    obs_out = adata.obs[cluster_cols].copy()
    obs_out.insert(0, "cell_id", adata.obs_names.astype(str))
    obs_out.to_csv(csv_path, index=False)
    metrics_path.write_text(json.dumps(clustering_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"h5ad": h5ad_path, "clusters_csv": csv_path, "metrics_json": metrics_path}


def _resolution_key(resolution: float) -> str:
    text = str(resolution).replace(".", "_")
    return f"leiden_r{text}"


def choose_best_resolution(
    metrics: list[dict[str, Any]],
    choose_by: str = "modularity",
    min_clusters: int = 5,
    max_clusters: int = 15,
) -> dict[str, Any]:
    if not metrics:
        raise ValueError("No clustering metrics available.")
    if choose_by != "modularity":
        raise ValueError(f"Unsupported clustering selection metric: {choose_by}")
    if min_clusters > max_clusters:
        raise ValueError(f"min_clusters ({min_clusters}) must be <= max_clusters ({max_clusters}).")

    valid = [
        item for item in metrics
        if min_clusters <= int(item.get("n_clusters", 0)) <= max_clusters
    ]
    if not valid:
        observed = ", ".join(
            f"r={item.get('resolution')}: n_clusters={item.get('n_clusters')}, "
            f"modularity={item.get('modularity')}"
            for item in metrics
        )
        raise ValueError(
            "No Leiden resolution satisfies the cluster-count constraint "
            f"[{min_clusters}, {max_clusters}]. Observed: {observed}"
        )
    return max(valid, key=lambda x: float(x.get("modularity", float("-inf"))))


def compute_modularity(connectivities: sparse.spmatrix, labels: list[str]) -> float:
    """Compute weighted graph modularity for Leiden labels.

    This avoids relying on scanpy internals and keeps resolution selection
    explicit. For very small/empty graphs, returns 0.
    """
    graph = connectivities.tocsr().astype(np.float64)
    m2 = float(graph.sum())
    if m2 <= 0:
        return 0.0
    labels_arr = np.asarray(labels)
    unique = pd.unique(labels_arr)
    degrees = np.asarray(graph.sum(axis=1)).ravel()
    modularity = 0.0
    for label in unique:
        idx = np.where(labels_arr == label)[0]
        if idx.size == 0:
            continue
        sub_weight = float(graph[idx][:, idx].sum())
        degree_sum = float(degrees[idx].sum())
        modularity += sub_weight / m2 - (degree_sum / m2) ** 2
    return float(modularity)
