#!/usr/bin/env python
"""Diagnose target-cell versus cluster-signature consistency on existing SCA outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", required=True)
    parser.add_argument("--clusters-csv", required=True)
    parser.add_argument("--de-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cluster-key", default="leiden")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--dataset-label", default="dataset")
    parser.add_argument("--veto-threshold", type=int, default=10)
    return parser.parse_args()


def score_presence(n_detected: np.ndarray) -> np.ndarray:
    return np.select(
        [n_detected >= 5, n_detected >= 3, n_detected >= 1],
        [15, 10, 5],
        default=0,
    ).astype(int)


def score_signature(percentiles: np.ndarray) -> np.ndarray:
    return np.select(
        [percentiles >= 25, percentiles >= 10, percentiles >= 5],
        [15, 10, 5],
        default=0,
    ).astype(int)


def to_dense(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray(matrix.toarray())
    return np.asarray(matrix)


def percentile_within(values: np.ndarray) -> np.ndarray:
    return np.asarray([(values <= value).mean() * 100.0 for value in values], dtype=float)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(args.h5ad)
    clusters = pd.read_csv(args.clusters_csv)
    de_summary = pd.read_csv(args.de_summary)
    if "cell_id" not in clusters or args.cluster_key not in clusters:
        raise ValueError(f"clusters CSV must contain cell_id and {args.cluster_key!r}")

    cell_to_cluster = clusters.set_index("cell_id")[args.cluster_key].astype(str)
    common_cells = [str(cell) for cell in adata.obs_names if str(cell) in cell_to_cluster.index]
    adata = adata[common_cells].copy()
    adata.obs["cellagent_cluster"] = [cell_to_cluster.loc[str(cell)] for cell in adata.obs_names]

    rows: list[dict] = []
    case_rows: list[dict] = []
    summary_rows: list[dict] = []

    for _, de_row in de_summary.iterrows():
        cluster = str(de_row["cluster"])
        de_csv = Path(str(de_row["csv"]))
        de = pd.read_csv(de_csv)
        top_genes = [str(g) for g in de["gene"].dropna().astype(str).tolist()[: args.top_n]]
        genes = [gene for gene in top_genes if gene in adata.var_names]
        missing = [gene for gene in top_genes if gene not in adata.var_names]
        cluster_mask = np.asarray(adata.obs["cellagent_cluster"].astype(str) == cluster)
        cell_ids = adata.obs_names[cluster_mask].astype(str).tolist()
        if not cell_ids or not genes:
            summary_rows.append(
                {
                    "cluster": cluster,
                    "n_cells": len(cell_ids),
                    "n_genes_used": len(genes),
                    "missing_genes": ",".join(missing),
                    "error": "no cells or no genes",
                }
            )
            continue

        matrix = to_dense(adata[cluster_mask, genes].X).astype(float)
        detected_counts = (matrix > 0).sum(axis=1)
        signatures = matrix.mean(axis=1)
        percentiles = percentile_within(signatures)
        presence_scores = score_presence(detected_counts)
        signature_scores = score_signature(percentiles)
        scores = presence_scores + signature_scores
        veto = scores <= args.veto_threshold
        obs = adata.obs.iloc[np.where(cluster_mask)[0]]

        cluster_rows = []
        for idx, cell_id in enumerate(cell_ids):
            row = {
                "cell_id": cell_id,
                "cluster": cluster,
                "author_cell_types": obs.iloc[idx].get("author_cell_types", ""),
                "cell_type": obs.iloc[idx].get("cell_type", ""),
                "top_genes": ",".join(top_genes),
                "used_genes": ",".join(genes),
                "missing_genes": ",".join(missing),
                "presence_detected_count": int(detected_counts[idx]),
                "presence_score": int(presence_scores[idx]),
                "signature_mean": float(signatures[idx]),
                "signature_percentile": float(percentiles[idx]),
                "signature_score": int(signature_scores[idx]),
                "target_cluster_consistency_score": int(scores[idx]),
                "veto": bool(veto[idx]),
            }
            rows.append(row)
            cluster_rows.append(row)

        dfc = pd.DataFrame(cluster_rows)
        summary_rows.append(
            {
                "cluster": cluster,
                "n_cells": len(cell_ids),
                "top_genes": ",".join(top_genes),
                "n_genes_used": len(genes),
                "missing_genes": ",".join(missing),
                "score_mean": float(dfc["target_cluster_consistency_score"].mean()),
                "score_median": float(dfc["target_cluster_consistency_score"].median()),
                "score_min": int(dfc["target_cluster_consistency_score"].min()),
                "score_p05": float(dfc["target_cluster_consistency_score"].quantile(0.05)),
                "score_p25": float(dfc["target_cluster_consistency_score"].quantile(0.25)),
                "score_p75": float(dfc["target_cluster_consistency_score"].quantile(0.75)),
                "veto_rate": float(dfc["veto"].mean()),
                "presence0_rate": float((dfc["presence_score"] == 0).mean()),
                "signature0_rate": float((dfc["signature_score"] == 0).mean()),
            }
        )

        for label, indices in [
            ("lowest", dfc.sort_values(["target_cluster_consistency_score", "signature_percentile"]).index[:2]),
            ("median", dfc.assign(abs_med=(dfc["signature_percentile"] - 50).abs()).sort_values("abs_med").index[:1]),
            ("highest", dfc.sort_values(["target_cluster_consistency_score", "signature_percentile"], ascending=False).index[:1]),
        ]:
            for index in indices:
                row = dfc.loc[index].to_dict()
                row["case_type"] = label
                case_rows.append(row)

    all_df = pd.DataFrame(rows)
    case_df = pd.DataFrame(case_rows)
    summary_df = pd.DataFrame(summary_rows)
    if "cluster" in summary_df:
        summary_df = summary_df.sort_values("cluster", key=lambda s: s.astype(int))

    prefix = f"target_cluster_consistency_{args.dataset_label}"
    all_path = output_dir / f"{prefix}_all_cells.csv"
    case_path = output_dir / f"{prefix}_cases.csv"
    summary_path = output_dir / f"{prefix}_summary.csv"
    json_path = output_dir / f"{prefix}_summary.json"
    all_df.to_csv(all_path, index=False)
    case_df.to_csv(case_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "n_cells": int(len(all_df)),
                "n_clusters": int(summary_df.shape[0]),
                "veto_threshold": int(args.veto_threshold),
                "overall_score_counts": {
                    str(k): int(v)
                    for k, v in all_df["target_cluster_consistency_score"].value_counts().sort_index().items()
                },
                "overall_veto_rate": float(all_df["veto"].mean()),
                "overall_presence0_rate": float((all_df["presence_score"] == 0).mean()),
                "overall_signature0_rate": float((all_df["signature_score"] == 0).mean()),
                "paths": {
                    "all_cells": str(all_path),
                    "cases": str(case_path),
                    "summary": str(summary_path),
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"all_cells={all_path}")
    print(f"cases={case_path}")
    print(f"summary={summary_path}")
    print(
        summary_df[
            [
                "cluster",
                "n_cells",
                "n_genes_used",
                "score_mean",
                "score_median",
                "score_min",
                "score_p05",
                "veto_rate",
                "presence0_rate",
                "signature0_rate",
            ]
        ].to_string(index=False)
    )
    print("overall_score_counts", all_df["target_cluster_consistency_score"].value_counts().sort_index().to_dict())
    print("overall_veto_rate", float(all_df["veto"].mean()))


if __name__ == "__main__":
    main()
