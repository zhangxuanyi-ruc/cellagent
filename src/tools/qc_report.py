"""QC report helpers for CellAgent preprocessing outputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd


QC_COLUMNS = (
    "n_genes_by_counts",
    "log1p_n_genes_by_counts",
    "total_counts",
    "log1p_total_counts",
    "total_counts_mt",
    "log1p_total_counts_mt",
    "pct_counts_mt",
)


def write_preprocessing_qc_report(
    adata: ad.AnnData,
    summary: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write machine-readable preprocessing QC summaries.

    The report intentionally avoids heavy plotting. It stores compact JSON/CSV
    files that notebooks and downstream agents can consume directly.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "preprocessing_summary.json"
    summary_path.write_text(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8")

    qc_cols = [col for col in QC_COLUMNS if col in adata.obs]
    metrics_path = out_dir / "qc_metrics_summary.csv"
    if qc_cols:
        rows = []
        for col in qc_cols:
            vals = pd.to_numeric(adata.obs[col], errors="coerce").dropna()
            if vals.empty:
                continue
            rows.append(
                {
                    "metric": col,
                    "count": int(vals.shape[0]),
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                    "min": float(vals.min()),
                    "q25": float(vals.quantile(0.25)),
                    "median": float(vals.median()),
                    "q75": float(vals.quantile(0.75)),
                    "max": float(vals.max()),
                }
            )
        pd.DataFrame(rows).to_csv(metrics_path, index=False)
    else:
        pd.DataFrame(columns=["metric", "count", "mean", "std", "min", "q25", "median", "q75", "max"]).to_csv(
            metrics_path,
            index=False,
        )

    obs_cols_path = out_dir / "qc_obs_columns.csv"
    pd.DataFrame({"obs_column": list(map(str, adata.obs.columns))}).to_csv(obs_cols_path, index=False)

    return {
        "summary_json": summary_path,
        "metrics_csv": metrics_path,
        "obs_columns_csv": obs_cols_path,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
