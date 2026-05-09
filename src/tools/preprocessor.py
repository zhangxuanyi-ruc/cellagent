"""External-data preprocessing entry points for CellAgent.

This module handles the first Phase4 step only: QC/filtering, batch-aware HVG
selection with fallback, optional scGPT-compatible binning, and writing a
`*_preprocessed.h5ad` dataset. Leiden clustering and DE export will be layered
on top after this preprocessing contract is stable.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad

from .preprocessing_utils import (
    PreprocessConfig,
    add_binned_layer,
    calculate_and_filter_qc,
    config_from_dict,
    default_preprocessed_path,
    detect_expression_status,
    ensure_unique_names,
    filter_cells_and_genes,
    filter_to_encoder_vocab,
    normalize_by_expression_status,
    read_h5ad,
    require_exact_gene_count,
    select_highly_variable_genes,
    standardize_genes,
    write_h5ad,
)


@dataclass
class PreprocessResult:
    adata: ad.AnnData
    output_path: Path
    summary: dict[str, Any]


class ExternalDataPreprocessor:
    """Preprocess external AnnData inputs into CellAgent-ready h5ad files."""

    def __init__(self, config: PreprocessConfig | dict[str, Any] | None = None):
        if isinstance(config, PreprocessConfig):
            self.config = config
        else:
            self.config = config_from_dict(config)

    def run(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
        output_dir: str | Path | None = None,
    ) -> PreprocessResult:
        start = time.time()
        print(f"[preprocess] read_h5ad start: {input_path}", flush=True)
        adata = read_h5ad(input_path)
        print(f"[preprocess] read_h5ad done: shape={adata.shape}, elapsed={time.time() - start:.1f}s", flush=True)
        return self.run_adata(
            adata,
            input_path=input_path,
            output_path=output_path,
            output_dir=output_dir,
        )

    def run_adata(
        self,
        adata: ad.AnnData,
        input_path: str | Path | None = None,
        output_path: str | Path | None = None,
        output_dir: str | Path | None = None,
    ) -> PreprocessResult:
        cfg = self.config
        total_start = time.time()
        input_shape = list(adata.shape)

        stage = time.time()
        adata.obs_names_make_unique()
        expression_status = detect_expression_status(adata)
        adata.uns["cellagent_expression_status"] = expression_status
        print(f"[preprocess] expression status: {expression_status['status']}, elapsed={time.time() - stage:.1f}s", flush=True)

        stage = time.time()
        adata, gene_info = standardize_genes(adata, cfg.gene_standardization)
        print(f"[preprocess] gene standardization done: shape={adata.shape}, elapsed={time.time() - stage:.1f}s", flush=True)
        ensure_unique_names(adata, make_var_names_unique=cfg.make_var_names_unique)

        stage = time.time()
        adata, encoder_vocab_info = filter_to_encoder_vocab(adata, cfg.encoder_vocab)
        print(f"[preprocess] encoder vocab filter done: shape={adata.shape}, elapsed={time.time() - stage:.1f}s", flush=True)

        stage = time.time()
        adata = filter_cells_and_genes(adata, cfg.filter)
        adata = calculate_and_filter_qc(adata, cfg.filter)
        post_qc_shape = list(adata.shape)
        print(f"[preprocess] filter/qc done: shape={adata.shape}, elapsed={time.time() - stage:.1f}s", flush=True)
        if adata.n_obs == 0 or adata.n_vars == 0:
            raise ValueError(
                f"No cells/genes remain after QC filtering. "
                f"input_shape={input_shape}, post_qc_shape={post_qc_shape}"
            )

        stage = time.time()
        adata = normalize_by_expression_status(
            adata,
            expression_status=expression_status,
            target_sum=cfg.target_sum,
            counts_layer=cfg.save_counts_layer,
            log1p=cfg.log1p,
        )
        print(f"[preprocess] normalization done: shape={adata.shape}, elapsed={time.time() - stage:.1f}s", flush=True)

        stage = time.time()
        adata, hvg_info = select_highly_variable_genes(
            adata,
            cfg.hvg,
            counts_layer=cfg.save_counts_layer,
        )
        print(f"[preprocess] hvg done: shape={adata.shape}, info={hvg_info}, elapsed={time.time() - stage:.1f}s", flush=True)
        if cfg.encoder_vocab.require_exact_n_genes:
            require_exact_gene_count(adata, cfg.hvg.n_top_genes, stage="post_hvg")
        post_hvg_shape = list(adata.shape)

        # Critical contract: feature extraction must consume this saved binned layer.
        stage = time.time()
        adata = add_binned_layer(adata, cfg.binning)
        print(f"[preprocess] binning done: shape={adata.shape}, elapsed={time.time() - stage:.1f}s", flush=True)

        hvg_summary = dict(hvg_info)
        hvg_summary["tried"] = json.dumps(hvg_summary.get("tried", []), ensure_ascii=False)
        summary = {
            "input_path": str(input_path) if input_path else None,
            "input_shape": input_shape,
            "post_qc_shape": post_qc_shape,
            "post_hvg_shape": post_hvg_shape,
            "expression_status": expression_status,
            "gene_standardization": gene_info,
            "encoder_vocab": encoder_vocab_info,
            "hvg": hvg_summary,
            "binning": adata.uns.get("cellagent_binning"),
            "feature_input_layer": cfg.binning.target_layer if cfg.binning.enabled else None,
        }
        adata.uns["cellagent_preprocessing"] = summary

        if output_path is None:
            if input_path is None:
                raise ValueError("output_path is required when preprocessing an in-memory AnnData.")
            output_path = default_preprocessed_path(input_path, output_dir=output_dir)
        stage = time.time()
        output_path = write_h5ad(adata, output_path)
        print(f"[preprocess] write_h5ad done: {output_path}, elapsed={time.time() - stage:.1f}s, total={time.time() - total_start:.1f}s", flush=True)

        return PreprocessResult(adata=adata, output_path=output_path, summary=summary)
