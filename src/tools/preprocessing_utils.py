"""Reusable preprocessing utilities for external single-cell datasets.

The functions here intentionally wrap Scanpy primitives instead of hiding them.
They provide stable defaults for CellAgent while keeping each step reusable by
the later clustering, DE, and feature-service stages.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from src.tools.rag.mapper import Mapper


DEFAULT_BATCH_KEY_CANDIDATES = (
    "batch",
    "batch_key",
    "batch_id",
    "sample",
    "sample_id",
    "sample_name",
    "donor",
    "donor_id",
    "patient",
    "patient_id",
    "library",
    "library_id",
    "orig.ident",
    "dataset",
    "study",
    "chemistry",
    "platform",
)


@dataclass
class FilterConfig:
    filter_gene_by_counts: int | bool = 3
    filter_cell_by_counts: int | bool = False
    min_genes: int | None = None
    max_counts: int | None = None
    max_pct_mt: float | None = 5.0
    mt_gene_prefixes: tuple[str, ...] = ("MT-", "mt-", "Mt-")


@dataclass
class HVGConfig:
    n_top_genes: int = 1200
    flavor: str = "seurat_v3"
    fallback_flavor: str = "seurat"
    preferred_batch_key: str | None = "donor_id"
    batch_key_candidates: tuple[str, ...] = DEFAULT_BATCH_KEY_CANDIDATES
    subset: bool = True


@dataclass
class BinningConfig:
    enabled: bool = True
    n_bins: int = 51
    source_layer: str | None = None
    target_layer: str = "X_binned"
    dtype: str = "int16"


@dataclass
class GeneStandardizationConfig:
    enabled: bool = True
    mapper: dict[str, Any] = field(default_factory=dict)
    input_species: str = "auto"
    target_species: str = "human"
    symbol_columns: tuple[str, ...] = (
        "gene_name",
        "gene_symbol",
        "symbol",
        "hgnc",
        "mgi",
        "gene",
        "external_gene_name",
    )
    aggregate_duplicates: str = "sum"
    min_mapping_rate: float = 0.2


@dataclass
class EncoderVocabConfig:
    vocab_path: str | None = None
    id_column: str = "scgpt_id"
    require_exact_n_genes: bool = True


@dataclass
class PreprocessConfig:
    filter: FilterConfig = field(default_factory=FilterConfig)
    hvg: HVGConfig = field(default_factory=HVGConfig)
    binning: BinningConfig = field(default_factory=BinningConfig)
    gene_standardization: GeneStandardizationConfig = field(default_factory=GeneStandardizationConfig)
    encoder_vocab: EncoderVocabConfig = field(default_factory=EncoderVocabConfig)
    target_sum: float = 1e4
    save_counts_layer: str = "counts"
    log1p: bool = True
    make_var_names_unique: bool = True


def read_h5ad(path: str | Path) -> ad.AnnData:
    return sc.read_h5ad(str(path))


def write_h5ad(adata: ad.AnnData, path: str | Path, **write_kwargs: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path, **write_kwargs)
    return path


def default_preprocessed_path(input_path: str | Path, output_dir: str | Path | None = None) -> Path:
    input_path = Path(input_path)
    stem = input_path.name[:-5] if input_path.name.endswith(".h5ad") else input_path.stem
    parent = Path(output_dir) if output_dir else input_path.parent
    return parent / f"{stem}_preprocessed.h5ad"


def ensure_unique_names(adata: ad.AnnData, make_var_names_unique: bool = True) -> ad.AnnData:
    adata.obs_names_make_unique()
    if make_var_names_unique:
        adata.var_names_make_unique()
    return adata


def detect_expression_status(adata: ad.AnnData, max_values: int = 200000) -> dict[str, Any]:
    """Detect raw/log1p/count-like matrix status with conservative heuristics."""
    x = adata.X
    if sparse.issparse(x):
        values = x.data
    else:
        arr = np.asarray(x)
        values = arr.ravel()
    if values.size > max_values:
        rng = np.random.default_rng(1)
        values = values[rng.choice(values.size, size=max_values, replace=False)]

    values = np.asarray(values)
    nonzero = values[values > 0]
    has_negative = bool(np.any(values < 0))
    dtype = str(adata.X.dtype)
    max_nonzero = float(np.max(nonzero)) if nonzero.size else 0.0
    median_nonzero = float(np.median(nonzero)) if nonzero.size else 0.0
    integer_like = bool(np.issubdtype(adata.X.dtype, np.integer))
    if not integer_like and nonzero.size:
        sample = nonzero[: min(nonzero.size, 10000)]
        integer_like = bool(np.allclose(sample, np.rint(sample), rtol=0, atol=1e-6))

    if has_negative:
        status = "invalid_negative"
    elif nonzero.size == 0:
        status = "empty"
    elif integer_like:
        status = "raw_counts"
    elif max_nonzero < 20:
        status = "log1p_like"
    else:
        status = "count_like_or_normalized"

    return {
        "status": status,
        "dtype": dtype,
        "has_negative": has_negative,
        "integer_like": integer_like,
        "max_nonzero": max_nonzero,
        "median_nonzero": median_nonzero,
        "nonzero_fraction": float(nonzero.size / max(values.size, 1)),
    }


def _select_gene_id_series(adata: ad.AnnData, cfg: GeneStandardizationConfig) -> tuple[pd.Series, str]:
    index_series = pd.Series(adata.var_names.astype(str), index=adata.var_names, dtype="object")
    index_id_type = _guess_gene_id_type(index_series.tolist())
    if index_id_type in {"ensembl", "entrez"}:
        for col in cfg.symbol_columns:
            if col not in adata.var:
                continue
            vals = adata.var[col].astype(str)
            sample = vals.dropna().head(100)
            if sample.empty:
                continue
            if _guess_gene_id_type(sample.tolist()) == "symbol":
                return vals, f"var.{col}"
    return index_series, "var.index"


def _guess_gene_id_type(gene_names: list[str]) -> str:
    sample = [str(g).strip() for g in gene_names[:1000] if str(g).strip() and str(g).strip().lower() != "nan"]
    if not sample:
        return "unknown"
    ensembl = sum(g.upper().startswith(("ENSG", "ENSMUSG")) for g in sample)
    entrez = sum(g.isdigit() for g in sample)
    if ensembl / len(sample) >= 0.5:
        return "ensembl"
    if entrez / len(sample) >= 0.5:
        return "entrez"
    return "symbol"


def standardize_genes(
    adata: ad.AnnData,
    cfg: GeneStandardizationConfig,
) -> tuple[ad.AnnData, dict[str, Any]]:
    """Standardize gene identifiers before filtering/HVG/binning.

    Human inputs are normalized to HGNC approved symbols. Mouse inputs are first
    normalized to MGI symbols and then mapped to human ortholog symbols because
    the downstream encoder/RAG stack is human-gene based.
    """
    if not cfg.enabled:
        return adata, {"enabled": False}

    mapper = Mapper(cfg.mapper)
    raw_gene_ids = pd.Series(adata.var_names.astype(str), index=adata.var_names, dtype="object")
    selected_ids, selected_source = _select_gene_id_series(adata, cfg)
    selected_list = selected_ids.fillna(raw_gene_ids).astype(str).tolist()

    detected_species = mapper.detect_gene_species(selected_list)
    input_species = detected_species if cfg.input_species == "auto" else cfg.input_species
    if input_species not in {"human", "mouse"}:
        input_species = "human"

    gene_id_types = [mapper.detect_gene_id_type(gene, species=input_species) for gene in selected_list]
    normalized_symbols: list[str | None] = []
    human_symbols: list[str | None] = []
    for gene in selected_list:
        normalized = mapper.normalize_gene(gene, species=input_species)
        normalized_symbols.append(normalized)
        if cfg.target_species == "human":
            human_symbols.append(mapper.normalize_gene_to_human(gene, species=input_species))
        else:
            human_symbols.append(normalized)

    adata.var["cellagent_original_gene_id"] = raw_gene_ids.values
    adata.var["cellagent_gene_id_source"] = selected_source
    adata.var["cellagent_gene_id_type"] = gene_id_types
    adata.var["cellagent_input_species"] = input_species
    adata.var["cellagent_gene_symbol"] = [x or "" for x in normalized_symbols]
    adata.var["cellagent_human_gene_symbol"] = [x or "" for x in human_symbols]

    keep_mask = pd.Series([bool(x) for x in human_symbols], index=adata.var_names)
    kept_before_agg = int(keep_mask.sum())
    mapping_rate = kept_before_agg / max(adata.n_vars, 1)
    if mapping_rate < cfg.min_mapping_rate:
        raise ValueError(
            f"Gene mapping rate too low: {mapping_rate:.2%}. "
            f"species={input_species}, source={selected_source}"
        )

    adata = adata[:, keep_mask.values].copy()
    target_symbols = pd.Series(adata.var["cellagent_human_gene_symbol"].astype(str).values)
    adata = aggregate_duplicate_genes(adata, target_symbols.tolist(), method=cfg.aggregate_duplicates)

    summary = {
        "enabled": True,
        "selected_gene_id_source": selected_source,
        "detected_species": detected_species,
        "input_species": input_species,
        "target_species": cfg.target_species,
        "input_gene_id_type_majority": _majority(gene_id_types),
        "n_input_genes": int(len(raw_gene_ids)),
        "n_mapped_genes_before_aggregation": kept_before_agg,
        "n_genes_after_aggregation": int(adata.n_vars),
        "mapping_rate_before_aggregation": mapping_rate,
        "duplicate_genes_aggregated": int(kept_before_agg - adata.n_vars),
    }
    adata.uns["cellagent_gene_standardization"] = summary
    return adata, summary


def _majority(values: list[str]) -> str:
    if not values:
        return "unknown"
    counts = pd.Series(values).value_counts()
    return str(counts.index[0]) if len(counts) else "unknown"


def aggregate_duplicate_genes(
    adata: ad.AnnData,
    target_symbols: list[str],
    method: str = "sum",
) -> ad.AnnData:
    if method != "sum":
        raise ValueError(f"Unsupported duplicate gene aggregation method: {method}")
    symbols = pd.Index(target_symbols, name="gene_symbol")
    if not symbols.has_duplicates:
        adata.var_names = symbols.astype(str)
        adata.var["gene_symbol"] = adata.var_names
        return adata

    unique_symbols = pd.Index(pd.unique(symbols), name="gene_symbol")
    group_codes = pd.Categorical(symbols, categories=unique_symbols).codes
    indicator = sparse.csr_matrix(
        (
            np.ones(len(group_codes), dtype=np.float32),
            (np.arange(len(group_codes)), group_codes),
        ),
        shape=(len(group_codes), len(unique_symbols)),
    )
    x_new = adata.X @ indicator
    if sparse.issparse(adata.X):
        x_new = x_new.tocsr()
    else:
        x_new = np.asarray(x_new)

    var = pd.DataFrame(index=unique_symbols.astype(str))
    var["gene_symbol"] = var.index
    var["cellagent_aggregated_n_source_genes"] = pd.Series(symbols).value_counts().reindex(var.index).fillna(1).astype(int).values
    obs = adata.obs.copy()
    new_adata = ad.AnnData(X=x_new, obs=obs, var=var, uns=adata.uns.copy())
    return new_adata


def load_encoder_vocab(vocab_path: str | Path) -> dict[str, int]:
    path = Path(vocab_path)
    if not path.exists():
        raise FileNotFoundError(f"Encoder vocab not found: {path}")
    data = pd.read_json(path, typ="series")
    return {str(gene): int(idx) for gene, idx in data.items()}


def filter_to_encoder_vocab(
    adata: ad.AnnData,
    cfg: EncoderVocabConfig,
) -> tuple[ad.AnnData, dict[str, Any]]:
    if not cfg.vocab_path:
        raise ValueError("encoder_vocab.vocab_path is required before HVG selection.")
    vocab = load_encoder_vocab(cfg.vocab_path)
    matched = [gene in vocab for gene in adata.var_names.astype(str)]
    n_before = int(adata.n_vars)
    n_matched = int(sum(matched))
    if n_matched == 0:
        raise ValueError(f"No genes overlap with encoder vocab: {cfg.vocab_path}")
    adata = adata[:, matched].copy()
    adata.var[cfg.id_column] = [int(vocab[gene]) for gene in adata.var_names.astype(str)]
    summary = {
        "vocab_path": str(cfg.vocab_path),
        "id_column": cfg.id_column,
        "n_genes_before_vocab_filter": n_before,
        "n_genes_after_vocab_filter": n_matched,
        "vocab_overlap_rate": n_matched / max(n_before, 1),
    }
    adata.uns["cellagent_encoder_vocab"] = summary
    return adata, summary


def require_exact_gene_count(adata: ad.AnnData, expected_n_genes: int, stage: str) -> None:
    if adata.n_vars != expected_n_genes:
        raise ValueError(
            f"{stage}: expected exactly {expected_n_genes} genes, got {adata.n_vars}. "
            "Cell encoder inference requires the preprocessed HVG set to match the configured size."
        )


def mark_mitochondrial_genes(
    adata: ad.AnnData,
    prefixes: Iterable[str] = ("MT-", "mt-", "Mt-"),
    key: str = "mt",
) -> None:
    prefixes = tuple(prefixes)
    adata.var[key] = np.asarray([str(gene).startswith(prefixes) for gene in adata.var_names], dtype=bool)


def filter_cells_and_genes(adata: ad.AnnData, cfg: FilterConfig) -> ad.AnnData:
    if cfg.filter_gene_by_counts:
        sc.pp.filter_genes(
            adata,
            min_counts=cfg.filter_gene_by_counts if type(cfg.filter_gene_by_counts) is int else None,
        )
    if type(cfg.filter_cell_by_counts) is int:
        sc.pp.filter_cells(adata, min_counts=cfg.filter_cell_by_counts)
    if cfg.min_genes is not None:
        sc.pp.filter_cells(adata, min_genes=cfg.min_genes)
    if cfg.max_counts is not None:
        sc.pp.filter_cells(adata, max_counts=cfg.max_counts)
    return adata


def calculate_and_filter_qc(adata: ad.AnnData, cfg: FilterConfig) -> ad.AnnData:
    mark_mitochondrial_genes(adata, cfg.mt_gene_prefixes)
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, inplace=True)
    if cfg.max_pct_mt is not None and "pct_counts_mt" in adata.obs:
        adata = adata[adata.obs["pct_counts_mt"] <= cfg.max_pct_mt].copy()
    return adata


def normalize_total_log1p(
    adata: ad.AnnData,
    target_sum: float = 1e4,
    counts_layer: str = "counts",
    log1p: bool = True,
) -> ad.AnnData:
    if counts_layer:
        adata.layers[counts_layer] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=target_sum)
    if log1p:
        sc.pp.log1p(adata)
    return adata


def normalize_by_expression_status(
    adata: ad.AnnData,
    expression_status: dict[str, Any],
    target_sum: float = 1e4,
    counts_layer: str = "counts",
    log1p: bool = True,
) -> ad.AnnData:
    status = expression_status.get("status")
    if status in {"invalid_negative", "empty"}:
        raise ValueError(f"Invalid expression matrix status: {expression_status}")
    if status == "log1p_like":
        adata.uns["cellagent_normalization"] = {
            "action": "skipped_already_log1p_like",
            "input_status": expression_status,
        }
        return adata

    adata = normalize_total_log1p(
        adata,
        target_sum=target_sum,
        counts_layer=counts_layer,
        log1p=log1p,
    )
    adata.uns["cellagent_normalization"] = {
        "action": "normalize_total_log1p" if log1p else "normalize_total",
        "target_sum": target_sum,
        "counts_layer": counts_layer,
        "log1p": log1p,
        "input_status": expression_status,
    }
    return adata


def valid_batch_keys(adata: ad.AnnData, candidates: Iterable[str]) -> list[str]:
    keys: list[str] = []
    for key in candidates:
        if key not in adata.obs:
            continue
        values = adata.obs[key].dropna()
        if values.nunique() > 1:
            keys.append(key)
    return keys


def ordered_batch_keys(adata: ad.AnnData, cfg: HVGConfig) -> list[str]:
    keys = valid_batch_keys(adata, cfg.batch_key_candidates)
    if cfg.preferred_batch_key and cfg.preferred_batch_key in keys:
        return [cfg.preferred_batch_key]
    return keys


def select_highly_variable_genes(
    adata: ad.AnnData,
    cfg: HVGConfig,
    counts_layer: str = "counts",
) -> tuple[ad.AnnData, dict[str, Any]]:
    """Select HVGs using batch-aware candidates first, then standard fallback."""
    tried: list[dict[str, Any]] = []
    batch_keys = ordered_batch_keys(adata, cfg)

    for batch_key in batch_keys:
        for flavor, layer in ((cfg.flavor, counts_layer), (cfg.fallback_flavor, None)):
            try:
                sc.pp.highly_variable_genes(
                    adata,
                    n_top_genes=cfg.n_top_genes,
                    flavor=flavor,
                    batch_key=batch_key,
                    layer=layer if layer in adata.layers else None,
                    subset=False,
                )
                if int(adata.var.get("highly_variable", []).sum()) > 0:
                    info = {
                        "mode": "batch_aware",
                        "batch_key": batch_key,
                        "flavor": flavor,
                        "n_hvg": int(adata.var["highly_variable"].sum()),
                        "tried": tried,
                    }
                    return _subset_hvg_if_needed(adata, cfg.subset), info
            except Exception as exc:  # Scanpy flavor dependencies vary by env.
                tried.append({"batch_key": batch_key, "flavor": flavor, "error": str(exc)})

    for flavor, layer in ((cfg.flavor, counts_layer), (cfg.fallback_flavor, None)):
        try:
            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=cfg.n_top_genes,
                flavor=flavor,
                layer=layer if layer in adata.layers else None,
                subset=False,
            )
            if int(adata.var.get("highly_variable", []).sum()) > 0:
                info = {
                    "mode": "standard",
                    "batch_key": None,
                    "flavor": flavor,
                    "n_hvg": int(adata.var["highly_variable"].sum()),
                    "tried": tried,
                }
                return _subset_hvg_if_needed(adata, cfg.subset), info
        except Exception as exc:
            tried.append({"batch_key": None, "flavor": flavor, "error": str(exc)})

    raise RuntimeError(f"Highly-variable gene selection failed. Attempts: {tried}")


def _subset_hvg_if_needed(adata: ad.AnnData, subset: bool) -> ad.AnnData:
    if not subset:
        return adata
    return adata[:, adata.var["highly_variable"]].copy()


def _digitize_like_scgpt(x: np.ndarray, bins: np.ndarray, side: str = "one") -> np.ndarray:
    if side == "one":
        return np.digitize(x, bins)
    left_digits = np.digitize(x, bins)
    right_digits = np.digitize(x, bins, right=True)
    rands = np.random.rand(len(x))
    digits = rands * (right_digits - left_digits) + left_digits
    return np.ceil(digits).astype(np.int64)


def scgpt_binning_row(row: np.ndarray, n_bins: int = 51) -> np.ndarray:
    """CellAgent-local implementation of scGPT per-cell expression binning."""
    dtype = row.dtype
    row = np.asarray(row)
    if row.size == 0:
        return row.astype(dtype)
    if row.min() <= 0:
        non_zero_ids = row.nonzero()
        non_zero_row = row[non_zero_ids]
        binned_row = np.zeros_like(row, dtype=np.int64)
        if len(non_zero_row) > 0:
            bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
            binned_row[non_zero_ids] = _digitize_like_scgpt(non_zero_row, bins)
    else:
        bins = np.quantile(row, np.linspace(0, 1, n_bins - 1))
        binned_row = _digitize_like_scgpt(row, bins)
    return binned_row.astype(dtype)


def scgpt_binning_matrix(matrix: Any, n_bins: int = 51, dtype: str = "int16") -> np.ndarray:
    """Apply scGPT-style binning to each cell row of a dense or sparse matrix."""
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix)
    binned = np.zeros(matrix.shape, dtype=dtype)
    for idx in range(matrix.shape[0]):
        binned[idx] = scgpt_binning_row(matrix[idx], n_bins=n_bins).astype(dtype)
    return binned


def add_binned_layer(
    adata: ad.AnnData,
    cfg: BinningConfig,
) -> ad.AnnData:
    if not cfg.enabled:
        return adata
    source = adata.layers[cfg.source_layer] if cfg.source_layer else adata.X
    adata.layers[cfg.target_layer] = scgpt_binning_matrix(
        source,
        n_bins=cfg.n_bins,
        dtype=cfg.dtype,
    )
    adata.uns["cellagent_binning"] = {
        "method": "scgpt_compatible_per_cell_quantile",
        "n_bins": cfg.n_bins,
        "source_layer": cfg.source_layer or "X",
        "target_layer": cfg.target_layer,
        "dtype": cfg.dtype,
    }
    return adata


def config_from_dict(raw: dict[str, Any] | None) -> PreprocessConfig:
    raw = raw or {}
    filter_cfg = FilterConfig(**raw.get("filter", {}))
    hvg_raw = dict(raw.get("hvg", {}))
    if "batch_key_candidates" in hvg_raw:
        hvg_raw["batch_key_candidates"] = tuple(hvg_raw["batch_key_candidates"])
    hvg_cfg = HVGConfig(**hvg_raw)
    binning_cfg = BinningConfig(**raw.get("binning", {}))
    gene_raw = dict(raw.get("gene_standardization", {}))
    if "symbol_columns" in gene_raw:
        gene_raw["symbol_columns"] = tuple(gene_raw["symbol_columns"])
    gene_cfg = GeneStandardizationConfig(**gene_raw)
    encoder_raw = dict(raw.get("encoder_vocab", {}))
    encoder_cfg = EncoderVocabConfig(**encoder_raw)
    return PreprocessConfig(
        filter=filter_cfg,
        hvg=hvg_cfg,
        binning=binning_cfg,
        gene_standardization=gene_cfg,
        encoder_vocab=encoder_cfg,
        target_sum=raw.get("target_sum", 1e4),
        save_counts_layer=raw.get("save_counts_layer", "counts"),
        log1p=raw.get("log1p", True),
        make_var_names_unique=raw.get("make_var_names_unique", True),
    )
