#!/usr/bin/env python
"""Extract cell encoder features from a CellAgent preprocessed h5ad.

This script intentionally does only cell encoder inference. It requires a
preprocessed h5ad containing `layers[X_binned]` and `var[scgpt_id]`.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import types
from collections import OrderedDict
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import torch
import torch.nn as nn
import yaml
from scipy import sparse
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_LOCAL_SCGPT_ROOT = Path("/root/wanghaoran/zxy/project/sc_showo/scgpt")


def load_local_transformer_model(scgpt_root: str | Path = DEFAULT_LOCAL_SCGPT_ROOT):
    """Load the project-local scGPT model without importing pip scgpt."""
    scgpt_root = Path(scgpt_root)
    for key in list(sys.modules.keys()):
        if key == "scgpt" or key.startswith("scgpt."):
            del sys.modules[key]

    scgpt_pkg = types.ModuleType("scgpt")
    scgpt_pkg.__path__ = [str(scgpt_root)]
    sys.modules["scgpt"] = scgpt_pkg

    model_dir = scgpt_root / "model"
    model_pkg = types.ModuleType("scgpt.model")
    model_pkg.__path__ = [str(model_dir)]
    sys.modules["scgpt.model"] = model_pkg
    scgpt_pkg.model = model_pkg

    module = importlib.import_module("scgpt.model.model_text_CL_flashattn_2")
    return module.TransformerModel


PAD_TOKEN = "<pad>"
SPECIAL_TOKENS = [PAD_TOKEN, "<cls>", "<eoc>"]


class SimpleGeneVocab:
    """Minimal token->id mapping needed by the local scGPT encoder."""

    def __init__(self, token_to_id: dict[str, int]):
        self.token_to_id = dict(token_to_id)

    @classmethod
    def from_file(cls, vocab_path: str | Path) -> "SimpleGeneVocab":
        with open(vocab_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls({str(k): int(v) for k, v in data.items()})

    def __contains__(self, token: str) -> bool:
        return token in self.token_to_id

    def __getitem__(self, token: str) -> int:
        return self.token_to_id[token]

    def __len__(self) -> int:
        return max(self.token_to_id.values()) + 1

    def append_token(self, token: str) -> None:
        if token not in self.token_to_id:
            self.token_to_id[token] = len(self)


class CellProjectionHead(nn.Module):
    def __init__(self, input_dim: int, intermediate_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, intermediate_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(intermediate_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.fc2(self.relu(self.fc1(x))))


def load_cell_encoder(
    model_path: str | Path,
    vocab: SimpleGeneVocab,
    device: torch.device,
    scgpt_root: str | Path = DEFAULT_LOCAL_SCGPT_ROOT,
) -> nn.Module:
    TransformerModel = load_local_transformer_model(scgpt_root)
    model_path = Path(model_path)
    model_dir = model_path.parent
    with open(model_dir / "args.json", "r", encoding="utf-8") as f:
        model_configs = json.load(f)

    gene_encoder = TransformerModel(
        ntoken=len(vocab),
        d_model=model_configs["embsize"],
        nhead=model_configs["nheads"],
        d_hid=model_configs["d_hid"],
        nlayers=model_configs["nlayers"],
        nlayers_cls=model_configs.get("n_layers_cls", 3),
        padding_idx=vocab[PAD_TOKEN],
        pad_token=PAD_TOKEN,
        cell_emb_style=model_configs.get("cell_emb_style", "cls"),
        use_fast_transformer=True,
        dim_text=768,
    )

    full_state_dict = torch.load(model_path, map_location="cpu")
    gene_encoder_state_dict = OrderedDict()
    for key, value in full_state_dict.items():
        if key.startswith(("encoder.", "flag_encoder.", "transformer_encoder.", "transformer_encoder_CL.")):
            gene_encoder_state_dict[key] = value
    gene_encoder.load_state_dict(gene_encoder_state_dict, strict=False)

    projection_head = CellProjectionHead(
        input_dim=model_configs["embsize"],
        intermediate_dim=768,
        output_dim=768,
    )
    projection_head_state_dict = OrderedDict()
    for key, value in full_state_dict.items():
        if key.startswith("cell2textAdapter."):
            projection_head_state_dict[key.replace("cell2textAdapter.", "")] = value
    if projection_head_state_dict:
        projection_head.load_state_dict(projection_head_state_dict)
    gene_encoder.cell2textAdapter = projection_head

    gene_encoder.to(device).half().eval()
    return gene_encoder


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def validate_preprocessed_adata(
    adata: ad.AnnData,
    feature_layer: str,
    gene_id_column: str,
    n_genes: int,
) -> None:
    if "cellagent_preprocessing" not in adata.uns:
        raise ValueError("Input h5ad is not a CellAgent preprocessed dataset: missing uns['cellagent_preprocessing'].")
    if feature_layer not in adata.layers:
        raise ValueError(f"Missing required binned feature layer: layers['{feature_layer}'].")
    if gene_id_column not in adata.var:
        raise ValueError(f"Missing required encoder gene id column: var['{gene_id_column}'].")
    if adata.n_vars != n_genes:
        raise ValueError(f"Expected exactly {n_genes} HVG genes, got {adata.n_vars}.")
    if adata.var[gene_id_column].isna().any():
        raise ValueError(f"var['{gene_id_column}'] contains NA values.")


def extract_features(
    preprocessed_h5ad: str | Path,
    output_dir: str | Path,
    model_path: str | Path,
    vocab_path: str | Path,
    feature_layer: str = "X_binned",
    gene_id_column: str = "scgpt_id",
    n_genes: int = 1200,
    batch_size: int = 1024,
    output_format: str = "npz",
    device: str | None = None,
    scgpt_root: str | Path = DEFAULT_LOCAL_SCGPT_ROOT,
) -> Path:
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(preprocessed_h5ad, backed=None)
    validate_preprocessed_adata(adata, feature_layer=feature_layer, gene_id_column=gene_id_column, n_genes=n_genes)

    vocab = SimpleGeneVocab.from_file(vocab_path)
    for token in SPECIAL_TOKENS:
        if token not in vocab:
            vocab.append_token(token)

    gene_ids = adata.var[gene_id_column].astype(int).to_numpy(dtype=np.int64)
    values = adata.layers[feature_layer]
    if sparse.issparse(values):
        values = values.toarray()
    values = np.asarray(values)
    if values.shape != (adata.n_obs, n_genes):
        raise ValueError(f"Expected layer shape {(adata.n_obs, n_genes)}, got {values.shape}.")

    gene_encoder = load_cell_encoder(model_path, vocab, device=device_obj, scgpt_root=scgpt_root)

    all_features: list[np.ndarray] = []
    gene_ids_batch_base = torch.from_numpy(gene_ids).long()
    n_batches = int(np.ceil(adata.n_obs / batch_size))

    for batch_idx in tqdm(range(n_batches), desc="cell encoder inference"):
        start = batch_idx * batch_size
        end = min(start + batch_size, adata.n_obs)
        batch_values = values[start:end]
        current_size = end - start

        expr_tensor = torch.from_numpy(batch_values).to(device=device_obj, dtype=torch.float32, non_blocking=True)
        gene_ids_tensor = gene_ids_batch_base.unsqueeze(0).expand(current_size, -1).to(device=device_obj, non_blocking=True)
        src_key_padding_mask = torch.zeros(current_size, n_genes, dtype=torch.bool, device=device_obj)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=device_obj.type == "cuda"):
                output = gene_encoder(
                    src=gene_ids_tensor,
                    values=expr_tensor,
                    src_key_padding_mask=src_key_padding_mask,
                    extract_cell_feature=True,
                    CLS=False,
                    MVC=False,
                    ECS=False,
                )
        all_features.append(output.detach().cpu().numpy().astype(np.float16))

    feature_matrix = np.concatenate(all_features, axis=0)
    stem = Path(preprocessed_h5ad).name
    stem = stem[:-5] if stem.endswith(".h5ad") else Path(preprocessed_h5ad).stem
    metadata = {
        "input_h5ad": str(preprocessed_h5ad),
        "model_path": str(model_path),
        "vocab_path": str(vocab_path),
        "feature_layer": feature_layer,
        "gene_id_column": gene_id_column,
        "n_cells": int(adata.n_obs),
        "n_genes": int(n_genes),
        "feature_dim": int(feature_matrix.shape[1]),
        "device": str(device_obj),
        "batch_size": int(batch_size),
        "scgpt_root": str(scgpt_root),
        "obs_names": adata.obs_names.astype(str).tolist(),
        "var_names": adata.var_names.astype(str).tolist(),
    }

    if output_format == "npy":
        out_path = output_dir / f"{stem}_cell_features.npy"
        np.save(out_path, feature_matrix)
        (output_dir / f"{stem}_cell_features.metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    elif output_format == "npz":
        out_path = output_dir / f"{stem}_cell_features.npz"
        np.savez_compressed(
            out_path,
            features=feature_matrix,
            obs_names=np.asarray(metadata["obs_names"], dtype=object),
            var_names=np.asarray(metadata["var_names"], dtype=object),
            scgpt_id=gene_ids.astype(np.int64),
            metadata=json.dumps(metadata, ensure_ascii=False),
        )
    else:
        raise ValueError(f"Unsupported output_format: {output_format}")

    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract CellAgent cell encoder features from preprocessed h5ad.")
    parser.add_argument("--input", required=True, help="CellAgent *_preprocessed.h5ad path.")
    parser.add_argument("--config", default="config/config.yaml", help="YAML config path.")
    parser.add_argument("--feature-dir", default=None, help="Override cell_encoder.feature_dir.")
    parser.add_argument("--model-path", default=None, help="Override cell_encoder.model_path.")
    parser.add_argument("--vocab-path", default=None, help="Override cell_encoder.vocab_path.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override cell_encoder.batch_size.")
    parser.add_argument("--device", default=None, help="Override cell_encoder.device, e.g. cuda, cuda:0, cpu.")
    parser.add_argument("--gpu-id", default=None, help="Override cell_encoder.gpu_id via CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--scgpt-root", default=None, help="Override cell_encoder.scgpt_root.")
    parser.add_argument("--output-format", choices=["npz", "npy"], default=None, help="Override cell_encoder.output_format.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    full_cfg = load_config(args.config)
    cfg = full_cfg.get("cell_encoder", full_cfg)
    pipeline_cfg = full_cfg.get("pipeline", {})
    output_root = Path(pipeline_cfg["output_root"]) if pipeline_cfg.get("output_root") else None
    gpu_id = args.gpu_id if args.gpu_id is not None else cfg.get("gpu_id")
    if gpu_id is not None and str(gpu_id).lower() not in {"", "none", "null"}:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    out_path = extract_features(
        preprocessed_h5ad=args.input,
        output_dir=args.feature_dir or (str(output_root / "features") if output_root else "output/features"),
        model_path=args.model_path or cfg["model_path"],
        vocab_path=args.vocab_path or cfg["vocab_path"],
        feature_layer=cfg.get("feature_layer", "X_binned"),
        gene_id_column=cfg.get("gene_id_column", "scgpt_id"),
        n_genes=int(cfg.get("n_genes", 1200)),
        batch_size=args.batch_size or int(cfg.get("batch_size", 1024)),
        output_format=args.output_format or cfg.get("output_format", "npz"),
        device=args.device or cfg.get("device"),
        scgpt_root=args.scgpt_root or cfg.get("scgpt_root", DEFAULT_LOCAL_SCGPT_ROOT),
    )
    print(json.dumps({"feature_path": str(out_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
