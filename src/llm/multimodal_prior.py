"""Multimodal prior inference with the CellWhisperer/LLaVA-Mistral checkpoint.

This module produces CellAgent's initial ``Prediction`` from a 768-dim encoder
feature. The checkpoint is a PEFT LoRA adapter plus a trained LLaVA
``mm_projector``; the raw feature is passed through the LLaVA ``images=`` path.
"""
from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_BASE_MODEL = "/data/bgi/data/projects/multimodal/zxy/models/mistral_7b"
DEFAULT_ADAPTER_PATH = "/data/bgi/data/projects/multimodal/zxy/features_and_ckpts/zxy/okrcell_cw_stage2"
DEFAULT_LLAVA_ROOT = "/root/wanghaoran/CellWhisperer/modules/LLaVA"


@dataclass
class MultimodalPriorResult:
    cell_type: str
    function: str
    confidence: float
    celltype_nll: float
    celltype_perplexity: float
    raw_text: str
    prompt: str

    def to_prediction_dict(self) -> dict[str, Any]:
        return {
            "cell_type": self.cell_type,
            "function": self.function,
            "confidence": self.confidence,
            "celltype_nll": self.celltype_nll,
            "celltype_perplexity": self.celltype_perplexity,
            "raw_text": self.raw_text,
            "prompt": self.prompt,
        }


class LlavaMistralPrior:
    """Thin inference wrapper for the LLaVA-Mistral cell prior model."""

    def __init__(
        self,
        base_model: str = DEFAULT_BASE_MODEL,
        adapter_path: str = DEFAULT_ADAPTER_PATH,
        llava_root: str = DEFAULT_LLAVA_ROOT,
        device: str | None = None,
        dtype: str = "bf16",
        merge_lora: bool = False,
    ):
        self.base_model = Path(base_model)
        self.adapter_path = Path(adapter_path)
        self.llava_root = Path(llava_root)
        import torch

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = self._resolve_dtype(dtype)

        self._prepare_llava_imports()
        self.tokenizer, self.model = self._load_model(merge_lora=merge_lora)

    @staticmethod
    def _resolve_dtype(dtype: str):
        import torch

        mapping = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        if dtype not in mapping:
            raise ValueError(f"Unsupported dtype: {dtype}")
        return mapping[dtype]

    def _prepare_llava_imports(self) -> None:
        if not self.llava_root.exists():
            raise FileNotFoundError(f"LLaVA root does not exist: {self.llava_root}")
        root = str(self.llava_root)
        if root not in sys.path:
            sys.path.insert(0, root)

    def _load_model(self, merge_lora: bool):
        import torch
        from peft import PeftModel
        from transformers import AutoTokenizer

        from llava.model.language_model.llava_mistral import LlavaMistralConfig, LlavaMistralForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            use_fast=False,
            local_files_only=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        cfg = LlavaMistralConfig.from_pretrained(self.adapter_path)
        model = LlavaMistralForCausalLM.from_pretrained(
            self.base_model,
            config=cfg,
            low_cpu_mem_usage=True,
            torch_dtype=self.dtype,
            local_files_only=True,
        )

        non_lora_path = self.adapter_path / "non_lora_trainables.bin"
        if not non_lora_path.exists():
            raise FileNotFoundError(f"Missing non-LoRA projector weights: {non_lora_path}")
        non_lora = torch.load(non_lora_path, map_location="cpu")
        non_lora = self._normalize_non_lora_keys(non_lora)
        missing, unexpected = model.load_state_dict(non_lora, strict=False)
        projector_missing = [k for k in missing if "mm_projector" in k]
        if projector_missing:
            raise RuntimeError(f"Failed to load mm_projector keys: {projector_missing[:8]}")
        if unexpected:
            print(f"[multimodal_prior] Ignored unexpected non-LoRA keys: {unexpected[:8]}")

        model = PeftModel.from_pretrained(model, self.adapter_path, torch_dtype=self.dtype)
        if merge_lora:
            model = model.merge_and_unload()

        model.eval()
        model.to(self.device)
        return tokenizer, model

    @staticmethod
    def _normalize_non_lora_keys(state: dict[str, Any]) -> dict[str, Any]:
        clean = {}
        for key, value in state.items():
            new_key = key
            if new_key.startswith("base_model."):
                new_key = new_key[len("base_model.") :]
            if new_key.startswith("model.model."):
                new_key = new_key[len("model.") :]
            clean[new_key] = value
        return clean

    def build_prompt(self, metadata: dict[str, Any] | None = None) -> str:
        metadata = metadata or {}
        metadata_text = json.dumps(metadata, ensure_ascii=False, sort_keys=True) if metadata else "not provided"
        question = (
            "<image>\n"
            "You are given a transcriptomic encoder feature for one target cell.\n"
            f"Target cell metadata: {metadata_text}\n"
            "Infer the most likely cell type and a concise biological function. "
            "Return only valid JSON with exactly these keys: "
            '{"celltype": string, "function": string}.'
        )
        return self._conversation_prompt(question)

    @staticmethod
    def _conversation_prompt(question: str) -> str:
        from llava.conversation import conv_templates

        conv = conv_templates["mistral_instruct"].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def infer(
        self,
        feature: np.ndarray | torch.Tensor,
        metadata: dict[str, Any] | None = None,
        max_new_tokens: int = 160,
        temperature: float = 0.0,
        top_p: float | None = None,
        num_beams: int = 1,
    ) -> MultimodalPriorResult:
        import torch

        prompt = self.build_prompt(metadata)
        image_feature = self._prepare_feature(feature)
        input_ids = self._tokenize_image_prompt(prompt).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                inputs=input_ids,
                images=image_feature,
                image_sizes=None,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        raw_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        parsed = self._parse_prediction(raw_text)
        nll, ppl = self.celltype_perplexity(prompt, image_feature, parsed["celltype"], parsed["function"])
        confidence = self._nll_to_confidence(nll)
        return MultimodalPriorResult(
            cell_type=parsed["celltype"],
            function=parsed["function"],
            confidence=confidence,
            celltype_nll=nll,
            celltype_perplexity=ppl,
            raw_text=raw_text,
            prompt=prompt,
        )

    def celltype_perplexity(
        self,
        prompt: str,
        image_feature: torch.Tensor,
        cell_type: str,
        function: str,
    ) -> tuple[float, float]:
        import torch

        answer = json.dumps({"celltype": cell_type, "function": function}, ensure_ascii=False)
        celltype_start = answer.index(cell_type)
        before_celltype = answer[:celltype_start]
        through_celltype = answer[: celltype_start + len(cell_type)]

        full_ids = self._tokenize_image_prompt(prompt + answer)
        start = len(self._tokenize_image_prompt(prompt + before_celltype))
        end = len(self._tokenize_image_prompt(prompt + through_celltype))
        if end <= start:
            raise RuntimeError("Could not identify celltype token span for perplexity.")

        labels = torch.full_like(full_ids, -100)
        labels[start:end] = full_ids[start:end]
        input_ids = full_ids.unsqueeze(0).to(self.device)
        labels = labels.unsqueeze(0).to(self.device)

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids,
                images=image_feature,
                image_sizes=None,
                labels=labels,
                use_cache=False,
                return_dict=True,
            )
        nll = float(outputs.loss.detach().float().cpu().item())
        ppl = float(math.exp(min(nll, 50.0)))
        return nll, ppl

    def _prepare_feature(self, feature: np.ndarray | torch.Tensor) -> torch.Tensor:
        import torch

        tensor = torch.as_tensor(feature, dtype=torch.float32).flatten()
        if tensor.numel() != 768:
            raise ValueError(f"Expected a 768-dim encoder feature, got shape {tuple(tensor.shape)}")
        return tensor.unsqueeze(0).to(device=self.device, dtype=self.dtype)

    def _tokenize_image_prompt(self, prompt: str) -> torch.LongTensor:
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.mm_utils import tokenizer_image_token

        return tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")

    @staticmethod
    def _parse_prediction(text: str) -> dict[str, str]:
        candidates = re.findall(r"\{.*?\}", text, flags=re.S)
        candidates.append(text)

        for candidate in reversed(candidates):
            try:
                obj = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            cell_type = obj.get("celltype") or obj.get("cell_type")
            function = obj.get("function")
            if cell_type and function:
                return {"celltype": str(cell_type).strip(), "function": str(function).strip()}

        raise ValueError(f"Model did not return parseable prediction JSON: {text[:500]}")

    @staticmethod
    def _nll_to_confidence(nll: float) -> float:
        # Mean token probability for the generated celltype span. This keeps the
        # confidence in [0, 1] and is directly tied to the requested PPL signal.
        return float(max(0.0, min(1.0, math.exp(-nll))))


def load_feature_from_path(path: str, cell_id: str | None = None, feature_key: str | None = None) -> np.ndarray:
    """Load one 768-dim feature from .npy/.npz/.json/.h5ad."""
    feature_path = Path(path)
    suffix = feature_path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(feature_path)
        return _select_feature_row(arr, cell_id)

    if suffix == ".npz":
        data = np.load(feature_path)
        key = feature_key or ("features" if "features" in data else data.files[0])
        return _select_feature_row(data[key], cell_id)

    if suffix == ".json":
        with open(feature_path, "r") as f:
            obj = json.load(f)
        if cell_id is not None and isinstance(obj, dict) and cell_id in obj:
            return np.asarray(obj[cell_id], dtype=np.float32)
        return np.asarray(obj, dtype=np.float32)

    if suffix == ".h5ad":
        import anndata as ad

        adata = ad.read_h5ad(feature_path, backed="r")
        try:
            if feature_key:
                source = adata.obsm[feature_key]
            elif "X_scFM" in adata.obsm:
                source = adata.obsm["X_scFM"]
            elif "X_pca" in adata.obsm:
                source = adata.obsm["X_pca"]
            else:
                source = adata.X

            if cell_id is None:
                row = source[0]
            else:
                obs_names = [str(x) for x in adata.obs_names]
                if cell_id in obs_names:
                    row = source[obs_names.index(cell_id)]
                elif "cell_id" in adata.obs:
                    values = [str(x) for x in adata.obs["cell_id"]]
                    row = source[values.index(cell_id)]
                else:
                    raise KeyError(f"Cell id not found in h5ad obs_names: {cell_id}")

            if hasattr(row, "toarray"):
                row = row.toarray()
            return np.asarray(row, dtype=np.float32).flatten()
        finally:
            if getattr(adata, "file", None) is not None:
                adata.file.close()

    raise ValueError(f"Unsupported feature file: {feature_path}")


def _select_feature_row(arr: np.ndarray, cell_id: str | None = None) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if cell_id is not None:
        try:
            idx = int(cell_id)
        except ValueError as exc:
            raise ValueError("For .npy/.npz matrix input, --cell-id must be an integer row index.") from exc
        return arr[idx]
    return arr[0]
