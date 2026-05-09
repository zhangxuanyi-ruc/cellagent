#!/usr/bin/env python3
"""Run CellAgent's multimodal prior model on one target cell feature."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.multimodal_prior import (  # noqa: E402
    DEFAULT_ADAPTER_PATH,
    DEFAULT_BASE_MODEL,
    DEFAULT_LLAVA_ROOT,
    LlavaMistralPrior,
    load_feature_from_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-path", required=True, help="Input .npy/.npz/.json/.h5ad containing a 768-dim feature.")
    parser.add_argument("--cell-id", default=None, help="Cell id for h5ad/json, or integer row index for npy/npz matrices.")
    parser.add_argument("--feature-key", default=None, help="Feature key for h5ad obsm or npz arrays. Defaults: X_scFM > X_pca > X.")
    parser.add_argument("--metadata-json", default=None, help="Optional metadata JSON string or path to a JSON file.")
    parser.add_argument("--output-json", required=True, help="Path to write the prediction JSON.")

    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-path", default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--llava-root", default=DEFAULT_LLAVA_ROOT)
    parser.add_argument("--device", default=None, help="Default: cuda if available else cpu.")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--merge-lora", action="store_true", help="Merge LoRA into the base model after loading.")

    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--num-beams", type=int, default=1)
    return parser.parse_args()


def load_metadata(value: str | None) -> dict:
    if not value:
        return {}
    candidate = Path(value)
    if candidate.exists():
        with open(candidate, "r") as f:
            return json.load(f)
    return json.loads(value)


def main() -> None:
    args = parse_args()
    feature = load_feature_from_path(args.feature_path, cell_id=args.cell_id, feature_key=args.feature_key)
    metadata = load_metadata(args.metadata_json)

    prior = LlavaMistralPrior(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        llava_root=args.llava_root,
        device=args.device,
        dtype=args.dtype,
        merge_lora=args.merge_lora,
    )
    result = prior.infer(
        feature,
        metadata=metadata,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
    )

    payload = result.to_prediction_dict()
    payload["cell_id"] = args.cell_id
    payload["feature_path"] = args.feature_path
    payload["metadata"] = metadata

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
