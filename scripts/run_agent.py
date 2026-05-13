#!/usr/bin/env python
"""Run the unified CellAgent post-SCA agent pipeline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.pipeline import AgentPipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--rag-config", default="config/rag_sources.yaml")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--prior-json", default=None)
    parser.add_argument("--top-de-genes", type=int, default=30)
    parser.add_argument("--enable-llm-judge", action="store_true", help="Enable configured LLM-as-judge for conflict/function/tissue scoring.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = AgentPipeline.from_paths(
        config_path=args.config,
        rag_config_path=args.rag_config,
        manifest_path=args.manifest,
        prior_json_path=args.prior_json,
        top_de_genes=args.top_de_genes,
        enable_llm_judge=args.enable_llm_judge,
    )
    result = pipeline.run()
    print(
        json.dumps(
            {
                "reasoning_paths": [str(p) for p in result.reasoning_paths],
                "judge_paths": [str(p) for p in result.judge_paths],
                "final_json": str(result.final_json) if result.final_json else None,
                "final_csv": str(result.final_csv) if result.final_csv else None,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
