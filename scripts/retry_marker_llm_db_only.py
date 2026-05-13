#!/usr/bin/env python3
"""Retry database-only marker LLM judge on saved real-data cases.

This is a diagnostic script. It intentionally removes obs labels from the
prompt and asks the LLM to judge only database marker lists plus precomputed
top10/top30 overlaps.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.clients import OpenAICompatibleClient
from src.tools.rag import RAGFacade


MULTI_CASES_PATH = Path("/home/qijinyin/wanghaoran/zxy/marker_llm_judge_multi_cases.json")
DE_ROOT = Path("/home/qijinyin/wanghaoran/zxy/cellagent_output/de")
OUTPUT_PATH = Path("/home/qijinyin/wanghaoran/zxy/marker_llm_judge_db_only_retry.json")


SYSTEM_PROMPT = (
    "You are a strict database-only marker evidence judge. "
    "You must not use biological memory or external knowledge. "
    "Only use marker_gene_list and the precomputed top10/top30 matches supplied by the user. "
    "If marker_gene_list is empty, or both top10_matches and top30_matches are empty, "
    "you must return decision INSUFFICIENT_DB_EVIDENCE or REJECT_NO_DATABASE_SUPPORT, "
    "recommended_marker_score 0, and should_marker_veto true. Return JSON only."
)


def build_client() -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        base_url="https://localagent.tail053d0c.ts.net/v1",
        model=None,
        api_key="unused",
        temperature=0.0,
        timeout=90,
        max_retries=1,
        tailscale_host="localagent.tail053d0c.ts.net",
        tailscale_ip="100.105.94.49",
        socks5_proxy="socks5://localhost:1080",
        verify_ssl=False,
        enable_thinking=False,
    )


def main() -> None:
    old_cases = json.loads(MULTI_CASES_PATH.read_text(encoding="utf-8"))
    rag = RAGFacade(PROJECT_ROOT / "config/rag_sources.yaml")
    mapper = rag.mapper
    client = build_client()
    print(f"model={client.model}")

    results: dict[str, dict] = {}
    for case_id, item in old_cases.items():
        summary = item["evidence_summary"]
        cluster = str(summary["cluster_id"])
        predicted_cell_type = summary["predicted_cell_type"]
        metadata = {"species": "human", "tissue": summary.get("metadata_tissue") or ""}

        de_csv = DE_ROOT / f"cluster_{cluster}_vs_all.csv"
        de_df = pd.read_csv(de_csv)
        top30 = [
            mapper.normalize_gene(str(gene)) or str(gene)
            for gene in de_df["gene"].dropna().astype(str).tolist()[:30]
        ]
        top10 = top30[:10]

        markers = rag.query_markers(
            predicted_cell_type,
            species="human",
            top_k=None,
            min_markers=5,
            metadata=metadata,
        )
        marker_genes = sorted(
            {
                mapper.normalize_gene(marker.gene_normalized or marker.gene)
                or str(marker.gene_normalized or marker.gene)
                for marker in markers
                if marker.gene_normalized or marker.gene
            }
        )
        marker_set = set(marker_genes)
        top10_matches = [gene for gene in top10 if gene in marker_set]
        top30_matches = [gene for gene in top30 if gene in marker_set]

        evidence = {
            "case_id": case_id,
            "predicted_cell_type": predicted_cell_type,
            "resolved_cl_id": mapper.normalize_cell_type(predicted_cell_type),
            "metadata": metadata,
            "de_top10": top10,
            "de_top30": top30,
            "marker_gene_list": marker_genes,
            "top10_matches": top10_matches,
            "top30_matches": top30_matches,
            "rules_to_follow": [
                "Use only marker_gene_list and precomputed matches.",
                "Do not infer marker support from gene names unless those genes are in marker_gene_list.",
                "If marker_gene_list is empty, return INSUFFICIENT_DB_EVIDENCE, score 0, veto true.",
                "If top10 has >=5 matches, score 40; if top10 has 3-4 matches, score 30; if top10 has 1-2 matches, score 10.",
                "If top10 has <3 but top30 has >=5, return BORDERLINE_DELAYED_SUPPORT with score no more than 20, not 40.",
                "If no matches, score 0 and veto true.",
            ],
            "required_json_schema": {
                "decision": (
                    "ACCEPT_STRONG | ACCEPT_MODERATE | ACCEPT_WEAK | "
                    "BORDERLINE_DELAYED_SUPPORT | REJECT_NO_DATABASE_SUPPORT | "
                    "INSUFFICIENT_DB_EVIDENCE"
                ),
                "recommended_marker_score": "integer 0/10/20/30/40",
                "should_marker_veto": "boolean",
                "evidence_used": "list of genes from top10_matches/top30_matches only",
                "reasoning": "short database-only explanation",
            },
        }
        prompt = "Judge this marker evidence strictly as database-only JSON.\n"
        prompt += json.dumps(evidence, ensure_ascii=False)
        response = client.chat(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            json_mode=True,
        )

        results[case_id] = {
            "llm_response": response,
            "database_evidence": {k: v for k, v in evidence.items() if k != "marker_gene_list"},
            "n_marker_genes": len(marker_genes),
            "marker_gene_preview": marker_genes[:50],
        }
        print(
            case_id,
            f"markers={len(marker_genes)}",
            f"top10={top10_matches}",
            f"top30={top30_matches}",
            f"response={response}",
        )

    OUTPUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved={OUTPUT_PATH}")


if __name__ == "__main__":
    main()
