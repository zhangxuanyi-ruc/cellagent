"""Unified CellAgent post-SCA pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from src.agent.state import AgentRunConfig, AgentRunResult
from src.agent.tools.finalizer import AnnotationFinalizer
from src.agent.tools.judging import EvidenceJudge, write_judge_result
from src.agent.tools.reasoning import (
    DeterministicReasoner,
    case_id_from_payload,
    cell_id_from_payload,
    load_cluster_assignments,
    load_de_genes,
    load_de_summary,
    load_prior_payloads,
    prediction_from_payload,
    resolve_cluster_id,
    safe_output_stem,
    write_reasoning_result,
)
from src.core.manifest import PipelineManifest, default_manifest_path
from src.core.schemas import JudgeResult
from src.llm.clients import build_llm_client_from_config
from src.tools.rag import RAGFacade


class AgentPipeline:
    """Main tree trunk for CellAgent annotation after SCA outputs exist."""

    def __init__(self, run_config: AgentRunConfig, config: dict[str, Any]):
        self.run_config = run_config
        self.config = config
        self.manifest = PipelineManifest.read(run_config.manifest_path)
        self.rag = RAGFacade(run_config.rag_config_path)
        self.llm = build_llm_client_from_config(config) if run_config.enable_llm_judge else None
        marker_cfg = config.get("rag", {}).get("markers", {})
        marker_top_k = marker_cfg.get("top_k")
        self.reasoner = DeterministicReasoner(
            rag=self.rag,
            mapper=self.rag.mapper,
            marker_top_k=None if marker_top_k is None else int(marker_top_k),
            min_markers=int(marker_cfg.get("min_markers", 5)),
        )
        self.judge = EvidenceJudge(
            mapper=self.rag.mapper,
            rag=self.rag,
            llm=self.llm,
            config=config,
        )
        self.finalizer = AnnotationFinalizer()

    @classmethod
    def from_paths(
        cls,
        config_path: str | Path = "config/config.yaml",
        rag_config_path: str | Path = "config/rag_sources.yaml",
        manifest_path: str | Path | None = None,
        prior_json_path: str | Path | None = None,
        top_de_genes: int = 30,
        enable_llm_judge: bool = False,
    ) -> "AgentPipeline":
        config_path = Path(config_path)
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        default_output_root = Path(config.get("pipeline", {}).get("output_root") or "output")
        manifest_path = Path(manifest_path) if manifest_path else default_manifest_path(default_output_root)
        manifest = PipelineManifest.read(manifest_path)
        output_root = Path(manifest.output_root)
        prior_json = Path(
            prior_json_path
            or manifest.multimodal_prior_json
            or output_root / "multimodal_prior" / "cluster_predictions.jsonl"
        )
        run_config = AgentRunConfig(
            config_path=config_path,
            rag_config_path=Path(rag_config_path),
            manifest_path=manifest_path,
            prior_json_path=prior_json,
            output_root=output_root,
            reasoning_dir=Path(manifest.reasoning_dir or output_root / "reasoning"),
            judging_dir=output_root / "judging",
            final_dir=Path(manifest.final_dir or output_root / "final"),
            top_de_genes=top_de_genes,
            enable_llm_judge=enable_llm_judge,
        )
        return cls(run_config, config)

    def run(self) -> AgentRunResult:
        de_summary = self.manifest.de_summary_csv or str(self.run_config.output_root / "de" / "de_summary.csv")
        de_by_cluster = load_de_summary(de_summary)
        clusters_csv = self.manifest.clusters_csv
        if not clusters_csv:
            raise ValueError("manifest.clusters_csv is required to map cell-level prior outputs to clusters.")
        cluster_key = self.config.get("de_analysis", {}).get("cluster_key", "leiden")
        cell_to_cluster = load_cluster_assignments(clusters_csv, cluster_key=cluster_key)
        prior_payloads = load_prior_payloads(self.run_config.prior_json_path)
        result = AgentRunResult()
        judge_results: list[JudgeResult] = []

        for payload in prior_payloads:
            cluster_id = resolve_cluster_id(payload, cell_to_cluster)
            if cluster_id not in de_by_cluster:
                raise ValueError(
                    f"No DE csv for cluster_id={cluster_id!r}. "
                    f"Available clusters: {sorted(de_by_cluster)[:20]}"
                )
            prediction = prediction_from_payload(payload)
            metadata = payload.get("metadata") or {}
            de_genes = load_de_genes(de_by_cluster[cluster_id], top_k=self.run_config.top_de_genes)
            cell_id = cell_id_from_payload(payload)
            case_id = case_id_from_payload(payload)
            output_stem = safe_output_stem(cluster_id=cluster_id, cell_id=cell_id, case_id=case_id)
            reasoning = self.reasoner.run(
                cluster_id=cluster_id,
                prediction=prediction,
                metadata=metadata,
                de_genes=de_genes,
                provenance={
                    "prior_json": str(self.run_config.prior_json_path),
                    "de_csv": de_genes.de_csv_path,
                    "manifest": str(self.run_config.manifest_path),
                    "rag_config": str(self.run_config.rag_config_path),
                    "cell_id": cell_id,
                    "case_id": case_id,
                    "clusters_csv": str(clusters_csv),
                },
            )
            reasoning_path = write_reasoning_result(
                reasoning,
                self.run_config.reasoning_dir,
                output_stem=output_stem,
            )
            judge_result = self.judge.judge(reasoning, reasoning_path=str(reasoning_path))
            judge_result.provenance["cell_type"] = prediction.cell_type
            judge_result.provenance["cell_type_cl_id"] = reasoning.standardized.cell_type_cl_id or ""
            judge_result.provenance["cell_id"] = cell_id or ""
            judge_result.provenance["case_id"] = case_id or ""
            judge_path = write_judge_result(judge_result, self.run_config.judging_dir, output_stem=output_stem)

            result.reasoning_paths.append(reasoning_path)
            result.judge_paths.append(judge_path)
            judge_results.append(judge_result)

        final_json, final_csv = self.finalizer.write(judge_results, self.run_config.final_dir)
        result.final_json = final_json
        result.final_csv = final_csv
        self._write_run_summary(result)
        return result

    def _write_run_summary(self, result: AgentRunResult) -> None:
        path = self.run_config.output_root / "agent_run_summary.json"
        payload = {
            "reasoning_paths": [str(p) for p in result.reasoning_paths],
            "judge_paths": [str(p) for p in result.judge_paths],
            "final_json": str(result.final_json) if result.final_json else None,
            "final_csv": str(result.final_csv) if result.final_csv else None,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


__all__ = ["AgentPipeline"]
