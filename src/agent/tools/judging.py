"""Judge/scoring branch for CellAgent."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.core.schemas import EvaluatorReport, JudgeResult, ReasoningResult
from src.llm.prompts import (
    conflict_arbitration_prompt,
    function_consistency_prompt,
    tissue_consistency_prompt,
)


class EvidenceJudge:
    DEFAULT_SCORING = {
        "marker_match": 40,
        "conflict_penalty": 30,
        "function_consistency": 20,
        "tissue_consistency": 10,
    }
    DEFAULT_VETO = {
        "strong_conflict_threshold": 3,
        "min_matched_markers": 1,
        "marker_full_match_threshold": 5,
    }

    def __init__(
        self,
        mapper: Any | None = None,
        rag: Any | None = None,
        llm: Any | None = None,
        config: dict[str, Any] | None = None,
    ):
        self.mapper = mapper
        self.rag = rag
        self.llm = llm
        engine_cfg = (config or {}).get("engine", config or {})
        self.scoring = {**self.DEFAULT_SCORING, **engine_cfg.get("scoring", {})}
        self.veto = {**self.DEFAULT_VETO, **engine_cfg.get("veto", {})}

    def judge(self, reasoning: ReasoningResult, reasoning_path: str | None = None) -> JudgeResult:
        warnings = list(reasoning.warnings)
        component_reasoning: dict[str, str] = {}

        marker_score, matches, misses, marker_details = self._score_markers(reasoning)
        marker_insufficient = self._marker_evidence_insufficient(reasoning)
        conflict_score, conflicts, conflict_veto, conflict_reason = self._score_conflicts(reasoning)
        function_score, function_veto, function_reason = self._score_function(reasoning)
        tissue_score, tissue_veto, tissue_reason = self._score_tissue(reasoning)

        marker_status = self._marker_status(marker_score, matches, marker_insufficient, marker_details)
        component_reasoning["marker"] = self._marker_reason(
            matches,
            misses,
            insufficient=marker_insufficient,
            status=marker_status,
            details=marker_details,
        )
        component_reasoning["conflict"] = conflict_reason
        component_reasoning["function"] = function_reason
        component_reasoning["tissue"] = tissue_reason

        report = EvaluatorReport(
            marker_match_score=marker_score,
            conflict_penalty=conflict_score,
            function_consistency_score=function_score,
            tissue_consistency_score=tissue_score,
        )
        veto_reasons = []
        if marker_insufficient:
            warnings.append("INSUFFICIENT_DATABASE_EVIDENCE: no marker genes found for the predicted CL ID.")
        elif marker_score == 0:
            veto_reasons.append("无 marker 匹配")
        if conflict_veto:
            veto_reasons.append(conflict_reason)
        if function_veto:
            veto_reasons.append(function_reason)
        if tissue_veto:
            veto_reasons.append(tissue_reason)

        report.veto_triggered = bool(veto_reasons)
        report.veto_reason = "; ".join(r for r in veto_reasons if r)
        report.total = marker_score + conflict_score + function_score + tissue_score
        report.detailed_reasoning = json.dumps(component_reasoning, ensure_ascii=False)

        if self.llm is None:
            warnings.append("LLM judge disabled: function/tissue/conflict scores used deterministic fallback.")

        return JudgeResult(
            cluster_id=reasoning.cluster_id,
            reasoning_path=reasoning_path,
            report=report,
            marker_matches=matches,
            marker_misses=misses,
            conflict_candidates=conflicts,
            component_reasoning=component_reasoning,
            warnings=warnings,
            provenance={
                "reasoning_provenance": reasoning.provenance,
                "llm_judge_enabled": self.llm is not None,
                "marker_status": marker_status,
                "marker_scoring": marker_details,
            },
        )

    def _score_markers(self, reasoning: ReasoningResult) -> tuple[int, list[str], list[str], dict[str, Any]]:
        max_score = int(self.scoring["marker_match"])
        full_threshold = int(self.veto.get("marker_full_match_threshold", 5))
        marker_genes = {
            self._normalize_gene(r.get("gene_normalized") or r.get("gene"))
            for r in reasoning.marker_records
            if r.get("gene_normalized") or r.get("gene")
        }
        marker_genes = {g for g in marker_genes if g}
        # DE csv already stores up to config.de_analysis.top_k significant genes.
        # Marker scoring uses top10 as the primary evidence and top30 as a
        # conservative rescue branch for noisy DE ranking.
        de_top10 = reasoning.standardized.de_genes_normalized[:10]
        de_top30 = reasoning.standardized.de_genes_normalized[:30]
        matches = [g for g in de_top10 if g in marker_genes]
        misses = [g for g in de_top10 if g not in marker_genes]
        top30_matches = [g for g in de_top30 if g in marker_genes]
        details = {
            "top10_hits": matches,
            "top10_n_hits": len(matches),
            "top10_score": 0,
            "top30_hits": top30_matches,
            "top30_n_hits": len(top30_matches),
            "top30_rescue_score": 0,
            "selected_branch": "none",
        }
        if not de_top10 or not marker_genes:
            return 0, matches, misses, details

        if len(matches) >= full_threshold:
            details["top10_score"] = max_score
        elif len(matches) >= 3:
            details["top10_score"] = int(max_score * 0.75)
        elif len(matches) >= 1:
            details["top10_score"] = int(max_score * 0.25)

        if len(top30_matches) >= 10:
            details["top30_rescue_score"] = 30
        elif len(top30_matches) >= 6:
            details["top30_rescue_score"] = 20
        elif len(top30_matches) >= 3:
            details["top30_rescue_score"] = 10

        if int(details["top30_rescue_score"]) > int(details["top10_score"]):
            details["selected_branch"] = "top30_rescue"
        elif int(details["top10_score"]) > 0:
            details["selected_branch"] = "top10"
        else:
            details["selected_branch"] = "none"
        return max(int(details["top10_score"]), int(details["top30_rescue_score"])), matches, misses, details

    def _score_conflicts(self, reasoning: ReasoningResult) -> tuple[int, list[dict[str, Any]], bool, str]:
        max_score = int(self.scoring["conflict_penalty"])
        candidates = self._reverse_marker_candidates(reasoning)
        if self.llm is not None and candidates:
            prompt = conflict_arbitration_prompt(
                predicted_cell_type=reasoning.prediction.cell_type,
                predicted_cl_id=reasoning.standardized.cell_type_cl_id,
                conflict_candidates=candidates,
                top10_de_genes=reasoning.standardized.de_genes_normalized[:10],
                top30_de_genes=reasoning.standardized.de_genes_normalized[:30],
                positive_marker_hits=[
                    self._normalize_gene(r.get("gene_normalized") or r.get("gene"))
                    for r in reasoning.marker_records
                    if self._normalize_gene(r.get("gene_normalized") or r.get("gene")) in reasoning.standardized.de_genes_normalized[:30]
                ],
            )
            return self._judge_with_llm(prompt, max_score, default_score=max_score, candidates=candidates)
        if not candidates:
            return max_score, candidates, False, "未发现反向 marker 冲突候选"
        candidate_genes = sorted({str(c.get("gene")) for c in candidates if c.get("gene")})
        return (
            max_score,
            candidates,
            False,
            f"基于未命中的 top10 DE genes 发现反向 marker 候选基因 {len(candidate_genes)} 个、"
            f"候选记录 {len(candidates)} 条；LLM judge 未启用，按规划不做规则 veto",
        )

    def _score_function(self, reasoning: ReasoningResult) -> tuple[int, bool, str]:
        max_score = int(self.scoring["function_consistency"])
        if self.llm is not None:
            prompt = function_consistency_prompt(
                llm_function=reasoning.prediction.function,
                function_records=reasoning.function_records,
            )
            score, _, veto, reason = self._judge_with_llm(prompt, max_score, default_score=int(max_score * 0.5))
            return score, veto, reason
        if reasoning.function_records:
            return int(max_score * 0.5), False, "存在权威功能记录；LLM judge 未启用，使用中性占位分"
        return 0, False, "无权威功能记录；LLM judge 未启用"

    def _score_tissue(self, reasoning: ReasoningResult) -> tuple[int, bool, str]:
        max_score = int(self.scoring["tissue_consistency"])
        if self.llm is not None:
            prompt = tissue_consistency_prompt(
                metadata={
                    "species": reasoning.standardized.species,
                    "tissue": reasoning.standardized.tissue_raw,
                    "tissue_uberon_id": reasoning.standardized.tissue_uberon_id,
                },
                tissue_records=reasoning.tissue_records,
            )
            score, _, veto, reason = self._judge_with_llm(prompt, max_score, default_score=int(max_score * 0.5))
            return score, veto, reason
        if not reasoning.tissue_records:
            return 0, False, "无组织分布记录；LLM judge 未启用"
        if reasoning.standardized.tissue_uberon_id:
            known_ids = {
                r.get("tissue_uberon_id") or r.get("tissue_ontology_id")
                for r in reasoning.tissue_records
            }
            if reasoning.standardized.tissue_uberon_id in known_ids:
                return max_score, False, "metadata tissue 与 RAG 组织 ID 直接匹配"
        return int(max_score * 0.5), False, "存在组织分布记录；LLM judge 未启用，使用中性占位分"

    def _judge_with_llm(
        self,
        prompt: str,
        max_score: int,
        default_score: int,
        candidates: list[dict[str, Any]] | None = None,
    ) -> tuple[int, list[dict[str, Any]], bool, str]:
        response = self.llm.chat([{"role": "user", "content": prompt}], json_mode=True)
        if not isinstance(response, dict) or response.get("error"):
            return default_score, candidates or [], False, f"LLM judge 返回异常，使用默认分: {response}"
        score = int(max(0, min(max_score, int(response.get("score", default_score)))))
        veto = bool(response.get("veto") or response.get("veto_triggered", False))
        reason = str(response.get("reasoning") or response.get("veto_reason") or "")
        if veto and response.get("veto_reason"):
            reason = str(response["veto_reason"])
        return score, candidates or [], veto, reason

    def _reverse_marker_candidates(self, reasoning: ReasoningResult) -> list[dict[str, Any]]:
        if self.rag is None:
            return []
        predicted_cl = reasoning.standardized.cell_type_cl_id
        marker_genes = {
            self._normalize_gene(r.get("gene_normalized") or r.get("gene"))
            for r in reasoning.marker_records
            if r.get("gene_normalized") or r.get("gene")
        }
        marker_genes = {g for g in marker_genes if g}
        unmatched = [g for g in reasoning.standardized.de_genes_normalized[:10] if g not in marker_genes]
        candidates: list[dict[str, Any]] = []
        for gene in unmatched:
            for cl_id in self.rag.query_cell_types_for_gene(gene, species=reasoning.standardized.species or "human"):
                if cl_id and cl_id != predicted_cl:
                    candidates.append({"gene": gene, "candidate_cl_id": cl_id})
        return candidates

    def _normalize_gene(self, gene: Any) -> str | None:
        if gene is None:
            return None
        if self.mapper is None:
            return str(gene)
        return self.mapper.normalize_gene(str(gene))

    @staticmethod
    def _marker_status(
        marker_score: int,
        matches: list[str],
        insufficient: bool = False,
        details: dict[str, Any] | None = None,
    ) -> str:
        if insufficient:
            return "INSUFFICIENT_DATABASE_EVIDENCE"
        if details and details.get("selected_branch") == "top30_rescue":
            return "RESCUED_SUPPORT"
        if len(matches) >= 5:
            return "STRONG_SUPPORT"
        if len(matches) >= 3:
            return "MODERATE_SUPPORT"
        if len(matches) >= 1:
            return "WEAK_SUPPORT"
        return "NO_SUPPORT"

    @staticmethod
    def _marker_reason(
        matches: list[str],
        misses: list[str],
        insufficient: bool = False,
        status: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> str:
        n_used = len(matches) + len(misses)
        fallback = " (fewer than 10 significant DE genes; used all available)" if n_used < 10 else ""
        if insufficient:
            return (
                f"marker_status={status or 'INSUFFICIENT_DATABASE_EVIDENCE'}; "
                f"used_top_de_genes={n_used}{fallback}; no marker genes found for predicted CL ID"
            )
        detail_text = f"; scoring={details}" if details else ""
        return f"marker_status={status or 'UNKNOWN'}; used_top_de_genes={n_used}{fallback}; matched={matches}; missed={misses}{detail_text}"

    @staticmethod
    def _marker_evidence_insufficient(reasoning: ReasoningResult) -> bool:
        return not any(
            r.get("gene_normalized") or r.get("gene")
            for r in reasoning.marker_records
        )


def read_reasoning_result(path: str | Path) -> ReasoningResult:
    return ReasoningResult.model_validate_json(Path(path).read_text(encoding="utf-8"))


def write_judge_result(
    result: JudgeResult,
    output_dir: str | Path,
    output_stem: str | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem or f"cluster_{str(result.cluster_id).replace('/', '_')}"
    path = output_dir / f"{stem}_judge.json"
    path.write_text(json.dumps(result.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


__all__ = ["EvidenceJudge", "read_reasoning_result", "write_judge_result"]
