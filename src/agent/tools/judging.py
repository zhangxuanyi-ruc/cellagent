"""Judge/scoring branch for CellAgent."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.core.schemas import EvaluatorReport, JudgeResult, ReasoningResult
from src.llm.prompts import (
    conflict_arbitration_prompt,
    function_consistency_prompt,
    tissue_consistency_prompt,
)
from src.agent.tools.marker_registry import MarkerRegistry, OfflineLLMMarkerRegistry
from src.agent.tools.target_cluster_concordance import TargetClusterConcordanceScorer


class EvidenceJudge:
    DEFAULT_SCORING = {
        "marker_match": 40,
        "target_cluster_consistency": 30,
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
        expression_h5ad_path: str | Path | None = None,
        clusters_csv: str | Path | None = None,
        cluster_key: str = "leiden",
        expression_layer: str | None = None,
    ):
        self.mapper = mapper
        self.rag = rag
        self.llm = llm
        self.config = config or {}
        engine_cfg = self.config.get("engine", self.config)
        self.scoring = {**self.DEFAULT_SCORING, **engine_cfg.get("scoring", {})}
        self.veto = {**self.DEFAULT_VETO, **engine_cfg.get("veto", {})}
        self.reverse_marker_cfg = self.config.get("reverse_marker_monitor", {})
        registry_cfg = self.config.get("marker_registry", {})
        self.marker_registry = (
            MarkerRegistry(registry_cfg.get("path"))
            if bool(registry_cfg.get("enabled", True))
            else MarkerRegistry("__disabled_marker_registry__.json")
        )
        offline_registry_cfg = self.config.get("offline_llm_marker_registry", {})
        self.offline_llm_marker_registry = (
            OfflineLLMMarkerRegistry(offline_registry_cfg.get("path"))
            if bool(offline_registry_cfg.get("enabled", False))
            else OfflineLLMMarkerRegistry("__disabled_offline_llm_marker_registry__.json")
        )
        self.target_cluster_scorer = TargetClusterConcordanceScorer(
            h5ad_path=expression_h5ad_path,
            clusters_csv=clusters_csv,
            cluster_key=cluster_key,
            expression_layer=expression_layer,
        )

    def judge(self, reasoning: ReasoningResult, reasoning_path: str | None = None) -> JudgeResult:
        warnings = list(reasoning.warnings)
        component_reasoning: dict[str, str] = {}

        marker_score, matches, misses, marker_details = self._score_markers(reasoning)
        marker_insufficient = self._marker_evidence_insufficient(reasoning, marker_details)
        target_cluster_score, target_cluster_report, target_cluster_veto, target_cluster_reason = (
            self._score_target_cluster_consistency(reasoning)
        )
        reverse_marker_monitor = self._monitor_reverse_markers(reasoning)
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
        component_reasoning["target_cluster_consistency"] = target_cluster_reason
        component_reasoning["reverse_marker_monitor"] = json.dumps(reverse_marker_monitor, ensure_ascii=False)
        component_reasoning["function"] = function_reason
        component_reasoning["tissue"] = tissue_reason

        report = EvaluatorReport(
            marker_match_score=marker_score,
            target_cluster_consistency_score=target_cluster_score,
            conflict_penalty=0,
            function_consistency_score=function_score,
            tissue_consistency_score=tissue_score,
        )
        veto_reasons = []
        if marker_insufficient:
            warnings.append("INSUFFICIENT_DATABASE_EVIDENCE: no marker genes found for the predicted CL ID.")
        elif self._marker_veto_required(marker_score, marker_details) and bool(self.veto.get("marker_zero_veto", True)):
            veto_reasons.append("无 marker 匹配")
        if target_cluster_veto:
            veto_reasons.append(target_cluster_reason)
        if function_veto:
            veto_reasons.append(function_reason)
        if tissue_veto:
            veto_reasons.append(tissue_reason)

        report.veto_triggered = bool(veto_reasons)
        report.veto_reason = "; ".join(r for r in veto_reasons if r)
        report.total = marker_score + target_cluster_score + function_score + tissue_score
        report.detailed_reasoning = json.dumps(component_reasoning, ensure_ascii=False)

        if self.llm is None:
            warnings.append("LLM judge disabled: function/tissue scores used deterministic fallback.")

        return JudgeResult(
            cluster_id=reasoning.cluster_id,
            reasoning_path=reasoning_path,
            report=report,
            marker_matches=matches,
            marker_misses=misses,
            target_cluster_report=target_cluster_report,
            reverse_marker_monitor=reverse_marker_monitor,
            conflict_candidates=reverse_marker_monitor.get("primary_candidates", []),
            component_reasoning=component_reasoning,
            warnings=warnings,
            provenance={
                "reasoning_provenance": reasoning.provenance,
                "llm_judge_enabled": self.llm is not None,
                "marker_status": marker_status,
                "marker_scoring": marker_details,
                "reverse_marker_monitor_used_for_score": False,
                "reverse_marker_monitor_used_for_veto": False,
            },
        )

    def _score_markers(self, reasoning: ReasoningResult) -> tuple[int, list[str], list[str], dict[str, Any]]:
        max_score = int(self.scoring["marker_match"])
        full_threshold = int(self.veto.get("marker_full_match_threshold", 5))
        marker_genes_full, marker_selection = self._marker_genes_for_scoring(reasoning)
        marker_genes = set(marker_selection.get("selected_genes") or marker_genes_full)
        # DE csv already stores up to config.de_analysis.top_k significant genes.
        # Marker scoring keeps top10 as the primary evidence and uses top30 as
        # a rescue branch. The selected branch is stored for reflection.
        de_top10 = reasoning.standardized.de_genes_normalized[:10]
        de_top30 = reasoning.standardized.de_genes_normalized[:30]
        matches = [g for g in de_top10 if g in marker_genes]
        misses = [g for g in de_top10 if g not in marker_genes]
        top30_matches = [g for g in de_top30 if g in marker_genes]
        details = {
            "n_marker_genes_full": len(marker_genes_full),
            "n_marker_genes_used": len(marker_genes),
            "marker_source": marker_selection.get("marker_source"),
            "fallback_used": bool(marker_selection.get("fallback_used", False)),
            "marker_selection": marker_selection,
            "top10_hits": matches,
            "top10_n_hits": len(matches),
            "top10_score": 0,
            "top30_hits": top30_matches,
            "top30_n_hits": len(top30_matches),
            "top30_rescue_score": 0,
            "selected_branch": "none",
            "selected_score": 0,
            "veto_eligible": bool(marker_selection.get("veto_eligible", True)),
        }
        if not de_top10 or not marker_genes:
            return 0, matches, misses, details

        if len(matches) >= full_threshold:
            details["top10_score"] = max_score
        elif len(matches) >= 3:
            details["top10_score"] = int(max_score * 0.75)
        elif len(matches) >= 1:
            details["top10_score"] = int(max_score * 0.25)

        if len(top30_matches) >= 5:
            details["top30_rescue_score"] = 30
        elif len(top30_matches) >= 3:
            details["top30_rescue_score"] = 15

        if int(details["top30_rescue_score"]) > int(details["top10_score"]):
            details["selected_branch"] = "top30_rescue"
        elif int(details["top10_score"]) > 0:
            details["selected_branch"] = "top10"
        else:
            details["selected_branch"] = "none"
        selected_score = max(int(details["top10_score"]), int(details["top30_rescue_score"]))
        if marker_selection.get("marker_source") == "offline_llm_curated_registry":
            selected_score = min(selected_score, int(marker_selection.get("llm_marker_score_cap", 30)))
        details["selected_score"] = selected_score
        return int(details["selected_score"]), matches, misses, details

    def _marker_genes_for_scoring(self, reasoning: ReasoningResult) -> tuple[set[str], dict[str, Any]]:
        registry_hit = self._query_marker_registry(reasoning)
        if registry_hit:
            markers = {self._normalize_gene(g) for g in registry_hit["markers"]}
            markers = {g for g in markers if g}
            return markers, {
                "marker_source": "sctype_registry",
                "registry_version": registry_hit.get("version"),
                "registry_path": registry_hit.get("registry_path"),
                "registry_query_key": registry_hit.get("query_key"),
                "registry_matched_cell_type": registry_hit.get("matched_cell_type"),
                "registry_sources": registry_hit.get("source") or [],
                "strategy": "registry_core_positive",
                "selected_genes": sorted(markers),
                "fallback_used": False,
                "veto_eligible": True,
                "llm_marker_branch_used": False,
                "llm_marker_branch_reason": "authoritative_marker_evidence_sufficient",
            }

        marker_genes_full = {
            self._normalize_gene(r.get("gene_normalized") or r.get("gene"))
            for r in reasoning.marker_records
            if r.get("gene_normalized") or r.get("gene")
        }
        marker_genes_full = {g for g in marker_genes_full if g}
        marker_genes, marker_selection = self._select_marker_genes_for_scoring(
            marker_genes_full,
            species=reasoning.standardized.species or "human",
            cell_type=reasoning.standardized.cell_type_raw or reasoning.prediction.cell_type,
        )
        marker_selection.update(
            {
                "marker_source": "rag_fallback",
                "fallback_used": True,
                "fallback_reason": "cell type not found in ScType registry",
                "veto_eligible": True,
                "llm_marker_branch_used": False,
                "llm_marker_branch_reason": "authoritative_marker_evidence_sufficient",
            }
        )
        rag_selection = {
            **marker_selection,
            "selected_genes": sorted(marker_genes),
        }
        offline_reason = self._offline_llm_marker_branch_reason(marker_genes_full, rag_selection)
        if offline_reason:
            offline_hit = self._query_offline_llm_marker_registry(reasoning)
            if offline_hit:
                offline_markers = {
                    self._normalize_gene(g)
                    for g in offline_hit["markers"]
                    if self._normalize_gene(g)
                }
                return offline_markers, {
                    "marker_source": "offline_llm_curated_registry",
                    "registry_version": offline_hit.get("version"),
                    "registry_path": offline_hit.get("registry_path"),
                    "registry_query_key": offline_hit.get("query_key"),
                    "registry_matched_cell_type": offline_hit.get("matched_cell_type"),
                    "registry_sources": offline_hit.get("source") or [],
                    "strategy": "offline_llm_curated_positive",
                    "selected_genes": sorted(offline_markers),
                    "fallback_used": True,
                    "fallback_reason": offline_reason,
                    "llm_marker_branch_used": True,
                    "llm_marker_branch_reason": offline_reason,
                    "llm_marker_score_cap": 30,
                    "llm_marker_used_for_veto": False,
                    "authoritative_marker_source": "rag_fallback",
                    "authoritative_marker_count": len(marker_genes_full),
                    "authoritative_marker_selection": rag_selection,
                    "veto_eligible": False,
                }
            rag_selection.update(
                {
                    "llm_marker_branch_used": False,
                    "llm_marker_branch_reason": f"{offline_reason}; offline registry miss",
                }
            )
        return marker_genes_full, rag_selection

    def _query_marker_registry(self, reasoning: ReasoningResult) -> dict[str, Any] | None:
        if not self.marker_registry.enabled:
            return None
        return self.marker_registry.query(self._marker_registry_query_candidates(reasoning))

    def _query_offline_llm_marker_registry(self, reasoning: ReasoningResult) -> dict[str, Any] | None:
        if not self.offline_llm_marker_registry.enabled:
            return None
        candidates = self._marker_registry_query_candidates(reasoning)
        return self.offline_llm_marker_registry.query(candidates)

    def _marker_registry_query_candidates(self, reasoning: ReasoningResult) -> list[str | None]:
        candidates: list[str | None] = [reasoning.prediction.cell_type]
        cl_id = reasoning.standardized.cell_type_cl_id
        if cl_id and self.rag is not None:
            try:
                candidates.append(self.rag.get_cell_type_name(cl_id))
            except Exception:
                pass
        if cl_id and self.mapper is not None and hasattr(self.mapper, "cell_type_synonyms"):
            try:
                candidates.extend(self.mapper.cell_type_synonyms(cl_id))
            except Exception:
                pass
        return candidates

    def _offline_llm_marker_branch_reason(
        self,
        marker_genes_full: set[str],
        marker_selection: dict[str, Any],
    ) -> str | None:
        cfg = self.config.get("offline_llm_marker_registry", {})
        if not bool(cfg.get("enabled", False)):
            return None
        min_markers = int(cfg.get("min_authoritative_markers", 5))
        if not marker_genes_full:
            return "INSUFFICIENT_DATABASE_EVIDENCE"
        if len(marker_genes_full) < min_markers:
            return f"RAG marker count < min_authoritative_markers ({len(marker_genes_full)} < {min_markers})"
        if bool(cfg.get("enable_for_low_quality_sampled_evidence", True)) and marker_selection.get("strategy") == "stable_hash_sample":
            return "low_quality_sampled_evidence"
        return None

    def _select_marker_genes_for_scoring(
        self,
        marker_genes: set[str],
        species: str = "human",
        cell_type: str | None = None,
    ) -> tuple[set[str], dict[str, Any]]:
        cfg = self.config.get("marker_scoring", {})
        max_marker_genes = int(cfg.get("max_marker_genes", 80))
        sample_seed = str(cfg.get("sample_seed", "cellagent_marker_v1"))
        if max_marker_genes <= 0 or len(marker_genes) <= max_marker_genes:
            return marker_genes, {
                "strategy": "all_markers",
                "max_marker_genes": max_marker_genes,
            }
        ranked = sorted(
            marker_genes,
            key=lambda gene: self._stable_marker_sample_key(
                gene=gene,
                cell_type=cell_type or "",
                sample_seed=sample_seed,
            ),
        )
        selected = set(ranked[:max_marker_genes])
        return selected, {
            "strategy": "stable_hash_sample",
            "max_marker_genes": max_marker_genes,
            "sample_seed": sample_seed,
            "sample_key": str(cell_type or ""),
            "selected_genes": ranked[:max_marker_genes],
            "excluded_gene_count": len(marker_genes) - max_marker_genes,
        }

    @staticmethod
    def _stable_marker_sample_key(gene: str, cell_type: str, sample_seed: str) -> str:
        text = f"{sample_seed}|{cell_type.strip().lower()}|{gene.strip().upper()}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _marker_veto_required(marker_score: int, marker_details: dict[str, Any]) -> bool:
        if marker_details.get("veto_eligible") is False:
            return False
        if marker_score > 0:
            return False
        return int(marker_details.get("top10_n_hits") or 0) == 0 and int(marker_details.get("top30_n_hits") or 0) == 0

    def _score_target_cluster_consistency(
        self,
        reasoning: ReasoningResult,
    ) -> tuple[int, dict[str, Any], bool, str]:
        max_score = int(self.scoring["target_cluster_consistency"])
        report = self.target_cluster_scorer.score(
            target_cell_id=reasoning.target_cell_id,
            cluster_id=reasoning.cluster_id,
            top_de_genes=reasoning.standardized.de_genes_normalized[
                : int((self.config or {}).get("target_cluster_consistency", {}).get("top_n_de_genes", 10))
            ],
        )
        raw_score = int(report.get("score", 0))
        score = int(max(0, min(max_score, raw_score)))
        veto_threshold = int(self.veto.get("target_cluster_veto_score_threshold", 10))
        veto = score <= veto_threshold and bool(self.veto.get("target_cluster_low_score_veto", True))
        if veto:
            report["veto"] = True
            report["veto_reason"] = (
                f"target-cluster consistency score <= {veto_threshold}: "
                f"presence_score={report.get('presence_score', 0)}, "
                f"signature_score={report.get('signature_score', 0)}"
            )
        reason = (
            str(report.get("veto_reason"))
            if veto
            else (
                "target-cluster consistency: "
                f"presence_score={report.get('presence_score', 0)}, "
                f"signature_score={report.get('signature_score', 0)}, "
                f"signature_percentile={report.get('signature_percentile', 0.0)}"
            )
        )
        return score, report, veto, reason

    def _monitor_reverse_markers(self, reasoning: ReasoningResult) -> dict[str, Any]:
        enabled = bool(self.reverse_marker_cfg.get("enabled", True))
        if not enabled:
            return {
                "enabled": False,
                "used_for_score": False,
                "used_for_veto": False,
                "primary_candidates": [],
                "screen_candidates": [],
                "n_primary_candidates": 0,
                "n_screen_candidates": 0,
            }
        primary = self._reverse_marker_candidates(reasoning)
        screen = self._exclusive_veto_screen_candidates(reasoning)
        return {
            "enabled": True,
            "used_for_score": False,
            "used_for_veto": False,
            "primary_candidates": primary,
            "screen_candidates": screen,
            "n_primary_candidates": len(primary),
            "n_screen_candidates": len(screen),
        }

    def _score_conflicts(self, reasoning: ReasoningResult) -> tuple[int, list[dict[str, Any]], bool, str]:
        max_score = int(self.scoring.get("conflict_penalty", 30))
        candidates = self._reverse_marker_candidates(reasoning)
        exclusive_screen_candidates = self._exclusive_veto_screen_candidates(reasoning)
        all_candidates = self._merge_conflict_candidates(candidates, exclusive_screen_candidates)
        if self.llm is not None and all_candidates:
            prompt = conflict_arbitration_prompt(
                predicted_cell_type=reasoning.prediction.cell_type,
                predicted_cl_id=reasoning.standardized.cell_type_cl_id,
                conflict_candidates=candidates,
                exclusive_veto_screen_candidates=exclusive_screen_candidates,
                top10_de_genes=reasoning.standardized.de_genes_normalized[:10],
                top30_de_genes=reasoning.standardized.de_genes_normalized[:30],
                positive_marker_hits=self._positive_marker_hits(reasoning, top_n=30),
            )
            score, judged_candidates, veto, reason = self._judge_with_llm(
                prompt,
                max_score,
                default_score=max_score,
                candidates=all_candidates,
            )
            return self._normalize_conflict_score(score, veto), judged_candidates, veto, reason
        if not all_candidates:
            return max_score, candidates, False, "未发现反向 marker 冲突候选"
        candidate_genes = sorted({str(c.get("gene")) for c in all_candidates if c.get("gene")})
        return (
            max_score,
            all_candidates,
            False,
            f"基于未命中的 top10/top30 DE genes 发现反向 marker 候选基因 {len(candidate_genes)} 个、"
            f"候选记录 {len(all_candidates)} 条；LLM judge 未启用，按规划不做规则 veto",
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

    def _positive_marker_hits(self, reasoning: ReasoningResult, top_n: int = 30) -> list[str]:
        marker_genes = self._predicted_marker_genes(reasoning)
        de_genes = reasoning.standardized.de_genes_normalized[:top_n]
        return [gene for gene in de_genes if gene in marker_genes]

    def _predicted_marker_genes(self, reasoning: ReasoningResult) -> set[str]:
        marker_genes = {
            self._normalize_gene(r.get("gene_normalized") or r.get("gene"))
            for r in reasoning.marker_records
            if r.get("gene_normalized") or r.get("gene")
        }
        return {g for g in marker_genes if g}

    def _reverse_marker_candidates(self, reasoning: ReasoningResult) -> list[dict[str, Any]]:
        """Primary reverse-marker candidates from top10 unmatched DE genes."""
        top_n = int(self.reverse_marker_cfg.get("top_n_primary", 10))
        return self._reverse_marker_candidates_for_top_n(reasoning, top_n=top_n, evidence_scope="top10_primary")

    def _exclusive_veto_screen_candidates(self, reasoning: ReasoningResult) -> list[dict[str, Any]]:
        """Auxiliary top30 screen, used only by the LLM for high-specificity veto context."""
        top_n = int(self.reverse_marker_cfg.get("top_n_screen", 30))
        return self._reverse_marker_candidates_for_top_n(reasoning, top_n=top_n, evidence_scope="top30_exclusive_veto_screen")

    def _reverse_marker_candidates_for_top_n(
        self,
        reasoning: ReasoningResult,
        top_n: int,
        evidence_scope: str,
    ) -> list[dict[str, Any]]:
        if self.rag is None:
            return []
        predicted_cl = reasoning.standardized.cell_type_cl_id
        marker_genes = self._predicted_marker_genes(reasoning)
        unmatched = [g for g in reasoning.standardized.de_genes_normalized[:top_n] if g not in marker_genes]
        candidates: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for gene in unmatched:
            for cl_id in self.rag.query_cell_types_for_gene(gene, species=reasoning.standardized.species or "human"):
                if cl_id and cl_id != predicted_cl:
                    key = (gene, cl_id)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(
                        {
                            "gene": gene,
                            "candidate_cl_id": cl_id,
                            "candidate_cell_type_name": self._candidate_cell_type_name(cl_id),
                            "evidence_scope": evidence_scope,
                        }
                    )
        return candidates

    def _candidate_cell_type_name(self, cl_id: str) -> str | None:
        if self.rag is None or not hasattr(self.rag, "get_cell_type_name"):
            return None
        try:
            return self.rag.get_cell_type_name(cl_id)
        except Exception:
            return None

    @staticmethod
    def _merge_conflict_candidates(
        primary: list[dict[str, Any]],
        screen: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for candidate in [*primary, *screen]:
            key = (
                str(candidate.get("gene") or ""),
                str(candidate.get("candidate_cl_id") or ""),
                str(candidate.get("evidence_scope") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(candidate)
        return merged

    @staticmethod
    def _normalize_conflict_score(score: int, veto: bool) -> int:
        """Keep reverse-marker scores on the planned 30/15/0 scale."""
        if veto:
            return 0
        if score <= 7:
            return 0
        if score < 23:
            return 15
        return 30

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
    def _marker_evidence_insufficient(
        reasoning: ReasoningResult,
        marker_details: dict[str, Any] | None = None,
    ) -> bool:
        if marker_details is not None:
            return int(marker_details.get("n_marker_genes_used") or 0) == 0
        return not any(r.get("gene_normalized") or r.get("gene") for r in reasoning.marker_records)


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
