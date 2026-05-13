"""Deprecated legacy CellAgent REACT engine.

This module is kept for historical compatibility only. New code should use
`src.agent.pipeline.AgentPipeline` plus `src.agent.tools.*`.

Legacy notes:

基于 State Graph 设计：
  - AgentState 在节点间流转（"黑板"模式）
  - 每个节点只读取 state，修改 state，返回 state
  - Router 节点做条件分支，控制迭代流程
  - 最大迭代次数 max_iter=3 防止死循环

节点流:
  START
    -> initial_guess_node (多模态 LLM 初步推断)
    -> rag_evaluate_node  (RAG 检索 + 评估打分)
    -> router_node         (条件分支: 通过/反思)
      -> "END"    -> 输出最终结果
      -> "REFLECT" -> reflection_node (LLM 反思修正)
        -> rag_evaluate_node (第二轮验证)
        -> ...
"""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Callable, Literal

from src.llm.prompts import reflection_prompt

from .schemas import (
    AgentState,
    DEGenes,
    EvaluatorReport,
    FinalResult,
    FunctionEvidence,
    MarkerEvidence,
    Prediction,
    ReflectionRecord,
    TissueEvidence,
)

if TYPE_CHECKING:
    from src.llm.clients import LLMClient
    from src.tools.go_annotation import GOAnnotator
    from src.tools.rag import RAGFacade
    from src.tools.rag.mapper import Mapper


class CellAnnotationEngine:
    """细胞注释 REACT 引擎.

    所有可调参数从 config.yaml 读取，避免硬编码。
    """

    DEFAULT_CONFIG = {
        "engine": {
            "max_iterations": 3,
            "pass_threshold": 80,
            "scoring": {"marker_match": 40, "conflict_penalty": 30, "function_consistency": 20, "tissue_consistency": 10},
            "veto": {"strong_conflict_threshold": 3, "min_matched_markers": 1},
        },
        "rag": {"markers": {"top_k": 10, "species_priority": "human", "min_markers": 5}},
        "go_annotation": {"aspect": "BP", "top_k_per_gene": 3},
    }

    def __init__(
        self,
        llm: "LLMClient",
        rag: "RAGFacade",
        go_annotator: "GOAnnotator",
        mapper: "Mapper",
        config: dict | None = None,
    ):
        self.llm = llm
        self.rag = rag
        self.go_annotator = go_annotator
        self.mapper = mapper
        self.config = self._merge_config(config or {})

    def _merge_config(self, user_config: dict) -> dict:
        """合并用户配置与默认配置，用户配置优先."""
        import copy
        merged = copy.deepcopy(self.DEFAULT_CONFIG)
        for key, val in user_config.items():
            if isinstance(val, dict) and key in merged and isinstance(merged[key], dict):
                merged[key].update(val)
            else:
                merged[key] = val
        return merged

    @property
    def _engine_cfg(self) -> dict:
        return self.config.get("engine", self.DEFAULT_CONFIG["engine"])

    @property
    def _rag_cfg(self) -> dict:
        return self.config.get("rag", self.DEFAULT_CONFIG["rag"])

    @property
    def _go_cfg(self) -> dict:
        return self.config.get("go_annotation", self.DEFAULT_CONFIG["go_annotation"])

    # ---- 节点函数 ----

    def _initial_guess_node(self, state: AgentState) -> AgentState:
        """Node A: 初始推断.

        多模态 LLM 输出（LLaVA-Mistral + encoder feature + 困惑度置信度）。
        当前用 Mock 占位，真实接入由 src/llm/multimodal_prior.py 完成。
        """
        de_genes = state.de_genes.top_genes
        metadata = state.metadata

        # 初始推断走多模态先验，此处保留 mock 作为占位
        prediction = self._mock_initial_guess(de_genes, metadata)

        state.current_prediction = prediction
        state.iteration_count = 0
        return state

    def _rag_evaluate_node(self, state: AgentState) -> AgentState:
        """Node B: RAG 检索 + 评估.

        1. 检索 marker / tissue / function 证据
        2. 查询 DE 基因 GO 注释
        3. LLM judge 打分
        """
        prediction = state.current_prediction
        de_genes = state.de_genes.top_genes

        # 1. 标准化细胞类型
        cl_id = self.mapper.normalize_cell_type(prediction.cell_type)

        # 2. RAG 检索（参数从配置读取）
        marker_cfg = self._rag_cfg.get("markers", {})
        markers = self.rag.query_markers(
            prediction.cell_type,
            top_k=marker_cfg.get("top_k", 10),
            min_markers=marker_cfg.get("min_markers", 5),
        )
        input_species = self._get_input_species(state.metadata)
        tissues = self.rag.query_tissues(
            prediction.cell_type,
            input_species=input_species,
            metadata=state.metadata,
        )
        conflicts = self._check_conflicts(de_genes)

        # 3. DE 基因 GO 证据（核心）— 参数从配置读取
        go_aspect = self._go_cfg.get("aspect", "BP")
        go_top_k = self._go_cfg.get("top_k_per_gene", 3)
        go_summary = self.go_annotator.summarize_for_judge(
            de_genes, aspect=go_aspect, top_k_per_gene=go_top_k
        )
        state.de_gene_go_evidence = go_summary.split("\n")

        # 4. 构建 EvaluatorReport
        report = self._evaluate(
            prediction=prediction,
            de_genes=de_genes,
            markers=markers,
            tissues=tissues,
            conflicts=conflicts,
            go_summary=go_summary,
        )
        state.evidence_report = report
        return state

    def _router_node(self, state: AgentState) -> Literal["END", "REFLECT"]:
        """Node C: 路由决策.

        根据 evidence_report 和配置参数决定是结束还是反思。
        """
        report = state.evidence_report
        pass_threshold = self._engine_cfg.get("pass_threshold", 80)
        max_iter = self._engine_cfg.get("max_iterations", 3)

        if report is None:
            return "REFLECT"

        if state.iteration_count >= max_iter:
            return "END"  # 达到最大迭代次数，fallback

        if report.veto_triggered:
            return "REFLECT"

        if report.total >= pass_threshold:
            return "END"

        return "REFLECT"

    def _reflection_node(self, state: AgentState) -> AgentState:
        """Node D: 反思修正.

        LLM 读取 evidence_report，分析失败原因，生成修正后的推断。
        """
        report = state.evidence_report
        prediction = state.current_prediction

        failure_reason = (
            report.veto_reason
            if report and report.veto_triggered
            else f"总分 {report.total if report else 0} 不足 80"
        )

        # 记录反思历史（先创建记录，再调用 LLM 填充 revised）
        record = ReflectionRecord(
            round_num=state.iteration_count + 1,
            failed_prediction=prediction.cell_type,
            failure_reason=failure_reason,
            revised_prediction="",
        )

        # 构建 evidence summary
        evidence_summary = ""
        if report:
            evidence_summary = (
                f"marker_match_score={report.marker_match_score}, "
                f"conflict_penalty={report.conflict_penalty}, "
                f"function_consistency={report.function_consistency_score}, "
                f"tissue_consistency={report.tissue_consistency_score}"
            )

        # 调用 LLM 生成修正后的推断
        prompt = reflection_prompt(
            failed_prediction=prediction.cell_type,
            failure_reason=failure_reason,
            de_genes=state.de_genes.top_genes,
            evidence_summary=evidence_summary,
            history=[f"R{r.round_num}: {r.failed_prediction} -> {r.revised_prediction} ({r.failure_reason})" for r in state.reflection_history],
        )

        try:
            response = self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                json_mode=True,
            )
            if isinstance(response, dict) and "cell_type" in response:
                revised = Prediction(
                    cell_type=str(response.get("cell_type", "unknown")),
                    function=str(response.get("function", "unknown")),
                    confidence=float(response.get("confidence", 0.5)),
                )
                record.revised_prediction = revised.cell_type
                record.focus = revised.cell_type
            else:
                # LLM 返回格式异常，fallback 到 mock
                revised = self._mock_reflection(prediction, report, state.reflection_history)
                record.revised_prediction = revised.cell_type
                record.focus = revised.cell_type
        except Exception as exc:
            # LLM 调用失败，fallback 到 mock
            revised = self._mock_reflection(prediction, report, state.reflection_history)
            record.revised_prediction = revised.cell_type
            record.focus = revised.cell_type
            record.failure_reason += f" [LLM error: {exc}]"

        state.reflection_history.append(record)
        state.current_prediction = revised
        state.iteration_count += 1
        state.evidence_report = None  # 清空，下一轮重新评估
        return state

    def _output_node(self, state: AgentState) -> AgentState:
        """最终输出节点.

        将 state 转换为 FinalResult 并落盘。
        """
        report = state.evidence_report
        prediction = state.current_prediction
        pass_threshold = self._engine_cfg.get("pass_threshold", 80)
        max_iter = self._engine_cfg.get("max_iterations", 3)

        if report and report.total >= pass_threshold and not report.veto_triggered:
            state.status = "success"
        elif state.iteration_count >= max_iter:
            state.status = "fallback"
        else:
            state.status = "failed"

        # 构建 FinalResult
        final = FinalResult(
            cluster_id=state.cluster_id,
            cell_type_cl_id=self.mapper.normalize_cell_type(prediction.cell_type) or "",
            cell_type_name=prediction.cell_type,
            function_description=prediction.function,
            overall_score=report.total if report else 0,
            reflection_summary=self._summarize_reflection(state.reflection_history),
            iteration_trace=[
                f"R{i+1}: {r.failed_prediction} -> {r.revised_prediction} ({r.failure_reason})"
                for i, r in enumerate(state.reflection_history)
            ],
        )
        state.final_output = final.model_dump()
        return state

    # ---- 图编译与执行 ----

    def compile(self) -> Callable[[AgentState], AgentState]:
        """编译状态图为可执行函数.

        返回的函数接收初始 state，执行完整 REACT 流程，返回最终 state。
        """

        def _execute(state: AgentState) -> AgentState:
            # Step 1: 初始推断
            state = self._initial_guess_node(state)

            # Step 2-4: 评估-路由-反思循环
            while True:
                state = self._rag_evaluate_node(state)
                decision = self._router_node(state)

                if decision == "END":
                    break

                state = self._reflection_node(state)

            # Step 5: 输出
            state = self._output_node(state)
            return state

        return _execute

    def invoke(self, state: AgentState) -> AgentState:
        """执行完整注释流程."""
        workflow = self.compile()
        return workflow(state)

    # ---- 评估逻辑 ----

    def _evaluate(
        self,
        prediction: Prediction,
        de_genes: list[str],
        markers: list,
        tissues: list,
        conflicts: list[str],
        go_summary: str,
    ) -> EvaluatorReport:
        """综合评估，生成 EvaluatorReport.

        TODO: 当前用规则打分，后续应接入 LLM judge。
        """
        scoring = self._engine_cfg.get("scoring", {})
        veto_cfg = self._engine_cfg.get("veto", {})
        max_marker = scoring.get("marker_match", 40)
        max_conflict = scoring.get("conflict_penalty", 30)
        max_function = scoring.get("function_consistency", 20)
        max_tissue = scoring.get("tissue_consistency", 10)
        strong_conflict_threshold = veto_cfg.get("strong_conflict_threshold", 3)
        min_matched = veto_cfg.get("min_matched_markers", 1)

        report = EvaluatorReport()

        # 1. marker 匹配 (0 - max_marker)
        marker_genes = {
            g for g in (self.mapper.normalize_gene(getattr(m, "gene", "")) for m in markers)
            if g
        }
        de_set = {g for g in (self.mapper.normalize_gene(gene) for gene in de_genes) if g}
        matched = de_set & marker_genes
        if len(de_set) == 0:
            report.marker_match_score = 0
            report.veto_triggered = True
            report.veto_reason = "无 DE 基因"
        elif len(matched) == len(de_set):
            report.marker_match_score = max_marker
        elif len(matched) >= 3:
            report.marker_match_score = int(max_marker * 0.75)
        elif len(matched) >= min_matched:
            report.marker_match_score = int(max_marker * 0.25)
        else:
            report.marker_match_score = 0
            report.veto_triggered = True
            report.veto_reason = "无 marker 匹配"

        # 2. 反向 marker 冲突 (0 - max_conflict)
        if conflicts:
            if len(conflicts) >= strong_conflict_threshold:
                report.conflict_penalty = 0
                report.veto_triggered = True
                report.veto_reason = f"强排他性冲突: {conflicts}"
            elif len(conflicts) >= 1:
                report.conflict_penalty = int(max_conflict * 0.33)
            else:
                report.conflict_penalty = max_conflict
        else:
            report.conflict_penalty = max_conflict

        # 3. 功能一致性 (0 - max_function) — 基于 DE 基因 GO 证据
        # TODO: 接入 LLM judge
        # 临时规则: 如果有 DE 基因 GO 证据，给 75% 分
        report.function_consistency_score = int(max_function * 0.75) if go_summary else int(max_function * 0.5)

        # 4. 组织分布 (0 - max_tissue)
        report.tissue_consistency_score = max_tissue if tissues else int(max_tissue * 0.5)

        report.total = (
            report.marker_match_score
            + report.conflict_penalty
            + report.function_consistency_score
            + report.tissue_consistency_score
        )

        return report

    def _check_conflicts(self, de_genes: list[str]) -> list[str]:
        """检查 DE 基因中是否存在排他性冲突 marker.

        对每个 DE 基因，反向查询它强烈指向哪些细胞类型。
        如果指向的细胞类型与当前假设不同，记录为冲突。
        """
        conflicts: list[str] = []
        for gene in de_genes:
            cell_types = self.rag.query_cell_types_for_gene(gene)
            # TODO: 更精细的冲突判断（需要当前假设的 CL ID 对比）
            if cell_types:
                conflicts.append(f"{gene} -> {cell_types}")
        return conflicts

    @staticmethod
    def _get_input_species(metadata: dict) -> str | None:
        """预处理器接入前兼容多种可能字段名，避免误启用 ImmGen."""
        for key in ("input_species", "species_detected", "detected_species", "gene_species"):
            value = metadata.get(key)
            if value:
                return str(value)
        return None

    # ---- Mock 辅助 ----

    def _mock_initial_guess(
        self, de_genes: list[str], metadata: dict
    ) -> Prediction:
        """Mock 初始推断（仅用于开发测试）."""
        # 简单启发: 根据 DE 基因中的 marker 做推断
        if "FOXP3" in de_genes:
            return Prediction(cell_type="regulatory T cell", function="免疫调节", confidence=0.7)
        if "CD3D" in de_genes and "CD4" in de_genes:
            return Prediction(cell_type="CD4-positive T cell", function="适应性免疫", confidence=0.6)
        if "CD3D" in de_genes and "CD8A" in de_genes:
            return Prediction(cell_type="CD8-positive T cell", function="细胞毒性", confidence=0.6)
        return Prediction(cell_type="unknown", function="unknown", confidence=0.0)

    def _mock_reflection(
        self,
        prediction: Prediction,
        report: EvaluatorReport | None,
        history: list[ReflectionRecord],
    ) -> Prediction:
        """Mock 反思修正（仅用于开发测试）."""
        # 简单启发: 如果上一轮是 CD4+ T cell 且被否决，尝试 Treg
        if "CD4" in prediction.cell_type and report and report.veto_triggered:
            return Prediction(cell_type="regulatory T cell", function="免疫调节", confidence=0.8)
        # 默认 fallback: 返回更宽泛的类型
        return Prediction(cell_type="T cell", function="免疫应答", confidence=0.5)

    def _summarize_reflection(self, history: list[ReflectionRecord]) -> str:
        """生成反思摘要."""
        if not history:
            return "无反思记录"
        parts = []
        for r in history:
            parts.append(f"R{r.round_num}: {r.failed_prediction} 因 {r.failure_reason} 被修正为 {r.revised_prediction}")
        return "; ".join(parts)
