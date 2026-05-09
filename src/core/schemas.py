"""CellAgent 核心数据模型.

所有在状态机节点间传递的状态结构定义.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---- 基础模型 ----

class Prediction(BaseModel):
    """多模态 LLM 的初步推断结果."""
    cell_type: str                      # 细胞类型文本描述
    cell_type_cl_id: str | None = None  # 标准化后的 CL ID (如有)
    function: str                       # 功能描述文本
    confidence: float = 0.0             # 置信度 (0-1)


class DEGenes(BaseModel):
    """差异分析结果."""
    top_genes: list[str] = Field(default_factory=list)  # top-k DE gene symbols
    de_csv_path: str | None = None                       # DE 结果 CSV 文件路径


# ---- Evidence 模型 ----

class MarkerEvidence(BaseModel):
    """Marker 基因匹配证据."""
    cell_type_cl_id: str
    supporting_markers: list[str] = Field(default_factory=list)
    conflicting_markers: list[str] = Field(default_factory=list)
    match_ratio: float = 0.0            # 匹配基因数 / DE 基因数
    sources: list[str] = Field(default_factory=list)


class FunctionEvidence(BaseModel):
    """功能一致性证据."""
    llm_function: str                   # LLM 原始功能描述
    de_gene_go_terms: list[str] = Field(default_factory=list)  # DE 基因关联 GO
    consistency_score: int = 0          # 0-20
    reasoning: str = ""


class TissueEvidence(BaseModel):
    """组织分布证据."""
    claimed_tissue: str | None = None
    rag_tissues: list[dict] = Field(default_factory=list)
    consistency_score: int = 0          # 0-10


class EvaluatorReport(BaseModel):
    """校验专家输出的评估报告."""
    marker_match_score: int = 0         # 0-40
    conflict_penalty: int = 0           # 0-30
    function_consistency_score: int = 0 # 0-20
    tissue_consistency_score: int = 0   # 0-10
    total: int = 0
    veto_triggered: bool = False
    veto_reason: str = ""
    detailed_reasoning: str = ""


class ReflectionRecord(BaseModel):
    """单轮反思记录."""
    round_num: int
    failed_prediction: str              # 上一轮失败的细胞类型
    failure_reason: str                 # 失败原因分析
    revised_prediction: str             # 修正后的细胞类型
    focus: str = ""                     # 下一轮验证焦点


# ---- 全局状态 ----

class AgentState(BaseModel):
    """在状态图节点间流转的全局状态.

    就像一块"黑板"，每个节点读取、修改、返回。
    """
    # 输入信息
    cluster_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    de_genes: DEGenes = Field(default_factory=DEGenes)

    # 当前推断 (随迭代更新)
    current_prediction: Prediction = Field(default_factory=lambda: Prediction(cell_type="", function=""))

    # 证据与评估 (由 RAG_and_Evaluate_Node 填写)
    evidence_report: EvaluatorReport | None = None
    de_gene_go_evidence: list[str] = Field(default_factory=list)

    # 迭代控制
    reflection_history: list[ReflectionRecord] = Field(default_factory=list)
    iteration_count: int = 0
    max_iterations: int = 3

    # 流程状态
    status: Literal["running", "success", "fallback", "failed"] = "running"
    final_output: dict[str, Any] | None = None

    def should_continue(self) -> bool:
        """判断是否应继续迭代."""
        return self.iteration_count < self.max_iterations and self.status == "running"


# ---- 最终输出 ----

class FinalResult(BaseModel):
    """最终输出结果卡."""
    cluster_id: str
    cell_type_cl_id: str
    cell_type_name: str
    core_markers: list[str] = Field(default_factory=list)
    marker_match_quality: str = ""      # "full"/"partial"/"poor"
    function_description: str = ""
    tissue_distribution: list[str] = Field(default_factory=list)
    overall_score: int = 0
    reflection_summary: str = ""
    iteration_trace: list[str] = Field(default_factory=list)


__all__ = [
    "Prediction",
    "DEGenes",
    "MarkerEvidence",
    "FunctionEvidence",
    "TissueEvidence",
    "EvaluatorReport",
    "ReflectionRecord",
    "AgentState",
    "FinalResult",
]
