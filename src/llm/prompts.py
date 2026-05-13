"""CellAgent LLM Prompts.

All prompts are plain strings for easy editing and version control.
"""
from __future__ import annotations


def reflection_prompt(
    failed_prediction: str,
    failure_reason: str,
    de_genes: list[str],
    evidence_summary: str,
    history: list[str],
) -> str:
    """Prompt for the reflection node: analyze failure and propose revised prediction."""
    history_text = "\n".join(f"  - {h}" for h in history) if history else "  (none)"
    return (
        "You are a bioinformatics expert reviewing a cell-type annotation hypothesis.\n\n"
        f"The previous hypothesis was: '{failed_prediction}'\n"
        f"It failed because: {failure_reason}\n\n"
        f"Top DE genes from the cluster: {', '.join(de_genes)}\n\n"
        f"Evidence summary:\n{evidence_summary}\n\n"
        f"Previous reflection history:\n{history_text}\n\n"
        "Propose a revised cell-type hypothesis and explain your reasoning.\n"
        "Return ONLY valid JSON with exactly these keys:\n"
        '{"cell_type": "<precise cell type>", "function": "<biological function>", "confidence": <0.0-1.0>, "reasoning": "<explanation>"}'
    )


def evaluation_prompt(
    prediction: str,
    de_genes: list[str],
    markers: list[str],
    conflicts: list[str],
    tissues: list[str],
    go_summary: str,
) -> str:
    """Prompt for the evaluation node: LLM judge scoring."""
    return (
        "You are a rigorous cell-type annotation evaluator.\n\n"
        f"Hypothesis: {prediction}\n"
        f"DE genes: {', '.join(de_genes)}\n"
        f"Supporting markers: {', '.join(markers)}\n"
        f"Conflicting markers: {', '.join(conflicts)}\n"
        f"Tissue distribution: {', '.join(tissues)}\n"
        f"GO evidence: {go_summary}\n\n"
        "Score the hypothesis (0-100) and determine if it should be vetoed.\n"
        "Return ONLY valid JSON with exactly these keys:\n"
        '{"score": <int>, "veto": <bool>, "veto_reason": "<str or empty>", "reasoning": "<str>"}'
    )


def function_consistency_prompt(
    llm_function: str,
    function_records: list[dict],
) -> str:
    """Prompt for function semantic consistency judge."""
    records = "\n".join(f"- {r}" for r in function_records[:20]) or "(none)"
    return (
        "You are a strict cell biology judge. Compare the model's function text "
        "against authoritative Cell Ontology function/definition records.\n\n"
        f"Model function text:\n{llm_function}\n\n"
        f"Authoritative function records:\n{records}\n\n"
        "Score function consistency from 0 to 20.\n"
        "Use 20 only for clearly consistent descriptions, 10 for vague but non-conflicting, "
        "and 0 with veto=true for clearly wrong or contradictory descriptions.\n"
        "Return ONLY valid JSON with keys: "
        '{"score": <int>, "veto": <bool>, "veto_reason": "<str>", "reasoning": "<str>"}'
    )


def tissue_consistency_prompt(
    metadata: dict,
    tissue_records: list[dict],
) -> str:
    """Prompt for tissue distribution consistency judge."""
    records = "\n".join(f"- {r}" for r in tissue_records[:50]) or "(none)"
    return (
        "You are a strict single-cell annotation judge. Decide whether the sample metadata "
        "is compatible with known tissue distributions for the predicted cell type.\n\n"
        f"Sample metadata:\n{metadata}\n\n"
        f"Known tissue records:\n{records}\n\n"
        "Score tissue consistency from 0 to 10. Use 10 for compatible, 0 for clearly incompatible. "
        "Do not invent tissues that are not in the records.\n"
        "Return ONLY valid JSON with keys: "
        '{"score": <int>, "veto": <bool>, "veto_reason": "<str>", "reasoning": "<str>"}'
    )


def conflict_arbitration_prompt(
    predicted_cell_type: str,
    predicted_cl_id: str | None,
    conflict_candidates: list[dict],
    top10_de_genes: list[str] | None = None,
    top30_de_genes: list[str] | None = None,
    positive_marker_hits: list[str] | None = None,
) -> str:
    """Prompt for reverse-marker conflict arbitration."""
    candidates = "\n".join(f"- {c}" for c in conflict_candidates[:30]) or "(none)"
    top10 = ", ".join(top10_de_genes or []) or "(none)"
    top30 = ", ".join(top30_de_genes or []) or "(none)"
    positives = ", ".join(positive_marker_hits or []) or "(none)"
    return (
        "You are a strict marker-gene conflict judge. Determine whether reverse marker "
        "candidates strongly contradict the predicted cell type.\n"
        "Use only the provided evidence. Do not introduce new genes, new markers, or "
        "new cell types. A veto requires strong cross-lineage or mutually exclusive "
        "core-marker evidence.\n\n"
        f"Predicted cell type: {predicted_cell_type}\n"
        f"Predicted CL ID: {predicted_cl_id}\n\n"
        f"Cluster top10 DE genes:\n{top10}\n\n"
        f"Cluster top30 DE genes:\n{top30}\n\n"
        f"Positive marker hits for predicted cell type:\n{positives}\n\n"
        f"Reverse marker candidates:\n{candidates}\n\n"
        "Score conflict consistency from 0 to 30. Use 30 if there is no meaningful conflict, "
        "15 for weak/non-exclusive conflicts, and 0 with veto=true for strong cross-lineage "
        "or mutually exclusive core marker conflicts.\n"
        "Return ONLY valid JSON with keys: "
        '{"score": <int>, "veto": <bool>, "veto_reason": "<str>", "reasoning": "<str>", "conflicting_genes": ["<gene>"]}'
    )


def initial_inference_prompt(
    de_genes: list[str],
    metadata: dict,
) -> str:
    """Prompt for initial cell-type inference from DE genes and metadata."""
    meta_text = "\n".join(f"  {k}: {v}" for k, v in metadata.items())
    return (
        "You are a single-cell RNA-seq expert. Based on the following information, "
        "infer the most likely cell type and its biological function.\n\n"
        f"DE genes: {', '.join(de_genes)}\n\n"
        f"Metadata:\n{meta_text}\n\n"
        "Return ONLY valid JSON with exactly these keys:\n"
        '{"cell_type": "<precise cell type>", "function": "<biological function>", "confidence": <0.0-1.0>}'
    )
