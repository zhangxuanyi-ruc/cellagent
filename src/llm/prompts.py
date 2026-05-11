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
