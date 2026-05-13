"""Agent runtime state and configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AgentRunConfig:
    """Resolved paths for one CellAgent annotation run."""

    config_path: Path
    rag_config_path: Path
    manifest_path: Path
    prior_json_path: Path
    output_root: Path
    reasoning_dir: Path
    judging_dir: Path
    final_dir: Path
    top_de_genes: int = 30
    enable_llm_judge: bool = False


@dataclass
class AgentRunResult:
    """Summary of one agent pipeline run."""

    reasoning_paths: list[Path] = field(default_factory=list)
    judge_paths: list[Path] = field(default_factory=list)
    final_json: Path | None = None
    final_csv: Path | None = None


__all__ = ["AgentRunConfig", "AgentRunResult"]
