"""Final output branch for CellAgent."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.core.schemas import JudgeResult


class AnnotationFinalizer:
    """Create compact final annotation tables from judge results."""

    def build_rows(self, judge_results: list[JudgeResult]) -> list[dict]:
        rows: list[dict] = []
        for result in judge_results:
            rows.append(
                {
                    "cluster_id": result.cluster_id,
                    "cell_type": result.provenance.get("cell_type", ""),
                    "cell_type_cl_id": result.provenance.get("cell_type_cl_id", ""),
                    "overall_score": result.report.total,
                    "status": self._status(result),
                    "veto_triggered": result.report.veto_triggered,
                    "veto_reason": result.report.veto_reason,
                    "marker_match_score": result.report.marker_match_score,
                    "conflict_penalty": result.report.conflict_penalty,
                    "function_consistency_score": result.report.function_consistency_score,
                    "tissue_consistency_score": result.report.tissue_consistency_score,
                    "marker_matches": ",".join(result.marker_matches),
                    "marker_misses": ",".join(result.marker_misses),
                }
            )
        return rows

    def write(self, judge_results: list[JudgeResult], output_dir: str | Path) -> tuple[Path, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        rows = self.build_rows(judge_results)
        json_path = output_dir / "annotations.json"
        csv_path = output_dir / "annotations.csv"
        json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        return json_path, csv_path

    @staticmethod
    def _status(result: JudgeResult) -> str:
        if result.report.veto_triggered:
            return "rejected"
        if result.report.total >= 80:
            return "accepted"
        return "needs_reflection"


__all__ = ["AnnotationFinalizer"]
