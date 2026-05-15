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
                    "target_cluster_consistency_score": result.report.target_cluster_consistency_score,
                    "function_consistency_score": result.report.function_consistency_score,
                    "tissue_consistency_score": result.report.tissue_consistency_score,
                    "target_cluster_presence_score": result.target_cluster_report.get("presence_score", 0),
                    "target_cluster_signature_score": result.target_cluster_report.get("signature_score", 0),
                    "target_cluster_signature_percentile": result.target_cluster_report.get("signature_percentile", 0.0),
                    "reverse_marker_n_primary_candidates": result.reverse_marker_monitor.get("n_primary_candidates", 0),
                    "reverse_marker_n_screen_candidates": result.reverse_marker_monitor.get("n_screen_candidates", 0),
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
            return "needs_reflection"
        if result.report.total >= 80:
            return "accepted"
        return "needs_reflection"


__all__ = ["AnnotationFinalizer"]
