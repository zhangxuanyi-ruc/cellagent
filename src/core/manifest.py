"""Pipeline artifact manifest for CellAgent.

The manifest is the stable contract between notebook/script stages. Config
contains defaults; the manifest records concrete artifacts produced by one run.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class PipelineManifest(BaseModel):
    run_id: str
    input_h5ad: str
    output_root: str
    config_path: str | None = None
    preprocessed_h5ad: str | None = None
    qc_report_dir: str | None = None
    feature_npz: str | None = None
    clusters_csv: str | None = None
    clustering_metrics_json: str | None = None
    cluster_source: str | None = None
    cluster_obs_key: str | None = None
    de_summary_csv: str | None = None
    de_dir: str | None = None
    multimodal_prior_json: str | None = None
    reasoning_dir: str | None = None
    final_dir: str | None = None
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.model_dump(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    @classmethod
    def read(cls, path: str | Path) -> "PipelineManifest":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def default_manifest_path(output_root: str | Path) -> Path:
    return Path(output_root) / "pipeline_manifest.json"


__all__ = ["PipelineManifest", "default_manifest_path"]
