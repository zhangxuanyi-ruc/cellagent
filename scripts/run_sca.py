#!/usr/bin/env python
"""Run the CellAgent SCA pipeline from an input h5ad to manifest outputs."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--input", required=True, help="Input h5ad path.")
    parser.add_argument("--output-root", required=True, help="Output root for this SCA run.")
    parser.add_argument("--cluster-source", choices=["cellagent", "obs"], default=None)
    parser.add_argument("--obs-key", default=None)
    parser.add_argument("--feature-timeout", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def run_command(cmd: list[str]) -> dict[str, Any]:
    print("[run_sca]", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    if proc.stdout:
        print(proc.stdout, flush=True)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")
    return parse_last_json(proc.stdout)


def parse_last_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        return {}
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            value = json.loads(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def write_run_config(base_config: Path, input_path: Path, output_root: Path, cluster_source: str | None, obs_key: str | None) -> Path:
    cfg = yaml.safe_load(base_config.read_text(encoding="utf-8")) or {}
    cfg.setdefault("pipeline", {})
    cfg["pipeline"]["input_path"] = str(input_path)
    cfg["pipeline"]["output_root"] = str(output_root)
    if cluster_source is not None:
        cfg.setdefault("clustering", {})
        cfg["clustering"]["source"] = cluster_source
    if obs_key is not None:
        cfg.setdefault("clustering", {})
        cfg["clustering"]["obs_key"] = obs_key
    output_root.mkdir(parents=True, exist_ok=True)
    run_config = output_root / "run_config.yaml"
    run_config.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return run_config


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_root = Path(args.output_root)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    run_config = write_run_config(
        base_config=Path(args.config),
        input_path=input_path,
        output_root=output_root,
        cluster_source=args.cluster_source,
        obs_key=args.obs_key,
    )
    preprocessed_dir = output_root / "preprocessed"
    qc_dir = output_root / "qc_report"
    feature_dir = output_root / "features"
    clustering_dir = output_root / "clustering"
    de_dir = output_root / "de"

    preprocessed = preprocessed_dir / f"{input_path.stem}_preprocessed.h5ad"
    if not (args.skip_existing and preprocessed.exists()):
        preprocess_out = run_command([
            sys.executable,
            "scripts/preprocess_external_data.py",
            "--config",
            str(run_config),
            "--input",
            str(input_path),
            "--output-dir",
            str(preprocessed_dir),
            "--qc-report-dir",
            str(qc_dir),
        ])
        preprocessed = Path(preprocess_out.get("output_path") or preprocessed)

    feature_candidates = sorted(feature_dir.glob(f"{preprocessed.stem}*_cell_features.npz"))
    feature_npz = feature_candidates[0] if feature_candidates else None
    if not (args.skip_existing and feature_npz and feature_npz.exists()):
        feature_cmd = [
            sys.executable,
            "scripts/request_feature_service.py",
            "--config",
            str(run_config),
            "--input",
            str(preprocessed),
            "--feature-dir",
            str(feature_dir),
        ]
        if args.feature_timeout is not None:
            feature_cmd.extend(["--timeout", str(args.feature_timeout)])
        feature_out = run_command(feature_cmd)
        feature_npz = Path(feature_out["feature_path"])

    cluster_out = run_command([
        sys.executable,
        "scripts/cluster_cell_features.py",
        "--config",
        str(run_config),
        "--features",
        str(feature_npz),
        "--preprocessed",
        str(preprocessed),
        "--output-dir",
        str(clustering_dir),
    ])
    clusters_csv = Path(cluster_out["clusters_csv"])
    clustering_metrics = Path(cluster_out["metrics_json"])

    de_out = run_command([
        sys.executable,
        "scripts/run_de_analysis.py",
        "--config",
        str(run_config),
        "--preprocessed",
        str(preprocessed),
        "--clusters",
        str(clusters_csv),
        "--output-dir",
        str(de_dir),
    ])
    de_summary = Path(de_out["summary"])

    manifest_out = run_command([
        sys.executable,
        "scripts/write_pipeline_manifest.py",
        "--config",
        str(run_config),
        "--preprocessed-h5ad",
        str(preprocessed),
        "--qc-report-dir",
        str(qc_dir),
        "--feature-npz",
        str(feature_npz),
        "--clusters-csv",
        str(clusters_csv),
        "--clustering-metrics-json",
        str(clustering_metrics),
        "--de-summary-csv",
        str(de_summary),
        "--de-dir",
        str(de_dir),
        "--reasoning-dir",
        str(output_root / "reasoning"),
        "--final-dir",
        str(output_root / "final"),
    ])
    print(json.dumps({"run_config": str(run_config), **manifest_out}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
