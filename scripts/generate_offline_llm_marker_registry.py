#!/usr/bin/env python
"""Generate raw offline LLM marker suggestions for candidate cell types."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.clients import build_llm_client_from_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default="resources/marker_registry/offline_llm_marker_candidates.json")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output", default="resources/marker_registry/offline_llm_marker_raw.jsonl")
    parser.add_argument("--prompt-dir", default="resources/marker_registry/offline_llm_marker_prompts")
    parser.add_argument("--limit", type=int, default=0, help="0 means all candidates.")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def slugify(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text.lower()).strip("_")[:120]


def build_prompt(item: dict[str, Any]) -> list[dict[str, str]]:
    cell_type = item["cell_type"]
    cl_id = item.get("cell_type_cl_id")
    system = (
        "You are a single-cell biology marker curation assistant. "
        "Return only valid JSON. Do not include markdown, explanations, or prose."
    )
    user = f"""
Task: curate a compact ranked positive marker gene list for the given cell type.

Cell type: {cell_type}
Cell Ontology ID: {cl_id}

Strict rules:
1. Output JSON only. No markdown, no explanations, no reasoning text.
2. Provide positive marker genes expected to be expressed in this cell type.
3. Use official human gene symbols only. Do not output proteins, pathways, descriptions, or CD names without gene symbols.
4. Rank genes by marker importance/specificity, most important first.
5. Return about 30 genes when possible.
6. Return at least 10 genes unless fewer than 10 high-confidence markers are known.
7. Never return more than 50 genes.
8. Do not refuse. Every candidate was pre-filtered and needs a marker list.
9. Do not output an empty marker list.

Required JSON schema:
{{
  "cell_type": "{cell_type}",
  "cell_type_cl_id": "{cl_id}",
  "positive_markers_ranked": ["GENE1", "GENE2"]
}}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def load_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("cell_type"):
            done.add(str(row["cell_type"]).strip().lower())
    return done


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    llm = build_llm_client_from_config(cfg)
    if llm is None:
        raise RuntimeError("LLM client is disabled in config.")

    candidates_path = resolve(args.candidates)
    payload = json.loads(candidates_path.read_text(encoding="utf-8"))
    candidates = list(payload.get("candidates") or [])
    if args.limit and args.limit > 0:
        candidates = candidates[: args.limit]

    output = resolve(args.output)
    prompt_dir = resolve(args.prompt_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    done = load_done(output) if args.resume else set()

    mode = "a" if args.resume else "w"
    n_written = 0
    with output.open(mode, encoding="utf-8") as fh:
        for item in candidates:
            cell_type = str(item["cell_type"])
            if cell_type.strip().lower() in done:
                continue
            messages = build_prompt(item)
            (prompt_dir / f"{slugify(cell_type)}.json").write_text(
                json.dumps({"candidate": item, "messages": messages}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            try:
                response = llm.chat(messages, json_mode=True)
            except Exception as exc:
                print(json.dumps({"cell_type": cell_type, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
                response = {"error": str(exc), "cell_type": cell_type}
            row = {
                "cell_type": cell_type,
                "cell_type_cl_id": item.get("cell_type_cl_id"),
                "candidate": item,
                "response": response,
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            n_written += 1
            print(json.dumps({"n_written": n_written, "cell_type": cell_type}, ensure_ascii=False))
    print(f"wrote={output}")


if __name__ == "__main__":
    main()
