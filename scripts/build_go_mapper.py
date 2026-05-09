#!/usr/bin/env python3
"""从 Gene Ontology OBO 文件构建功能名称 → GO_ID 映射表.

官方文档:
  - Gene Ontology: https://geneontology.org/docs/download-ontology/
  - 当前版本: http://purl.obolibrary.org/obo/go.obo

输出:
  rag/mappers/function_to_go.json

映射规则与 build_cell_type_mapper.py 相同，只是目标 ontology 换成 GO.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def build_go_mapper(go_obo_path: str, output_path: str) -> dict:
    """读取 go.obo，构建 {name/synonym: GO_ID} 映射表."""
    go_obo = Path(go_obo_path)
    if not go_obo.exists():
        print(f"[ERROR] go.obo 不存在: {go_obo_path}", file=sys.stderr)
        sys.exit(1)

    content = go_obo.read_text(encoding="utf-8")
    terms = content.split("\n[Term]")
    terms = terms[1:]

    name_to_go: dict[str, str] = {}
    go_to_synonyms: dict[str, list[str]] = {}

    for term_block in terms:
        lines = term_block.strip().splitlines()
        term_id = None
        term_name = None
        synonyms: list[str] = []
        is_obsolete = False

        for line in lines:
            line = line.strip()
            if line.startswith("id:"):
                term_id = line.split(":", 1)[1].strip()
            elif line.startswith("name:"):
                term_name = line.split(":", 1)[1].strip()
            elif line.startswith("synonym:"):
                match = re.search(r'synonym:\s+"([^"]+)"', line)
                if match:
                    synonyms.append(match.group(1))
            elif line.startswith("is_obsolete: true"):
                is_obsolete = True
            elif line.startswith("["):
                break

        if term_id and term_name and term_id.startswith("GO:") and not is_obsolete:
            name_to_go[term_name.lower()] = term_id
            for syn in synonyms:
                name_to_go.setdefault(syn.lower(), term_id)
            go_to_synonyms[term_id] = [term_name] + synonyms

    mapper = {
        "name_to_go": name_to_go,
        "go_to_synonyms": go_to_synonyms,
        "source": go_obo_path,
        "n_terms": len(go_to_synonyms),
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(mapper, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[DONE] GO 功能映射表构建完成")
    print(f"  源文件: {go_obo_path}")
    print(f"  输出:   {output_path}")
    print(f"  条目数: {len(go_to_synonyms)} 个 GO term")
    print(f"  映射数: {len(name_to_go)} 个 name/synonym → GO_ID")

    return mapper


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="构建 GO 功能名称映射表")
    parser.add_argument(
        "--obo",
        default="/data/bgi/data/projects/multimodal/RNA_data/cellagent_database/ontology/go/go.obo",
        help="Gene Ontology OBO 文件路径",
    )
    parser.add_argument(
        "--output",
        default="/root/wanghaoran/zxy/project/cellagent/rag/mappers/function_to_go.json",
        help="输出 JSON 路径",
    )
    args = parser.parse_args()

    build_go_mapper(args.obo, args.output)
