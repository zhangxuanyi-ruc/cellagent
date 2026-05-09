#!/usr/bin/env python3
"""从 Uberon OBO 文件构建组织名称 → UBERON_ID 映射表.

官方文档:
  - Uberon: https://obofoundry.org/ontology/uberon.html
  - 文件: uberon-basic.obo (或 uberon.obo)

输出:
  rag/mappers/tissue_to_uberon.json

映射规则:
  1. 遍历 uberon.obo 中所有 [Term]
  2. 提取 name 字段 → 作为主键
  3. 提取所有 synonym 字段 → 作为别名
  4. 全部转小写后建立 name/synonym → UBERON_ID 映射

注意:
  Uberon 包含解剖结构、组织、器官、系统等多种概念。
  本映射表覆盖全部 term，不做过滤。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def build_tissue_mapper(obo_path: str, output_path: str) -> dict:
    """读取 uberon obo，构建 {name/synonym: UBERON_ID} 映射表."""
    obo_file = Path(obo_path)
    if not obo_file.exists():
        print(f"[ERROR] OBO 文件不存在: {obo_path}", file=sys.stderr)
        sys.exit(1)

    content = obo_file.read_text(encoding="utf-8")
    terms = content.split("\n[Term]")
    terms = terms[1:]

    name_to_id: dict[str, str] = {}
    id_to_synonyms: dict[str, list[str]] = {}

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

        if term_id and term_name and term_id.startswith("UBERON:") and not is_obsolete:
            name_to_id[term_name.lower()] = term_id
            for syn in synonyms:
                name_to_id.setdefault(syn.lower(), term_id)
            id_to_synonyms[term_id] = [term_name] + synonyms

    mapper = {
        "name_to_id": name_to_id,
        "id_to_synonyms": id_to_synonyms,
        "source": obo_path,
        "n_terms": len(id_to_synonyms),
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(mapper, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[DONE] 组织映射表构建完成")
    print(f"  源文件: {obo_path}")
    print(f"  输出:   {output_path}")
    print(f"  条目数: {len(id_to_synonyms)} 个 UBERON term")
    print(f"  映射数: {len(name_to_id)} 个 name/synonym → UBERON_ID")

    return mapper


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="构建组织名称映射表")
    parser.add_argument(
        "--obo",
        default="/data/bgi/data/projects/multimodal/RNA_data/tissue_ontology/uberon-basic.obo",
        help="Uberon OBO 文件路径",
    )
    parser.add_argument(
        "--output",
        default="/root/wanghaoran/zxy/project/cellagent/rag/mappers/tissue_to_uberon.json",
        help="输出 JSON 路径",
    )
    args = parser.parse_args()

    build_tissue_mapper(args.obo, args.output)
