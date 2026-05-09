#!/usr/bin/env python3
"""从 Cell Ontology OBO 文件构建细胞类型名称 → CL_ID 映射表.

官方文档:
  - Cell Ontology: https://obofoundry.org/ontology/cl.html
  - OBO 格式说明: http://purl.obolibrary.org/obo/cl.obo

输出:
  rag/mappers/cell_type_to_cl.json

映射规则:
  1. 遍历 cl.obo 中所有 [Term]
  2. 提取 name 字段 → 作为主键
  3. 提取所有 synonym 字段 → 作为别名
  4. 全部转小写后建立 name/synonym → CL_ID 映射
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def parse_obo_synonyms(text: str) -> list[str]:
    """从 synonym 行提取文本.

    OBO 格式示例:
      synonym: "T-cell" EXACT []
      synonym: "T lymphocyte" RELATED []
    """
    synonyms = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("synonym:"):
            # 提取引号内的文本
            match = re.search(r'synonym:\s+"([^"]+)"', line)
            if match:
                synonyms.append(match.group(1))
    return synonyms


def build_cell_type_mapper(cl_obo_path: str, output_path: str) -> dict:
    """读取 cl.obo，构建 {name/synonym: CL_ID} 映射表."""
    cl_obo = Path(cl_obo_path)
    if not cl_obo.exists():
        print(f"[ERROR] cl.obo 不存在: {cl_obo_path}", file=sys.stderr)
        sys.exit(1)

    content = cl_obo.read_text(encoding="utf-8")

    # 按 [Term] 分割
    terms = content.split("\n[Term]")
    # 第一个块是文件头，跳过
    terms = terms[1:]

    name_to_cl: dict[str, str] = {}
    cl_to_synonyms: dict[str, list[str]] = {}

    for term_block in terms:
        lines = term_block.strip().splitlines()
        term_id = None
        term_name = None
        synonyms: list[str] = []

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
            # 遇到下一个 stanza 或空行则终止当前 term 解析
            elif line.startswith("["):
                break

        if term_id and term_name and term_id.startswith("CL:"):
            # 主 name
            name_to_cl[term_name.lower()] = term_id
            # 所有 synonym
            for syn in synonyms:
                name_to_cl.setdefault(syn.lower(), term_id)
            cl_to_synonyms[term_id] = [term_name] + synonyms

    mapper = {
        "name_to_cl": name_to_cl,
        "cl_to_synonyms": cl_to_synonyms,
        "source": cl_obo_path,
        "n_terms": len(cl_to_synonyms),
    }

    # 写入 JSON
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(mapper, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[DONE] 细胞类型映射表构建完成")
    print(f"  源文件: {cl_obo_path}")
    print(f"  输出:   {output_path}")
    print(f"  条目数: {len(cl_to_synonyms)} 个 CL term")
    print(f"  映射数: {len(name_to_cl)} 个 name/synonym → CL_ID")

    return mapper


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="构建细胞类型名称映射表")
    parser.add_argument(
        "--obo",
        default="/data/bgi/data/projects/multimodal/RNA_data/cellagent_database/ontology/cl/cl.obo",
        help="Cell Ontology OBO 文件路径",
    )
    parser.add_argument(
        "--output",
        default="/root/wanghaoran/zxy/project/cellagent/rag/mappers/cell_type_to_cl.json",
        help="输出 JSON 路径",
    )
    args = parser.parse_args()

    build_cell_type_mapper(args.obo, args.output)
