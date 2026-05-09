#!/usr/bin/env python3
"""构建基因名称标准化映射表.

支持物种:
  - human: HGNC approved symbol
  - mouse: MGI official symbol

官方数据来源:
  - HGNC (人): https://www.genenames.org/
    EBI FTP 镜像: ftp://ftp.ebi.ac.uk/pub/databases/genenames/hgnc/tsv/hgnc_complete_set.txt
  - MGI (鼠): http://www.informatics.jax.org/
    FTP: http://www.informatics.jax.org/downloads/reports/MGI_EntrezGene.rpt

输出:
  rag/mappers/gene_symbol_mapper.json

映射结构:
  {
    "human": {
      "approved": {"TP53": "TP53", "p53": "TP53", ...},
      "alias_to_approved": {"BCC7": "TP53", "LFS1": "TP53", ...},
      "ensembl_to_approved": {"ENSG00000141510": "TP53"},
      "entrez_to_approved": {"7157": "TP53"}
    },
    "mouse": {
      "approved": {"Trp53": "Trp53", ...},
      "alias_to_approved": {...},
      "ensembl_to_approved": {"ENSMUSG00000059552": "Trp53"},
      "entrez_to_approved": {"22059": "Trp53"},
      "mouse_to_human_ortholog": {
        "Trp53": [{"human_symbol": "TP53", "orthology_type": "one_to_one", ...}]
      }
    }
  }

注意:
  HGNC complete set 列包括:
    - hgnc_id, symbol, name, alias_symbol, prev_symbol, entrez_id, ensembl_gene_id
  本脚本使用 symbol 作为 approved symbol，alias_symbol 和 prev_symbol 作为别名.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# HGNC complete set 官方公开下载地址 (Google Cloud Storage)
# 来源: https://www.genenames.org/download/archive/
HGNC_URL = "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"
DEFAULT_MOUSE_ORTHOLOGY_DIR = Path("/data/bgi/data/projects/multimodal/RNA_data/cellagent_database/mouse_human_orthology")


def download_hgnc(output_path: Path) -> Path:
    """下载 HGNC complete set."""
    import urllib.request

    print(f"[DOWNLOAD] HGNC complete set from {HGNC_URL}")
    try:
        urllib.request.urlretrieve(HGNC_URL, output_path)
        print(f"[DONE] 下载完成: {output_path} ({output_path.stat().st_size} bytes)")
        return output_path
    except Exception as e:
        print(f"[ERROR] 下载失败: {e}", file=sys.stderr)
        print("[HINT] 可手动从 https://www.genenames.org/download/archive/ 下载", file=sys.stderr)
        sys.exit(1)


def parse_hgnc_tsv(path: Path) -> dict:
    """解析 HGNC TSV，构建 approved + alias + Ensembl + Entrez 映射."""
    import csv

    approved: dict[str, str] = {}      # symbol -> approved symbol (统一大小写)
    alias_to_approved: dict[str, str] = {}  # 别名 -> approved symbol
    ensembl_to_approved: dict[str, str] = {}  # ENSG -> approved symbol
    entrez_to_approved: dict[str, str] = {}  # Entrez ID -> approved symbol

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            symbol = row.get("symbol", "").strip()
            if not symbol:
                continue

            # approved symbol 自身映射
            approved[symbol] = symbol
            approved[symbol.lower()] = symbol
            approved[symbol.upper()] = symbol

            # alias_symbol: 可能包含多个，用 | 分隔
            alias_str = row.get("alias_symbol", "").strip()
            if alias_str and alias_str != "-":
                for alias in alias_str.split("|"):
                    alias = alias.strip()
                    if alias:
                        alias_to_approved.setdefault(alias, symbol)
                        alias_to_approved.setdefault(alias.lower(), symbol)
                        alias_to_approved.setdefault(alias.upper(), symbol)

            # prev_symbol: 旧 symbol
            prev_str = row.get("prev_symbol", "").strip()
            if prev_str and prev_str != "-":
                for prev in prev_str.split("|"):
                    prev = prev.strip()
                    if prev:
                        alias_to_approved.setdefault(prev, symbol)
                        alias_to_approved.setdefault(prev.lower(), symbol)
                        alias_to_approved.setdefault(prev.upper(), symbol)

            # ensembl_gene_id: HGNC usually stores a single ENSG ID.
            ensembl_str = row.get("ensembl_gene_id", "").strip()
            if ensembl_str and ensembl_str != "-":
                for ensembl_id in ensembl_str.split("|"):
                    ensembl_id = ensembl_id.strip()
                    if ensembl_id:
                        ensembl_to_approved.setdefault(ensembl_id, symbol)
                        ensembl_to_approved.setdefault(ensembl_id.upper(), symbol)

            # entrez_id: numeric ID as string.
            entrez_str = row.get("entrez_id", "").strip()
            if entrez_str and entrez_str != "-":
                for entrez_id in entrez_str.split("|"):
                    entrez_id = entrez_id.strip()
                    if entrez_id:
                        entrez_to_approved.setdefault(entrez_id, symbol)

    return {
        "approved": approved,
        "alias_to_approved": alias_to_approved,
        "ensembl_to_approved": ensembl_to_approved,
        "entrez_to_approved": entrez_to_approved,
    }


def _add_case_keys(mapping: dict[str, str], key: str, value: str) -> None:
    if not key:
        return
    mapping.setdefault(key, value)
    mapping.setdefault(key.lower(), value)
    mapping.setdefault(key.upper(), value)


def parse_mgi_mouse_reports(
    gene_model_path: Path | None = None,
    entrez_path: Path | None = None,
    homology_path: Path | None = None,
) -> dict:
    """Parse official MGI reports for mouse IDs and mouse->human orthologs."""
    import csv
    import collections

    approved: dict[str, str] = {}
    alias_to_approved: dict[str, str] = {}
    ensembl_to_approved: dict[str, str] = {}
    entrez_to_approved: dict[str, str] = {}
    mouse_to_human_ortholog: dict[str, list[dict]] = {}

    if entrez_path and entrez_path.exists():
        cols = [
            "mgi_id",
            "symbol",
            "status",
            "name",
            "cm_position",
            "chromosome",
            "marker_type",
            "secondary_accession_ids",
            "entrez_id",
            "synonyms",
            "feature_types",
            "genome_start",
            "genome_end",
            "strand",
            "biotypes",
        ]
        with entrez_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t", fieldnames=cols)
            for row in reader:
                if row.get("status") != "O" or row.get("marker_type") != "Gene":
                    continue
                symbol = row.get("symbol", "").strip()
                if not symbol:
                    continue
                _add_case_keys(approved, symbol, symbol)
                entrez_id = row.get("entrez_id", "").strip()
                if entrez_id:
                    entrez_to_approved.setdefault(entrez_id, symbol)
                for alias in row.get("synonyms", "").split("|"):
                    alias = alias.strip()
                    if alias and alias != symbol:
                        _add_case_keys(alias_to_approved, alias, symbol)

    if gene_model_path and gene_model_path.exists():
        with gene_model_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if row.get("2. marker type") != "Gene":
                    continue
                symbol = row.get("3. marker symbol", "").strip()
                if not symbol:
                    continue
                _add_case_keys(approved, symbol, symbol)
                entrez_id = row.get("6. Entrez gene id", "").strip()
                if entrez_id:
                    entrez_to_approved.setdefault(entrez_id, symbol)
                ensembl_id = row.get("11. Ensembl gene id", "").strip()
                if ensembl_id:
                    ensembl_to_approved.setdefault(ensembl_id, symbol)
                    ensembl_to_approved.setdefault(ensembl_id.upper(), symbol)

    if homology_path and homology_path.exists():
        groups: dict[str, dict[str, list[dict]]] = collections.defaultdict(lambda: {"mouse": [], "human": []})
        with homology_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                species = row.get("Common Organism Name", "")
                if species == "mouse, laboratory":
                    groups[row["DB Class Key"]]["mouse"].append(row)
                elif species == "human":
                    groups[row["DB Class Key"]]["human"].append(row)

        for db_class_key, group in groups.items():
            mouse_rows = group["mouse"]
            human_rows = group["human"]
            if not mouse_rows or not human_rows:
                continue
            if len(mouse_rows) == 1 and len(human_rows) == 1:
                orthology_type = "one_to_one"
            elif len(mouse_rows) == 1 and len(human_rows) > 1:
                orthology_type = "one_to_many"
            elif len(mouse_rows) > 1 and len(human_rows) == 1:
                orthology_type = "many_to_one"
            else:
                orthology_type = "many_to_many"

            for mouse_row in mouse_rows:
                mouse_symbol = mouse_row.get("Symbol", "").strip()
                if not mouse_symbol:
                    continue
                _add_case_keys(approved, mouse_symbol, mouse_symbol)
                mouse_entrez = mouse_row.get("EntrezGene ID", "").strip()
                if mouse_entrez:
                    entrez_to_approved.setdefault(mouse_entrez, mouse_symbol)

                records = mouse_to_human_ortholog.setdefault(mouse_symbol, [])
                for human_row in human_rows:
                    human_symbol = human_row.get("Symbol", "").strip()
                    if not human_symbol:
                        continue
                    records.append(
                        {
                            "human_symbol": human_symbol,
                            "human_entrez_id": human_row.get("EntrezGene ID", "").strip() or None,
                            "hgnc_id": human_row.get("HGNC ID", "").strip() or None,
                            "mouse_symbol": mouse_symbol,
                            "mouse_entrez_id": mouse_entrez or None,
                            "mouse_mgi_id": mouse_row.get("Mouse MGI ID", "").strip() or None,
                            "db_class_key": db_class_key,
                            "orthology_type": orthology_type,
                            "source": "MGI_HOM_MouseHumanSequence",
                        }
                    )

    return {
        "approved": approved,
        "alias_to_approved": alias_to_approved,
        "ensembl_to_approved": ensembl_to_approved,
        "entrez_to_approved": entrez_to_approved,
        "mouse_to_human_ortholog": mouse_to_human_ortholog,
    }


def build_gene_mapper(
    hgnc_path: str | None = None,
    mgi_gene_model_path: str | None = None,
    mgi_entrez_path: str | None = None,
    mgi_homology_path: str | None = None,
    output_path: str = "/root/wanghaoran/zxy/project/cellagent/rag/mappers/gene_symbol_mapper.json",
) -> dict:
    """构建基因映射表."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # 下载或读取 HGNC
    if hgnc_path:
        hgnc_file = Path(hgnc_path)
    else:
        hgnc_file = out.parent / "hgnc_complete_set.txt"
        if not hgnc_file.exists():
            download_hgnc(hgnc_file)

    if not hgnc_file.exists():
        print(f"[ERROR] HGNC 文件不存在: {hgnc_file}", file=sys.stderr)
        sys.exit(1)

    human_mapper = parse_hgnc_tsv(hgnc_file)
    mouse_mapper = parse_mgi_mouse_reports(
        gene_model_path=Path(mgi_gene_model_path) if mgi_gene_model_path else DEFAULT_MOUSE_ORTHOLOGY_DIR / "MGI_Gene_Model_Coord.rpt",
        entrez_path=Path(mgi_entrez_path) if mgi_entrez_path else DEFAULT_MOUSE_ORTHOLOGY_DIR / "MGI_EntrezGene.rpt",
        homology_path=Path(mgi_homology_path) if mgi_homology_path else DEFAULT_MOUSE_ORTHOLOGY_DIR / "HOM_MouseHumanSequence.rpt",
    )

    mapper = {
        "human": human_mapper,
        "mouse": mouse_mapper,
        "sources": {
            "human": str(hgnc_file),
            "mouse_gene_model": str(mgi_gene_model_path or DEFAULT_MOUSE_ORTHOLOGY_DIR / "MGI_Gene_Model_Coord.rpt"),
            "mouse_entrez": str(mgi_entrez_path or DEFAULT_MOUSE_ORTHOLOGY_DIR / "MGI_EntrezGene.rpt"),
            "mouse_homology": str(mgi_homology_path or DEFAULT_MOUSE_ORTHOLOGY_DIR / "HOM_MouseHumanSequence.rpt"),
        },
    }

    out.write_text(json.dumps(mapper, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[DONE] 基因映射表构建完成")
    print(f"  源文件: {hgnc_file}")
    print(f"  输出:   {output_path}")
    print(f"  人基因 approved:   {len(set(human_mapper['approved'].values()))} 个")
    print(f"  人基因 alias:      {len(human_mapper['alias_to_approved'])} 个")
    print(f"  人基因 Ensembl:    {len(human_mapper['ensembl_to_approved'])} 个")
    print(f"  人基因 Entrez:     {len(human_mapper['entrez_to_approved'])} 个")
    print(f"  鼠基因 approved:   {len(set(mouse_mapper['approved'].values()))} 个")
    print(f"  鼠基因 alias:      {len(mouse_mapper['alias_to_approved'])} 个")
    print(f"  鼠基因 Ensembl:    {len(mouse_mapper['ensembl_to_approved'])} 个")
    print(f"  鼠基因 Entrez:     {len(mouse_mapper['entrez_to_approved'])} 个")
    print(f"  鼠->人 ortholog:   {len(mouse_mapper['mouse_to_human_ortholog'])} 个 mouse symbols")

    return mapper


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="构建基因名称标准化映射表")
    parser.add_argument(
        "--hgnc",
        default=None,
        help="HGNC complete set TSV 文件路径（不指定则自动下载）",
    )
    parser.add_argument("--mgi-gene-model", default=None, help="MGI_Gene_Model_Coord.rpt 路径")
    parser.add_argument("--mgi-entrez", default=None, help="MGI_EntrezGene.rpt 路径")
    parser.add_argument("--mgi-homology", default=None, help="HOM_MouseHumanSequence.rpt 路径")
    parser.add_argument(
        "--output",
        default="/root/wanghaoran/zxy/project/cellagent/rag/mappers/gene_symbol_mapper.json",
        help="输出 JSON 路径",
    )
    args = parser.parse_args()

    build_gene_mapper(
        hgnc_path=args.hgnc,
        mgi_gene_model_path=args.mgi_gene_model,
        mgi_entrez_path=args.mgi_entrez,
        mgi_homology_path=args.mgi_homology,
        output_path=args.output,
    )
