"""从 DE 基因查询关联的 Gene Ontology terms.

用于 REACT 引擎的功能一致性验证.
核心思路: 不依赖"假设细胞类型"的 GO terms,
而是直接查 DE 基因各自关联的 GO terms,
汇总得到 "基因表达谱暗示的功能", 再与 LLM 功能描述比对.

数据来源: mygene.info (BioThings API, 免费, 无需 key)
  https://mygene.info/
  https://docs.mygene.info/en/latest/doc/query_service.html

使用方式:
  from src.tools.go_annotation import GOAnnotator
  annot = GOAnnotator()
  result = annot.query_genes(["TP53", "CD3D", "FOXP3"])
  # result = {
  #   "TP53": {"BP": [{"id":"GO:...","term":"..."},...], "MF": [...], "CC": [...]},
  #   "CD3D": {...},
  #   ...
  # }
  go_terms = annot.extract_go_terms(result, aspect="BP", top_k=5)
  # go_terms = ["T cell activation", "immune response", ...]
"""
from __future__ import annotations

import urllib.request
import urllib.parse
import json
from typing import Literal


class GOAnnotator:
    """Gene Ontology 注释查询器 (基于 mygene.info API)."""

    BASE_URL = "https://mygene.info/v3/query"

    def __init__(self, species: str = "human"):
        self.species = species

    def query_gene(self, gene_symbol: str) -> dict:
        """查询单个基因的 GO 注释.

        Returns:
            {
                "gene": str,
                "go": {
                    "BP": [{"id": "GO:...", "term": "..."}, ...],
                    "MF": [{"id": "GO:...", "term": "..."}, ...],
                    "CC": [{"id": "GO:...", "term": "..."}, ...],
                }
            }
        """
        params = urllib.parse.urlencode({
            "q": f"symbol:{gene_symbol}",
            "species": self.species,
            "fields": "go",
            "size": 1,
        })
        url = f"{self.BASE_URL}?{params}"

        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            return {"gene": gene_symbol, "go": {}, "error": str(e)}

        hits = data.get("hits", [])
        if not hits:
            return {"gene": gene_symbol, "go": {}}

        go_data = hits[0].get("go", {})
        return {
            "gene": gene_symbol,
            "go": {
                "BP": [{"id": t.get("id", ""), "term": t.get("term", "")}
                       for t in go_data.get("BP", [])],
                "MF": [{"id": t.get("id", ""), "term": t.get("term", "")}
                       for t in go_data.get("MF", [])],
                "CC": [{"id": t.get("id", ""), "term": t.get("term", "")}
                       for t in go_data.get("CC", [])],
            },
        }

    def query_genes(self, gene_symbols: list[str]) -> dict[str, dict]:
        """批量查询多个基因的 GO 注释.

        优先使用 mygene.info 批量 POST API（一次请求查所有基因），
        失败时 fallback 到串行查询。

        Returns:
            {gene_symbol: {"go": {"BP": [...], "MF": [...], "CC": [...]}}}
        """
        if not gene_symbols:
            return {}

        # 尝试批量 POST 查询（更高效）
        try:
            return self._query_genes_batch(gene_symbols)
        except Exception:
            # fallback 到串行查询
            results: dict[str, dict] = {}
            for gene in gene_symbols:
                result = self.query_gene(gene)
                results[gene] = result
            return results

    def _query_genes_batch(self, gene_symbols: list[str]) -> dict[str, dict]:
        """使用 mygene.info 批量 POST API 查询."""
        import urllib.request

        url = f"{self.BASE_URL}"
        data = urllib.parse.urlencode({
            "q": ",".join(gene_symbols),
            "scopes": "symbol",
            "species": self.species,
            "fields": "go,symbol",
        }).encode("utf-8")

        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/x-www-form-urlencoded")

        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = json.loads(resp.read().decode("utf-8"))

        # 解析响应（列表形式，每个元素对应一个查询基因）
        results: dict[str, dict] = {}
        for item in raw:
            gene = item.get("query", "")
            if not gene:
                continue
            go_data = item.get("go", {})
            results[gene] = {
                "gene": gene,
                "go": {
                    "BP": [{"id": t.get("id", ""), "term": t.get("term", "")}
                           for t in go_data.get("BP", [])],
                    "MF": [{"id": t.get("id", ""), "term": t.get("term", "")}
                           for t in go_data.get("MF", [])],
                    "CC": [{"id": t.get("id", ""), "term": t.get("term", "")}
                           for t in go_data.get("CC", [])],
                },
            }

        # 为未返回的基因填充空结果
        for gene in gene_symbols:
            if gene not in results:
                results[gene] = {"gene": gene, "go": {}}

        return results

    def extract_go_terms(
        self,
        results: dict[str, dict],
        aspect: Literal["BP", "MF", "CC", "all"] = "all",
        top_k: int = 10,
    ) -> list[str]:
        """从批量查询结果中提取 GO term 名称列表 (去重).

        Args:
            results: query_genes() 的返回值
            aspect: "BP"(生物学过程), "MF"(分子功能), "CC"(细胞组分), "all"
            top_k: 每个基因每个 aspect 最多取几条
        """
        aspects = ["BP", "MF", "CC"] if aspect == "all" else [aspect]
        terms: set[str] = set()

        for gene, data in results.items():
            go_data = data.get("go", {})
            for asp in aspects:
                for t in go_data.get(asp, [])[:top_k]:
                    term = t.get("term", "").strip()
                    if term:
                        terms.add(term)

        return sorted(terms)

    def extract_go_ids(
        self,
        results: dict[str, dict],
        aspect: Literal["BP", "MF", "CC", "all"] = "all",
        top_k: int = 10,
    ) -> list[str]:
        """从批量查询结果中提取 GO ID 列表 (去重)."""
        aspects = ["BP", "MF", "CC"] if aspect == "all" else [aspect]
        ids: set[str] = set()

        for gene, data in results.items():
            go_data = data.get("go", {})
            for asp in aspects:
                for t in go_data.get(asp, [])[:top_k]:
                    go_id = t.get("id", "").strip()
                    if go_id:
                        ids.add(go_id)

        return sorted(ids)

    def summarize_for_judge(
        self,
        gene_symbols: list[str],
        aspect: Literal["BP", "MF", "CC", "all"] = "BP",
        top_k_per_gene: int = 3,
    ) -> str:
        """生成供 LLM judge 使用的 GO 证据摘要文本.

        输出格式:
          "DE 基因关联功能 (Gene Ontology):
           - CD3D: T cell activation, alpha-beta T cell differentiation
           - FOXP3: regulatory T cell differentiation, immune response
           ..."
        """
        results = self.query_genes(gene_symbols)
        lines = ["DE 基因关联功能 (Gene Ontology):"]

        for gene in gene_symbols:
            data = results.get(gene, {})
            go_data = data.get("go", {})
            aspects = ["BP", "MF", "CC"] if aspect == "all" else [aspect]

            gene_terms: list[str] = []
            for asp in aspects:
                for t in go_data.get(asp, [])[:top_k_per_gene]:
                    term = t.get("term", "").strip()
                    if term:
                        gene_terms.append(term)

            if gene_terms:
                lines.append(f"  - {gene}: {', '.join(gene_terms)}")
            else:
                lines.append(f"  - {gene}: (未找到 GO 注释)")

        return "\n".join(lines)
