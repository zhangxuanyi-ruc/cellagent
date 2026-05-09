"""Cell Ontology + Gene Ontology OBO 解析.

官方来源:
  - CL: http://purl.obolibrary.org/obo/cl.obo
  - GO: http://purl.obolibrary.org/obo/go.obo

使用 obonet 替代 pronto，对 OBO 语法兼容性更好.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import obonet


class OntologyLoader:
    """CL 和 GO OBO 本体加载器."""

    def __init__(self, config: dict[str, Any]):
        self.cl_path = Path(config["cl_path"])
        self.go_path = Path(config["go_path"]) if config.get("go_path") else None
        self._cl_graph_obj: Any | None = None
        self._go_graph_obj: Any | None = None

    def _get_cl_graph(self) -> Any:
        if self._cl_graph_obj is None:
            self._cl_graph_obj = obonet.read_obo(self.cl_path)
        return self._cl_graph_obj

    def _get_go_graph(self) -> Any | None:
        if self._go_graph_obj is None and self.go_path:
            self._go_graph_obj = obonet.read_obo(self.go_path)
        return self._go_graph_obj

    def get_term_data(self, term_id: str) -> dict | None:
        """获取 term 的原始数据 dict."""
        graph = self._get_cl_graph() if term_id.startswith("CL:") else self._get_go_graph()
        if graph and term_id in graph.nodes:
            return graph.nodes[term_id]
        return None

    def get_name(self, term_id: str) -> str | None:
        """获取 term 的人类可读名称."""
        data = self.get_term_data(term_id)
        return data.get("name") if data else None

    def get_definition(self, term_id: str) -> str | None:
        """获取 term 的定义文本."""
        data = self.get_term_data(term_id)
        return data.get("def") if data else None

    def get_parent(self, term_id: str) -> str | None:
        """获取 term 的直接父类 ID.

        用于 3 轮 REACT 不通过时的 fallback 机制。
        """
        data = self.get_term_data(term_id)
        if not data:
            return None
        is_a = data.get("is_a", [])
        return is_a[0] if is_a else None

    def get_parents(self, term_id: str) -> list[str]:
        """获取 term 的所有父类 ID."""
        data = self.get_term_data(term_id)
        if not data:
            return []
        return data.get("is_a", [])

    def get_children(self, term_id: str) -> list[str]:
        """获取 term 的所有子类 ID."""
        graph = self._get_cl_graph() if term_id.startswith("CL:") else self._get_go_graph()
        if not graph:
            return []
        return [child for child in graph.predecessors(term_id)]

    def get_synonyms(self, term_id: str) -> list[str]:
        """获取 term 的同义词列表."""
        data = self.get_term_data(term_id)
        if not data:
            return []
        return data.get("synonym", [])

    def lookup_by_name(self, name: str, ontology: str = "cl") -> str | None:
        """通过名称查找 term ID.

        Args:
            name: 细胞类型名称
            ontology: "cl" 或 "go"
        """
        graph = self._get_cl_graph() if ontology == "cl" else self._get_go_graph()
        if not graph:
            return None

        target = name.lower().strip()
        for term_id, data in graph.nodes(data=True):
            if ontology == "cl" and not term_id.startswith("CL:"):
                continue
            if ontology == "go" and not term_id.startswith("GO:"):
                continue

            # 匹配主名
            term_name = data.get("name", "").lower()
            if term_name == target:
                return term_id

            # 匹配同义词
            for syn in data.get("synonym", []):
                # synonym 格式: "text" EXACT []
                syn_text = syn.split('"')[1] if '"' in syn else syn
                if syn_text.lower() == target:
                    return term_id

        return None

    def query(self, cell_type: str, cl_id: str | None = None) -> list[dict]:
        """查询细胞类型的功能信息."""
        graph = self._get_cl_graph()

        if cl_id and cl_id in graph.nodes:
            term_id = cl_id
            data = graph.nodes[term_id]
        else:
            term_id = self.lookup_by_name(cell_type, ontology="cl")
            if term_id is None:
                return []
            data = graph.nodes[term_id]

        parents = self.get_parents(term_id)
        synonyms = self.get_synonyms(term_id)

        # 从 CL 的 relationship 或 xref 中尝试提取 GO terms
        go_terms: list[str] = []
        for rel in data.get("relationship", []):
            if "capable_of" in rel or "part_of" in rel:
                # relationship 格式如: capable_of(GO:0008283)
                import re
                go_match = re.search(r'(GO:\d+)', rel)
                if go_match:
                    go_terms.append(go_match.group(1))

        return [
            {
                "cell_type_cl_id": term_id,
                "cell_type_name": data.get("name", cell_type),
                "definition": data.get("def"),
                "go_terms": go_terms,
                "parents": parents,
                "synonyms": synonyms,
                "source": "cell_ontology",
            }
        ]
