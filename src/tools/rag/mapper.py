"""基因名称 / 细胞类型名称 / 组织名称 / GO 功能术语 标准化映射器.

核心职责:
  1. 将各数据库原始的基因名统一为 HGNC/MGI approved symbol
  2. 将各数据库原始的细胞类型名统一为 Cell Ontology ID (CL:XXXXXXX)
  3. 将各数据库原始的组织名统一为 Uberon ID (UBERON:XXXXXXX)
  4. 将功能描述文本统一为 Gene Ontology ID (GO:XXXXXXX)
  5. 提供跨库对齐：不同数据库对同一细胞类型/组织的不同写法映射到同一 ID

依赖:
  - rag/mappers/gene_symbol_mapper.json    (由 scripts/build_gene_mapper.py 构建)
  - rag/mappers/cell_type_to_cl.json       (由 scripts/build_cell_type_mapper.py 构建)
  - rag/mappers/tissue_to_uberon.json      (由 scripts/build_tissue_mapper.py 构建)
  - rag/mappers/function_to_go.json        (由 scripts/build_go_mapper.py 构建)

使用方式:
  from src.tools.rag.mapper import Mapper
  mapper = Mapper(config_path="config/rag_sources.yaml")
  mapper.normalize_gene("p53", species="human")      # -> "TP53"
  mapper.normalize_cell_type("T cell")               # -> "CL:0000084"
  mapper.normalize_tissue("liver")                   # -> "UBERON:0002107"
  mapper.normalize_function("cell cycle")            # -> "GO:0007049"
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


class GeneMapper:
    """基因名称标准化: 原始基因名 -> HGNC/MGI approved symbol."""

    def __init__(self, mapper_json_path: str | Path | None = None):
        self._species_maps: dict[str, dict] = {}

        path = Path(mapper_json_path) if mapper_json_path else None
        if path and path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            self._species_maps = {
                species: data.get(species, {})
                for species in ("human", "mouse")
            }

    def normalize(self, gene_name: str, species: str = "human") -> str | None:
        """将原始基因名标准化为 approved symbol.

        匹配策略:
          1. 直接匹配 approved symbol (大小写不敏感)
          2. 查询 alias/prev symbol 映射
          3. 若均未匹配，返回原始输入
        """
        if not gene_name or not isinstance(gene_name, str):
            return None

        gene_clean = gene_name.strip()
        if not gene_clean:
            return None

        maps = self._get_species_map(species)
        approved = maps.get("approved", {})
        alias = maps.get("alias_to_approved", {})
        ensembl = maps.get("ensembl_to_approved", {})
        entrez = maps.get("entrez_to_approved", {})

        ensembl_key = self._strip_ensembl_version(gene_clean)
        if ensembl_key in ensembl:
            return ensembl[ensembl_key]
        upper_ensembl = ensembl_key.upper()
        if upper_ensembl in ensembl:
            return ensembl[upper_ensembl]

        if gene_clean.isdigit() and gene_clean in entrez:
            return entrez[gene_clean]

        # 如果已经是标准形式，直接返回
        upper = gene_clean.upper()
        if upper in approved:
            return approved[upper]

        lower = gene_clean.lower()
        if lower in approved:
            return approved[lower]

        # alias 映射
        if upper in alias:
            return alias[upper]
        if lower in alias:
            return alias[lower]

        # 未匹配到，返回原始输入（保留大小写）
        return gene_clean

    @staticmethod
    def _strip_ensembl_version(gene_name: str) -> str:
        """Convert ENSG00000141510.18 to ENSG00000141510."""
        return re.sub(r"^((?:ENS[A-Z]*G|ENSG|ENSMUSG)\d+)\.\d+$", r"\1", gene_name.strip(), flags=re.IGNORECASE)

    def _get_species_map(self, species: str) -> dict:
        return self._species_maps.get(species, self._species_maps.get("human", {}))

    def detect_id_type(self, gene_name: str, species: str = "human") -> str:
        """Classify a gene identifier for DE audit columns."""
        if not gene_name or not isinstance(gene_name, str):
            return "unknown"
        text = gene_name.strip()
        if re.match(r"^(?:ENS[A-Z]*G|ENSG)\d+(?:\.\d+)?$", text, flags=re.IGNORECASE):
            return "ensembl"
        if text.isdigit():
            return "entrez"
        normalized = self.normalize(text, species=species)
        if normalized and normalized != text:
            return "alias"
        if self.is_known(text, species=species):
            return "symbol"
        return "unknown"

    def is_known(self, gene_name: str, species: str = "human") -> bool:
        """检查基因名是否在映射表中."""
        if not gene_name:
            return False
        key = gene_name.strip().upper()
        ensembl_key = self._strip_ensembl_version(gene_name)
        maps = self._get_species_map(species)
        approved = maps.get("approved", {})
        alias = maps.get("alias_to_approved", {})
        ensembl = maps.get("ensembl_to_approved", {})
        entrez = maps.get("entrez_to_approved", {})
        return (
            key in approved
            or key in alias
            or key in ensembl
            or ensembl_key in ensembl
            or ensembl_key.upper() in ensembl
            or gene_name.strip() in entrez
        )

    def mouse_to_human_orthologs(self, mouse_gene: str) -> list[dict]:
        """Return human ortholog records for a mouse gene symbol/id."""
        mouse_symbol = self.normalize(mouse_gene, species="mouse")
        if not mouse_symbol:
            return []
        mouse_map = self._get_species_map("mouse")
        return mouse_map.get("mouse_to_human_ortholog", {}).get(mouse_symbol, [])

    def normalize_to_human(self, gene_name: str, species: str = "human", prefer_one_to_one: bool = True) -> str | None:
        """Normalize a gene to the human approved symbol used by downstream RAG/encoder."""
        if species == "human":
            return self.normalize(gene_name, species="human")
        if species != "mouse":
            return self.normalize(gene_name, species=species)

        records = self.mouse_to_human_orthologs(gene_name)
        if not records:
            return None
        if prefer_one_to_one:
            for record in records:
                if record.get("orthology_type") == "one_to_one":
                    return record.get("human_symbol")
        return records[0].get("human_symbol")

    def detect_species(self, gene_names: list[str]) -> str:
        """Heuristically detect whether a gene list is human, mouse, mixed, or unknown."""
        human_hits = 0
        mouse_hits = 0
        for gene in gene_names:
            text = str(gene).strip()
            if re.match(r"^ENSMUSG\d+(?:\.\d+)?$", text, flags=re.IGNORECASE):
                mouse_hits += 1
                continue
            if re.match(r"^ENSG\d+(?:\.\d+)?$", text, flags=re.IGNORECASE):
                human_hits += 1
                continue
            human_symbols = set(self._get_species_map("human").get("approved", {}).values())
            mouse_symbols = set(self._get_species_map("mouse").get("approved", {}).values())
            if text in human_symbols:
                human_hits += 1
            if text in mouse_symbols:
                mouse_hits += 1
        total = max(len(gene_names), 1)
        human_ratio = human_hits / total
        mouse_ratio = mouse_hits / total
        if human_ratio >= 0.5 and mouse_ratio >= 0.5:
            return "mixed"
        if human_ratio >= 0.5:
            return "human"
        if mouse_ratio >= 0.5:
            return "mouse"
        return "unknown"


class CellTypeMapper:
    """细胞类型名称标准化: 原始名称 -> Cell Ontology ID."""

    def __init__(self, mapper_json_path: str | Path | None = None):
        self._name_to_cl: dict[str, str] = {}
        self._cl_to_synonyms: dict[str, list[str]] = {}

        path = Path(mapper_json_path) if mapper_json_path else None
        if path and path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            self._name_to_cl = data.get("name_to_cl", {})
            self._cl_to_synonyms = data.get("cl_to_synonyms", {})

    def normalize(self, name: str) -> str | None:
        """将原始细胞类型名称标准化为 CL ID.

        匹配策略:
          1. 若输入已是 CL: 格式，直接返回
          2. 查映射表 (大小写不敏感)
          3. 去除标点后的模糊匹配
        """
        if not name or not isinstance(name, str):
            return None

        name_clean = name.strip()
        if not name_clean:
            return None

        # 已是 CL ID 格式
        if name_clean.upper().startswith("CL:"):
            return name_clean.upper()

        # 精确匹配 (lower case)
        lower = name_clean.lower()
        if lower in self._name_to_cl:
            return self._name_to_cl[lower]

        # 去除标点的模糊匹配
        simplified = re.sub(r"[^\w\s]", "", lower).strip()
        if simplified in self._name_to_cl:
            return self._name_to_cl[simplified]

        # 去除尾缀 s 的单复数匹配 (如 "t cells" -> "t cell")
        if simplified.endswith("s") and len(simplified) > 1:
            singular = simplified[:-1].strip()
            if singular in self._name_to_cl:
                return self._name_to_cl[singular]

        return None

    def synonyms(self, cl_id: str) -> list[str]:
        """获取 CL ID 对应的所有同义词 (含主名)."""
        return self._cl_to_synonyms.get(cl_id, [])

    def is_known(self, name: str) -> bool:
        """检查细胞类型名是否在映射表中."""
        return self.normalize(name) is not None


class FunctionMapper:
    """GO 功能术语标准化: 文本描述 -> GO ID."""

    def __init__(self, mapper_json_path: str | Path | None = None):
        self._name_to_go: dict[str, str] = {}
        self._go_to_synonyms: dict[str, list[str]] = {}

        path = Path(mapper_json_path) if mapper_json_path else None
        if path and path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            self._name_to_go = data.get("name_to_go", {})
            self._go_to_synonyms = data.get("go_to_synonyms", {})

    def normalize(self, text: str) -> str | None:
        """将功能描述文本标准化为 GO ID."""
        if not text or not isinstance(text, str):
            return None

        text_clean = text.strip().lower()
        if not text_clean:
            return None

        # 已是 GO ID 格式
        if text_clean.upper().startswith("GO:"):
            return text_clean.upper()

        # 精确匹配
        if text_clean in self._name_to_go:
            return self._name_to_go[text_clean]

        # 去除标点的模糊匹配
        simplified = re.sub(r"[^\w\s]", "", text_clean).strip()
        if simplified in self._name_to_go:
            return self._name_to_go[simplified]

        return None

    def search(self, text: str) -> list[str]:
        """模糊搜索：返回所有包含该文本的 GO term ID 列表."""
        if not text:
            return []

        query = text.lower()
        results: list[str] = []
        for name, go_id in self._name_to_go.items():
            if query in name:
                results.append(go_id)
        return list(dict.fromkeys(results))  # 去重保持顺序

    def synonyms(self, go_id: str) -> list[str]:
        """获取 GO ID 对应的所有同义词 (含主名)."""
        return self._go_to_synonyms.get(go_id, [])


class TissueMapper:
    """组织名称标准化: 原始名称 -> Uberon ID."""

    def __init__(self, mapper_json_path: str | Path | None = None):
        self._name_to_id: dict[str, str] = {}
        self._id_to_synonyms: dict[str, list[str]] = {}

        path = Path(mapper_json_path) if mapper_json_path else None
        if path and path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            self._name_to_id = data.get("name_to_id", {})
            self._id_to_synonyms = data.get("id_to_synonyms", {})

    def normalize(self, name: str) -> str | None:
        """将原始组织名称标准化为 UBERON ID.

        匹配策略:
          1. 若输入已是 UBERON: 格式，直接返回
          2. 查映射表 (大小写不敏感)
          3. 去除标点后的模糊匹配
          4. 去除尾缀 s 的单复数匹配
        """
        if not name or not isinstance(name, str):
            return None

        name_clean = name.strip()
        if not name_clean:
            return None

        # 已是 UBERON ID 格式
        if name_clean.upper().startswith("UBERON:"):
            return name_clean.upper()

        # 精确匹配 (lower case)
        lower = name_clean.lower()
        if lower in self._name_to_id:
            return self._name_to_id[lower]

        # 去除标点的模糊匹配
        simplified = re.sub(r"[^\w\s]", "", lower).strip()
        if simplified in self._name_to_id:
            return self._name_to_id[simplified]

        # 去除尾缀 s 的单复数匹配 (如 "kidneys" -> "kidney")
        if simplified.endswith("s") and len(simplified) > 1:
            singular = simplified[:-1].strip()
            if singular in self._name_to_id:
                return self._name_to_id[singular]

        return None

    def synonyms(self, uberon_id: str) -> list[str]:
        """获取 UBERON ID 对应的所有同义词 (含主名)."""
        return self._id_to_synonyms.get(uberon_id, [])

    def is_known(self, name: str) -> bool:
        """检查组织名是否在映射表中."""
        return self.normalize(name) is not None


class Mapper:
    """统一映射器入口，聚合基因 / 细胞类型 / 组织 / GO 功能四类标准化."""

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}

        # 基因映射
        gene_json = config.get("gene_mapper_json")
        self.gene = GeneMapper(gene_json)

        # 细胞类型映射
        cell_type_json = config.get("cell_type_mapper_json")
        self.cell_type = CellTypeMapper(cell_type_json)

        # 组织映射
        tissue_json = config.get("tissue_mapper_json")
        self.tissue = TissueMapper(tissue_json)

        # GO 功能映射
        go_json = config.get("go_mapper_json")
        self.function = FunctionMapper(go_json)

    def normalize_gene(self, gene_name: str, species: str = "human") -> str | None:
        return self.gene.normalize(gene_name, species=species)

    def normalize_gene_to_human(self, gene_name: str, species: str = "human") -> str | None:
        return self.gene.normalize_to_human(gene_name, species=species)

    def mouse_to_human_orthologs(self, mouse_gene: str) -> list[dict]:
        return self.gene.mouse_to_human_orthologs(mouse_gene)

    def detect_gene_species(self, gene_names: list[str]) -> str:
        return self.gene.detect_species(gene_names)

    def detect_gene_id_type(self, gene_name: str, species: str = "human") -> str:
        return self.gene.detect_id_type(gene_name, species=species)

    def normalize_cell_type(self, name: str) -> str | None:
        return self.cell_type.normalize(name)

    def normalize_tissue(self, name: str) -> str | None:
        return self.tissue.normalize(name)

    def normalize_function(self, text: str) -> str | None:
        return self.function.normalize(text)

    def cell_type_synonyms(self, cl_id: str) -> list[str]:
        return self.cell_type.synonyms(cl_id)

    def tissue_synonyms(self, uberon_id: str) -> list[str]:
        return self.tissue.synonyms(uberon_id)

    def function_synonyms(self, go_id: str) -> list[str]:
        return self.function.synonyms(go_id)

    def cross_reference_cell_type(self, name: str, source: str = "") -> str | None:
        """跨库细胞类型对齐：不同数据库写法 -> 统一 CL ID."""
        return self.cell_type.normalize(name)

    def cross_reference_tissue(self, name: str, source: str = "") -> str | None:
        """跨库组织对齐：不同数据库写法 -> 统一 UBERON ID."""
        return self.tissue.normalize(name)

    def get_unmapped_genes(self, gene_list: list[str], species: str = "human") -> list[str]:
        """返回无法映射到 approved symbol 的基因名列表（用于日志和补录）."""
        return [g for g in gene_list if not self.gene.is_known(g)]

    def get_unmapped_cell_types(self, names: list[str]) -> list[str]:
        """返回无法映射到 CL ID 的细胞类型名列表（用于日志和补录）."""
        return [n for n in names if not self.cell_type.is_known(n)]

    def get_unmapped_tissues(self, names: list[str]) -> list[str]:
        """返回无法映射到 UBERON ID 的组织名列表（用于日志和补录）."""
        return [n for n in names if not self.tissue.is_known(n)]
