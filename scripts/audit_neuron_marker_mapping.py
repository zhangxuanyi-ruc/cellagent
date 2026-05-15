#!/usr/bin/env python
"""Audit where ALS neuron DE genes map in the marker mapper."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.rag import RAGFacade


def main() -> None:
    rag = RAGFacade("/root/wanghaoran/zxy/project/cellagent/config/rag_sources.yaml")
    mapper_path = Path("/root/wanghaoran/zxy/project/cellagent/rag/mappers/marker_celltype_to_genes.json")
    data = json.loads(mapper_path.read_text(encoding="utf-8"))
    de_genes = [
        "SYT1", "SNAP25", "CSMD1", "DLGAP1", "OPCML", "FGF14", "FGF12", "RIMS2", "LRRTM4", "NRXN1",
        "RBFOX1", "GABRB2", "GRM5", "NRG3", "KCNIP4", "RALYL", "MEG3", "BASP1", "KCNQ5", "CCSER1",
        "CADPS", "DPP10", "PCDH7", "CELF2", "CNTNAP2", "VSNL1", "MAP2", "CELF4", "SCN2A", "RGS7",
    ]
    queries = [
        "neuron",
        "central nervous system neuron",
        "interneuron",
        "pyramidal neuron",
        "glutamatergic neuron",
        "GABAergic neuron",
        "neural progenitor cell",
        "Purkinje cell",
        "retinal ganglion cell",
    ]
    print("mapper_n_cl", len(data))
    for cell_type in queries:
        cl_id = rag.mapper.normalize_cell_type(cell_type)
        record = data.get(cl_id or "")
        genes = set((record or {}).get("all_genes") or (record or {}).get("genes") or [])
        print("\nQUERY", cell_type, "CL", cl_id, "name", (record or {}).get("cell_type_name"), "n_genes", len(genes))
        print("hits", [gene for gene in de_genes if gene in genes])
        print("sources", (record or {}).get("sources"), "counts", (record or {}).get("source_counts"))

    print("\nREVERSE_LOOKUP")
    for gene in de_genes:
        cl_ids = rag.query_cell_types_for_gene(gene)
        names = [rag.get_cell_type_name(cl_id) for cl_id in cl_ids]
        print(gene, list(zip(cl_ids, names))[:30])


if __name__ == "__main__":
    main()
