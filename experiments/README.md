# DERMC Experiments

> **Comprehensive experimental evaluation of DERMC framework comparing Flat vs Hierarchical parsing modes**

---

## Overview

This repository provides an experimental hierarchy-aware framework for automatic knowledge graph construction from **simulated** hierarchical knowledge documents for proof of concept purposes. It builds on the ERD+R paradigm **[1]**. The system, **DERMC** (`Document Processing`, `Extraction`, `Resolution`, `Merging`, `Canonicalization`), extends **[1]** by adding functionalities such as:

- Multilingual coreference resolution
- Hierarchy-aware parsing for structured documents
- Similarity-based triple refinement
- Domain-specific ontology alignment

The framework is currently optimized for markdown-structured documents and aligns extracted triples with domain-specific ontologies.

**NOTE**: For illustration purposes, the current experiments use **simulated datasets**: `E-Commerce_Documents.csv` and `domain_ontology.json`. All documents used in these experiments are **synthetic e-commerce customer service documents**, not real company files. The datasets simulate realistic business scenarios (return policies, warranties, refunds, shipping procedures) to validate DERMC's knowledge extraction capabilities in a controlled research environment.

### Experimental Pipeline

This folder contains the complete experimental pipeline for DERMC research, including:

- Multiple simulated test datasets (10, 20, 50, 200 documents)
- Dual evaluation frameworks (TAA + Prometheus)
- Automated experiment runners
- Visualization generators
- Quality analysis tools
- Complete experimental outputs

**Key Note**: All datasets in `inputs/` are artificially generated for research purposes. They represent realistic e-commerce documentation patterns but contain no proprietary or confidential business information.

**Planned Work**: More granular relation profiles (label + synonyms + definition + type cues), cross-lingual entity-type alignment, attribute-aware canonicalization, and support for full, multi-module ontologies.

---

## Quick Start

### Run Complete Pipeline

```bash
# From project root
cd experiments

# Run full pipeline (experiments + evaluations + visualizations)
python run_all_experiments.py

# Run specific datasets only
python run_all_experiments.py --datasets test10 test20

# Run only experiments (skip evaluations)
python run_all_experiments.py --experiments-only

# Run only visualizations
python run_all_experiments.py --viz-only
```

### Requirements

```bash
# Install PyTorch with CUDA support (for RTX 5070)
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# Install other dependencies
pip install transformers sentence-transformers spacy fastcoref scikit-learn openai httpx python-dotenv langdetect
python -m spacy download en_core_web_sm

# Ensure Ollama is running (for LLM backend)
ollama serve
ollama pull llama3.2
```

---

## Folder Structure

```
experiments/
├── inputs/                             # Input datasets
│   ├── E-Commerce_Documents_test10.csv
│   ├── E-Commerce_Documents_test20.csv
│   ├── E-Commerce_Documents_test50.csv
│   └── E-Commerce_Documents_test200.csv
│
├── outputs/                            # All generated results
│   ├── analysis_visuals/               # PDF visualizations
│   │   ├── DERMC_Quality_Flat_vs_Hierarchical.pdf
│   │   ├── DERMC_Triplet_Progression.pdf
│   │   └── quality_scores_combined.pdf
│   │
│   ├── evaluations/                    # Evaluation results
│   │   ├── comprehensive_flat_hierarchical_evaluation/
│   │   ├── true_taa_evaluation/
│   │   └── taa_inspired_evaluation/
│   │
│   ├── enhanced_dermc_output_test10/   # Experiment results
│   ├── enhanced_dermc_output_test20/
│   ├── enhanced_dermc_output_test50/
│   └── enhanced_dermc_output_test200/
│
├── scripts/                            # Processing scripts
│   ├── evaluation/                     # Evaluation scripts
│   │   ├── taa_inspired_evaluation.py
│   │   ├── true_taa_evaluation.py
│   │   ├── prometheus_shacl_evaluation.py
│   │   └── quick_prometheus_evaluation.py
│   │
│   ├── visualization/                  # Visualization scripts
│   │   ├── create_quality_visuals.py
│   │   ├── poster results.py
│   │   ├── corrected_dermc_analysis.py
│   │   └── dermc_performance_visualizer.py
│   │
│   └── utilities/                      # Utility scripts
│       ├── generate_synthetic_docs.py
│       ├── generate_rule_based_dataset.py
│       └── validators.py
│
├── data/                               # Manual assessments
│   ├── FLAT.xlsx
│   ├── Hierarchical.xlsx
│   └── Triple Progression.xlsx
│
├── run_all_experiments.py              # Master execution script
├── dermc_comprehensive_logger.py       # Logging utility
├── domain_ontology.json                # Domain ontology
├── requirements.txt                    # Python dependencies
├── .env.example                        # Configuration template
│
└── README.md                           # This file
```

---

## Experimental Pipeline

### Phase 1: Run Experiments

**Master Script**: `run_all_experiments.py` (recommended)

```bash
# Run experiments with Ollama backend
python run_all_experiments.py --experiments-only --datasets test10
```

**Input**: Test datasets from `inputs/`

**Output**: Triplets for each stage (OIE → Refined → Canonical) in both modes

**Generates**:
- `outputs/enhanced_dermc_output_test10/oie_triplets_{flat,hierarchical}.csv`
- `outputs/enhanced_dermc_output_test10/refined_triplets_{flat,hierarchical}.csv`
- `outputs/enhanced_dermc_output_test10/canonical_kg_{flat,hierarchical}.csv`
- `outputs/enhanced_dermc_output_test10/dermc_metrics_{flat,hierarchical}.json`

**LLM Backend**: Uses Ollama (llama3.2) by default - no API key needed!

**GPU Support**: Automatically uses NVIDIA GPU when available (RTX 5070 supported)

---

### Phase 2: Evaluation

#### 2A. TAA-Inspired Evaluation

**Script**: `scripts/evaluation/taa_inspired_evaluation.py`

**Method**: Llama 3.2-based semantic validation following TAA principles

```bash
# Automatically runs as part of full pipeline
python run_all_experiments.py

# Or run standalone
python scripts/evaluation/taa_inspired_evaluation.py
```

**Evaluates**:
1. **Structural Validation** - Triplet completeness
2. **Entity-Relation Compatibility** - Semantic coherence
3. **Cross-Stage Consistency** - Relation preservation
4. **Domain Plausibility** - E-commerce relevance

**Output**: `outputs/evaluations/true_taa_evaluation/taa_results_flat_vs_hierarchical.json`

---

#### 2B. Prometheus + SHACL Evaluation

**Script**: `scripts/evaluation/prometheus_shacl_evaluation.py`

**Method**: Prometheus-7B-v2.0 model + RDF structural validation

```bash
# Automatically runs as part of full pipeline
python run_all_experiments.py

# Or run standalone
python scripts/evaluation/prometheus_shacl_evaluation.py
```

**Evaluates**:
1. **Semantic Coherence** (1-10)
2. **Factual Accuracy** (1-10)
3. **Relationship Clarity** (1-10)
4. **Overall Quality** (average)

**Output**: `outputs/evaluations/comprehensive_flat_hierarchical_evaluation/prometheus_evaluation_results.json`

---

### Phase 3: Visualization

#### Generate Quality Visualizations

```bash
# Run all visualizations (part of full pipeline)
python run_all_experiments.py --viz-only

# Or run individual scripts
python scripts/visualization/create_quality_visuals.py
python scripts/visualization/poster\ results.py
python scripts/visualization/corrected_dermc_analysis.py
python scripts/visualization/dermc_performance_visualizer.py
```

**Output**: PDF charts in `outputs/analysis_visuals/`

---

## Key Results

### Winner: Hierarchical Parsing

| Metric | Flat | Hierarchical | Advantage |
|--------|------|--------------|-----------|
| **Mean Compatibility** | 6.31/10 | **7.12/10** | **+12.8%** |
| **Mean Plausibility** | 7.79/10 | **8.16/10** | **+4.7%** |
| **Structural Quality** | 1.00/1.00 | **1.00/1.00** | Equal |
| **Low-Quality Triplets** | 3 | **0** | **100% better** |

**Verdict**: Hierarchical parsing produces higher-quality knowledge graphs with superior semantic coherence and zero low-quality triplets.

---

### Coreference Model Warnings

If you see warnings like "Coreference model not available - returning original text":

**This is expected behavior** - the system gracefully falls back to processing without coreference resolution. The pipeline will continue working normally, just without pronoun/reference resolution.

**To enable coreference resolution** (optional):

```bash
# Install fastcoref if not already installed
pip install fastcoref

# First run will download model files (~500MB)
# This happens automatically when coreference is first used
# Subsequent runs will use cached models from coref/cache/
```

**Note**: Coreference resolution is computationally intensive and optional. The main DERMC pipeline (extraction, ontology alignment, canonicalization) works fully without it.


---

## Citation

This work has been presented at **ISWC 2026 Industrial Track**. If you use DERMC in your research, please cite:

```bibtex
@inproceedings{sauka2025building,
  title = {Building Hierarchy-Aware Knowledge Graphs: Ontology-Grounded Triple Extraction with LLMs},
  author = {Sauka, K. and Bardelloni, G. and Bulto, J. and Situmeang, F. B.},
  booktitle = {Proceedings of the International Semantic Web Conference (ISWC) - Industrial Track},
  year = {2025},
  note = {Presented at ISWC 2025}
}
```

**For more information**, refer to:

Sauka, K., Bardelloni, G., Bulto, J., & Situmeang, F. B. (2025). *Building Hierarchy-Aware Knowledge Graphs: Ontology-Grounded Triple Extraction with LLMs*. ISWC 2025 Industrial Track.

### Acknowledgments

This work builds upon:

[1] Zhang, B., & Soh, H. (2024). Extract, define, canonicalize: An llm-based framework for knowledge graph construction. [arXiv:2404.03868](https://arxiv.org/abs/2404.03868)


---

**Main Package**: See `/dermc` for the core DERMC package  
**This Folder**: Complete experimental pipeline and evaluation framework
