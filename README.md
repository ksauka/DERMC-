# DERMC: Document-Enhanced Retrieval with Multi-level Contextual Ranking

> **Hierarchy-aware framework for automatic knowledge graph construction from structured documents**

---

## Overview

This repository provides an experimental hierarchy-aware framework for automatic knowledge graph construction from **simulated** hierarchical knowledge documents for proof of concept purposes. It builds on the ERD+R paradigm **[1]**. The system, **DERMC** (`Document Processing`, `Extraction`, `Resolution`, `Merging`, `Canonicalization`), extends **[1]** by adding functionalities such as:

- Multilingual coreference resolution
- Hierarchy-aware parsing for structured documents
- Similarity-based triple refinement
- Domain-specific ontology alignment

The framework is currently optimized for markdown-structured documents and aligns extracted triples with domain-specific ontologies.

### Pipeline Stages

1. **Document Processing** - Hierarchy-aware or flat parsing
2. **Extraction** - Open Information Extraction (OIE)
3. **Resolution** - Multilingual coreference resolution
4. **Merging** - Semantic similarity-based refinement
5. **Canonicalization** - Ontology-aligned KG construction

**NOTE**: For illustration purposes, the current experiments use **simulated datasets**: `E-Commerce_Documents.csv` and `domain_ontology.json`. The ontology used in examples is not a full production vocabulary.

**Planned Work**: More granular relation profiles (label + synonyms + definition + type cues), cross-lingual entity-type alignment, attribute-aware canonicalization, and support for full, multi-module ontologies.

### Key Features

- **Dual Parsing Modes**: Hierarchical (structure-aware) vs Flat (sequential)  
- **Multilingual Coreference**: Cross-lingual entity resolution  
- **Ontology Alignment**: Domain-specific schema integration  
- **Quality Filtering**: Adaptive semantic-based refinement  
- **Flexible LLM Backend**: Ollama (default, local) or OpenAI API  
- **GPU Acceleration**: Automatic CUDA detection for RTX GPUs  

---

## Installation

### Requirements
- Python 3.10 or higher
- pip or conda
- NVIDIA GPU (recommended, RTX 30/40/50 series with CUDA 12.x)
- Ollama (for local LLM backend - default)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd dermc

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with GPU support (CUDA 12.8 for CUDA 12.x GPUs)
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# Download spaCy model
python -m spacy download en_core_web_sm

# Install and configure Ollama (local LLM backend)
# 1. Download from https://ollama.ai
# 2. Pull model: ollama pull llama3.2

# Optional: Set OpenAI API key (if using OpenAI backend)
export OPENAI_API_KEY="your-api-key"
```

---

## Quick Start

### Basic Usage

```bash
# Run with hierarchical parsing
python -m dermc.main --mode hierarchical --single_mode_only

# Run with flat parsing
python -m dermc.main --mode flat --single_mode_only

# Run both modes and compare
python -m dermc.main --mode both
```

### Custom Dataset

```bash
python -m dermc.main \
  --input your_documents.csv \
  --ontology your_ontology.json \
  --output_dir your_output \
  --mode both
```

### LLM Backend Options

```bash
# Ollama (default - local, free, GPU-accelerated)
python -m dermc.main --llm_backend ollama --ollama_model llama3.2

# OpenAI (requires API key)
python -m dermc.main --llm_backend openai --openai_model gpt-4
```

---

## Package Structure

```
dermc/                          # Main package (importable)
├── __init__.py
├── main.py                     # CLI entry point
├── document_parser.py          # Hierarchical/flat parsing
├── hierarchy_aware_erdc.py     # Main pipeline orchestrator
├── llm_client.py               # LLM abstraction layer
├── schema_retriever.py         # Ontology alignment
└── coreference_resolver.py     # Multilingual coref resolution
```

---

## Output Structure

When you run DERMC, it generates the following outputs:

```
{output_dir}/
├── dermc_results_{mode}.csv              # Main pipeline results
├── oie_triplets_{mode}.csv              # Raw OIE extractions
├── refined_triplets_{mode}.csv          # Quality-filtered triplets
├── canonical_kg_{mode}.csv              # Final knowledge graph
├── knowledge_graph_{mode}.csv           # Alias for canonical KG
├── dermc_metrics_{mode}.json            # Performance metrics
└── dermc_comprehensive_log.json         # Detailed pipeline logs
```

**Mode** = `flat` or `hierarchical`

---

## Advanced Usage

### Import as Package

```python
from dermc.hierarchy_aware_erdc import HierarchyAwareERDC
from dermc.llm_client import LLMClient

# Initialize LLM
llm = LLMClient(backend='openai', model='gpt-3.5-turbo')

# Initialize pipeline
pipeline = HierarchyAwareERDC(
    llm=llm,
    ontology_path='domain_ontology.json',
    enable_hierarchical_parsing=True
)

# Process documents
results_df = pipeline.process_csv('documents.csv')
```

### Test Coreference Resolution

```bash
# Test coreference resolution module
python -m dermc.main --test
```

---

## Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | `E-Commerce_Documents.csv` | Input CSV file |
| `--ontology` | `domain_ontology.json` | Ontology file |
| `--output_dir` | `enhanced_dermc_output` | Output directory |
| `--mode` | `both` | Parsing mode: `flat`, `hierarchical`, or `both` |
| `--single_mode_only` | False | Run only one mode (use with --mode) |
| `--llm_backend` | `ollama` | LLM backend: `ollama` or `openai` |
| `--openai_model` | `gpt-3.5-turbo` | OpenAI model to use |
| `--ollama_model` | `llama3.2` | Ollama model to use |
| `--test` | False | Run coreference tests |

---

## Experiments

For comprehensive experiments, evaluations, and visualizations, see the **`experiments/`** folder:

- **Run experiments**: `cd experiments && python run_all_experiments.py`
- **Input datasets**: `experiments/inputs/` (simulated e-commerce documents)
- **Generated outputs**: `experiments/outputs/`
- **Evaluation scripts**: `experiments/scripts/evaluation/`
- **Visualization scripts**: `experiments/scripts/visualization/`
- **Detailed workflow**: See `experiments/README.md`

The experiments folder contains:
- Multiple test datasets (10, 20, 50, 200 simulated documents)
- TAA-inspired and Prometheus evaluation frameworks
- Automated visualization generators
- Quality analysis tools
- Complete experimental pipeline with GPU support

**Note**: All datasets are **simulated** for research purposes, not real company data.

---

## Input Format

### Documents CSV

```csv
Doc ID,Answer ID,Document
DOC001,ANS001,"# Product Return Policy
## Conditions
- Item must be unused
- Original packaging required
## Timeframe
Returns accepted within 30 days"
```

### Ontology JSON

```json
{
  "entity_types": {
    "Product": {
      "label": "Product",
      "definition": "A commercial product or service"
    }
  },
  "relations": {
    "has_policy": {
      "label": "has_policy",
      "definition": "Entity has an associated policy"
    }
  }
}
```

---

## Citation

This work has been presented at **ISWC 2026 Industrial Track**. If you use DERMC in your research, please cite:

```bibtex
@inproceedings{sauka2025building,
  title = {Building Hierarchy-Aware Knowledge Graphs: Ontology-Grounded Triple Extraction with LLMs},
  author = {Sauka, K. and Bardelloni, G. and Bulto, J. and Situmeang, F. B.},
  booktitle = {Proceedings of the International Semantic Web Conference (ISWC) - Industrial Track},
  year = {2025},
  note = {Presented at ISWC 2026}
}
```

**For more information**, refer to:

Sauka, K., Bardelloni, G., Bulto, J., & Situmeang, F. B. (2025). *Building Hierarchy-Aware Knowledge Graphs: Ontology-Grounded Triple Extraction with LLMs*. ISWC 2026 Industrial Track.

### Acknowledgments

This work builds upon:

[1] Zhang, B., & Soh, H. (2024). Extract, define, canonicalize: An llm-based framework for knowledge graph construction. [arXiv:2404.03868](https://arxiv.org/abs/2404.03868)

---

## Research Context

**Status**: Experimental / Proof of Concept  
**Domain**: E-commerce policy documents (simulated data for research)  
**GPU Support**: Optimized for NVIDIA RTX 30/40/50 series with CUDA 12.x  
**Note**: The provided ontology (`domain_ontology.json`) is illustrative, not production-ready

**Optimizations**:
- GPU-accelerated schema retrieval with SentenceTransformers
- FastCoref for multilingual coreference resolution (GPU-enabled)
- PEP8 compliance and DRY principles applied
- Professional logging without excessive visual clutter

**Ongoing Work**:
- Granular relation profiles (labels + synonyms + definitions)
- Cross-lingual entity-type alignment
- Attribute-aware canonicalization
- Multi-module ontology support

---

## Contact

For questions about DERMC methodology or implementation, please open an issue or refer to the documentation in the `experiments/` folder.

---

**Project Structure**:
- `/dermc/` - Main package (importable Python modules)
- `/experiments/` - Self-contained experimental workspace
  - `/inputs/` - Test datasets (simulated documents)
  - `/outputs/` - All generated results and visualizations
  - `/scripts/` - Evaluation and visualization scripts
  - `run_all_experiments.py` - Master execution script
- `/requirements.txt` - Python dependencies
- `/domain_ontology.json` - Sample ontology
- `/E-Commerce_Documents.csv` - Sample dataset
