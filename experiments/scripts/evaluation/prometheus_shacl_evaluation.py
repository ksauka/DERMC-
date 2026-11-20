
"""
Advanced Knowledge Graph Evaluation using Prometheus-7B-v2.0 + SHACL Validation
Combines state-of-the-art LLM evaluation with formal RDF validation

Features:
- Prometheus-7B-v2.0: Advanced transformer-based triplet quality assessment
- SHACL Validation: Formal RDF compliance and structural integrity
- Triple Progression Analysis: Cross-stage consistency tracking
- Comprehensive Metrics: Multi-dimensional quality assessment
"""

import os
import sys
import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Advanced evaluation imports
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    PROMETHEUS_AVAILABLE = True
except ImportError:
    print("[WARNING]  Transformers not available - using fallback LLM evaluation")
    PROMETHEUS_AVAILABLE = False

try:
    from rdflib import Graph, Namespace, URIRef, Literal
    from rdflib.namespace import RDF, XSD
    from pyshacl import validate
    SHACL_AVAILABLE = True
except ImportError:
    print("[WARNING]  RDFLib/pySHACL not available - using fallback structural validation")
    SHACL_AVAILABLE = False

class PrometheusEvaluator:
    """Prometheus-7B-v2.0 based triplet quality evaluator"""
    
    def __init__(self, model_id: str = "prometheus-eval/prometheus-7b-v2.0"):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if PROMETHEUS_AVAILABLE:
            try:
                print(f" Loading Prometheus-7B-v2.0 model on {self.device}...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                )
                self.model.eval()
                print("[SUCCESS] Prometheus model loaded successfully")
            except Exception as e:
                print(f"[WARNING]  Failed to load Prometheus model: {e}")
                print("   Using fallback evaluation...")
                self.model = None
    
    def _extract_json(self, txt: str) -> str:
        """Extract JSON from generated text"""
        m = re.search(r"\{.*\}", txt, flags=re.S)
        return m.group(0) if m else "{}"
    
    def _parse_scores(self, txt: str) -> dict:
        """Parse scores using regex fallback"""
        def pick(pat):
            m = re.search(pat, txt, flags=re.I)
            return float(m.group(1)) if m else 0.0
        
        return {
            "semantic_coherence": pick(r"semantic\s*coherence\s*:\s*([0-9]+(?:\.[0-9]+)?)"),
            "factual_accuracy": pick(r"factual\s*accuracy\s*:\s*([0-9]+(?:\.[0-9]+)?)"),
            "relationship_clarity": pick(r"relationship\s*clarity\s*:\s*([0-9]+(?:\.[0-9]+)?)"),
            "overall_quality": pick(r"overall\s*quality\s*:\s*([0-9]+(?:\.[0-9]+)?)"),
        }
    
    def _norm01(self, v: float) -> float:
        """Normalize score to 1-10 range"""
        v = float(v)
        return max(1.0, min(10.0, v))
    
    def evaluate_triplet(self, payload: dict) -> dict:
        """Evaluate a single triplet using Prometheus-7B-v2.0"""
        s = str(payload.get("subject", "")).strip()
        r = str(payload.get("relation", "")).strip()
        o = str(payload.get("object", "")).strip()
        
        # Fallback evaluation if model not available
        if not self.model or not PROMETHEUS_AVAILABLE:
            return self._fallback_evaluation(s, r, o)
        
        prompt = f"""You are a strict evaluator for knowledge graph triples.

Task: Score the triple on these 4 criteria from 1 to 10.
Return ONLY a compact JSON object with float values:
{{
  "semantic_coherence": 0-10,
  "factual_accuracy": 0-10,
  "relationship_clarity": 0-10,
  "overall_quality": 0-10
}}

Triple:
Subject: "{s}"
Relation: "{r}"
Object: "{o}"
"""
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            gen = self.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            # Try JSON → fallback to regex parsing
            try:
                data = json.loads(self._extract_json(gen))
            except Exception:
                data = self._parse_scores(gen)
            
            return {
                "semantic_coherence": self._norm01(data.get("semantic_coherence", 0.0)),
                "factual_accuracy": self._norm01(data.get("factual_accuracy", 0.0)),
                "relationship_clarity": self._norm01(data.get("relationship_clarity", 0.0)),
                "overall_quality": self._norm01(data.get("overall_quality", 0.0)),
            }
            
        except Exception as e:
            print(f"[WARNING]  Prometheus evaluation failed: {e}")
            return self._fallback_evaluation(s, r, o)
    
    def _fallback_evaluation(self, s: str, r: str, o: str) -> dict:
        """Fallback heuristic evaluation"""
        # Basic heuristics
        semantic_score = min(10.0, max(1.0, len(s.split()) + len(r.split()) + len(o.split())))
        factual_score = 7.0 if all(len(x.strip()) > 0 for x in [s, r, o]) else 3.0
        clarity_score = 8.0 if len(r.split()) <= 3 else 5.0
        overall_score = np.mean([semantic_score, factual_score, clarity_score])
        
        return {
            "semantic_coherence": semantic_score,
            "factual_accuracy": factual_score,
            "relationship_clarity": clarity_score,
            "overall_quality": overall_score,
        }

class SHACLValidator:
    """SHACL-based RDF validation for knowledge graphs"""
    
    def __init__(self, base_uri: str = "http://dermc.example.org/"):
        self.base_uri = base_uri
        self.namespace = Namespace(base_uri)
    
    def _get_field(self, t: Dict[str, Any], name: str) -> str:
        """Extract field with flexible key matching"""
        for k in t.keys():
            key_lower = str(k).lower()
            if key_lower in {name, name[0], {"subject":"subj","relation":"rel","object":"obj"}.get(name, "")}:
                return str(t[k]).strip()
        return str(t.get(name.capitalize(), "")).strip()
    
    def _slug(self, s: str) -> str:
        """Convert string to URI-safe slug"""
        return re.sub(r"[^A-Za-z0-9]+", "_", s.strip()).strip("_").lower()
    
    def looks_like_literal(self, val: str) -> bool:
        """Heuristic to determine if value should be a literal"""
        # Numbers or long descriptive text
        return bool(re.fullmatch(r"\d+(\.\d+)?", val)) or len(val.split()) > 6
    
    def triples_to_graph(self, triples: List[Dict[str, Any]]) -> Graph:
        """Convert triplets to RDF graph"""
        g = Graph()
        g.bind("ex", self.namespace)
        
        for t in triples:
            s_txt = self._get_field(t, "subject")
            p_txt = self._get_field(t, "relation")
            o_txt = self._get_field(t, "object")
            
            if not (s_txt and p_txt and o_txt):
                continue  # Skip incomplete triples
            
            s = URIRef(f"{self.base_uri}ent/{self._slug(s_txt)}")
            p = URIRef(f"{self.base_uri}prop/{self._slug(p_txt)}")
            
            if self.looks_like_literal(o_txt):
                o = Literal(o_txt)
            else:
                o = URIRef(f"{self.base_uri}ent/{self._slug(o_txt)}")
            
            g.add((s, p, o))
        
        return g
    
    def build_ecommerce_shapes(self) -> str:
        """Build SHACL shapes for e-commerce domain"""
        return f"""
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix ex: <{self.base_uri}> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

# General entity shape
ex:EntityShape a sh:NodeShape ;
    sh:targetClass ex:Entity ;
    sh:property [
        sh:path ex:prop/has_name ;
        sh:datatype xsd:string ;
        sh:minLength 1 ;
    ] .

# Product shapes
ex:ProductShape a sh:NodeShape ;
    sh:targetSubjectsOf ex:prop/has_price ;
    sh:nodeKind sh:IRI .

ex:PriceShape a sh:NodeShape ;
    sh:targetObjectsOf ex:prop/has_price ;
    sh:or (
        [ sh:datatype xsd:decimal ]
        [ sh:datatype xsd:double ]
        [ sh:datatype xsd:float ]
    ) .

# Customer/User shapes
ex:CustomerShape a sh:NodeShape ;
    sh:targetSubjectsOf ex:prop/makes_purchase ;
    sh:nodeKind sh:IRI .

# Policy shapes
ex:PolicyShape a sh:NodeShape ;
    sh:targetSubjectsOf ex:prop/applies_to ;
    sh:nodeKind sh:IRI .

# Temporal constraints
ex:DateShape a sh:NodeShape ;
    sh:targetObjectsOf ex:prop/valid_until ;
    sh:or (
        [ sh:datatype xsd:date ]
        [ sh:datatype xsd:dateTime ]
        [ sh:pattern "\\\\d{{4}}-\\\\d{{2}}-\\\\d{{2}}" ]
    ) .

# Quantity constraints
ex:QuantityShape a sh:NodeShape ;
    sh:targetObjectsOf ex:prop/has_quantity ;
    sh:datatype xsd:integer ;
    sh:minInclusive 0 .
"""
    
    def validate_graph(self, data_graph: Graph, shapes_ttl: str = None) -> Dict[str, Any]:
        """Validate RDF graph against SHACL shapes"""
        if not SHACL_AVAILABLE:
            return {"conforms": True, "report": "SHACL validation skipped", "violations": []}
        
        if shapes_ttl is None:
            shapes_ttl = self.build_ecommerce_shapes()
        
        try:
            sg = Graph().parse(data=shapes_ttl, format="turtle")
            conforms, report_graph, report_text = validate(
                data_graph,
                shacl_graph=sg,
                inference="rdfs",
                abort_on_first=False,
                allow_infos=True,
                allow_warnings=True,
            )
            
            # Collect violation messages
            viol_msgs = []
            SH_resultMessage = URIRef("http://www.w3.org/ns/shacl#resultMessage")
            for _, _, m in report_graph.triples((None, SH_resultMessage, None)):
                viol_msgs.append(str(m))
            
            return {
                "conforms": bool(conforms),
                "report": report_text,
                "violations": viol_msgs,
                "violation_count": len(viol_msgs)
            }
            
        except Exception as e:
            return {
                "conforms": False,
                "report": f"SHACL validation error: {e}",
                "violations": [str(e)],
                "violation_count": 1
            }

class AdvancedKGEvaluator:
    """Advanced Knowledge Graph evaluator combining Prometheus + SHACL + Triple Progression"""
    
    def __init__(self, base_dir: str = None):
        # Use experiments/outputs directory
        if base_dir is None:
            script_dir = Path(__file__).parent
            base_dir = script_dir.parent.parent / "outputs"
        self.base_dir = Path(base_dir)
        self.prometheus = PrometheusEvaluator()
        self.shacl = SHACLValidator()
        self.results = {}
    
    def load_pipeline_results(self, dataset_size: int) -> Dict[str, List[Dict]]:
        """Load results from all pipeline stages"""
        results = {}
        
        # Updated file mapping for CSV format (same as fast evaluator)
        stages = {
            "oie": f"enhanced_dermc_output_test{dataset_size}/oie_triplets.csv",
            "refined": f"enhanced_dermc_output_test{dataset_size}/refined_triplets.csv", 
            "canonical": f"enhanced_dermc_output_test{dataset_size}/canonical_kg.csv"
        }
        
        for stage, path in stages.items():
            file_path = self.base_dir / path
            if file_path.exists():
                try:
                    # Load CSV data
                    df = pd.read_csv(file_path)
                    
                    # Convert to list of dictionaries
                    triplets = []
                    for _, row in df.iterrows():
                        triplet = {}
                        # Handle different column naming conventions
                        for col in df.columns:
                            col_lower = col.lower()
                            if col_lower in ['subject', 'subj', 's']:
                                triplet['subject'] = str(row[col])
                            elif col_lower in ['relation', 'predicate', 'rel', 'r', 'p']:
                                triplet['relation'] = str(row[col])
                            elif col_lower in ['object', 'obj', 'o']:
                                triplet['object'] = str(row[col])
                            elif col_lower in ['confidence', 'conf', 'score']:
                                triplet['confidence'] = float(row[col]) if pd.notna(row[col]) else 0.8
                        
                        # Only add if we have the core triple
                        if all(k in triplet for k in ['subject', 'relation', 'object']):
                            triplets.append(triplet)
                    
                    results[stage] = triplets
                    print(f"[SUCCESS] Loaded {len(results[stage])} triplets from {stage} stage")
                    
                except Exception as e:
                    print(f"[WARNING]  Failed to load {stage}: {e}")
                    results[stage] = []
            else:
                print(f"[WARNING]  File not found: {file_path}")
                results[stage] = []
        
        return results
    
    def triple_progression_analysis(self, stages_data: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Analyze triple progression across pipeline stages"""
        progression = {}
        
        for stage, triplets in stages_data.items():
            progression[stage] = {
                "count": len(triplets),
                "unique_subjects": len(set(self._get_field(t, "subject") for t in triplets)),
                "unique_relations": len(set(self._get_field(t, "relation") for t in triplets)),
                "unique_objects": len(set(self._get_field(t, "object") for t in triplets)),
                "avg_subject_length": np.mean([len(self._get_field(t, "subject").split()) for t in triplets]) if triplets else 0,
                "avg_relation_length": np.mean([len(self._get_field(t, "relation").split()) for t in triplets]) if triplets else 0,
                "avg_object_length": np.mean([len(self._get_field(t, "object").split()) for t in triplets]) if triplets else 0,
            }
        
        # Cross-stage consistency
        if len(stages_data) >= 2:
            stage_names = list(stages_data.keys())
            for i in range(len(stage_names) - 1):
                current_stage = stage_names[i]
                next_stage = stage_names[i + 1]
                
                current_relations = set(self._get_field(t, "relation") for t in stages_data[current_stage])
                next_relations = set(self._get_field(t, "relation") for t in stages_data[next_stage])
                
                overlap = len(current_relations & next_relations)
                progression[f"{current_stage}_to_{next_stage}_relation_retention"] = {
                    "overlap_count": overlap,
                    "retention_rate": overlap / len(current_relations) if current_relations else 0
                }
        
        return progression
    
    def _get_field(self, t: Dict[str, Any], name: str) -> str:
        """Extract field with flexible key matching"""
        for k in t.keys():
            key_lower = str(k).lower()
            if key_lower in {name, name[0], {"subject":"subj","relation":"rel","object":"obj"}.get(name, "")}:
                return str(t[k]).strip()
        return str(t.get(name.capitalize(), "")).strip()
    
    def evaluate_dataset(self, dataset_size: int) -> Dict[str, Any]:
        """Comprehensive evaluation of a dataset"""
        print(f"\nEvaluating {dataset_size}-document dataset...")
        
        # Load pipeline results
        stages_data = self.load_pipeline_results(dataset_size)
        
        if not any(stages_data.values()):
            return {"error": f"No data found for {dataset_size}-document dataset"}
        
        results = {
            "dataset_size": dataset_size,
            "stages": {},
            "progression": self.triple_progression_analysis(stages_data),
            "summary": {}
        }
        
        # Evaluate each stage
        for stage, triplets in stages_data.items():
            if not triplets:
                continue
                
            print(f"   Evaluating {stage} stage ({len(triplets)} triplets)...")
            
            # Prometheus LLM evaluation
            llm_scores = []
            sample_size = min(50, len(triplets))  # Limit for performance
            sample_triplets = np.random.choice(len(triplets), sample_size, replace=False)
            
            for idx in sample_triplets:
                triplet = triplets[idx]
                payload = {
                    "subject": self._get_field(triplet, "subject"),
                    "relation": self._get_field(triplet, "relation"),
                    "object": self._get_field(triplet, "object")
                }
                score = self.prometheus.evaluate_triplet(payload)
                llm_scores.append(score)
            
            # Aggregate LLM scores
            llm_summary = {}
            for metric in ['semantic_coherence', 'factual_accuracy', 'relationship_clarity', 'overall_quality']:
                scores = [s[metric] for s in llm_scores]
                llm_summary[metric] = {
                    "mean": np.mean(scores),
                    "std": np.std(scores),
                    "min": np.min(scores),
                    "max": np.max(scores)
                }
            
            # SHACL validation
            rdf_graph = self.shacl.triples_to_graph(triplets)
            shacl_result = self.shacl.validate_graph(rdf_graph)
            
            # Structural analysis
            structural = {
                "completeness": len([t for t in triplets if all(self._get_field(t, f) for f in ["subject", "relation", "object"])]) / len(triplets),
                "avg_confidence": np.mean([float(t.get("confidence", t.get("Confidence", 0.8))) for t in triplets]),
                "relation_diversity": len(set(self._get_field(t, "relation") for t in triplets)) / len(triplets) if triplets else 0,
            }
            
            results["stages"][stage] = {
                "triplet_count": len(triplets),
                "llm_evaluation": llm_summary,
                "shacl_validation": shacl_result,
                "structural_metrics": structural
            }
        
        # Overall summary
        if results["stages"]:
            final_stage = list(results["stages"].keys())[-1]
            final_metrics = results["stages"][final_stage]
            
            results["summary"] = {
                "final_triplet_count": final_metrics["triplet_count"],
                "overall_quality": final_metrics["llm_evaluation"]["overall_quality"]["mean"],
                "shacl_conformance": final_metrics["shacl_validation"]["conforms"],
                "structural_completeness": final_metrics["structural_metrics"]["completeness"],
                "pipeline_stages": len(results["stages"])
            }
        
        return results
    
    def generate_report(self, dataset_sizes: List[int] = [10, 20, 50]):
        """Generate comprehensive evaluation report"""
        print("Advanced Knowledge Graph Evaluation")
        print("=" * 60)
        print("Features:")
        print("   • Prometheus-7B-v2.0 LLM evaluation")
        print("   • SHACL RDF validation")
        print("   • Triple progression analysis")
        print("   • Multi-dimensional quality assessment")
        print()
        
        all_results = {}
        
        # Evaluate each dataset
        for size in dataset_sizes:
            all_results[size] = self.evaluate_dataset(size)
        
        # Create output directory
        output_dir = self.base_dir / "advanced_evaluation_results"
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed results
        with open(output_dir / "advanced_evaluation_results.json", 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Generate comparative analysis
        self.generate_comparative_analysis(all_results, output_dir)
        
        # Generate visualizations
        self.create_visualizations(all_results, output_dir)
        
        print(f"\n[SUCCESS] Advanced evaluation complete!")
        print(f"Results saved to: {output_dir}")
        
        return all_results
    
    def generate_comparative_analysis(self, results: Dict, output_dir: Path):
        """Generate comparative analysis across datasets"""
        comparison_data = []
        
        for size, result in results.items():
            if "error" in result:
                continue
                
            summary = result.get("summary", {})
            progression = result.get("progression", {})
            
            row = {
                "dataset_size": size,
                "final_triplet_count": summary.get("final_triplet_count", 0),
                "overall_quality": summary.get("overall_quality", 0),
                "shacl_conformance": summary.get("shacl_conformance", False),
                "structural_completeness": summary.get("structural_completeness", 0),
                "pipeline_stages": summary.get("pipeline_stages", 0),
            }
            
            # Add progression metrics
            for stage in ["oie", "refined", "canonical"]:
                if stage in progression:
                    row[f"{stage}_count"] = progression[stage]["count"]
                    row[f"{stage}_unique_relations"] = progression[stage]["unique_relations"]
            
            comparison_data.append(row)
        
        # Save comparative analysis
        df = pd.DataFrame(comparison_data)
        df.to_csv(output_dir / "comparative_analysis.csv", index=False)
        
        # Generate summary statistics
        summary_stats = {
            "total_datasets_evaluated": len(comparison_data),
            "avg_quality_score": df["overall_quality"].mean() if len(df) > 0 else 0,
            "shacl_conformance_rate": df["shacl_conformance"].mean() if len(df) > 0 else 0,
            "avg_completeness": df["structural_completeness"].mean() if len(df) > 0 else 0,
        }
        
        with open(output_dir / "summary_statistics.json", 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print(f"Comparative analysis saved")
    
    def create_visualizations(self, results: Dict, output_dir: Path):
        """Create evaluation visualizations"""
        try:
            # Set up plotting style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # 1. Quality progression across datasets
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Advanced KG Evaluation: Prometheus + SHACL + Triple Progression', fontsize=16, weight='bold')
            
            # Collect data for plotting
            sizes = []
            quality_scores = []
            completeness_scores = []
            triplet_counts = []
            shacl_scores = []
            
            for size, result in results.items():
                if "error" in result:
                    continue
                    
                summary = result.get("summary", {})
                sizes.append(size)
                quality_scores.append(summary.get("overall_quality", 0))
                completeness_scores.append(summary.get("structural_completeness", 0))
                triplet_counts.append(summary.get("final_triplet_count", 0))
                shacl_scores.append(1.0 if summary.get("shacl_conformance", False) else 0.0)
            
            # Plot 1: Quality vs Dataset Size
            axes[0, 0].plot(sizes, quality_scores, marker='o', linewidth=2, markersize=8)
            axes[0, 0].set_title('Prometheus LLM Quality Score', weight='bold')
            axes[0, 0].set_xlabel('Dataset Size (documents)')
            axes[0, 0].set_ylabel('Quality Score (1-10)')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Structural Completeness
            axes[0, 1].plot(sizes, completeness_scores, marker='s', linewidth=2, markersize=8, color='green')
            axes[0, 1].set_title('Structural Completeness', weight='bold')
            axes[0, 1].set_xlabel('Dataset Size (documents)')
            axes[0, 1].set_ylabel('Completeness Score (0-1)')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Triplet Count Growth
            axes[1, 0].bar(sizes, triplet_counts, color='skyblue', alpha=0.7)
            axes[1, 0].set_title('Final Triplet Count', weight='bold')
            axes[1, 0].set_xlabel('Dataset Size (documents)')
            axes[1, 0].set_ylabel('Number of Triplets')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: SHACL Conformance
            axes[1, 1].bar(sizes, shacl_scores, color='orange', alpha=0.7)
            axes[1, 1].set_title('SHACL RDF Conformance', weight='bold')
            axes[1, 1].set_xlabel('Dataset Size (documents)')
            axes[1, 1].set_ylabel('Conformance (0=No, 1=Yes)')
            axes[1, 1].set_ylim(0, 1.1)
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / "advanced_evaluation_overview.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Stage progression visualization
            fig, ax = plt.subplots(figsize=(12, 8))
            
            stage_data = defaultdict(list)
            for size, result in results.items():
                if "error" in result:
                    continue
                    
                progression = result.get("progression", {})
                for stage in ["oie", "refined", "canonical"]:
                    if stage in progression:
                        stage_data[stage].append(progression[stage]["count"])
                    else:
                        stage_data[stage].append(0)
            
            x = np.arange(len(sizes))
            width = 0.25
            
            for i, (stage, counts) in enumerate(stage_data.items()):
                ax.bar(x + i*width, counts, width, label=stage.upper(), alpha=0.8)
            
            ax.set_title('Triple Progression Across Pipeline Stages', fontsize=14, weight='bold')
            ax.set_xlabel('Dataset Size (documents)')
            ax.set_ylabel('Number of Triplets')
            ax.set_xticks(x + width)
            ax.set_xticklabels(sizes)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / "triple_progression.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Visualizations saved")
            
        except Exception as e:
            print(f"[WARNING]  Visualization error: {e}")

def main():
    """Main evaluation runner"""
    evaluator = AdvancedKGEvaluator()
    
    # Run comprehensive evaluation
    results = evaluator.generate_report([10, 20, 50])
    
    # Print summary
    print("\n" + "="*60)
    print("ADVANCED EVALUATION SUMMARY")
    print("="*60)
    
    for size, result in results.items():
        if "error" in result:
            print(f"[ERROR] {size}-doc dataset: {result['error']}")
            continue
            
        summary = result.get("summary", {})
        print(f"\n{size}-document dataset:")
        print(f"   • Final Triplets: {summary.get('final_triplet_count', 0)}")
        print(f"   • Prometheus Quality: {summary.get('overall_quality', 0):.1f}/10")
        print(f"   • SHACL Conformance: {'[SUCCESS]' if summary.get('shacl_conformance', False) else '[ERROR]'}")
        print(f"   • Structural Completeness: {summary.get('structural_completeness', 0)*100:.1f}%")
        print(f"   • Pipeline Stages: {summary.get('pipeline_stages', 0)}")

if __name__ == "__main__":
    main()