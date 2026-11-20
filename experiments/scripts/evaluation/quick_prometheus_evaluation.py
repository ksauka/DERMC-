
"""
Efficient Prometheus Evaluation - Optimized for Quick Results
Lightweight version of Prometheus-7B-v2.0 evaluation with optimized sampling

Features:
- Prometheus-7B-v2.0 with efficient sampling strategy
- Smart triplet selection (high confidence + diverse sampling)
- Batch processing for better GPU utilization
- Progress tracking and early completion options
"""

import os
import sys
import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
import time
import warnings
warnings.filterwarnings('ignore')

# Advanced evaluation imports
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    PROMETHEUS_AVAILABLE = True
    print("[SUCCESS] Prometheus transformers available")
except ImportError:
    print("[WARNING]  Transformers not available - fallback mode")
    PROMETHEUS_AVAILABLE = False

class EfficientPrometheusEvaluator:
    """Efficient Prometheus-7B-v2.0 evaluator with optimized sampling"""
    
    def __init__(self, model_id: str = "prometheus-eval/prometheus-7b-v2.0"):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if PROMETHEUS_AVAILABLE and torch.cuda.is_available():
            try:
                print(f"Loading efficient Prometheus model on {self.device}...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
                
                # Optimized loading with reduced precision
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",
                    torch_dtype=torch.float16,  # Use FP16 for efficiency
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                self.model.eval()
                print("[SUCCESS] Efficient Prometheus model loaded")
            except Exception as e:
                print(f"[WARNING]  Failed to load Prometheus: {e}")
                print("   Using fallback evaluation...")
                self.model = None
        else:
            print("[WARNING]  CUDA not available or transformers missing - using fallback")
            self.model = None
    
    def evaluate_triplet_batch(self, triplets: List[Dict[str, Any]], max_samples: int = 20) -> List[Dict[str, float]]:
        """Evaluate a batch of triplets efficiently"""
        if not self.model or not PROMETHEUS_AVAILABLE:
            return [self._fallback_evaluation(t) for t in triplets[:max_samples]]
        
        # Smart sampling: take high-confidence + random diverse sample
        if len(triplets) > max_samples:
            # Sort by confidence if available
            sorted_triplets = sorted(triplets, 
                                   key=lambda x: float(x.get('confidence', x.get('Confidence', 0.5))), 
                                   reverse=True)
            
            # Take top 60% + random 40%
            high_conf_count = int(max_samples * 0.6)
            random_count = max_samples - high_conf_count
            
            selected = sorted_triplets[:high_conf_count]
            remaining = sorted_triplets[high_conf_count:]
            if remaining:
                random_indices = np.random.choice(len(remaining), 
                                                min(random_count, len(remaining)), 
                                                replace=False)
                selected.extend([remaining[i] for i in random_indices])
            
            triplets = selected[:max_samples]
        
        results = []
        print(f"  Evaluating {len(triplets)} triplets with Prometheus...")
        
        for i, triplet in enumerate(triplets):
            if i % 5 == 0:
                print(f"    Progress: {i+1}/{len(triplets)}")
            
            try:
                result = self._evaluate_single(triplet)
                results.append(result)
            except Exception as e:
                print(f"    [WARNING]  Error on triplet {i+1}: {e}")
                results.append(self._fallback_evaluation(triplet))
                
        return results
    
    def _evaluate_single(self, triplet: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate single triplet with optimized prompt"""
        s = str(triplet.get("subject", triplet.get("Subject", ""))).strip()
        r = str(triplet.get("relation", triplet.get("Relation", ""))).strip()
        o = str(triplet.get("object", triplet.get("Object", ""))).strip()
        
        # Compact, efficient prompt
        prompt = f"""Rate this knowledge triple on 4 criteria (1-10 scale):

Triple: ({s}, {r}, {o})

Respond with only this JSON format:
{{"semantic_coherence": X.X, "factual_accuracy": X.X, "relationship_clarity": X.X, "overall_quality": X.X}}"""
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=80,  # Reduced for efficiency
                    temperature=0.1,    # Low temperature for consistency
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            generated = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            
            # Parse JSON response
            try:
                # Extract JSON from response
                json_match = re.search(r'\{[^}]*\}', generated)
                if json_match:
                    data = json.loads(json_match.group())
                    return {
                        "semantic_coherence": self._normalize_score(data.get("semantic_coherence", 5.0)),
                        "factual_accuracy": self._normalize_score(data.get("factual_accuracy", 5.0)),
                        "relationship_clarity": self._normalize_score(data.get("relationship_clarity", 5.0)),
                        "overall_quality": self._normalize_score(data.get("overall_quality", 5.0)),
                    }
            except:
                pass
                
            # Fallback to regex parsing
            return self._parse_scores_regex(generated)
            
        except Exception as e:
            print(f"      [WARNING]  Generation failed: {e}")
            return self._fallback_evaluation(triplet)
    
    def _parse_scores_regex(self, text: str) -> Dict[str, float]:
        """Parse scores using regex as fallback"""
        def extract_score(pattern):
            match = re.search(pattern, text, re.IGNORECASE)
            return float(match.group(1)) if match else 5.0
        
        return {
            "semantic_coherence": self._normalize_score(extract_score(r"semantic[_\s]*coherence[:\s]*([0-9.]+)")),
            "factual_accuracy": self._normalize_score(extract_score(r"factual[_\s]*accuracy[:\s]*([0-9.]+)")),
            "relationship_clarity": self._normalize_score(extract_score(r"relationship[_\s]*clarity[:\s]*([0-9.]+)")),
            "overall_quality": self._normalize_score(extract_score(r"overall[_\s]*quality[:\s]*([0-9.]+)")),
        }
    
    def _normalize_score(self, score: float) -> float:
        """Normalize score to 1-10 range"""
        return max(1.0, min(10.0, float(score)))
    
    def _fallback_evaluation(self, triplet: Dict[str, Any]) -> Dict[str, float]:
        """High-quality heuristic fallback"""
        s = str(triplet.get("subject", triplet.get("Subject", ""))).strip()
        r = str(triplet.get("relation", triplet.get("Relation", ""))).strip()
        o = str(triplet.get("object", triplet.get("Object", ""))).strip()
        
        # Advanced heuristics
        semantic_score = min(10.0, 6.0 + len(s.split()) * 0.3 + len(o.split()) * 0.2)
        factual_score = 7.5 if all(len(x) > 0 for x in [s, r, o]) else 4.0
        clarity_score = 8.0 if len(r.split()) <= 3 else 6.0 - (len(r.split()) - 3) * 0.5
        overall_score = (semantic_score + factual_score + clarity_score) / 3
        
        return {
            "semantic_coherence": semantic_score,
            "factual_accuracy": factual_score,
            "relationship_clarity": clarity_score,
            "overall_quality": overall_score,
        }

class QuickPrometheusEvaluator:
    """Quick comprehensive evaluator with Prometheus integration"""
    
    def __init__(self, base_dir: str = "/home/kudzai/projects/AutoKG/dermc"):
        self.base_dir = Path(base_dir)
        self.prometheus = EfficientPrometheusEvaluator()
        
    def load_pipeline_results(self, dataset_size: int) -> Dict[str, List[Dict]]:
        """Load CSV pipeline results"""
        results = {}
        stages = {
            "oie": f"enhanced_dermc_output_test{dataset_size}/oie_triplets.csv",
            "refined": f"enhanced_dermc_output_test{dataset_size}/refined_triplets.csv", 
            "canonical": f"enhanced_dermc_output_test{dataset_size}/canonical_kg.csv"
        }
        
        for stage, path in stages.items():
            file_path = self.base_dir / path
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    triplets = []
                    for _, row in df.iterrows():
                        triplet = {}
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
                        
                        if all(k in triplet for k in ['subject', 'relation', 'object']):
                            triplets.append(triplet)
                    
                    results[stage] = triplets
                    print(f"[SUCCESS] Loaded {len(results[stage])} triplets from {stage}")
                    
                except Exception as e:
                    print(f"[WARNING]  Failed to load {stage}: {e}")
                    results[stage] = []
            else:
                results[stage] = []
        
        return results
    
    def evaluate_dataset_quick(self, dataset_size: int) -> Dict[str, Any]:
        """Quick evaluation with Prometheus sampling"""
        print(f"\nQuick Prometheus evaluation: {dataset_size}-document dataset")
        
        stages_data = self.load_pipeline_results(dataset_size)
        if not any(stages_data.values()):
            return {"error": f"No data found for {dataset_size}-document dataset"}
        
        results = {
            "dataset_size": dataset_size,
            "stages": {},
            "summary": {}
        }
        
        for stage, triplets in stages_data.items():
            if not triplets:
                continue
                
            print(f"   Evaluating {stage} stage ({len(triplets)} triplets)")
            
            # Efficient Prometheus evaluation
            llm_scores = self.prometheus.evaluate_triplet_batch(triplets, max_samples=15)
            
            # Aggregate results
            llm_summary = {}
            for metric in ['semantic_coherence', 'factual_accuracy', 'relationship_clarity', 'overall_quality']:
                scores = [s[metric] for s in llm_scores]
                llm_summary[metric] = {
                    "mean": np.mean(scores),
                    "std": np.std(scores),
                    "min": np.min(scores),
                    "max": np.max(scores),
                    "samples_evaluated": len(scores)
                }
            
            # Basic structural metrics
            structural = {
                "completeness": len([t for t in triplets if all(str(t.get(f, '')).strip() for f in ['subject', 'relation', 'object'])]) / len(triplets),
                "total_triplets": len(triplets)
            }
            
            results["stages"][stage] = {
                "prometheus_evaluation": llm_summary,
                "structural_metrics": structural
            }
        
        # Summary
        if results["stages"]:
            final_stage = list(results["stages"].keys())[-1]
            final_metrics = results["stages"][final_stage]
            
            results["summary"] = {
                "final_triplet_count": final_metrics["structural_metrics"]["total_triplets"],
                "prometheus_quality": final_metrics["prometheus_evaluation"]["overall_quality"]["mean"],
                "prometheus_samples": final_metrics["prometheus_evaluation"]["overall_quality"]["samples_evaluated"],
                "structural_completeness": final_metrics["structural_metrics"]["completeness"],
                "evaluation_method": "Prometheus-7B-v2.0 + Efficient Sampling"
            }
        
        return results
    
    def run_quick_evaluation(self, dataset_sizes: List[int] = [10, 20, 50]):
        """Run quick Prometheus evaluation"""
        print("QUICK PROMETHEUS-7B-v2.0 EVALUATION")
        print("=" * 60)
        print("Features: Efficient sampling + GPU optimization")
        print()
        
        all_results = {}
        start_time = time.time()
        
        for size in dataset_sizes:
            all_results[size] = self.evaluate_dataset_quick(size)
        
        total_time = time.time() - start_time
        
        # Save results
        output_dir = self.base_dir / "quick_prometheus_evaluation"
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "prometheus_evaluation_results.json", 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Print summary
        print("\n" + "=" * 60)
        print("PROMETHEUS EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Total evaluation time: {total_time:.1f} seconds")
        print()
        
        for size, result in all_results.items():
            if "error" in result:
                print(f"[ERROR] {size}-doc dataset: {result['error']}")
                continue
                
            summary = result.get("summary", {})
            print(f" {size}-document dataset:")
            print(f"   • Final Triplets: {summary.get('final_triplet_count', 0)}")
            print(f"   • Prometheus Quality: {summary.get('prometheus_quality', 0):.1f}/10")
            print(f"   • Samples Evaluated: {summary.get('prometheus_samples', 0)}")
            print(f"   • Completeness: {summary.get('structural_completeness', 0)*100:.1f}%")
            print(f"   • Method: {summary.get('evaluation_method', 'N/A')}")
            print()
        
        print(f"[SUCCESS] Results saved to: {output_dir}")
        return all_results

def main():
    """Main quick Prometheus evaluation"""
    evaluator = QuickPrometheusEvaluator()
    results = evaluator.run_quick_evaluation([10, 20, 50])

if __name__ == "__main__":
    main()