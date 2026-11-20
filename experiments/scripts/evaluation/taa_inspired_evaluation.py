
"""
TAA-Inspired Triplet Validation Framework
=========================================

Based on the paper: "Triples Accuracy Assessment (TAA): An Automatic Approach 
for Validating RDF Triples in Knowledge Graphs"

This implementation adapts TAA principles for DERMC pipeline evaluation:
1. Triplet structural validation
2. Entity-relation compatibility assessment  
3. Cross-stage consistency analysis
4. Domain-specific plausibility scoring

Key insight: Evaluate triplets as structured knowledge representations,
not as pseudo-sentences (which was methodologically incorrect).

"""

import os
import json
import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Tuple, Any, Set
import logging
from pathlib import Path
from collections import defaultdict, Counter
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TAAInspiredValidator:
    """TAA-inspired triplet validation for DERMC pipeline"""
    
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.model_name = "llama3.2"
        
    def validate_triplet_structure(self, triplets: List[Dict]) -> Dict[str, Any]:
        """Validate structural integrity of triplets (TAA principle 1)"""
        logger.info(f"Validating structural integrity of {len(triplets)} triplets...")
        
        structural_metrics = {
            'total_triplets': len(triplets),
            'complete_triplets': 0,
            'incomplete_triplets': 0,
            'empty_subjects': 0,
            'empty_relations': 0,
            'empty_objects': 0,
            'malformed_relations': 0,
            'valid_entity_types': 0,
            'structural_quality_score': 0.0
        }
        
        for triplet in triplets:
            subject = str(triplet.get('Subject', '')).strip()
            relation = str(triplet.get('Relation', '')).strip()
            obj = str(triplet.get('Object', '')).strip()
            
            # Check completeness
            if subject and relation and obj:
                structural_metrics['complete_triplets'] += 1
            else:
                structural_metrics['incomplete_triplets'] += 1
                if not subject:
                    structural_metrics['empty_subjects'] += 1
                if not relation:
                    structural_metrics['empty_relations'] += 1
                if not obj:
                    structural_metrics['empty_objects'] += 1
            
            # Check relation format (should be meaningful, not just underscores)
            if relation and not re.match(r'^[_]+$', relation):
                if '_' in relation or relation.islower():
                    # Acceptable relation format
                    pass
                else:
                    structural_metrics['malformed_relations'] += 1
            
            # Check entity types if available
            subj_type = triplet.get('Subject Type', '')
            obj_type = triplet.get('Object Type', '')
            if subj_type or obj_type:
                structural_metrics['valid_entity_types'] += 1
        
        # Calculate structural quality score
        if structural_metrics['total_triplets'] > 0:
            completeness_ratio = structural_metrics['complete_triplets'] / structural_metrics['total_triplets']
            structural_metrics['structural_quality_score'] = completeness_ratio
        
        return structural_metrics
    
    def assess_entity_relation_compatibility(self, triplets: List[Dict]) -> Dict[str, Any]:
        """Assess if entities and relations are semantically compatible (TAA principle 2)"""
        logger.info(f"Assessing entity-relation compatibility for {len(triplets)} triplets...")
        
        # Check if LLM is available
        try:
            requests.get(f"{self.ollama_url}/api/tags", timeout=2)
        except:
            raise ConnectionError("LLM server not available")
        
        # Sample for efficiency
        sample_size = min(50, len(triplets))
        sample_triplets = np.random.choice(triplets, sample_size, replace=False).tolist()
        
        compatibility_scores = []
        relation_patterns = defaultdict(list)
        
        for triplet in sample_triplets:
            subject = str(triplet.get('Subject', '')).strip()
            relation = str(triplet.get('Relation', '')).strip()
            obj = str(triplet.get('Object', '')).strip()
            
            if subject and relation and obj:
                # Analyze relation patterns
                relation_patterns[relation].append((subject, obj))
                
                # Get compatibility score
                score = self._assess_compatibility(subject, relation, obj)
                if score is not None:
                    compatibility_scores.append(score)
        
        # Analyze relation consistency
        relation_consistency = self._analyze_relation_consistency(relation_patterns)
        
        return {
            'sample_size': sample_size,
            'evaluated_triplets': len(compatibility_scores),
            'mean_compatibility': np.mean(compatibility_scores) if compatibility_scores else 0.0,
            'median_compatibility': np.median(compatibility_scores) if compatibility_scores else 0.0,
            'std_compatibility': np.std(compatibility_scores) if compatibility_scores else 0.0,
            'relation_consistency': relation_consistency,
            'unique_relations': len(relation_patterns),
            'compatibility_scores': compatibility_scores
        }
    
    def _assess_compatibility(self, subject: str, relation: str, obj: str) -> float:
        """Assess semantic compatibility of subject-relation-object"""
        try:
            # TAA-inspired compatibility assessment
            prompt = f"""Evaluate the semantic compatibility of this knowledge triplet:

Subject: "{subject}"
Relation: "{relation}"
Object: "{obj}"

Assessment criteria:
1. Semantic coherence: Do these entities logically connect via this relation?
2. Domain appropriateness: Is this relation appropriate for these entity types?
3. Factual plausibility: Could this relationship exist in the real world?

Rate compatibility on a scale of 1-10:
- 9-10: Highly compatible and plausible
- 7-8: Generally compatible with minor issues
- 5-6: Somewhat compatible but questionable
- 3-4: Low compatibility, likely incorrect
- 1-2: Incompatible, definitely incorrect

Provide only the numerical score."""

            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get('message', {}).get('content', '').strip()
                
                try:
                    score = float(content.split()[0])
                    return score if 1 <= score <= 10 else None
                except:
                    return None
            else:
                return None
                
        except Exception as e:
            logger.warning(f"Error assessing compatibility: {e}")
            return None
    
    def _analyze_relation_consistency(self, relation_patterns: Dict[str, List[Tuple]]) -> Dict[str, Any]:
        """Analyze consistency of relation usage patterns"""
        consistency_analysis = {
            'total_relations': len(relation_patterns),
            'consistent_relations': 0,
            'inconsistent_relations': 0,
            'relation_usage_stats': {}
        }
        
        for relation, pairs in relation_patterns.items():
            usage_count = len(pairs)
            
            # Analyze subject and object patterns for this relation
            subjects = [pair[0] for pair in pairs]
            objects = [pair[1] for pair in pairs]
            
            subject_diversity = len(set(subjects)) / len(subjects) if subjects else 0
            object_diversity = len(set(objects)) / len(objects) if objects else 0
            
            consistency_analysis['relation_usage_stats'][relation] = {
                'usage_count': usage_count,
                'subject_diversity': subject_diversity,
                'object_diversity': object_diversity,
                'sample_pairs': pairs[:3]  # Store first 3 examples
            }
            
            # Simple consistency heuristic
            if usage_count >= 2 and (subject_diversity > 0.5 or object_diversity > 0.5):
                consistency_analysis['consistent_relations'] += 1
            else:
                consistency_analysis['inconsistent_relations'] += 1
        
        return consistency_analysis
    
    def cross_stage_consistency_analysis(self, oie_triplets: List[Dict], 
                                       refined_triplets: List[Dict], 
                                       canonical_triplets: List[Dict]) -> Dict[str, Any]:
        """Analyze consistency across pipeline stages (TAA principle 3)"""
        logger.info("Analyzing cross-stage consistency...")
        
        # Extract relation sets from each stage
        oie_relations = set(str(t.get('Relation', t.get('relation', ''))).strip() 
                           for t in oie_triplets if t.get('Relation', t.get('relation', '')))
        refined_relations = set(str(t.get('Relation', t.get('relation', ''))).strip() 
                              for t in refined_triplets if t.get('Relation', t.get('relation', '')))
        canonical_relations = set(str(t.get('Relation', t.get('relation', ''))).strip() 
                                for t in canonical_triplets if t.get('Relation', t.get('relation', '')))
        
        # Analyze relation evolution
        oie_to_refined_preserved = len(oie_relations.intersection(refined_relations))
        refined_to_canonical_preserved = len(refined_relations.intersection(canonical_relations))
        oie_to_canonical_preserved = len(oie_relations.intersection(canonical_relations))
        
        # Calculate consistency metrics
        consistency_analysis = {
            'oie_relations_count': len(oie_relations),
            'refined_relations_count': len(refined_relations),
            'canonical_relations_count': len(canonical_relations),
            
            'oie_to_refined_preservation_rate': oie_to_refined_preserved / len(oie_relations) if oie_relations else 0,
            'refined_to_canonical_preservation_rate': refined_to_canonical_preserved / len(refined_relations) if refined_relations else 0,
            'oie_to_canonical_preservation_rate': oie_to_canonical_preserved / len(oie_relations) if oie_relations else 0,
            
            'relations_lost_in_refinement': len(oie_relations) - oie_to_refined_preserved,
            'relations_lost_in_canonicalization': len(refined_relations) - refined_to_canonical_preserved,
            
            'new_relations_in_refined': len(refined_relations - oie_relations),
            'new_relations_in_canonical': len(canonical_relations - refined_relations),
            
            'stage_triplet_counts': {
                'oie': len(oie_triplets),
                'refined': len(refined_triplets),
                'canonical': len(canonical_triplets)
            }
        }
        
        return consistency_analysis
    
    def domain_plausibility_assessment(self, triplets: List[Dict], domain: str = "e-commerce") -> Dict[str, Any]:
        """Assess domain-specific plausibility (TAA principle 4)"""
        logger.info(f"Assessing {domain} domain plausibility for {len(triplets)} triplets...")
        
        # Sample for efficiency
        sample_size = min(30, len(triplets))
        sample_triplets = np.random.choice(triplets, sample_size, replace=False).tolist()
        
        plausibility_scores = []
        domain_relevance_scores = []
        
        for triplet in sample_triplets:
            subject = str(triplet.get('Subject', '')).strip()
            relation = str(triplet.get('Relation', '')).strip()
            obj = str(triplet.get('Object', '')).strip()
            
            if subject and relation and obj:
                plausibility_score = self._assess_domain_plausibility(subject, relation, obj, domain)
                if plausibility_score is not None:
                    plausibility_scores.append(plausibility_score)
        
        return {
            'domain': domain,
            'sample_size': sample_size,
            'evaluated_triplets': len(plausibility_scores),
            'mean_plausibility': np.mean(plausibility_scores) if plausibility_scores else 0.0,
            'median_plausibility': np.median(plausibility_scores) if plausibility_scores else 0.0,
            'std_plausibility': np.std(plausibility_scores) if plausibility_scores else 0.0,
            'high_plausibility_count': sum(1 for score in plausibility_scores if score >= 7),
            'low_plausibility_count': sum(1 for score in plausibility_scores if score <= 4)
        }
    
    def _assess_domain_plausibility(self, subject: str, relation: str, obj: str, domain: str) -> float:
        """Assess plausibility within specific domain context"""
        try:
            prompt = f"""Evaluate the plausibility of this knowledge triplet within the {domain} domain:

Subject: "{subject}"
Relation: "{relation}"
Object: "{obj}"

Context: This triplet was extracted from {domain} policy documents.

Assessment:
1. Domain relevance: Is this relationship relevant to {domain}?
2. Real-world plausibility: Does this relationship make sense in practice?
3. Extraction quality: Is this a meaningful relationship or extraction noise?

Rate plausibility on a scale of 1-10:
- 9-10: Highly plausible and domain-relevant
- 7-8: Plausible with good domain fit
- 5-6: Somewhat plausible but questionable
- 3-4: Low plausibility, likely extraction error
- 1-2: Implausible, definitely incorrect

Provide only the numerical score."""

            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get('message', {}).get('content', '').strip()
                
                try:
                    score = float(content.split()[0])
                    return score if 1 <= score <= 10 else None
                except:
                    return None
            else:
                return None
                
        except Exception as e:
            logger.warning(f"Error assessing domain plausibility: {e}")
            return None

class TAAInspiredEvaluator:
    """Main TAA-inspired evaluation orchestrator"""
    
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            base_dir = Path(__file__).parent.parent
        self.base_dir = Path(base_dir)
        self.validator = TAAInspiredValidator()
        
        # Create TAA-inspired output directory
        self.output_dir = self.base_dir / "taa_inspired_evaluation"
        self.output_dir.mkdir(exist_ok=True)
        
    def run_taa_inspired_evaluation(self):
        """Run comprehensive TAA-inspired evaluation"""
        logger.info("Starting TAA-inspired triplet validation...")
        
        # Look in outputs directory
        outputs_dir = self.base_dir.parent / "outputs"
        experiment_dirs = [
            outputs_dir / "enhanced_dermc_output_test10",
            outputs_dir / "enhanced_dermc_output_test20", 
            outputs_dir / "enhanced_dermc_output_test50"
        ]
        
        parsing_modes = ['flat', 'hierarchical']
        
        taa_results = {}
        
        for exp_path in experiment_dirs:
            if exp_path.exists():
                for mode in parsing_modes:
                    result_key = f"{exp_dir}_{mode}"
                    logger.info(f"TAA-inspired evaluation for: {result_key}")
                    results = self.evaluate_experiment_taa_style(exp_path, exp_dir, mode)
                    taa_results[result_key] = results
            else:
                logger.warning(f"Experiment directory not found: {exp_path}")
        
        # Create TAA-inspired comparative analysis
        self.create_taa_comparative_analysis(taa_results)
        
        logger.info("TAA-inspired evaluation completed!")
        return taa_results
    
    def evaluate_experiment_taa_style(self, exp_path: Path, exp_name: str, parsing_mode: str = 'flat') -> Dict[str, Any]:
        """Evaluate single experiment using TAA principles"""
        results = {
            'experiment_name': exp_name,
            'parsing_mode': parsing_mode,
            'dataset_size': int(exp_name.split('test')[1]) if 'test' in exp_name else 0,
            'evaluation_methodology': 'TAA-inspired'
        }
        
        try:
            # Load all pipeline stages - mode-specific files
            oie_triplets = self.load_triplets_safe(exp_path / f"oie_triplets_{parsing_mode}.csv")
            refined_triplets = self.load_triplets_safe(exp_path / f"refined_triplets_{parsing_mode}.csv")
            canonical_triplets = self.load_triplets_safe(exp_path / f"canonical_kg_{parsing_mode}.csv")
            
            if not canonical_triplets:
                logger.warning(f"No canonical triplets loaded for {exp_name}")
                return results
            
            # TAA Principle 1: Structural validation
            logger.info(f"Running structural validation for {exp_name}...")
            results['structural_validation'] = self.validator.validate_triplet_structure(canonical_triplets)
            
            # TAA Principle 2: Entity-relation compatibility (skip if LLM not available)
            try:
                logger.info(f"Assessing entity-relation compatibility for {exp_name}...")
                results['compatibility_assessment'] = self.validator.assess_entity_relation_compatibility(canonical_triplets)
            except Exception as e:
                logger.warning(f"Skipping compatibility assessment (LLM not available): {e}")
                results['compatibility_assessment'] = {
                    'mean_compatibility': 0.0,
                    'unique_relations': len(set(t.get('Relation', '') for t in canonical_triplets)),
                    'evaluated_triplets': 0,
                    'skipped': True
                }
            
            # TAA Principle 3: Cross-stage consistency (if all stages available)
            if oie_triplets and refined_triplets and canonical_triplets:
                logger.info(f"Analyzing cross-stage consistency for {exp_name}...")
                results['cross_stage_consistency'] = self.validator.cross_stage_consistency_analysis(
                    oie_triplets, refined_triplets, canonical_triplets)
            
            # TAA Principle 4: Domain plausibility (skip if LLM not available)
            try:
                logger.info(f"Assessing domain plausibility for {exp_name}...")
                results['domain_plausibility'] = self.validator.domain_plausibility_assessment(
                    canonical_triplets, "e-commerce")
            except Exception as e:
                logger.warning(f"Skipping domain plausibility (LLM not available): {e}")
                results['domain_plausibility'] = {
                    'domain': 'e-commerce',
                    'sample_size': 0,
                    'evaluated_triplets': 0,
                    'mean_plausibility': 0.0,
                    'skipped': True
                }
            
            # Save results
            results_file = self.output_dir / f"{exp_name}_taa_evaluation.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"TAA-inspired evaluation completed for {exp_name}")
            
        except Exception as e:
            logger.error(f"Error in TAA evaluation for {exp_name}: {e}")
            results['error'] = str(e)
        
        return results
    
    def load_triplets_safe(self, file_path: Path) -> List[Dict]:
        """Safely load triplets with flexible column handling"""
        if not file_path.exists():
            return []
            
        try:
            df = pd.read_csv(file_path)
            triplets = df.to_dict('records')
            logger.info(f"Loaded {len(triplets)} triplets from {file_path.name}")
            return triplets
        except Exception as e:
            logger.error(f"Error loading triplets from {file_path}: {e}")
            return []
    
    def create_taa_comparative_analysis(self, all_results: Dict[str, Any]):
        """Create comparative analysis using TAA-inspired metrics"""
        logger.info("Creating TAA-inspired comparative analysis...")
        
        comparison_data = []
        
        for exp_name, results in all_results.items():
            if 'error' in results:
                continue
                
            dataset_size = results.get('dataset_size', 0)
            
            row = {
                'experiment': exp_name,
                'dataset_size': dataset_size,
            }
            
            # Structural validation metrics
            if 'structural_validation' in results:
                sv = results['structural_validation']
                row.update({
                    'structural_quality_score': sv.get('structural_quality_score', 0),
                    'complete_triplets': sv.get('complete_triplets', 0),
                    'incomplete_triplets': sv.get('incomplete_triplets', 0),
                    'completeness_rate': sv.get('complete_triplets', 0) / sv.get('total_triplets', 1)
                })
            
            # Compatibility assessment metrics
            if 'compatibility_assessment' in results:
                ca = results['compatibility_assessment']
                row.update({
                    'mean_compatibility': ca.get('mean_compatibility', 0),
                    'unique_relations': ca.get('unique_relations', 0),
                    'compatibility_evaluated': ca.get('evaluated_triplets', 0)
                })
            
            # Cross-stage consistency metrics
            if 'cross_stage_consistency' in results:
                csc = results['cross_stage_consistency']
                row.update({
                    'oie_to_canonical_preservation': csc.get('oie_to_canonical_preservation_rate', 0),
                    'relations_lost_refinement': csc.get('relations_lost_in_refinement', 0),
                    'relations_lost_canonicalization': csc.get('relations_lost_in_canonicalization', 0)
                })
            
            # Domain plausibility metrics
            if 'domain_plausibility' in results:
                dp = results['domain_plausibility']
                row.update({
                    'mean_domain_plausibility': dp.get('mean_plausibility', 0),
                    'high_plausibility_count': dp.get('high_plausibility_count', 0),
                    'low_plausibility_count': dp.get('low_plausibility_count', 0)
                })
            
            comparison_data.append(row)
        
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            comparison_file = self.output_dir / "taa_inspired_flat_vs_hierarchical.csv"
            df_comparison.to_csv(comparison_file, index=False)
            
            logger.info(f"TAA-inspired flat vs hierarchical comparison saved to {comparison_file}")
            
            # Print summary
            self._print_taa_summary(df_comparison)
        
        else:
            logger.warning("No valid results for TAA-inspired comparative analysis")
    
    def _print_taa_summary(self, df: pd.DataFrame):
        """Print TAA-inspired evaluation summary"""
        print("\nTAA-INSPIRED TRIPLET VALIDATION RESULTS")
        print("=" * 55)
        print("Based on: 'Triples Accuracy Assessment (TAA): An Automatic")
        print("Approach for Validating RDF Triples in Knowledge Graphs'")
        print()
        
        print(" STRUCTURAL VALIDATION (TAA Principle 1)")
        print("-" * 45)
        for _, row in df.iterrows():
            size = int(row['dataset_size'])
            print(f"{size}-Document Dataset:")
            print(f"   • Structural Quality: {row['structural_quality_score']:.3f}/1.000")
            print(f"   • Complete Triplets: {int(row['complete_triplets']):,}")
            print(f"   • Completeness Rate: {row['completeness_rate']:.1%}")
            print()
        
        print("ENTITY-RELATION COMPATIBILITY (TAA Principle 2)")
        print("-" * 55)
        for _, row in df.iterrows():
            size = int(row['dataset_size'])
            print(f"{size}-Document Dataset:")
            print(f"   • Mean Compatibility: {row['mean_compatibility']:.2f}/10.0")
            print(f"   • Unique Relations: {int(row['unique_relations'])}")
            print(f"   • Evaluated Triplets: {int(row['compatibility_evaluated'])}")
            print()
        
        if 'oie_to_canonical_preservation' in df.columns:
            print("CROSS-STAGE CONSISTENCY (TAA Principle 3)")
            print("-" * 50)
            for _, row in df.iterrows():
                size = int(row['dataset_size'])
                print(f"{size}-Document Dataset:")
                print(f"   • OIE→Canonical Preservation: {row['oie_to_canonical_preservation']:.1%}")
                print(f"   • Relations Lost (Refinement): {int(row['relations_lost_refinement'])}")
                print(f"   • Relations Lost (Canonicalization): {int(row['relations_lost_canonicalization'])}")
                print()
        
        print(" DOMAIN PLAUSIBILITY (TAA Principle 4)")
        print("-" * 45)
        for _, row in df.iterrows():
            size = int(row['dataset_size'])
            print(f"{size}-Document Dataset:")
            print(f"   • Mean Plausibility: {row['mean_domain_plausibility']:.2f}/10.0")
            print(f"   • High Plausibility Count: {int(row['high_plausibility_count'])}")
            print(f"   • Low Plausibility Count: {int(row['low_plausibility_count'])}")
            print()
        
        print("[SUCCESS] METHODOLOGICAL ADVANTAGES")
        print("-" * 35)
        print("• Validates triplets as structured knowledge (not pseudo-sentences)")
        print("• Based on established TAA principles from literature")
        print("• Multi-dimensional assessment across 4 key principles")
        print("• Domain-aware evaluation for e-commerce context")
        print("• Cross-stage pipeline consistency analysis")

def main():
    """Main execution with TAA-inspired methodology"""
    print("TAA-INSPIRED TRIPLET VALIDATION FRAMEWORK")
    print("=" * 55)
    print("Implementing principles from:")
    print("'Triples Accuracy Assessment (TAA): An Automatic Approach") 
    print("for Validating RDF Triples in Knowledge Graphs'")
    print()
    print("This approach correctly validates triplets as structured")
    print("knowledge representations, not pseudo-sentences.")
    print()
    
    evaluator = TAAInspiredEvaluator()
    results = evaluator.run_taa_inspired_evaluation()
    
    return results

if __name__ == "__main__":
    main()