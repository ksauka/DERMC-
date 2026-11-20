#!/usr/bin/env python3
"""
True TAA Implementation for DERMC Pipeline
==========================================

Implements actual TAA (Triples Accuracy Assessment) methodology:
1. Source triples: DERMC canonical output
2. Target triples: External knowledge graphs (DBpedia, Wikidata, etc.)
3. Consensus calculation: Agreement between source and target
4. Confidence scoring: Based on consensus strength

This addresses the limitation that our previous "TAA-inspired" approach
lacked external target triples for true consensus validation.

Author: DERMC TAA Implementation (True)
Date: September 2025
"""

import os
import json
import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Tuple, Any, Set
import logging
from pathlib import Path
from SPARQLWrapper import SPARQLWrapper, JSON
import time
from difflib import SequenceMatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExternalKGConnector:
    """Connect to external knowledge graphs for target triples"""
    
    def __init__(self):
        self.dbpedia_endpoint = "http://dbpedia.org/sparql"
        self.wikidata_endpoint = "https://query.wikidata.org/sparql"
        
    def query_dbpedia(self, entity: str, relation: str = None) -> List[Dict]:
        """Query DBpedia for related triples"""
        try:
            sparql = SPARQLWrapper(self.dbpedia_endpoint)
            
            if relation:
                # Query for specific relation
                query = f"""
                SELECT ?subject ?predicate ?object WHERE {{
                    ?subject ?predicate ?object .
                    FILTER (contains(lcase(str(?subject)), lcase("{entity}")) ||
                            contains(lcase(str(?object)), lcase("{entity}"))) .
                    FILTER (contains(lcase(str(?predicate)), lcase("{relation}"))) .
                }}
                LIMIT 50
                """
            else:
                # Query for any relations involving the entity
                query = f"""
                SELECT ?subject ?predicate ?object WHERE {{
                    ?subject ?predicate ?object .
                    FILTER (contains(lcase(str(?subject)), lcase("{entity}")) ||
                            contains(lcase(str(?object)), lcase("{entity}"))) .
                }}
                LIMIT 100
                """
            
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
            
            triples = []
            for result in results["results"]["bindings"]:
                triple = {
                    'subject': result['subject']['value'],
                    'relation': result['predicate']['value'],
                    'object': result['object']['value'],
                    'source': 'dbpedia'
                }
                triples.append(triple)
            
            return triples
            
        except Exception as e:
            logger.warning(f"Error querying DBpedia: {e}")
            return []
    
    def query_wikidata(self, entity: str) -> List[Dict]:
        """Query Wikidata for related triples"""
        try:
            sparql = SPARQLWrapper(self.wikidata_endpoint)
            
            query = f"""
            SELECT ?subject ?subjectLabel ?predicate ?predicateLabel ?object ?objectLabel WHERE {{
                ?subject ?predicate ?object .
                ?subject rdfs:label ?subjectLabel .
                ?predicate rdfs:label ?predicateLabel .
                ?object rdfs:label ?objectLabel .
                FILTER (contains(lcase(?subjectLabel), lcase("{entity}")) ||
                        contains(lcase(?objectLabel), lcase("{entity}"))) .
                FILTER (lang(?subjectLabel) = "en") .
                FILTER (lang(?predicateLabel) = "en") .
                FILTER (lang(?objectLabel) = "en") .
            }}
            LIMIT 50
            """
            
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
            
            triples = []
            for result in results["results"]["bindings"]:
                triple = {
                    'subject': result['subjectLabel']['value'],
                    'relation': result['predicateLabel']['value'],
                    'object': result['objectLabel']['value'],
                    'source': 'wikidata'
                }
                triples.append(triple)
            
            return triples
            
        except Exception as e:
            logger.warning(f"Error querying Wikidata: {e}")
            return []

class TrueTAAValidator:
    """Implements true TAA methodology with external target triples"""
    
    def __init__(self):
        self.kg_connector = ExternalKGConnector()
        self.similarity_threshold = 0.7
        
    def calculate_consensus_score(self, source_triples: List[Dict]) -> Dict[str, Any]:
        """Calculate TAA consensus scores against external KGs"""
        logger.info(f"Calculating TAA consensus for {len(source_triples)} source triples...")
        
        consensus_results = {
            'total_source_triples': len(source_triples),
            'external_matches_found': 0,
            'high_confidence_triples': 0,
            'medium_confidence_triples': 0,
            'low_confidence_triples': 0,
            'no_consensus_triples': 0,
            'average_confidence': 0.0,
            'triplet_confidence_scores': []
        }
        
        # Sample for efficiency (TAA on large datasets is computationally expensive)
        sample_size = min(20, len(source_triples))
        sample_triples = np.random.choice(source_triples, sample_size, replace=False).tolist()
        
        for i, source_triple in enumerate(sample_triples):
            if i % 5 == 0:
                logger.info(f"Processing source triplet {i+1}/{len(sample_triples)}")
            
            confidence_score = self._calculate_triplet_consensus(source_triple)
            consensus_results['triplet_confidence_scores'].append(confidence_score)
            
            # Categorize confidence levels
            if confidence_score >= 0.8:
                consensus_results['high_confidence_triples'] += 1
            elif confidence_score >= 0.5:
                consensus_results['medium_confidence_triples'] += 1
            elif confidence_score >= 0.2:
                consensus_results['low_confidence_triples'] += 1
            else:
                consensus_results['no_consensus_triples'] += 1
            
            if confidence_score > 0:
                consensus_results['external_matches_found'] += 1
            
            # Rate limiting to avoid overwhelming external APIs
            time.sleep(0.5)
        
        # Calculate average confidence
        if consensus_results['triplet_confidence_scores']:
            consensus_results['average_confidence'] = np.mean(consensus_results['triplet_confidence_scores'])
        
        return consensus_results
    
    def _calculate_triplet_consensus(self, source_triple: Dict) -> float:
        """Calculate consensus score for a single triplet"""
        try:
            subject = str(source_triple.get('Subject', '')).strip()
            relation = str(source_triple.get('Relation', '')).strip()
            obj = str(source_triple.get('Object', '')).strip()
            
            if not (subject and relation and obj):
                return 0.0
            
            # Extract key terms for external KG queries
            subject_terms = self._extract_key_terms(subject)
            object_terms = self._extract_key_terms(obj)
            
            # Query external KGs for target triples
            target_triples = []
            
            # Query DBpedia
            for term in subject_terms[:2]:  # Limit to avoid too many queries
                dbpedia_triples = self.kg_connector.query_dbpedia(term, relation)
                target_triples.extend(dbpedia_triples)
            
            # Query Wikidata  
            for term in object_terms[:2]:
                wikidata_triples = self.kg_connector.query_wikidata(term)
                target_triples.extend(wikidata_triples)
            
            if not target_triples:
                return 0.0
            
            # Calculate similarity with target triples
            max_similarity = 0.0
            for target_triple in target_triples:
                similarity = self._calculate_triple_similarity(source_triple, target_triple)
                max_similarity = max(max_similarity, similarity)
            
            return max_similarity
            
        except Exception as e:
            logger.warning(f"Error calculating consensus for triplet: {e}")
            return 0.0
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text for KG queries"""
        # Simple term extraction - could be enhanced with NER
        terms = []
        
        # Remove common e-commerce policy language
        clean_text = text.lower()
        stop_phrases = ['policy', 'document', 'agreement', 'terms', 'conditions']
        for phrase in stop_phrases:
            clean_text = clean_text.replace(phrase, '')
        
        # Extract meaningful terms
        words = clean_text.split()
        meaningful_words = [w for w in words if len(w) > 3 and w.isalpha()]
        
        return meaningful_words[:3]  # Return top 3 terms
    
    def _calculate_triple_similarity(self, source_triple: Dict, target_triple: Dict) -> float:
        """Calculate similarity between source and target triples"""
        try:
            # Extract components
            src_subj = str(source_triple.get('Subject', '')).lower()
            src_rel = str(source_triple.get('Relation', '')).lower()
            src_obj = str(source_triple.get('Object', '')).lower()
            
            tgt_subj = str(target_triple.get('subject', '')).lower()
            tgt_rel = str(target_triple.get('relation', '')).lower()
            tgt_obj = str(target_triple.get('object', '')).lower()
            
            # Calculate component similarities
            subj_sim = SequenceMatcher(None, src_subj, tgt_subj).ratio()
            rel_sim = SequenceMatcher(None, src_rel, tgt_rel).ratio()
            obj_sim = SequenceMatcher(None, src_obj, tgt_obj).ratio()
            
            # Weighted average (relations are most important for TAA)
            weighted_similarity = (0.3 * subj_sim) + (0.4 * rel_sim) + (0.3 * obj_sim)
            
            return weighted_similarity
            
        except Exception as e:
            logger.warning(f"Error calculating similarity: {e}")
            return 0.0

class TrueTAAEvaluator:
    """Main orchestrator for true TAA evaluation"""
    
    def __init__(self, base_dir: str = None):
        if base_dir is None:
            self.base_dir = Path(__file__).parent.parent
        else:
            self.base_dir = Path(base_dir)
        self.validator = TrueTAAValidator()
        
        # Create output directory
        self.output_dir = self.base_dir / "true_taa_evaluation"
        self.output_dir.mkdir(exist_ok=True)
        
    def run_true_taa_evaluation(self):
        """Run true TAA evaluation with external knowledge graphs"""
        logger.info("Starting TRUE TAA evaluation with external knowledge graphs...")
        
        print("TRUE TAA IMPLEMENTATION")
        print("=" * 40)
        print("[WARNING]  WARNING: This requires external KG access")
        print("• DBpedia SPARQL endpoint")
        print("• Wikidata SPARQL endpoint") 
        print("• Rate limiting applied (0.5s per query)")
        print("• Sample size limited to 20 triplets per dataset")
        print()
        
        experiment_dirs = [
            "enhanced_dermc_output_test10",
            "enhanced_dermc_output_test20", 
            "enhanced_dermc_output_test50"
        ]
        
        parsing_modes = ['flat', 'hierarchical']
        
        true_taa_results = {}
        
        for exp_dir in experiment_dirs:
            exp_path = self.base_dir / exp_dir
            if exp_path.exists():
                for mode in parsing_modes:
                    result_key = f"{exp_dir}_{mode}"
                    logger.info(f"True TAA evaluation for: {result_key}")
                    results = self.evaluate_experiment_true_taa(exp_path, exp_dir, mode)
                    true_taa_results[result_key] = results
            else:
                logger.warning(f"Experiment directory not found: {exp_path}")
        
        # Create TAA comparative analysis
        self.create_true_taa_analysis(true_taa_results)
        
        logger.info("True TAA evaluation completed!")
        return true_taa_results
    
    def evaluate_experiment_true_taa(self, exp_path: Path, exp_name: str, parsing_mode: str = 'flat') -> Dict[str, Any]:
        """Evaluate single experiment using true TAA methodology"""
        results = {
            'experiment_name': exp_name,
            'parsing_mode': parsing_mode,
            'dataset_size': int(exp_name.split('test')[1]) if 'test' in exp_name else 0,
            'evaluation_methodology': 'True TAA with external KGs'
        }
        
        try:
            # Load canonical triplets (our source triples) - mode-specific file
            canonical_triplets = self.load_triplets_safe(exp_path / f"canonical_kg_{parsing_mode}.csv")
            
            if not canonical_triplets:
                logger.warning(f"No canonical triplets loaded for {exp_name}")
                return results
            
            # Calculate TAA consensus scores against external KGs
            logger.info(f"Running TRUE TAA consensus calculation for {exp_name}...")
            results['taa_consensus'] = self.validator.calculate_consensus_score(canonical_triplets)
            
            # Save results
            results_file = self.output_dir / f"{exp_name}_true_taa_evaluation.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"True TAA evaluation completed for {exp_name}")
            
        except Exception as e:
            logger.error(f"Error in true TAA evaluation for {exp_name}: {e}")
            results['error'] = str(e)
        
        return results
    
    def load_triplets_safe(self, file_path: Path) -> List[Dict]:
        """Safely load triplets"""
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
    
    def create_true_taa_analysis(self, all_results: Dict[str, Any]):
        """Create analysis of true TAA results"""
        logger.info("Creating true TAA comparative analysis...")
        
        print("\n TRUE TAA CONSENSUS RESULTS - FLAT VS HIERARCHICAL")
        print("=" * 60)
        
        for exp_name, results in all_results.items():
            if 'error' in results:
                print(f"[ERROR] {exp_name}: {results['error']}")
                continue
            
            if 'taa_consensus' in results:
                taa = results['taa_consensus']
                size = results['dataset_size']
                mode = results.get('parsing_mode', 'unknown')
                
                print(f"\n{size}-Document Dataset ({mode.upper()} mode):")
                print(f"   • Average Confidence: {taa['average_confidence']:.3f}")
                print(f"   • External Matches: {taa['external_matches_found']}")
                print(f"   • High Confidence: {taa['high_confidence_triples']} triplets")
                print(f"   • Medium Confidence: {taa['medium_confidence_triples']} triplets")
                print(f"   • Low Confidence: {taa['low_confidence_triples']} triplets")
                print(f"   • No Consensus: {taa['no_consensus_triples']} triplets")
        
        # Save CSV with flat vs hierarchical comparison
        csv_data = []
        for exp_name, results in all_results.items():
            if 'taa_consensus' in results:
                taa = results['taa_consensus']
                csv_data.append({
                    'experiment': exp_name,
                    'dataset_size': results['dataset_size'],
                    'parsing_mode': results.get('parsing_mode', 'unknown'),
                    'average_confidence': taa['average_confidence'],
                    'external_matches': taa['external_matches_found'],
                    'high_confidence': taa['high_confidence_triples'],
                    'medium_confidence': taa['medium_confidence_triples'],
                    'low_confidence': taa['low_confidence_triples'],
                    'no_consensus': taa['no_consensus_triples']
                })
        
        if csv_data:
            df = pd.DataFrame(csv_data)
            output_path = Path(self.base_dir) / "true_taa_flat_vs_hierarchical_results.csv"
            df.to_csv(output_path, index=False)
            print(f"\n[SUCCESS] Results saved to: {output_path}")
        
        print("\n TRUE TAA METHODOLOGY")
        print("-" * 30)
        print("• Source Triples: DERMC canonical output")
        print("• Target Triples: DBpedia + Wikidata")
        print("• Consensus: Similarity-based matching")
        print("• Confidence: Weighted similarity scores")

def main():
    """Main execution for true TAA evaluation"""
    print("TRUE TAA (TRIPLES ACCURACY ASSESSMENT)")
    print("=" * 50)
    print("Implementing actual TAA methodology:")
    print("• Source triples: DERMC canonical output")
    print("• Target triples: External knowledge graphs")
    print("• Consensus calculation: Cross-KG validation")
    print()
    
    # Check if user wants to proceed (external KG queries are slow)
    response = input("Proceed with external KG queries? (y/N): ").lower().strip()
    if response != 'y':
        print("True TAA evaluation cancelled.")
        print("\nNote: Our previous evaluation was 'TAA-inspired' but lacked")
        print("external target triples for true consensus validation.")
        return None
    
    evaluator = TrueTAAEvaluator()
    results = evaluator.run_true_taa_evaluation()
    
    return results

if __name__ == "__main__":
    main()