
"""
Enhanced logging system for DERMC pipeline that captures:
1. Schema retrieval and canonicalization operations
2. Coreference resolution operations
3. Pipeline processing metrics
4. Quality assessment data
"""

import json
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from pathlib import Path

class DERMCLogger:
    """Comprehensive logger for DERMC pipeline operations"""
    
    def __init__(self, log_file: str = "dermc_comprehensive_log.json", 
                 enable_schema_logging: bool = True,
                 enable_coref_logging: bool = True,
                 enable_pipeline_logging: bool = True):
        self.log_file = log_file
        self.enable_schema_logging = enable_schema_logging
        self.enable_coref_logging = enable_coref_logging
        self.enable_pipeline_logging = enable_pipeline_logging
        
        # Initialize log structure
        self.log_data = {
            "session_info": {
                "session_id": f"dermc_{int(time.time())}",
                "start_time": datetime.now(timezone.utc).isoformat(),
                "pipeline_version": "folks_2026_enhanced",
                "logging_config": {
                    "schema_logging": enable_schema_logging,
                    "coref_logging": enable_coref_logging,
                    "pipeline_logging": enable_pipeline_logging
                }
            },
            "schema_retrieval": {
                "operations": [],
                "statistics": {
                    "total_entity_canonicalizations": 0,
                    "successful_entity_canonicalizations": 0,
                    "total_relation_canonicalizations": 0,
                    "successful_relation_canonicalizations": 0,
                    "relation_retrievals": 0
                }
            },
            "coreference_resolution": {
                "operations": [],
                "statistics": {
                    "total_texts_processed": 0,
                    "successful_resolutions": 0,
                    "total_clusters_found": 0,
                    "total_character_changes": 0,
                    "cache_hits": 0,
                    "cache_misses": 0
                }
            },
            "pipeline_metrics": {
                "processing_stages": [],
                "document_statistics": {
                    "total_documents": 0,
                    "successful_documents": 0,
                    "failed_documents": 0,
                    "avg_triplets_per_document": 0.0,
                    "total_processing_time": 0.0
                },
                "triplet_statistics": {
                    "oie_triplets": 0,
                    "refined_triplets": 0,
                    "canonical_triplets": 0,
                    "quality_filtered": 0,
                    "merged_duplicates": 0
                }
            },
            "quality_metrics": {
                "enhancement_operations": [],
                "refinement_improvements": 0,
                "low_quality_filtered": 0,
                "similarity_merges": 0
            }
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)
        
        # Write initial log
        self._save_log()
    
    def log_schema_operation(self, operation_type: str, input_data: Any, 
                           output_data: Any, metadata: Dict = None) -> None:
        """Log schema retrieval operations"""
        if not self.enable_schema_logging:
            return
            
        operation = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "operation_type": operation_type,
            "input": self._sanitize_for_json(input_data),
            "output": self._sanitize_for_json(output_data),
            "metadata": self._sanitize_for_json(metadata or {})
        }
        
        self.log_data["schema_retrieval"]["operations"].append(operation)
        
        # Update statistics
        stats = self.log_data["schema_retrieval"]["statistics"]
        if operation_type == "canonicalize_entity":
            stats["total_entity_canonicalizations"] += 1
            if metadata and metadata.get("was_canonicalized", False):
                stats["successful_entity_canonicalizations"] += 1
        elif operation_type == "canonicalize_relation":
            stats["total_relation_canonicalizations"] += 1
            if metadata and metadata.get("was_canonicalized", False):
                stats["successful_relation_canonicalizations"] += 1
        elif operation_type == "retrieve_relations":
            stats["relation_retrievals"] += 1
        
        self._save_log()
    
    def log_coreference_operation(self, operation_type: str, input_text: str, 
                                output_text: str, metadata: Dict = None) -> None:
        """Log coreference resolution operations"""
        if not self.enable_coref_logging:
            return
            
        # Calculate changes
        char_changes = sum(1 for a, b in zip(input_text, output_text) if a != b) if input_text and output_text else 0
        text_modified = input_text != output_text
        
        operation = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "operation_type": operation_type,
            "input_length": len(input_text) if input_text else 0,
            "output_length": len(output_text) if output_text else 0,
            "character_changes": char_changes,
            "text_modified": text_modified,
            "input_sample": input_text[:100] + "..." if len(input_text) > 100 else input_text,
            "output_sample": output_text[:100] + "..." if len(output_text) > 100 else output_text,
            "metadata": self._sanitize_for_json(metadata or {})
        }
        
        self.log_data["coreference_resolution"]["operations"].append(operation)
        
        # Update statistics
        stats = self.log_data["coreference_resolution"]["statistics"]
        stats["total_texts_processed"] += 1
        if text_modified:
            stats["successful_resolutions"] += 1
            stats["total_character_changes"] += char_changes
        
        if metadata:
            if metadata.get("cache_hit", False):
                stats["cache_hits"] += 1
            else:
                stats["cache_misses"] += 1
            
            clusters = metadata.get("clusters", [])
            if clusters:
                stats["total_clusters_found"] += len(clusters)
        
        self._save_log()
    
    def log_pipeline_stage(self, stage_name: str, stage_data: Dict) -> None:
        """Log pipeline processing stages"""
        if not self.enable_pipeline_logging:
            return
            
        stage_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage_name": stage_name,
            "data": self._sanitize_for_json(stage_data)
        }
        
        self.log_data["pipeline_metrics"]["processing_stages"].append(stage_entry)
        self._save_log()
    
    def log_document_processing(self, doc_id: str, success: bool, 
                              triplet_counts: Dict, processing_time: float) -> None:
        """Log individual document processing results"""
        if not self.enable_pipeline_logging:
            return
            
        # Update document statistics
        doc_stats = self.log_data["pipeline_metrics"]["document_statistics"]
        doc_stats["total_documents"] += 1
        if success:
            doc_stats["successful_documents"] += 1
        else:
            doc_stats["failed_documents"] += 1
        
        doc_stats["total_processing_time"] += processing_time
        
        # Update triplet statistics
        triplet_stats = self.log_data["pipeline_metrics"]["triplet_statistics"]
        triplet_stats["oie_triplets"] += triplet_counts.get("oie", 0)
        triplet_stats["refined_triplets"] += triplet_counts.get("refined", 0)
        triplet_stats["canonical_triplets"] += triplet_counts.get("canonical", 0)
        triplet_stats["quality_filtered"] += triplet_counts.get("filtered", 0)
        triplet_stats["merged_duplicates"] += triplet_counts.get("merged", 0)
        
        # Recalculate averages
        if doc_stats["total_documents"] > 0:
            total_triplets = (triplet_stats["oie_triplets"] + 
                            triplet_stats["refined_triplets"] + 
                            triplet_stats["canonical_triplets"])
            doc_stats["avg_triplets_per_document"] = total_triplets / doc_stats["total_documents"]
        
        self._save_log()
    
    def log_quality_enhancement(self, enhancement_type: str, 
                              improvement_data: Dict) -> None:
        """Log quality enhancement operations"""
        enhancement = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "enhancement_type": enhancement_type,
            "data": self._sanitize_for_json(improvement_data)
        }
        
        self.log_data["quality_metrics"]["enhancement_operations"].append(enhancement)
        
        # Update counters
        quality_stats = self.log_data["quality_metrics"]
        if enhancement_type == "refinement_improvement":
            quality_stats["refinement_improvements"] += 1
        elif enhancement_type == "low_quality_filter":
            quality_stats["low_quality_filtered"] += improvement_data.get("filtered_count", 0)
        elif enhancement_type == "similarity_merge":
            quality_stats["similarity_merges"] += improvement_data.get("merged_count", 0)
        
        self._save_log()
    
    def finalize_session(self) -> None:
        """Finalize the logging session with summary statistics"""
        self.log_data["session_info"]["end_time"] = datetime.now(timezone.utc).isoformat()
        
        # Calculate final statistics
        start_time = datetime.fromisoformat(self.log_data["session_info"]["start_time"].replace('Z', '+00:00'))
        end_time = datetime.fromisoformat(self.log_data["session_info"]["end_time"].replace('Z', '+00:00'))
        total_session_time = (end_time - start_time).total_seconds()
        
        self.log_data["session_info"]["total_session_time_seconds"] = total_session_time
        
        # Add summary
        self.log_data["session_summary"] = self._generate_session_summary()
        
        self._save_log()
    
    def _generate_session_summary(self) -> Dict:
        """Generate comprehensive session summary"""
        schema_stats = self.log_data["schema_retrieval"]["statistics"]
        coref_stats = self.log_data["coreference_resolution"]["statistics"]
        pipeline_stats = self.log_data["pipeline_metrics"]["document_statistics"]
        triplet_stats = self.log_data["pipeline_metrics"]["triplet_statistics"]
        quality_stats = self.log_data["quality_metrics"]
        
        return {
            "schema_retrieval_effectiveness": {
                "entity_canonicalization_rate": (
                    schema_stats["successful_entity_canonicalizations"] / 
                    max(schema_stats["total_entity_canonicalizations"], 1)
                ),
                "relation_canonicalization_rate": (
                    schema_stats["successful_relation_canonicalizations"] / 
                    max(schema_stats["total_relation_canonicalizations"], 1)
                ),
                "total_schema_operations": len(self.log_data["schema_retrieval"]["operations"])
            },
            "coreference_resolution_effectiveness": {
                "resolution_success_rate": (
                    coref_stats["successful_resolutions"] / 
                    max(coref_stats["total_texts_processed"], 1)
                ),
                "avg_character_changes_per_resolution": (
                    coref_stats["total_character_changes"] / 
                    max(coref_stats["successful_resolutions"], 1)
                ),
                "cache_hit_rate": (
                    coref_stats["cache_hits"] / 
                    max(coref_stats["cache_hits"] + coref_stats["cache_misses"], 1)
                ),
                "avg_clusters_per_text": (
                    coref_stats["total_clusters_found"] / 
                    max(coref_stats["total_texts_processed"], 1)
                )
            },
            "pipeline_performance": {
                "document_success_rate": (
                    pipeline_stats["successful_documents"] / 
                    max(pipeline_stats["total_documents"], 1)
                ),
                "avg_processing_time_per_document": (
                    pipeline_stats["total_processing_time"] / 
                    max(pipeline_stats["total_documents"], 1)
                ),
                "triplet_generation_efficiency": {
                    "oie_to_refined_ratio": (
                        triplet_stats["refined_triplets"] / 
                        max(triplet_stats["oie_triplets"], 1)
                    ),
                    "refined_to_canonical_ratio": (
                        triplet_stats["canonical_triplets"] / 
                        max(triplet_stats["refined_triplets"], 1)
                    )
                }
            },
            "quality_improvements": {
                "total_enhancements": len(quality_stats["enhancement_operations"]),
                "refinement_improvements": quality_stats["refinement_improvements"],
                "quality_filtering_rate": (
                    quality_stats["low_quality_filtered"] / 
                    max(triplet_stats["oie_triplets"], 1)
                ),
                "similarity_merge_rate": (
                    quality_stats["similarity_merges"] / 
                    max(triplet_stats["refined_triplets"], 1)
                )
            }
        }
    
    def _sanitize_for_json(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format"""
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, dict):
            return {key: self._sanitize_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._sanitize_for_json(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self._sanitize_for_json(obj.__dict__)
        else:
            return str(obj)
    
    def _save_log(self) -> None:
        """Save log data to file"""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.log_data, f, indent=2, default=str)
        except Exception as e:
            print(f"Warning: Failed to save log to {self.log_file}: {e}")
    
    def get_summary_report(self) -> str:
        """Generate a human-readable summary report"""
        if "session_summary" not in self.log_data:
            self.finalize_session()
        
        summary = self.log_data["session_summary"]
        
        report = f"""
# DERMC Pipeline Comprehensive Log Report

## Session Information
- Session ID: {self.log_data['session_info']['session_id']}
- Pipeline Version: {self.log_data['session_info']['pipeline_version']}
- Start Time: {self.log_data['session_info']['start_time']}
- End Time: {self.log_data['session_info'].get('end_time', 'In Progress')}
- Total Duration: {self.log_data['session_info'].get('total_session_time_seconds', 0):.2f} seconds

## Schema Retrieval Performance
- Entity Canonicalization Rate: {summary['schema_retrieval_effectiveness']['entity_canonicalization_rate']:.2%}
- Relation Canonicalization Rate: {summary['schema_retrieval_effectiveness']['relation_canonicalization_rate']:.2%}
- Total Schema Operations: {summary['schema_retrieval_effectiveness']['total_schema_operations']}

## Coreference Resolution Performance
- Resolution Success Rate: {summary['coreference_resolution_effectiveness']['resolution_success_rate']:.2%}
- Average Character Changes: {summary['coreference_resolution_effectiveness']['avg_character_changes_per_resolution']:.1f}
- Cache Hit Rate: {summary['coreference_resolution_effectiveness']['cache_hit_rate']:.2%}
- Average Clusters per Text: {summary['coreference_resolution_effectiveness']['avg_clusters_per_text']:.1f}

## Pipeline Performance
- Document Success Rate: {summary['pipeline_performance']['document_success_rate']:.2%}
- Average Processing Time: {summary['pipeline_performance']['avg_processing_time_per_document']:.2f}s/doc
- OIE to Refined Ratio: {summary['pipeline_performance']['triplet_generation_efficiency']['oie_to_refined_ratio']:.2f}
- Refined to Canonical Ratio: {summary['pipeline_performance']['triplet_generation_efficiency']['refined_to_canonical_ratio']:.2f}

## Quality Improvements
- Total Enhancements: {summary['quality_improvements']['total_enhancements']}
- Refinement Improvements: {summary['quality_improvements']['refinement_improvements']}
- Quality Filtering Rate: {summary['quality_improvements']['quality_filtering_rate']:.2%}
- Similarity Merge Rate: {summary['quality_improvements']['similarity_merge_rate']:.2%}
"""
        return report


# Example usage function
def test_comprehensive_logging():
    """Test the comprehensive logging system"""
    logger = DERMCLogger("test_dermc_log.json")
    
    # Test schema operations
    logger.log_schema_operation(
        "canonicalize_entity",
        "customer service",
        "Customer Service",
        {"similarity_score": 0.95, "was_canonicalized": True}
    )
    
    logger.log_schema_operation(
        "retrieve_relations",
        "customer feedback",
        ["provides_feedback", "requires_support", "has_concern"],
        {"top_k": 3, "query_time": 0.02}
    )
    
    # Test coreference operations
    logger.log_coreference_operation(
        "resolve_coreferences",
        "John went to the store. He bought milk.",
        "John went to the store. John bought milk.",
        {"clusters": [["John", "He"]], "cache_hit": False}
    )
    
    # Test pipeline stages
    logger.log_pipeline_stage(
        "document_parsing",
        {"parsing_mode": "hierarchical", "sections_found": 5}
    )
    
    # Test document processing
    logger.log_document_processing(
        "doc_1",
        True,
        {"oie": 10, "refined": 8, "canonical": 6, "filtered": 2, "merged": 1},
        1.5
    )
    
    # Test quality enhancements
    logger.log_quality_enhancement(
        "refinement_improvement",
        {"original": "has", "improved": "requires_approval", "confidence": 0.9}
    )
    
    # Finalize and generate report
    logger.finalize_session()
    print(logger.get_summary_report())
    
    print(f"Comprehensive log saved to: {logger.log_file}")


if __name__ == "__main__":
    test_comprehensive_logging()
