"""
Ontology-based schema retrieval that provides:
1. Entity and relation canonicalization via semantic similarity
2. Domain-specific ontology alignment
3. Embedding-based similarity search (cached)
4. Domain/range constraint validation
"""
import json
import os
import numpy as np
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

# Type hint for logger with log_schema_operation method
from typing import Protocol

class SchemaLoggerProtocol(Protocol):
    def log_schema_operation(
        self,
        op_type: str,
        input_data: object,
        output_data: object,
        metadata: dict
    ) -> None: ...

class SchemaRetriever:
    """
    SchemaRetriever provides ontology-based entity and relation retrieval, canonicalization, and validation.
    All orchestration/CLI logic has been removed for modularity.
    """
    def validate_domain_range(self, subject: str, relation: str, obj: str) -> bool:
        """
        Validate subject/object types against ontology domain/range constraints (if available).
        Returns True if valid or if no constraints are present.
        """
        # Check if ontology defines type constraints for this relation
        domains = self.ontology.get('relation_domains', {})
        ranges = self.ontology.get('relation_ranges', {})
        rel = relation
        subj_type = subject
        obj_type = obj
        # Validate both domain (subject type) and range (object type) if constraints exist
        if rel in domains and rel in ranges:
            valid_domain = subj_type in domains[rel]
            valid_range = obj_type in ranges[rel]
            return valid_domain and valid_range
        # No constraints defined - consider valid by default
        return True
    def __init__(self, ontology_path: str, entity_sim_threshold: float = 0.7, 
                 relation_sim_threshold: float = 0.75, log_file: str = "schema_retriever_log.json",
                 comprehensive_logger: Optional[SchemaLoggerProtocol] = None):
        with open(ontology_path) as f:
            self.ontology = json.load(f)
        self.version = self.ontology.get('version', '1.0')
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.entity_sim_threshold = entity_sim_threshold
        self.relation_sim_threshold = relation_sim_threshold
        self._precompute_embeddings()
        self.comprehensive_logger = comprehensive_logger
        self.log_file = log_file
        self.log_data = {
            "version": self.version,
            "entity_threshold": entity_sim_threshold,
            "relation_threshold": relation_sim_threshold,
            "operations": []
        }
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(self.log_file, 'w') as f:
            json.dump(self.log_data, f, indent=2)
        self.entity_canonicalized = 0
        self.relation_canonicalized = 0
        # Section/content embedding cache
        self.section_embedding_cache = {}

    def _precompute_embeddings(self):
        """Precompute embeddings for all ontology entities and relations to enable fast similarity search."""
        # Combine entity name + description for richer semantic matching
        self.entity_embeddings = {
            e: self.embedder.encode(f"{e}: {desc}")
            for e, desc in self.ontology.get('entities', {}).items()
        }
        # Combine relation name + description for richer semantic matching
        self.relation_embeddings = {
            r: self.embedder.encode(f"{r}: {desc}")
            for r, desc in self.ontology.get('relations', {}).items()
        }
    def get_section_embedding(self, section_text: str, section_path: Optional[str] = None):
        """Get or compute embedding for section text, with optional hierarchy path for context. Uses cache to avoid recomputation."""
        # Use (text, path) tuple as cache key for context-aware embeddings
        cache_key = (section_text, section_path) if section_path else section_text
        if cache_key in self.section_embedding_cache:
            return self.section_embedding_cache[cache_key]
        # Include hierarchy path in embedding for better context matching
        if section_path:
            text_for_embedding = f"{section_path} | {section_text}"
        else:
            text_for_embedding = section_text
        embedding = self.embedder.encode(text_for_embedding)
        import numpy as np
        embedding = np.asarray(embedding)
        self.section_embedding_cache[cache_key] = embedding
        return embedding

    def _log_operation(self, op_type: str, input_data: Any, output_data: Any, metadata: Optional[dict] = None):
        # Temporarily disable logging to prevent infinite loops
        return

    def retrieve_relevant_relations(self, query: str, top_k: int = 5) -> List[str]:
        if not self.relation_embeddings:
            return []
        query_embed = self.embedder.encode(query)
        import numpy as np
        query_embed = np.asarray(query_embed).reshape(1, -1)
        similarities = [
            (rel, float(cosine_similarity(query_embed, np.asarray(emb).reshape(1, -1))[0][0]))
            for rel, emb in self.relation_embeddings.items()
        ]
        sorted_relations = sorted(similarities, key=lambda x: -x[1])[:top_k]
        result = [rel for rel, _ in sorted_relations]
        metadata = {
            "top_k": top_k,
            "candidates_with_scores": sorted_relations
        }
        self._log_operation(
            op_type="retrieve_relations",
            input_data=query,
            output_data=result,
            metadata=metadata
        )
        if self.comprehensive_logger:
            self.comprehensive_logger.log_schema_operation(
                "retrieve_relations",
                query,
                result,
                metadata
            )
        return result

    def canonicalize_entity(self, entity: str) -> str:
        if not entity or not self.entity_embeddings:
            return entity
        entity_embed = self.embedder.encode(entity)
        import numpy as np
        entity_embed = np.asarray(entity_embed).reshape(1, -1)
        best_match, highest_sim = max(
            ((ent, float(cosine_similarity(entity_embed, np.asarray(emb).reshape(1, -1))[0][0]))
             for ent, emb in self.entity_embeddings.items()),
            key=lambda x: x[1]
        )
        result = best_match if highest_sim > self.entity_sim_threshold else entity
        canonicalized = highest_sim > self.entity_sim_threshold
        metadata = {
            "best_match": best_match,
            "similarity_score": float(highest_sim),
            "threshold": self.entity_sim_threshold,
            "was_canonicalized": canonicalized
        }
        self._log_operation(
            op_type="canonicalize_entity",
            input_data=entity,
            output_data=result,
            metadata=metadata
        )
        if self.comprehensive_logger:
            self.comprehensive_logger.log_schema_operation(
                "canonicalize_entity",
                entity,
                result,
                metadata
            )
        if canonicalized:
            self.entity_canonicalized += 1
        return result

    def canonicalize_relation(self, relation: str) -> str:
        if not relation or not self.relation_embeddings:
            return relation
        relation_embed = self.embedder.encode(relation)
        import numpy as np
        relation_embed = np.asarray(relation_embed).reshape(1, -1)
        best_match, highest_sim = max(
            ((rel, float(cosine_similarity(relation_embed, np.asarray(emb).reshape(1, -1))[0][0]))
             for rel, emb in self.relation_embeddings.items()),
            key=lambda x: x[1]
        )
        result = best_match if highest_sim > self.relation_sim_threshold else relation
        canonicalized = highest_sim > self.relation_sim_threshold
        metadata = {
            "best_match": best_match,
            "similarity_score": float(highest_sim),
            "threshold": self.relation_sim_threshold,
            "was_canonicalized": canonicalized
        }
        self._log_operation(
            op_type="canonicalize_relation",
            input_data=relation,
            output_data=result,
            metadata=metadata
        )
        if self.comprehensive_logger:
            self.comprehensive_logger.log_schema_operation(
                "canonicalize_relation",
                relation,
                result,
                metadata
            )
        if canonicalized:
            self.relation_canonicalized += 1
        return result
