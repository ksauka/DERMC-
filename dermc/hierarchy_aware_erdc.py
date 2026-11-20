"""
DERMC pipeline orchestrator that coordinates:
1. Document parsing (hierarchical/flat modes)
2. Open Information Extraction (OIE)
3. Triplet refinement with ontology alignment
4. Coreference resolution integration
5. Knowledge graph canonicalization
"""
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from tqdm import tqdm

from .document_parser import DocumentParser
from .schema_retriever import SchemaRetriever
from .coreference_resolver import CoreferenceResolver
from .llm_client import LLMClient

logger = logging.getLogger(__name__)

class HierarchyAwareERDC:

    def _is_low_quality_triplet(self, subject, relation, obj):
        """Filter out low-quality triplets with empty/generic entities or overly vague relations."""
        subject = str(subject) if subject is not None else ""
        relation = str(relation) if relation is not None else ""
        obj = str(obj) if obj is not None else ""
        # Reject if subject/relation too short
        if len(subject.strip()) < 2 or len(relation.strip()) < 2:
            return True
        # Reject placeholder subjects
        if subject.lower() in {"none", "n/a", "unknown", "-"}:
            return True
        # Reject generic relations with short objects
        if relation.lower() in {"is", "has", "was", "were", "be", "are", "am"} and len(obj.strip()) < 3:
            return True
        # Reject placeholder objects
        if obj.lower() in {"none", "n/a", "unknown", "-"}:
            return True
        return False

    def _enhance_entity(self, entity, context):
        # Example: expand abbreviations, add context if needed (stub, can be extended)
        entity = str(entity)
        if len(entity) < 4 and context:
            return f"{entity} ({context[:20]}...)"
        return entity

    def _enhance_relation(self, relation, context, hierarchy):
        # Example: prefer more specific relations, add hierarchy cues (stub, can be extended)
        relation = str(relation)
        if relation in {"is", "has"} and hierarchy:
            return f"{relation}_{hierarchy.replace(' ', '_')[:10]}"
        return relation

    def _validate_enhanced_triplet(self, subject, relation, obj):
        # Example: ensure all fields are non-empty and not generic
        if not subject or not relation or not obj:
            return False
        if subject.lower() in {"none", "n/a", "unknown", "-"}:
            return False
        if relation.lower() in {"is", "has", "was", "were", "be", "are", "am"} and len(obj.strip()) < 3:
            return False
        if obj.lower() in {"none", "n/a", "unknown", "-"}:
            return False
        return True
    def refine_triplets_batch(self, content_items, triplets):
        """Refine extracted triplets using ontology-aligned relations via batch LLM calls."""
        import json
        from typing import Tuple, List, Dict
        if not triplets:
            return [[] for _ in content_items]
        # Build refinement prompts with ontology candidates for each triplet
        refinement_prompts = []
        prompt_indices = []
        for i, item in enumerate(content_items):
            if i >= len(triplets) or not triplets[i]:
                continue
            for j, (s, r, o) in enumerate(triplets[i]):
                try:
                    # Retrieve top-k ontology relations matching current relation + context
                    candidates = self.schema_retriever.retrieve_relevant_relations(f"{r} {item['text']}", top_k=3)
                    refinement_prompts.append(self.create_refinement_prompt(s, r, o, candidates, item['text'], item['hierarchy']))
                    prompt_indices.append((i, j))
                except Exception as e:
                    logger = logging.getLogger("dermc.hierarchy_aware_erdc")
                    logger.error(f"Refinement prompt creation failed: {e}")
                    refinement_prompts.append("")
                    prompt_indices.append((i, j))
        responses = self.llm.generate_batch(refinement_prompts)
        refined_triplets = [[] for _ in content_items]
        for idx, response in enumerate(responses):
            if idx >= len(prompt_indices):
                continue
            i, j = prompt_indices[idx]
            if i < len(refined_triplets):
                parsed = self.parse_triplets(response)
                if parsed:
                    refined_triplets[i].append(parsed[0])
                elif j < len(triplets[i]):
                    refined_triplets[i].append(triplets[i][j])
        return refined_triplets

    def create_extraction_prompt(self, text, hierarchy_context, hierarchy_path, relations):
        """Generate LLM prompt for triplet extraction with hierarchy context and suggested relations."""
        return f"""
### Document Structure
{hierarchy_context}

### Current Section
{hierarchy_path}

### Suggested Relations
{', '.join(relations) if relations else 'No relations available'}

### Task
Extract relation triplets from the text below. Return ONLY a JSON list of triplets.

### Text
"{text}"

### Response Format:
[["subject", "relation", "object"], ...]
"""

    def create_refinement_prompt(self, s, r, o, candidates, context, hierarchy):
        return f"""
### Refinement Context
Section: {hierarchy}
Original Text: "{context}"

### Current Triplet
["{s}", "{r}", "{o}"]

### Available Relations from Ontology
{', '.join(candidates) if candidates else 'No specific relations available'}

### Instructions
You must improve this triplet for maximum quality and precision:

1. **Subject Enhancement**: Keep the subject meaningful and specific
2. **Relation Improvement**: Choose the BEST relation from candidates OR create a more precise relation
3. **Object Refinement**: Ensure the object is complete and informative
4. **Quality Check**: The result must be factually accurate and semantically meaningful

### Quality Criteria
- Relations should be specific, not generic (avoid "is", "has" unless very appropriate)
- Entities should be complete (expand abbreviations, add context if needed)
- The triplet should capture the actual semantic relationship

### Examples of Good Refinements
- "Items" → "Returned items", "Processing time" → "Refund processing time"  
- "is" → "requires", "has" → "includes_feature", "excluded" → "excluded_from"

Return ONLY the improved triplet as JSON: ["enhanced_subject", "best_relation", "enhanced_object"]
"""

    def parse_triplets(self, raw_response):
        """Parse LLM response to extract triplets, handling JSON and text formats."""
        import re, json, ast
        triplets = []
        if not raw_response.strip():
            return triplets
        try:
            # Try JSON parsing first (most reliable)
            json_match = re.search(r'\[.*\]', raw_response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                if isinstance(parsed, list):
                    return [tuple(item) for item in parsed if len(item) == 3]
        except json.JSONDecodeError:
            pass
        for line in raw_response.split('\n'):
            line = line.strip()
            if line.startswith('[') and line.endswith(']'):
                try:
                    triplet = ast.literal_eval(line)
                    if isinstance(triplet, (list, tuple)) and len(triplet) == 3:
                        triplets.append(tuple(triplet))
                except (SyntaxError, ValueError):
                    continue
        return triplets

    def _merge_similar_triplets(self, triplets, similarity_threshold=0.85):
        from sklearn.metrics.pairwise import cosine_similarity
        if not triplets:
            return triplets
        hierarchy_groups = {}
        for triplet in triplets:
            hierarchy = triplet.get('hierarchy', 'unknown')
            if hierarchy not in hierarchy_groups:
                hierarchy_groups[hierarchy] = []
            hierarchy_groups[hierarchy].append(triplet)
        merged_triplets = []
        for hierarchy, group_triplets in hierarchy_groups.items():
            if len(group_triplets) <= 1:
                merged_triplets.extend(group_triplets)
                continue
            merged_indices = set()
            for i, triplet1 in enumerate(group_triplets):
                if i in merged_indices:
                    continue
                similar_triplets = [triplet1]
                merged_indices.add(i)
                for j, triplet2 in enumerate(group_triplets[i+1:], i+1):
                    if j in merged_indices:
                        continue
                    similarity = self._calculate_triplet_similarity(triplet1, triplet2)
                    if similarity >= similarity_threshold:
                        similar_triplets.append(triplet2)
                        merged_indices.add(j)
                if len(similar_triplets) > 1:
                    merged_triplet = self._merge_triplet_group(similar_triplets)
                    merged_triplets.append(merged_triplet)
                else:
                    merged_triplets.append(triplet1)
        return merged_triplets

    def _calculate_triplet_similarity(self, triplet1, triplet2):
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        try:
            if 'triplet' in triplet1:
                s1, r1, o1 = triplet1['triplet']
                s2, r2, o2 = triplet2['triplet']
            else:
                s1, r1, o1 = triplet1.get('subject', ''), triplet1.get('relation', ''), triplet1.get('object', '')
                s2, r2, o2 = triplet2.get('subject', ''), triplet2.get('relation', ''), triplet2.get('object', '')
            text1 = f"{s1} {r1} {o1}".strip()
            text2 = f"{s2} {r2} {o2}".strip()
            if not text1 or not text2:
                return 0.0
            embeddings = self.embedder.encode([text1, text2])
            arr1 = np.asarray(embeddings[0]).reshape(1, -1)
            arr2 = np.asarray(embeddings[1]).reshape(1, -1)
            similarity = cosine_similarity(arr1, arr2)[0][0]
            return float(similarity)
        except Exception as e:
            logger = logging.getLogger("dermc.hierarchy_aware_erdc")
            logger.debug(f"Similarity calculation failed: {e}")
            return 0.0

    def _merge_triplet_group(self, triplets):
        if len(triplets) == 1:
            return triplets[0]
        base_triplet = triplets[0].copy()
        if 'subject' in base_triplet:
            subjects = [t.get('subject', '') for t in triplets if t.get('subject')]
            relations = [t.get('relation', '') for t in triplets if t.get('relation')]
            objects = [t.get('object', '') for t in triplets if t.get('object')]
            base_triplet['subject'] = max(subjects, key=len) if subjects else base_triplet.get('subject', '')
            base_triplet['object'] = max(objects, key=len) if objects else base_triplet.get('object', '')
            base_triplet['relation'] = min(relations, key=len) if relations else base_triplet.get('relation', '')
        elif 'triplet' in base_triplet:
            subjects = [t['triplet'][0] for t in triplets if len(t['triplet']) > 0]
            relations = [t['triplet'][1] for t in triplets if len(t['triplet']) > 1]
            objects = [t['triplet'][2] for t in triplets if len(t['triplet']) > 2]
            merged_subject = max(subjects, key=len) if subjects else base_triplet['triplet'][0]
            merged_relation = min(relations, key=len) if relations else base_triplet['triplet'][1]
            merged_object = max(objects, key=len) if objects else base_triplet['triplet'][2]
            base_triplet['triplet'] = (merged_subject, merged_relation, merged_object)
        base_triplet['merged_count'] = len(triplets)
        base_triplet['confidence'] = min(1.0, len(triplets) * 0.2)
        return base_triplet

    def _enhance_refinement_quality(self, content_items, triplets):
        if not triplets:
            return triplets
        enhanced_triplets = []
        for i, item in enumerate(content_items):
            if i >= len(triplets) or not triplets[i]:
                enhanced_triplets.append([])
                continue
            item_enhanced = []
            for triplet in triplets[i]:
                s, r, o = triplet
                if self._is_low_quality_triplet(s, r, o):
                    continue
                enhanced_s = self._enhance_entity(s, item['text']) if hasattr(self, '_enhance_entity') else s
                enhanced_o = self._enhance_entity(o, item['text']) if hasattr(self, '_enhance_entity') else o
                enhanced_r = self._enhance_relation(r, item['text'], item.get('hierarchy', '')) if hasattr(self, '_enhance_relation') else r
                if self._validate_enhanced_triplet(enhanced_s, enhanced_r, enhanced_o) if hasattr(self, '_validate_enhanced_triplet') else True:
                    item_enhanced.append((enhanced_s, enhanced_r, enhanced_o))
            enhanced_triplets.append(item_enhanced)
        return enhanced_triplets
    def process_csv(self, csv_path: str, text_column: str = 'Document'):
        import pandas as pd
        import json
        from tqdm import tqdm
        logger = logging.getLogger("dermc.hierarchy_aware_erdc")
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Processing {len(df)} documents from {csv_path}")
            if self.enable_coref and self.coref_resolver:
                df[text_column] = [
                    self.coref_resolver.resolve_coreferences(text)
                    for text in tqdm(df[text_column], desc="Coreference resolution")
                ]
            parsed_df = self.parser.parse_dataframe(df, text_column)
            return self._process_parsed_documents(parsed_df)
        except Exception as e:
            logger.error(f"Document processing failed: {e}", exc_info=True)
            raise

    def _process_parsed_documents(self, parsed_df):
        from tqdm import tqdm
        results = []
        for i in tqdm(range(0, len(parsed_df), self.batch_size), desc="Processing documents"):
            batch = parsed_df.iloc[i:i+self.batch_size]
            batch_results = self._process_batch(batch)
            results.extend(batch_results)
        import pandas as pd
        return pd.DataFrame(results)

    def _process_batch(self, batch_df):
        import json
        batch_results = []
        all_content_items = self._prepare_content_items(batch_df)
        oie_triplets = self._safe_extract_triplets(all_content_items)
        refined_triplets = self._refine_triplets(all_content_items, oie_triplets)
        for doc_idx, row in batch_df.iterrows():
            doc_results = self._process_document_results(doc_idx, row, all_content_items, oie_triplets, refined_triplets)
            if doc_results:
                batch_results.append(doc_results)
        return batch_results

    def _prepare_content_items(self, batch_df):
        import json
        items = []
        for doc_idx, row in batch_df.iterrows():
            try:
                content = json.loads(row['content_with_hierarchy'])
                items.extend({
                    'text': item['text'],
                    'hierarchy': item['hierarchy'],
                    'doc_idx': doc_idx,
                    'hierarchy_context': row['hierarchy_context'],
                    'doc_id': row.get('Doc ID', None),
                    'answer_id': row.get('Answer ID', None)
                } for item in content if not item.get('is_header'))
            except Exception:
                logger = logging.getLogger("dermc.hierarchy_aware_erdc")
                logger.error(f"JSON parsing failed for document {row.get('Doc ID', '')}")
        return items

    def _safe_extract_triplets(self, content_items):
        try:
            return self.extract_triplets_batch(content_items)
        except Exception as e:
            logger = logging.getLogger("dermc.hierarchy_aware_erdc")
            logger.error(f"Triplet extraction failed: {e}")
            return [[] for _ in content_items]

    def _refine_triplets(self, content_items, initial_triplets):
        refined = initial_triplets
        for i in range(self.refinement_iterations):
            try:
                refined = self.refine_triplets_batch(content_items, refined)
            except Exception as e:
                logger = logging.getLogger("dermc.hierarchy_aware_erdc")
                logger.error(f"Refinement iteration {i+1} failed: {e}")
                break
        return refined

    def _process_document_results(self, doc_idx, row, content_items, oie_triplets, refined_triplets):
        doc_content = [item for item in content_items if item['doc_idx'] == doc_idx]
        if not doc_content:
            return None
        enhanced_triplets = self._enhance_refinement_quality(doc_content, refined_triplets) if hasattr(self, '_enhance_refinement_quality') else refined_triplets
        logger = logging.getLogger("dermc.hierarchy_aware_erdc")
        logger.info(f"Enhanced refinement completed for doc {doc_idx}")
        doc_oie = self._collect_triplets(doc_idx, doc_content, oie_triplets, content_items)
        doc_refined = self._collect_triplets(doc_idx, doc_content, enhanced_triplets, content_items)
        original_refined_count = len(doc_refined)
        doc_refined_merged = self._merge_similar_triplets(doc_refined, similarity_threshold=0.85) if hasattr(self, '_merge_similar_triplets') else doc_refined
        refined_merged_count = len(doc_refined_merged)
        if original_refined_count > refined_merged_count:
            logger.info(f"Merged {original_refined_count - refined_merged_count} similar triplets in refined stage for doc {doc_idx}")
        doc_canonical = self._canonicalize_triplets(doc_refined_merged)
        original_canonical_count = len(doc_canonical)
        doc_canonical_merged = self._merge_similar_triplets(doc_canonical, similarity_threshold=0.90) if hasattr(self, '_merge_similar_triplets') else doc_canonical
        canonical_merged_count = len(doc_canonical_merged)
        if original_canonical_count > canonical_merged_count:
            logger.info(f"Merged {original_canonical_count - canonical_merged_count} similar triplets in canonical stage for doc {doc_idx}")
        return {
            'Doc ID': row.get('Doc ID', ''),
            'Answer ID': row.get('Answer ID', ''),
            'Parsing Mode': 'hierarchical' if getattr(self, 'enable_hierarchical_parsing', False) else 'flat',
            'Ontology Version': getattr(self.schema_retriever, 'version', ''),
            'oie_triplets': doc_oie,
            'refined_triplets': doc_refined_merged,
            'canonical_triplets': doc_canonical_merged,
            'processing_metadata': {
                'hierarchy_enabled': getattr(self, 'enable_hierarchical_parsing', False),
                'coref_enabled': getattr(self, 'enable_coref', False),
                'content_items': len(doc_content),
                'merged_refined_triplets': original_refined_count - refined_merged_count,
                'merged_canonical_triplets': original_canonical_count - canonical_merged_count
            }
        }

    def _collect_triplets(self, doc_idx, doc_content, all_triplets, all_content_items):
        results = []
        for item in doc_content:
            try:
                global_idx = all_content_items.index(item)
            except ValueError:
                continue
            if global_idx < len(all_triplets):
                for triplet in all_triplets[global_idx]:
                    results.append({
                        'triplet': triplet,
                        'hierarchy': item['hierarchy']
                    })
        return results
    def extract_triplets_stream(self, content_items: List[Dict], batch_size: int = 0):
        """Generator for streaming/online ingestion of triplet extraction in batches."""
        if not batch_size:
            batch_size = self.batch_size
        for i in range(0, len(content_items), batch_size):
            batch = content_items[i:i+batch_size]
            prompts = []
            for item in batch:
                try:
                    relations = self.schema_retriever.retrieve_relevant_relations(item['text'], top_k=5)
                    prompt = self.create_extraction_prompt(
                        item['text'],
                        item['hierarchy_context'],
                        item['hierarchy'],
                        relations
                    )
                    prompts.append(prompt)
                except Exception as e:
                    logger.error(f"Prompt creation failed: {e}")
                    prompts.append("")
            responses = self.llm.generate_batch(prompts)
            for resp in responses:
                yield self.parse_triplets(resp)
    def __init__(self, llm: LLMClient, ontology_path: str, batch_size: int = 8,
                 max_hierarchy_depth: int = 4, enable_coref: bool = True,
                 enable_hierarchical_parsing: bool = True,
                 entity_sim_threshold: float = 0.7, relation_sim_threshold: float = 0.75,
                 refinement_iterations: int = 1, schema_log_file: str = "schema_retriever_log.json",
                 comprehensive_logger: Optional[Any] = None):
        self.llm = llm
        self.enable_hierarchical_parsing = enable_hierarchical_parsing
        self.max_hierarchy_depth = max_hierarchy_depth
        self.comprehensive_logger = comprehensive_logger
        self.parser = DocumentParser(max_depth=max_hierarchy_depth)
        self.schema_retriever = SchemaRetriever(
            ontology_path,
            entity_sim_threshold=entity_sim_threshold,
            relation_sim_threshold=relation_sim_threshold,
            log_file=schema_log_file,
            comprehensive_logger=comprehensive_logger
        )
        self.batch_size = batch_size
        self.enable_coref = enable_coref
        self.coref_resolver = None
        if enable_coref:
            try:
                logger.info("Initializing CoreferenceResolver for multilingual support...")
                self.coref_resolver = CoreferenceResolver(comprehensive_logger=comprehensive_logger)
                logger.info("[SUCCESS] CoreferenceResolver initialized successfully")
            except Exception as e:
                logger.warning(f"[WARNING]  Failed to initialize CoreferenceResolver: {e}")
                logger.warning("Continuing without coreference resolution...")
                self.enable_coref = False
        self.refinement_iterations = refinement_iterations
        try:
            import spacy  # type: ignore
            self.entity_typer = spacy.load("en_core_web_trf")
        except Exception as e:
            logger.warning(f"Entity typing model not available: {e}")
            self.entity_typer = None
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        parsing_mode = "hierarchical" if enable_hierarchical_parsing else "flat"
        logger.info(f"ERDRC pipeline initialized with {parsing_mode} parsing mode")


    def _canonicalize_triplets(self, refined_triplets: List[Dict]) -> List[Dict]:
        """Canonicalize refined triplets using SchemaRetriever (with name+description embedding)"""
        canonical = []
        for item in refined_triplets:
            s, r, o = item['triplet']
            typed_s = self._add_entity_type(s)
            typed_o = self._add_entity_type(o)
            try:
                canonical.append({
                    'subject': self.schema_retriever.canonicalize_entity(s),
                    'relation': self.schema_retriever.canonicalize_relation(r),
                    'object': self.schema_retriever.canonicalize_entity(o),
                    'hierarchy': item['hierarchy'],
                    'subject_type': self._extract_entity_type(typed_s),
                    'object_type': self._extract_entity_type(typed_o)
                })
            except Exception as e:
                logger.error(f"Canonicalization failed: {e}")
                canonical.append({
                    'subject': s,
                    'relation': r,
                    'object': o,
                    'hierarchy': item['hierarchy'],
                    'subject_type': self._extract_entity_type(typed_s),
                    'object_type': self._extract_entity_type(typed_o)
                })
        return canonical

    def _add_entity_type(self, entity: str) -> str:
        """Add entity type using NER if available"""
        if not entity or not self.entity_typer:
            return entity
        try:
            doc = self.entity_typer(entity)
            if doc.ents:
                main_ent = doc.ents[0]
                return f"{entity} ({main_ent.label_})"
        except Exception as e:
            logger.error(f"Entity typing failed for '{entity}': {e}")
        return entity

    def _extract_entity_type(self, entity: str) -> str:
        """Extract entity type from parenthesized string"""
        import re
        if not entity:
            return ""
        match = re.search(r'\(([A-Z]+)\)$', entity)
        return match.group(1) if match else ""

    def extract_triplets_batch(self, content_items: List[Dict]) -> List[List[Tuple[str, str, str]]]:
        """Batch extract triplets with schema-aware prompting"""
        prompts = []
        for item in content_items:
            try:
                relations = self.schema_retriever.retrieve_relevant_relations(item['text'], top_k=5)
                prompt = self.create_extraction_prompt(
                    item['text'],
                    item['hierarchy_context'],
                    item['hierarchy'],
                    relations
                )
                prompts.append(prompt)
            except Exception as e:
                logger.error(f"Prompt creation failed: {e}")
                prompts.append("")
        responses = self.llm.generate_batch(prompts)
        return [self.parse_triplets(resp) for resp in responses]
