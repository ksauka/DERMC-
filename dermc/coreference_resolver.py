"""
Multilingual coreference resolution module that handles:
1. Pronoun and entity reference resolution
2. GPU-accelerated FastCoref integration
3. Multilingual support with translation fallback
4. Result caching for performance optimization

Requirements:
    pip install spacy fastcoref langdetect transformers
    python -m spacy download en_core_web_sm
"""
from typing import Protocol

class CorefLoggerProtocol(Protocol):
    def log_coreference_operation(self, *args, **kwargs) -> None: ...
import os
import time
import logging
import hashlib
from typing import Optional
from spacy.util import filter_spans
from spacy import load as spacy_load
from transformers import pipeline
from langdetect import detect
import torch

logger = logging.getLogger(__name__)

class CoreferenceResolver:
    """Optimized coreference resolver with enhanced resolution handling and comprehensive logging"""
    def __init__(self, cache_dir="coref/cache", model_name="biu-nlp/lingmess-coref", device=None,
                 comprehensive_logger: 'Optional[CorefLoggerProtocol]' = None):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.comprehensive_logger = comprehensive_logger
        self._initialize_models(model_name)
        self.debug = True  # Always enable debugging for coref

    def _initialize_models(self, model_name: str):
        try:
            self.nlp = spacy_load("en_core_web_sm", exclude=["parser", "lemmatizer", "ner", "textcat"])
            self.nlp.add_pipe(
                "fastcoref",
                config={
                    "model_architecture": "LingMessCoref",
                    "model_path": model_name,
                    "device": self.device,
                    "enable_progress_bar": True
                }
            )
            self.translator = None
            if self._needs_translation():
                self.translator = pipeline(
                    "translation",
                    model="facebook/m2m100_418M",
                    tokenizer="facebook/m2m100_418M",
                    src_lang="nl",
                    tgt_lang="en",
                    device=self.device
                )
            logger.info(f"Coreference resolver initialized on {self.device.upper()}")
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            self.nlp = None
            self.translator = None

    def _needs_translation(self) -> bool:
        return True  # Enable for mixed English/Dutch content

    def resolve_coreferences(self, doc_text: str) -> str:
        start_time = time.time()
        if not isinstance(doc_text, str) or not doc_text.strip():
            return doc_text
        if not self.nlp or "fastcoref" not in self.nlp.pipe_names:
            logger.warning("Coreference model not available - returning original text")
            return doc_text
        cache_path = self._get_cache_path(doc_text)
        cache_hit = os.path.exists(cache_path)
        if cache_hit:
            with open(cache_path, "r") as f:
                cached_result = f.read()
            if self.comprehensive_logger:
                self.comprehensive_logger.log_coreference_operation(
                    "resolve_coreferences_cached",
                    doc_text,
                    cached_result,
                    {
                        "cache_hit": True,
                        "character_count": len(doc_text),
                        "word_count": len(doc_text.split()),
                        "processing_time": time.time() - start_time
                    }
                )
            return cached_result
        try:
            char_count = len(doc_text)
            word_count = len(doc_text.split())
            logger.debug(f"Processing text: {char_count} chars, {word_count} words")
            text_to_process = self._translate_if_needed(doc_text)
            doc = self.nlp(text_to_process, component_cfg={"fastcoref": {"resolve_text": True}})
            resolved_text = doc._.resolved_text if hasattr(doc._, 'resolved_text') else text_to_process
            if resolved_text == text_to_process:
                clusters = doc._.coref_clusters if hasattr(doc._, 'coref_clusters') else []
                self._log_resolution_failure(text_to_process, clusters)
            else:
                changed_chars = sum(1 for a, b in zip(text_to_process, resolved_text) if a != b)
                change_percent = (changed_chars / len(text_to_process)) * 100
                logger.debug(f"Resolution modified {change_percent:.2f}% of characters")
                if self.debug and hasattr(doc._, 'coref_clusters'):
                    self._log_clusters(doc._.coref_clusters)
            with open(cache_path, "w") as f:
                f.write(resolved_text)
            if self.comprehensive_logger:
                clusters = doc._.coref_clusters if hasattr(doc._, 'coref_clusters') else []
                cache_hit = os.path.exists(cache_path)
                self.comprehensive_logger.log_coreference_operation(
                    "resolve_coreferences",
                    doc_text,
                    resolved_text,
                    {
                        "clusters": [[str(mention) for mention in cluster] for cluster in clusters],
                        "cache_hit": cache_hit,
                        "character_count": len(doc_text),
                        "word_count": len(doc_text.split()),
                        "processing_time": time.time() - start_time
                    }
                )
            return resolved_text
        except Exception as e:
            logger.error(f"Coreference resolution failed: {str(e)}")
            if self.comprehensive_logger:
                self.comprehensive_logger.log_coreference_operation(
                    "resolve_coreferences_failed",
                    doc_text,
                    doc_text,
                    {
                        "error": str(e),
                        "cache_hit": False,
                        "character_count": len(doc_text),
                        "word_count": len(doc_text.split())
                    }
                )
            return doc_text

    def _log_resolution_failure(self, text: str, clusters: list):
        logger.warning("Coreference resolution didn't modify text")
        logger.debug(f"Text length: {len(text)} characters, {len(text.split())} words")
        truncated = text[:200] + ('...' if len(text) > 200 else '')
        logger.debug(f"Text sample: '{truncated}'")
        if clusters:
            logger.debug(f"Detected {len(clusters)} clusters but no resolution applied:")
            for i, cluster in enumerate(clusters):
                logger.debug(f"  Cluster {i+1}:")
                for mention in cluster:
                    try:
                        if hasattr(mention, 'text'):
                            mention_text = mention.text
                            position = f"{mention.start_char}-{mention.end_char}" if hasattr(mention, 'start_char') else "unknown"
                        elif isinstance(mention, tuple) and len(mention) >= 2:
                            mention_text = str(mention[0]) if len(mention) > 0 else "unknown"
                            position = f"{mention[1]}-{mention[2]}" if len(mention) > 2 else "unknown"
                        else:
                            mention_text = str(mention)
                            position = "unknown"
                        logger.debug(f"    - '{mention_text}' (position: {position})")
                    except Exception as e:
                        logger.debug(f"    - Error logging mention: {e}")
        else:
            logger.debug("No coreference clusters detected")

    def _log_clusters(self, clusters: list):
        logger.debug(f"Resolved {len(clusters)} coreference clusters:")
        for i, cluster in enumerate(clusters):
            logger.debug(f"  Cluster {i+1} ({len(cluster)} mentions):")
            try:
                main_ref = cluster.main.text if hasattr(cluster, 'main') and hasattr(cluster.main, 'text') else "N/A"
            except:
                main_ref = "N/A"
            logger.debug(f"    Main reference: '{main_ref}'")
            for j, mention in enumerate(cluster):
                try:
                    if hasattr(mention, 'text'):
                        mention_text = mention.text
                        position = f"{mention.start_char}-{mention.end_char}" if hasattr(mention, 'start_char') else "unknown"
                    elif isinstance(mention, tuple) and len(mention) >= 2:
                        mention_text = str(mention[0]) if len(mention) > 0 else "unknown"
                        position = f"{mention[1]}-{mention[2]}" if len(mention) > 2 else "unknown"
                    else:
                        mention_text = str(mention)
                        position = "unknown"
                    logger.debug(f"    Mention {j+1}: '{mention_text}' (position: {position})")
                except Exception as e:
                    logger.debug(f"    Mention {j+1}: Error logging mention: {e}")

    def _translate_if_needed(self, text: str) -> str:
        if not self.translator:
            return text
        try:
            lang = detect(text)
            logger.debug(f"Detected language: {lang}")
            if lang.lower() == "nl":
                logger.debug("Translating Dutch text to English for coreference resolution")
                max_chunk_size = 400
                if len(text) > max_chunk_size:
                    chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
                    translated_chunks = []
                    for chunk in chunks:
                        translated = self.translator(chunk, max_length=512, truncation=True)
                        translated_chunks.append(translated[0]["translation_text"])
                    return " ".join(translated_chunks)
                else:
                    translated = self.translator(text, max_length=512, truncation=True)
                    return translated[0]["translation_text"]
            elif lang.lower() in ["en", "english"]:
                logger.debug("Text is already in English")
                return text
            else:
                logger.debug(f"Unsupported language {lang}, processing as-is")
                return text
        except Exception as e:
            logger.warning(f"Translation failed: {str(e)}, processing original text")
        return text

    def _get_cache_path(self, text: str) -> str:
        return os.path.join(self.cache_dir, f"{hashlib.sha256(text.encode('utf-8')).hexdigest()}.txt")
