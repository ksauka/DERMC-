"""
Unified LLM client that supports:
1. Multiple backends (OpenAI API, Ollama local)
2. Automatic retry logic with exponential backoff
3. Batch generation for efficiency
4. Fallback mechanisms for quota/rate limits

Requirements:
    pip install openai httpx
"""
import os
import time
import logging
from typing import List
from tqdm import tqdm
import httpx
from openai import OpenAI

logger = logging.getLogger(__name__)

class LLMClient:
    """
    Unified LLM client supporting both OpenAI and Ollama backends.
    Set backend to 'openai' or 'ollama' for clarity.
    """
    def __init__(self, model: str = "gpt-3.5-turbo", max_retries: int = 3, retry_delay: float = 1.0,
                 backend: str = "openai", ollama_model: str = "llama3.2",
                 ollama_url: str = "http://localhost:11434/api/chat"):
        """
        Args:
            model: OpenAI model name.
            max_retries: Max retries for LLM calls.
            retry_delay: Delay between retries (seconds).
            backend: 'openai' or 'ollama'.
            ollama_model: Ollama model name.
            ollama_url: Ollama API URL.
        """
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backend = backend.lower()
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        # Allow environment variables to override constructor defaults
        env_backend = os.getenv("LLM_BACKEND", "").strip().lower()
        if env_backend in ("openai", "ollama"):
            self.backend = env_backend
        self.ollama_model = os.getenv("OLLAMA_MODEL", self.ollama_model)
        self.ollama_url = os.getenv("OLLAMA_URL", self.ollama_url)

        # Initialize appropriate backend client
        self.use_ollama = self.backend == "ollama"
        if self.use_ollama:
            logger.info(f"Using Ollama model: {ollama_model}")
            self._test_ollama_connection()  # Verify Ollama service is running
        else:
            logger.info(f"Using OpenAI model: {model}")
            self.client = self._initialize_openai_client()

        # Whether to attempt automatic fallback to Ollama on quota/rate errors
        self.enable_fallback = os.getenv("LLM_ENABLE_OLLAMA_FALLBACK", "true").lower() in ("1", "true", "yes")

    def _initialize_openai_client(self) -> OpenAI:
        proxy_url = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
        transport = httpx.HTTPTransport(proxy=proxy_url) if proxy_url else None
        base_url = os.getenv("OPENAI_BASE_URL")
        return OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            http_client=httpx.Client(transport=transport) if transport else None,
            base_url=base_url if base_url else None,
        )

    def _test_ollama_connection(self) -> None:
        try:
            with httpx.Client() as client:
                response = client.post(
                    self.ollama_url,
                    json={
                        "model": self.ollama_model,
                        "messages": [{"role": "user", "content": "Hello"}],
                        "stream": False,
                    },
                    timeout=10,
                )
                if response.status_code == 200:
                    logger.info("Ollama connection successful")
                else:
                    logger.warning(f"Ollama responded with status {response.status_code}")
        except Exception as e:
            logger.error(f"Ollama connection failed: {e}")
            logger.error("Please ensure Ollama is running with: ollama serve")
            raise ConnectionError("Ollama is not available. Please start Ollama service.")

    def _generate_openai(self, prompt: str) -> str:
        """Call OpenAI API with retry logic and optional Ollama fallback on quota errors."""
        for attempt in range(self.max_retries + 1):
            try:
                # Auto-detect JSON mode if prompt mentions JSON
                use_json_format = "json" in prompt.lower()
                request_params = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 512
                }
                if use_json_format:
                    request_params["response_format"] = {"type": "json_object"}
                response = self.client.chat.completions.create(**request_params)
                return response.choices[0].message.content.strip()
            except Exception as e:
                # Handle rate limits / quota exceeded by falling back to Ollama when enabled
                msg = str(e).lower()
                is_quota = "insufficient_quota" in msg or "you exceeded your current quota" in msg
                is_rate = "429" in msg or "rate limit" in msg
                if self.enable_fallback and (is_quota or is_rate):
                    logger.warning("OpenAI quota/rate issue detected; attempting fallback to Ollama...")
                    try:
                        # Switch backend to Ollama dynamically
                        self.backend = "ollama"
                        self.use_ollama = True
                        self._test_ollama_connection()
                        return self._generate_ollama(prompt)
                    except Exception as fe:
                        logger.error(f"Ollama fallback failed: {fe}")
                        # Continue retrying OpenAI below
                if attempt == self.max_retries:
                    logger.error(f"OpenAI generation failed after {self.max_retries + 1} attempts: {e}")
                    return ""
                delay = self.retry_delay * (2 ** attempt)
                logger.warning(f"OpenAI attempt {attempt + 1} failed. Retrying in {delay}s... Error: {e}")
                time.sleep(delay)
        return ""

    def _generate_ollama(self, prompt: str) -> str:
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client() as client:
                    response = client.post(
                        self.ollama_url,
                        json={
                            "model": self.ollama_model,
                            "messages": [
                                {"role": "system", "content": "You are a helpful AI assistant specialized in knowledge extraction and natural language processing."},
                                {"role": "user", "content": prompt}
                            ],
                            "stream": False,
                            "options": {
                                "temperature": 0.3,
                                "top_p": 0.9,
                                "num_predict": 512
                            }
                        },
                        timeout=60,
                    )
                    response.raise_for_status()
                    data = response.json()
                    return data["message"]["content"].strip()
            except Exception as e:
                if attempt == self.max_retries:
                    logger.error(f"Ollama generation failed after {self.max_retries + 1} attempts: {e}")
                    return ""
                delay = self.retry_delay * (2 ** attempt)
                logger.warning(f"Ollama attempt {attempt + 1} failed. Retrying in {delay}s... Error: {e}")
                time.sleep(delay)
        return ""

    def generate(self, prompt: str) -> str:
        """
        Generate a single response from the LLM backend.
        """
        if self.use_ollama:
            return self._generate_ollama(prompt)
        else:
            return self._generate_openai(prompt)

    def generate_batch(self, prompts: List[str], parallel: bool = True, max_workers: int = 8) -> List[str]:
        """
        Generate responses for a batch of prompts, optionally in parallel.
        Args:
            prompts: List of prompt strings.
            parallel: Whether to use parallel execution.
            max_workers: Number of parallel workers.
        Returns:
            List of response strings (empty string for failed prompts).
        """
        # Allow environment to control concurrency to avoid rate limits
        env_parallel = os.getenv("LLM_PARALLEL")
        if env_parallel is not None:
            parallel = env_parallel.strip().lower() in ("1", "true", "yes")
        try:
            env_workers = int(os.getenv("LLM_MAX_WORKERS", "0"))
            if env_workers > 0:
                max_workers = max(1, env_workers)
        except Exception:
            pass
        backend = "Ollama" if self.use_ollama else "OpenAI"
        results: List[str] = ["" for _ in prompts]
        if parallel:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {executor.submit(self.generate, prompt): idx for idx, prompt in enumerate(prompts)}
                for future in tqdm(as_completed(future_to_idx), total=len(prompts), desc=f"Generating responses ({backend}, parallel)"):
                    idx = future_to_idx[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        logger.error(f"LLM batch generation failed for prompt {idx}: {e}")
                        results[idx] = ""
        else:
            for idx, prompt in enumerate(tqdm(prompts, desc=f"Generating responses ({backend})")):
                try:
                    results[idx] = self.generate(prompt)
                except Exception as e:
                    logger.error(f"LLM batch generation failed for prompt {idx}: {e}")
                    results[idx] = ""
        return results

    # Allow env to tune retries without code changes
    def _apply_env_tuning(self):
        try:
            mr = int(os.getenv("LLM_MAX_RETRIES", "0"))
            if mr > 0:
                self.max_retries = mr
        except Exception:
            pass
        try:
            rd = float(os.getenv("LLM_RETRY_DELAY", "0"))
            if rd > 0:
                self.retry_delay = rd
        except Exception:
            pass
