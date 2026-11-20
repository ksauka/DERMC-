#!/usr/bin/env python3
"""
Synthetic E‑Commerce policy document generator for DERMC.

Features:
- Calls an LLM to generate hierarchical Markdown documents suitable for OIE.
- Validates structure (H1/H2, declarative statements, length bounds).
- De-duplicates against prior outputs and/or an existing CSV file.
- Writes CSV with columns: Doc ID, Answer ID, Document.

Usage (examples):
  python scripts/generate_synthetic_docs.py --count 50 --output E-Commerce_Documents_synth.csv
  python scripts/generate_synthetic_docs.py --count 100 --append --dedupe-against E-Commerce_Documents.csv

Environment:
  - Load API config from .env or environment variables.
    OPENAI_API_KEY, OPENAI_BASE_URL (optional), OPENAI_MODEL (default: gpt-4o-mini or gpt-4o-mini-2024-07-18)

Note: This script constructs Doc IDs automatically and sets Answer ID to 0 by default.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

from dotenv import load_dotenv

# Local validators
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
try:
    from scripts.validators import (
        normalize_text,
        validate_markdown_document,
        read_existing_csv_docs,
    )
except Exception as e:
    print(f"[WARN] Failed to import validators: {e}")
    # Fallback minimal validators (very basic)
    def normalize_text(t: str) -> str:
        return re.sub(r"\s+", " ", t.lower()).strip()

    def validate_markdown_document(text: str) -> Tuple[bool, List[str]]:
        msgs = []
        ok = True
        if not re.search(r"^# ", text, re.M):
            ok = False; msgs.append("missing H1 heading")
        if len(re.findall(r"^## ", text, re.M)) < 2:
            ok = False; msgs.append("needs at least two H2 headings")
        words = len(re.findall(r"\w+", text))
        if words < 250 or words > 1500:
            ok = False; msgs.append(f"word count out of range: {words}")
        if not re.search(r"\b(must|shall|is|are|includes|provides|requires)\b", text, re.I):
            ok = False; msgs.append("lacking declarative/constraint statements")
        return ok, msgs

    def read_existing_csv_docs(path: Path) -> List[str]:
        if not path.exists():
            return []
        docs = []
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                docs.append(row.get("Document", ""))
        return docs


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def load_ontology(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Could not load ontology at {path}: {e}")
        return {}


def infer_next_doc_id(csv_path: Path) -> int:
    if not csv_path.exists():
        return 1
    try:
        max_id = 0
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    did = int(row.get("Doc ID", "0").strip())
                    if did > max_id:
                        max_id = did
                except Exception:
                    continue
        return max_id + 1
    except Exception as e:
        print(f"[WARN] Failed to infer next Doc ID: {e}")
        return 1


def build_system_prompt(ontology: dict) -> str:
    entity_list = ", ".join(sorted(ontology.get("entities", {}).keys())) if ontology else (
        "Policy, Section, Requirement, Customer, Product, Payment Method, Return, Refund, Shipping, Order, Review, Support Channel, Security Measure, Data, Delivery Zone, Promotion, Tax, Warranty, Subscription, Account, Transaction"
    )
    relation_list = ", ".join(sorted(ontology.get("relations", {}).keys())) if ontology else (
        "has_section, has_requirement, applies_to, offers_feature, requires_payment_method, supports_return, grants_refund, covers_shipping, places_order, writes_review, uses_support_channel, implements_security_measure, processes_data, defines_delivery_zone, includes_promotion, charges_tax, includes_warranty, activates_subscription, owns_account, records_transaction"
    )

    return f"""
You are a careful technical writer creating realistic, diverse, hierarchical e‑commerce policy documents for an information extraction pipeline.

Hard rules:
- Output only raw Markdown documents, no code fences, no YAML, no CSV. Separate documents with a single line containing exactly: ---DOC---
- Each document MUST:
  - Start with one H1 (# Title) describing the policy type and scope.
  - Include at least three H2 (##) sections; use optional H3 (###) where helpful.
  - Write clear, declarative statements suitable for triplet extraction (e.g., "Customer must provide receipt", "Refund is issued within 5 business days"). Avoid ambiguous language.
  - Be 300–1200 words. Prefer concise, factual sentences; avoid marketing fluff.
  - Use bullets for lists of rules, exceptions, or steps. Keep sentences standalone.
  - Cover a wide range of policy domains (returns, shipping, warranty, payment, subscription, promotions, taxes, data privacy, security, account access, reviews) and vary geography or category (e.g., EU/US/APAC, electronics/apparel/grocery, digital vs physical).
  - Avoid personally identifiable data. Use neutral placeholders where needed.
- Do NOT include any text before the first H1 or after the final document separator.

Context for ontology alignment (do not output this literally):
- Entities: {entity_list}
- Relations: {relation_list}
Write statements that can imply these relations without labeling them.
""".strip()


def build_user_prompt(n_docs: int) -> str:
    return f"""
Generate {n_docs} distinct e‑commerce policy documents as raw Markdown.
Vary policy type, region, and product category. Ensure each document is self-contained and non-overlapping in content.
Output format: document1\n---DOC---\ndocument2\n---DOC---\n... (no numbering, no extra prose).
""".strip()


@dataclass
class LLMClient:
    model: str
    base_url: str | None
    api_key: str

    def _client(self):
        """Lazily import OpenAI SDK and return a unified client callable."""
        try:
            # New SDK style
            from openai import OpenAI  # type: ignore

            return OpenAI(api_key=self.api_key, base_url=self.base_url)
        except Exception:
            # Fallback to legacy interface
            import openai  # type: ignore

            if self.base_url:
                openai.api_base = self.base_url
            openai.api_key = self.api_key
            return openai

    def chat(self, system_prompt: str, user_prompt: str, max_tokens: int = 4096, temperature: float = 0.6) -> str:
        client = self._client()
        # Try new SDK signature first
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except AttributeError:
            # Legacy SDK
            resp = client.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp["choices"][0]["message"]["content"]


def split_documents(raw: str) -> List[str]:
    parts = re.split(r"^---DOC---\s*$", raw.strip(), flags=re.M)
    # Filter empty pieces and trim
    docs = [p.strip() for p in parts if p and p.strip()]
    return docs


def generate_batch(client: LLMClient, n_docs: int, ontology: dict, retries: int = 3, backoff: float = 2.0) -> List[str]:
    sys_prompt = build_system_prompt(ontology)
    usr_prompt = build_user_prompt(n_docs)
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            raw = client.chat(system_prompt=sys_prompt, user_prompt=usr_prompt)
            docs = split_documents(raw)
            if len(docs) < n_docs:
                raise ValueError(f"Expected {n_docs} docs, got {len(docs)}")
            return docs[:n_docs]
        except Exception as e:
            last_err = e
            sleep = backoff * attempt
            print(f"[WARN] Generation failed (attempt {attempt}/{retries}): {e}. Retrying in {sleep:.1f}s...")
            time.sleep(sleep)
    raise RuntimeError(f"Failed to generate batch after {retries} attempts: {last_err}")


def write_csv(output_path: Path, rows: Iterable[Tuple[int, int, str]], append: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["Doc ID", "Answer ID", "Document"]
    mode = "a" if append and output_path.exists() else "w"
    with output_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        for did, aid, doc in rows:
            writer.writerow({"Doc ID": did, "Answer ID": aid, "Document": doc})


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic e‑commerce policy documents as CSV")
    parser.add_argument("--count", type=int, default=50, help="Total documents to generate")
    parser.add_argument("--batch-size", type=int, default=10, help="LLM generation batch size")
    parser.add_argument("--output", type=str, default=str(ROOT_DIR / "E-Commerce_Documents_synth.csv"), help="Output CSV path")
    parser.add_argument("--append", action="store_true", help="Append to existing CSV and continue Doc IDs")
    parser.add_argument("--dedupe-against", type=str, default="", help="Optional CSV to dedupe against (e.g., existing corpus)")
    parser.add_argument("--answer-id", type=int, default=0, help="Answer ID value to write for each row")
    parser.add_argument("--model", type=str, default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), help="Model name")
    parser.add_argument("--base-url", type=str, default=os.getenv("OPENAI_BASE_URL", ""), help="Custom base URL (e.g., Azure/OpenAI-compatible)")
    parser.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY", ""), help="API key (or set OPENAI_API_KEY)")
    parser.add_argument("--ontology", type=str, default=str(ROOT_DIR / "domain_ontology.json"), help="Ontology JSON path for context")

    args = parser.parse_args()

    load_dotenv(override=False)

    if not args.api_key:
        print("[ERROR] Missing OPENAI_API_KEY. Set it in environment or .env.")
        sys.exit(1)

    output_path = Path(args.output).resolve()
    ontology = load_ontology(Path(args.ontology))

    client = LLMClient(
        model=args.model,
        base_url=(args.base_url or None),
        api_key=args.api_key,
    )

    # Build dedupe set
    seen_hashes = set()
    # Optionally include existing target CSV when appending
    if args.append and output_path.exists():
        existing_docs = read_existing_csv_docs(output_path)
        for d in existing_docs:
            seen_hashes.add(sha1(normalize_text(d)))
        print(f"[INFO] Loaded {len(existing_docs)} existing docs from target for dedupe")

    # Optionally include external CSV to dedupe against
    if args.dedupe_against:
        other = Path(args.dedupe_against).resolve()
        if other.exists():
            other_docs = read_existing_csv_docs(other)
            for d in other_docs:
                seen_hashes.add(sha1(normalize_text(d)))
            print(f"[INFO] Loaded {len(other_docs)} docs from --dedupe-against for dedupe")
        else:
            print(f"[WARN] --dedupe-against path not found: {other}")

    # Determine starting Doc ID
    start_doc_id = infer_next_doc_id(output_path) if args.append else 1

    remaining = args.count
    batch_size = max(1, args.batch_size)
    to_write: List[Tuple[int, int, str]] = []
    next_doc_id = start_doc_id

    while remaining > 0:
        n = min(batch_size, remaining)
        print(f"[INFO] Generating batch of {n} (remaining: {remaining})...")
        docs = generate_batch(client, n_docs=n, ontology=ontology)

        for doc in docs:
            norm = normalize_text(doc)
            h = sha1(norm)
            if h in seen_hashes:
                print("[SKIP] Duplicate detected; skipping a document")
                continue
            ok, msgs = validate_markdown_document(doc)
            if not ok:
                print(f"[SKIP] Validation failed: {', '.join(msgs)}")
                continue
            to_write.append((next_doc_id, args.answer_id, doc))
            seen_hashes.add(h)
            next_doc_id += 1

        remaining = args.count - (len(to_write))

        # Persist in chunks to avoid losing progress
        if to_write:
            write_csv(output_path, to_write, append=args.append or output_path.exists())
            args.append = True  # after first write, always append
            print(f"[INFO] Wrote {len(to_write)} rows to {output_path}")
            to_write.clear()

    print("[DONE] Synthetic generation complete.")


if __name__ == "__main__":
    main()
