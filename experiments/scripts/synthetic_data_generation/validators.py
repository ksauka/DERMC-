"""Validation helpers for synthetic document generation."""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import List, Tuple


def normalize_text(t: str) -> str:
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


def validate_markdown_document(
    text: str,
    *,
    min_words: int = 300,
    max_words: int = 1200,
    min_h2: int = 3,
) -> Tuple[bool, List[str]]:
    msgs: List[str] = []
    ok = True

    # H1 present at start or within first 5 lines
    if not re.search(r"^# ", text, re.M):
        ok = False; msgs.append("missing H1 heading")

    # At least min_h2 H2 sections
    if len(re.findall(r"^## ", text, re.M)) < min_h2:
        ok = False; msgs.append(f"needs at least {min_h2} H2 headings")

    # Prefer short, declarative sentences
    if not re.search(r"\b(must|shall|is|are|includes|provides|requires|applies|covers)\b", text, re.I):
        ok = False; msgs.append("lacking declarative/constraint statements")

    # Word count bounds
    words = len(re.findall(r"\w+", text))
    if words < min_words or words > max_words:
        ok = False; msgs.append(f"word count out of range: {words}")

    # Ensure no fenced code blocks or YAML frontmatter
    if re.search(r"^---\s*$", text, re.M) and not re.search(r"^---DOC---\s*$", text, re.M):
        ok = False; msgs.append("unexpected '---' blocks; use only ---DOC--- as separator")
    if re.search(r"^```", text, re.M):
        ok = False; msgs.append("remove code fences")

    return ok, msgs


def read_existing_csv_docs(path: Path) -> List[str]:
    if not path.exists():
        return []
    docs: List[str] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            docs.append(row.get("Document", ""))
    return docs
