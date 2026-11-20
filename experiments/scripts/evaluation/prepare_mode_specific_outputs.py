#!/usr/bin/env python3

"""
Prepare mode-specific per-dataset outputs
========================================

This script uses the consolidated outputs in `enhanced_dermc_output/` and
filters them by Doc ID to materialize per-dataset (10, 20, 50 docs) files
with the expected filenames:

  - oie_triplets_flat.csv / oie_triplets_hierarchical.csv
  - refined_triplets_flat.csv / refined_triplets_hierarchical.csv
  - canonical_kg_flat.csv / canonical_kg_hierarchical.csv

Outputs are written into existing directories:
  - enhanced_dermc_output_test10/
  - enhanced_dermc_output_test20/
  - enhanced_dermc_output_test50/

This avoids rerunning the full pipeline and leverages existing data.
"""

from pathlib import Path
import pandas as pd

BASE = Path(__file__).parent.parent

CONSOLIDATED = BASE / "enhanced_dermc_output"
TEST_DIRS = {
    10: BASE / "enhanced_dermc_output_test10",
    20: BASE / "enhanced_dermc_output_test20",
    50: BASE / "enhanced_dermc_output_test50",
}

STAGES = [
    ("oie_triplets", "oie_triplets_flat.csv", "oie_triplets_hierarchical.csv"),
    ("refined_triplets", "refined_triplets_flat.csv", "refined_triplets_hierarchical.csv"),
    ("canonical_kg", "canonical_kg_flat.csv", "canonical_kg_hierarchical.csv"),
]

def ensure_dirs():
    for p in TEST_DIRS.values():
        p.mkdir(parents=True, exist_ok=True)

def get_doc_ids(n: int) -> list:
    # We assume sequential doc IDs: doc_1..doc_n
    return [f"doc_{i}" for i in range(1, n + 1)]

def filter_by_doc_ids(df: pd.DataFrame, doc_ids: list) -> pd.DataFrame:
    col = None
    for candidate in ["Doc ID", "doc_id", "docID", "doc"]:
        if candidate in df.columns:
            col = candidate
            break
    if col is None:
        # If missing column, return empty to be safe
        return df.iloc[0:0].copy()
    return df[df[col].astype(str).isin(doc_ids)].copy()

def prepare_subset(n: int):
    doc_ids = get_doc_ids(n)
    out_dir = TEST_DIRS[n]

    for stage, flat_name, hier_name in STAGES:
        flat_file = CONSOLIDATED / flat_name
        hier_file = CONSOLIDATED / hier_name

        if flat_file.exists():
            df = pd.read_csv(flat_file)
            sub = filter_by_doc_ids(df, doc_ids)
            sub.to_csv(out_dir / flat_name, index=False)
            print(f"[SUCCESS] Wrote {len(sub)} rows -> {out_dir / flat_name}")
        else:
            print(f"[WARNING]  Missing consolidated file: {flat_file}")

        if hier_file.exists():
            df = pd.read_csv(hier_file)
            sub = filter_by_doc_ids(df, doc_ids)
            sub.to_csv(out_dir / hier_name, index=False)
            print(f"[SUCCESS] Wrote {len(sub)} rows -> {out_dir / hier_name}")
        else:
            print(f"[WARNING]  Missing consolidated file: {hier_file}")

def main():
    ensure_dirs()
    for n in [10, 20, 50]:
        print(f"\nPreparing subset for first {n} documents...")
        prepare_subset(n)
    print("\nPreparation complete. Mode-specific files are ready in test directories.")

if __name__ == "__main__":
    main()
