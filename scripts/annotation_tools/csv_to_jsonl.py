#!/usr/bin/env python3
"""
Convert reviewed CSV back to annotated JSONL for Phase 2.2.
"""

import json
import pandas as pd
from pathlib import Path

ANNOTATIONS_DIR = Path(__file__).parent.parent.parent / "data" / "annotations"
CSV_FILE = ANNOTATIONS_DIR / "annotation_review.csv"
MASTER_FILE = ANNOTATIONS_DIR / "annotation_master.jsonl"
OUTPUT_FILE = ANNOTATIONS_DIR / "annotation_master_labeled.jsonl"

def main():
    print("Loading CSV...")
    df = pd.read_csv(CSV_FILE)
    print(f"Loaded {len(df)} rows")
    
    # Load original for full transcript
    originals = {}
    with open(MASTER_FILE, "r") as f:
        for line in f:
            d = json.loads(line)
            originals[d["conversation_id"]] = d
    
    labeled = []
    for _, row in df.iterrows():
        conv_id = row["conversation_id"]
        if conv_id not in originals:
            print(f"Warning: {conv_id} not in original")
            continue
        
        base = originals[conv_id].copy()
        
        # Use corrected_intent if provided, else predicted
        intent = row.get("corrected_intent", row.get("predicted_intent"))
        if pd.isna(intent):
            continue
            
        base["annotated_intent"] = intent
        base["annotated_entities"] = eval(row["entities"]) if pd.notna(row.get("entities")) and row["entities"] else None
        base["annotated_resolution"] = row.get("resolution", "unclear") if pd.notna(row.get("resolution")) else "unclear"
        base["annotator_notes"] = row.get("notes", "") if pd.notna(row.get("notes")) else ""
        base["annotator_id"] = row.get("annotator_id", "reviewer") if pd.notna(row.get("annotator_id")) else "reviewer"
        
        labeled.append(base)
    
    with open(OUTPUT_FILE, "w") as f:
        for item in labeled:
            f.write(json.dumps(item, default=str) + "\n")
    
    print(f"Saved {len(labeled)} labeled conversations to {OUTPUT_FILE}")
    
    # Quick stats
    intents = [d["annotated_intent"] for d in labeled]
    print("\nIntent distribution:")
    for intent, count in pd.Series(intents).value_counts().items():
        print(f"  {intent}: {count}")

if __name__ == "__main__":
    main()