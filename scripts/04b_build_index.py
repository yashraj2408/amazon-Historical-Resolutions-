#!/usr/bin/env python3
"""
Phase 4b: Build FAISS index from pre-computed embeddings (checkpoint-based).
Run this after embeddings are generated.
"""

import json
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict

import numpy as np
import faiss

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
INDEX_DIR = PROJECT_ROOT / "index"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_DIM = 384

# Checkpoint files
EMBEDDINGS_PATH = INDEX_DIR / "resolution_embeddings.npy"
EXAMPLES_PATH = INDEX_DIR / "resolution_examples.jsonl"
META_PATH = INDEX_DIR / "resolution_metadata.jsonl"
CHECKPOINT_DIR = INDEX_DIR / "checkpoints"

def load_latest_checkpoint() -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Load latest checkpoint."""
    embeddings_files = list(CHECKPOINT_DIR.glob("embeddings_*.npy"))
    if not embeddings_files:
        logger.error(f"No checkpoint embeddings found in {CHECKPOINT_DIR}")
        logger.error("Run Phase 4a first to generate embeddings")
        sys.exit(1)
    
    latest_emb = max(embeddings_files, key=lambda p: int(p.stem.split("_")[1]))
    checkpoint_id = int(latest_emb.stem.split("_")[1])
    ex_file = CHECKPOINT_DIR / f"examples_{checkpoint_id}.jsonl"
    
    embeddings = np.load(latest_emb)
    examples = []
    with open(ex_file, "r") as f:
        for line in f:
            examples.append(json.loads(line))
    
    logger.info(f"Loaded checkpoint {checkpoint_id}: {len(examples)} examples, embeddings shape {embeddings.shape}")
    return examples, embeddings


def build_and_save_index(embeddings: np.ndarray, examples: List[Dict[str, Any]]):
    """Build FAISS index and save everything."""
    
    logger.info("Building FAISS index...")
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings.astype(np.float32))
    logger.info(f"Built index with {index.ntotal} vectors")
    
    # Save FAISS index
    index_path = INDEX_DIR / "resolution_index.faiss"
    faiss.write_index(index, str(index_path))
    logger.info(f"Saved FAISS index to {index_path}")
    
    # Save intent-filtered indices
    logger.info("Creating intent-filtered indices...")
    by_intent = defaultdict(list)
    for i, ex in enumerate(examples):
        intent = ex.get("predicted_intent", "unknown")
        by_intent[intent].append((i, ex))
    
    for intent, items in by_intent.items():
        if len(items) < 2:
            continue
        
        indices = [item[0] for item in items]
        intent_embeddings = embeddings[indices].astype(np.float32)
        
        intent_idx = faiss.IndexFlatIP(EMBEDDING_DIM)
        intent_idx.add(intent_embeddings)
        
        intent_idx_path = INDEX_DIR / f"index_{intent}.faiss"
        faiss.write_index(intent_idx, str(intent_idx_path))
        
        # Save intent metadata
        meta_path = INDEX_DIR / f"metadata_{intent}.jsonl"
        with open(meta_path, "w") as f:
            for _, ex in items:
                f.write(json.dumps(ex, default=str) + "\n")
        
        logger.info(f"  {intent}: {len(items)} vectors saved")
    
    # Save metadata
    meta_path = INDEX_DIR / "resolution_metadata.jsonl"
    with open(meta_path, "w") as f:
        for i, ex in enumerate(examples):
            meta = ex.copy()
            meta["index_id"] = i
            f.write(json.dumps(meta, default=str) + "\n")
    logger.info(f"Saved metadata to {meta_path}")
    
    # Save intent distribution
    intent_dist = defaultdict(int)
    for ex in examples:
        intent_dist[ex.get("predicted_intent", "unknown")] += 1
    
    with open(INDEX_DIR / "intent_distribution.json", "w") as f:
        json.dump(dict(intent_dist), f, indent=2)
    
    logger.info("Phase 4b complete!")


def test_retrieval():
    """Test the built index."""
    logger.info("Testing retrieval...")
    
    # Load index
    index_path = INDEX_DIR / "resolution_index.faiss"
    if not index_path.exists():
        logger.error("Index not found")
        return
    
    index = faiss.read_index(str(index_path))
    logger.info(f"Loaded index with {index.ntotal} vectors")
    
    # Load examples and embeddings
    examples, embeddings = load_latest_checkpoint()
    
    # Load model for query encoding
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    def clean_text(text: str) -> str:
        import re
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    # Load classifier
    import joblib
    clf = joblib.load(PROJECT_ROOT / "models" / "intent_classifier_v2.joblib")
    
    # Test queries
    test_queries = [
        "@AmazonHelp my package says delivered but I never received it!",
        "@AmazonHelp I was charged twice for my Prime membership",
        "@AmazonHelp my Echo Dot won't connect to wifi",
        "@AmazonHelp I want to return this item, how do I get a refund?",
        "@AmazonHelp I can't log into my account, password reset not working"
    ]
    
    print("\n" + "=" * 80)
    print("RETRIEVAL TEST")
    print("=" * 80)
    
    for query in test_queries:
        query_clean = clean_text(query)
        pred_intent = clf.predict([query_clean])[0]
        query_emb = model.encode([query_clean], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)
        
        # General search
        scores, indices = index.search(query_emb.astype(np.float32), 5)
        
        print(f"\nQuery: {query[:80]}...")
        print(f"Predicted Intent: {pred_intent}")
        print("-" * 60)
        
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
            if idx >= len(examples):
                continue
            ex = examples[idx]
            print(f"  #{rank} score={score:.3f} [{ex['predicted_intent']}]")
            print(f"    Customer: {ex['customer_query'][:100]}")
            print(f"    Resolution: {ex['resolution'][:150]}")
        
        # Intent-filtered
        if pred_intent in [ex.get("predicted_intent") for ex in examples]:
            filtered_scores = []
            for i, ex in enumerate(examples):
                if ex.get("predicted_intent") == pred_intent:
                    sim = np.dot(query_emb[0], embeddings[i])  # cosine since normalized
                    filtered_scores.append((i, sim))
            
            filtered_scores.sort(key=lambda x: -x[1])
            
            print(f"\n  Filtered by intent '{pred_intent}':")
            for rank, (idx, score) in enumerate(filtered_scores[:3], 1):
                ex = examples[idx]
                print(f"    #{rank} score={score:.3f}: {ex['resolution'][:120]}")


def main():
    logger.info("Starting Phase 4b: Build Index from Checkpoints")
    
    # Load data from checkpoints
    examples, embeddings = load_latest_checkpoint()
    
    if len(examples) != embeddings.shape[0]:
        logger.error(f"Mismatch: {len(examples)} examples vs {embeddings.shape[0]} embeddings")
        sys.exit(1)
    
    # Build and save
    build_and_save_index(embeddings, examples)
    
    # Test
    test_retrieval()
    
    logger.info("Phase 4b complete!")


if __name__ == "__main__":
    main()