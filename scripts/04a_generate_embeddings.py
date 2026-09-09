#!/usr/bin/env python3
"""
Phase 4a: Generate embeddings with checkpointing.
Saves embeddings and examples every N examples to allow resumption.
"""

import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import pickle

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INDEX_DIR = PROJECT_ROOT / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
CHECKPOINT_INTERVAL = 5000

# Checkpoint files
EMBEDDINGS_PATH = INDEX_DIR / "resolution_embeddings.npy"
EXAMPLES_PATH = INDEX_DIR / "resolution_examples.jsonl"
CHECKPOINT_DIR = INDEX_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def load_resolved_conversations() -> List[Dict[str, Any]]:
    """Load all resolved candidate conversations."""
    conv_path = PROCESSED_DIR / "amazonhelp_conversations.jsonl"
    conversations = []
    with open(conv_path, "r") as f:
        for line in f:
            conv = json.loads(line)
            if conv.get("resolved_candidate", False):
                conversations.append(conv)
    logger.info(f"Loaded {len(conversations)} resolved candidate conversations")
    return conversations


def load_intent_classifier():
    """Load trained intent classifier."""
    import joblib
    model_path = PROJECT_ROOT / "models" / "intent_classifier_v2.joblib"
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)
    clf = joblib.load(model_path)
    logger.info("Loaded intent classifier v2")
    return clf


def prepare_resolution_examples(conversations: List[Dict[str, Any]], clf) -> List[Dict[str, Any]]:
    """Prepare training examples from resolved conversations."""
    logger.info("Preparing resolution examples...")
    
    examples = []
    
    for conv in conversations:
        first_customer = None
        for msg in conv["messages"]:
            if msg["speaker"] == "customer":
                first_customer = msg["text"]
                break
        
        if not first_customer:
            continue
        
        import re
        def clean_text(text):
            text = re.sub(r"https?://\S+", "", text)
            text = re.sub(r"@\w+", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text
        
        text_clean = clean_text(first_customer)
        pred_intent = clf.predict([text_clean])[0]
        confidence = max(clf.predict_proba([text_clean])[0])
        
        support_msgs = [msg for msg in conv["messages"] if msg["speaker"] == "support"]
        if not support_msgs:
            continue
        
        resolution_text = " | ".join([m["text"] for m in support_msgs])
        resolution_clean = clean_text(resolution_text)
        
        if len(resolution_clean) < 20:
            continue
        
        examples.append({
            "conversation_id": conv["conversation_id"],
            "customer_query": first_customer,
            "customer_query_clean": text_clean,
            "predicted_intent": pred_intent,
            "intent_confidence": float(confidence),
            "resolution": resolution_text,
            "resolution_clean": resolution_clean,
            "support_message_count": len(support_msgs),
            "resolution_score": conv.get("resolution_score", 0),
            "message_count": len(conv["messages"])
        })
    
    logger.info(f"Prepared {len(examples)} resolution examples")
    return examples


def save_checkpoint(embeddings: np.ndarray, examples: List[Dict], checkpoint_id: int):
    """Save checkpoint."""
    emb_path = CHECKPOINT_DIR / f"embeddings_{checkpoint_id}.npy"
    ex_path = CHECKPOINT_DIR / f"examples_{checkpoint_id}.jsonl"
    
    np.save(emb_path, embeddings)
    with open(ex_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, default=str) + "\n")
    logger.info(f"Checkpoint {checkpoint_id} saved: {len(examples)} examples")


def load_latest_checkpoint() -> Tuple[List[Dict], np.ndarray, int]:
    """Load latest checkpoint if exists."""
    checkpoints = list(CHECKPOINT_DIR.glob("embeddings_*.npy"))
    if not checkpoints:
        return [], np.array([]), 0
    
    latest = max(checkpoints, key=lambda p: int(p.stem.split("_")[1]))
    checkpoint_id = int(latest.stem.split("_")[1])
    
    embeddings = np.load(latest)
    ex_path = CHECKPOINT_DIR / f"examples_{checkpoint_id}.jsonl"
    
    examples = []
    with open(ex_path, "r") as f:
        for line in f:
            examples.append(json.loads(line))
    
    logger.info(f"Loaded checkpoint {checkpoint_id}: {len(examples)} examples")
    return examples, embeddings, checkpoint_id


def generate_embeddings_incremental(
    texts: List[str],
    model: SentenceTransformer,
    start_idx: int,
    existing_embeddings: np.ndarray,
    existing_examples: List[Dict]
) -> Tuple[np.ndarray, List[Dict]]:
    """Generate embeddings incrementally with checkpointing."""
    
    batch_size = 64
    all_embeddings = list(existing_embeddings) if len(existing_embeddings) > 0 else []
    all_examples = existing_examples[:]
    
    for i in range(start_idx, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch_texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
        
        all_embeddings.extend(batch_embeddings)
        
        # Save checkpoint every CHECKPOINT_INTERVAL
        if (i + 1) % CHECKPOINT_INTERVAL == 0 or i + batch_size >= len(texts):
            checkpoint_id = (i + batch_size) // CHECKPOINT_INTERVAL
            save_checkpoint(np.array(all_embeddings), all_examples, checkpoint_id)
            logger.info(f"Progress: {min(i+batch_size, len(texts))}/{len(texts)}")
    
    return np.array(all_embeddings), all_examples


def main():
    logger.info("Starting Phase 4a: Generate Embeddings (with checkpoints)")
    
    # Load conversations and classifier
    conversations = load_resolved_conversations()
    clf = load_intent_classifier()
    
    # Prepare examples
    examples = prepare_resolution_examples(conversations, clf)
    
    # Extract texts
    texts = [ex["customer_query_clean"] for ex in examples]
    logger.info(f"Total texts to embed: {len(texts)}")
    
    # Check for existing checkpoint
    existing_examples, existing_embeddings, checkpoint_id = load_latest_checkpoint()
    start_idx = len(existing_examples)
    
    if start_idx > 0:
        logger.info(f"Resuming from checkpoint {checkpoint_id}: {start_idx}/{len(texts)}")
        examples = existing_examples
    else:
        logger.info("Starting fresh")
    
    # Load embedding model
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    # Generate embeddings
    logger.info("Generating embeddings...")
    embeddings, examples = generate_embeddings_incremental(
        texts, model, start_idx, existing_embeddings, examples
    )
    
    # Normalize
    faiss.normalize_L2(embeddings)
    
    # Save final
    logger.info("Saving final results...")
    np.save(EMBEDDINGS_PATH, embeddings)
    
    with open(EXAMPLES_PATH, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, default=str) + "\n")
    
    logger.info(f"Saved {len(examples)} examples and embeddings to {INDEX_DIR}")
    logger.info("Phase 4a complete! Run Phase 4b to build index.")


if __name__ == "__main__":
    main()