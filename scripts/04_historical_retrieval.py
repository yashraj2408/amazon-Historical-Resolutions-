#!/usr/bin/env python3
"""
Phase 4: Historical Resolution Retrieval
Build vector index of resolved conversations for grounded response generation.
"""

import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
INDEX_DIR = PROJECT_ROOT / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
TOP_K = 5


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
    model_path = MODELS_DIR / "intent_classifier_v2.joblib"
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)
    clf = joblib.load(model_path)
    logger.info("Loaded intent classifier v2")
    return clf


def prepare_resolution_examples(conversations: List[Dict[str, Any]], clf) -> List[Dict[str, Any]]:
    """Prepare training examples from resolved conversations with intent labels."""
    logger.info("Preparing resolution examples...")
    
    examples = []
    intent_counts = defaultdict(int)
    
    for conv in conversations:
        # Get first customer message
        first_customer = None
        for msg in conv["messages"]:
            if msg["speaker"] == "customer":
                first_customer = msg["text"]
                break
        
        if not first_customer:
            continue
        
        # Predict intent
        import re
        def clean_text(text):
            text = re.sub(r"https?://\S+", "", text)
            text = re.sub(r"@\w+", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text
        
        text_clean = clean_text(first_customer)
        pred_intent = clf.predict([text_clean])[0]
        confidence = max(clf.predict_proba([text_clean])[0])
        
        # Get support responses (resolution)
        support_msgs = [msg for msg in conv["messages"] if msg["speaker"] == "support"]
        if not support_msgs:
            continue
        
        # Combine support responses as resolution
        resolution_text = " | ".join([m["text"] for m in support_msgs])
        resolution_clean = clean_text(resolution_text)
        
        # Only keep if resolution is substantial
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
        intent_counts[pred_intent] += 1
    
    logger.info(f"Prepared {len(examples)} resolution examples")
    for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {intent}: {count}")
    
    return examples


def create_embeddings(texts: List[str], model_name: str = EMBEDDING_MODEL) -> np.ndarray:
    """Create embeddings for texts."""
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    logger.info(f"Encoding {len(texts)} texts...")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Build FAISS index for similarity search."""
    logger.info("Building FAISS index...")
    
    # Use IndexFlatIP for cosine similarity (since embeddings are normalized)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings.astype(np.float32))
    
    logger.info(f"Built index with {index.ntotal} vectors")
    return index


def save_index_and_metadata(
    index: faiss.Index,
    examples: List[Dict[str, Any]],
    embeddings: np.ndarray
):
    """Save FAISS index and metadata."""
    
    # Save FAISS index
    index_path = INDEX_DIR / "resolution_index.faiss"
    faiss.write_index(index, str(index_path))
    logger.info(f"Saved FAISS index to {index_path}")
    
    # Save embeddings
    emb_path = INDEX_DIR / "resolution_embeddings.npy"
    np.save(emb_path, embeddings)
    logger.info(f"Saved embeddings to {emb_path}")
    
    # Save metadata (without embeddings)
    metadata = []
    for i, ex in enumerate(examples):
        meta = ex.copy()
        meta["index_id"] = i
        metadata.append(meta)
    
    meta_path = INDEX_DIR / "resolution_metadata.jsonl"
    with open(meta_path, "w") as f:
        for m in metadata:
            f.write(json.dumps(m, default=str) + "\n")
    logger.info(f"Saved metadata to {meta_path}")
    
    # Save as CSV for inspection
    df = pd.DataFrame(metadata)
    df.to_csv(INDEX_DIR / "resolution_metadata.csv", index=False)
    
    # Save intent distribution
    intent_dist = df["predicted_intent"].value_counts().to_dict()
    with open(INDEX_DIR / "intent_distribution.json", "w") as f:
        json.dump(intent_dist, f, indent=2)


def test_retrieval(
    index: faiss.Index,
    examples: List[Dict[str, Any]],
    clf,
    model: SentenceTransformer,
    test_queries: List[str] = None
):
    """Test retrieval with sample queries."""
    
    if test_queries is None:
        test_queries = [
            "@AmazonHelp my package says delivered but I never received it!",
            "@AmazonHelp I was charged twice for my Prime membership",
            "@AmazonHelp my Echo Dot won't connect to wifi",
            "@AmazonHelp I want to return this item, how do I get a refund?",
            "@AmazonHelp I can't log into my account, password reset not working"
        ]
    
    logger.info("Testing retrieval...")
    print("\n" + "=" * 80)
    print("RETRIEVAL TEST RESULTS")
    print("=" * 80)
    
    import re
    def clean_text(text):
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    for query in test_queries:
        # Predict intent
        query_clean = clean_text(query)
        pred_intent = clf.predict([query_clean])[0]
        
        # Embed query
        query_emb = model.encode([query_clean], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)
        
        # Search
        scores, indices = index.search(query_emb.astype(np.float32), TOP_K)
        
        print(f"\nQuery: {query[:80]}...")
        print(f"Predicted Intent: {pred_intent}")
        print(f"Top {TOP_K} Similar Resolved Conversations:")
        print("-" * 80)
        
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
            if idx >= len(examples):
                continue
            ex = examples[idx]
            print(f"\n  #{rank} (score: {score:.3f}) | Intent: {ex['predicted_intent']} | Conv: {ex['conversation_id']}")
            print(f"  Customer: {ex['customer_query'][:120]}")
            print(f"  Resolution: {ex['resolution'][:200]}")


def filter_by_intent(
    index: faiss.Index,
    examples: List[Dict[str, Any]],
    query_intent: str,
    query_emb: np.ndarray,
    top_k: int = TOP_K
) -> List[Dict[str, Any]]:
    """Retrieve top-k results filtered by intent."""
    # Search with larger k to allow filtering
    search_k = min(top_k * 5, index.ntotal)
    scores, indices = index.search(query_emb.astype(np.float32), search_k)
    
    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx >= len(examples):
            continue
        ex = examples[idx]
        if ex["predicted_intent"] == query_intent:
            result = ex.copy()
            result["similarity_score"] = float(score)
            results.append(result)
            if len(results) >= top_k:
                break
    
    return results


def create_intent_filtered_index(examples: List[Dict[str, Any]], embeddings: np.ndarray) -> Dict[str, faiss.Index]:
    """Create separate FAISS index per intent for filtered retrieval."""
    logger.info("Creating intent-filtered indices...")
    
    intent_indices = {}
    for intent in set(ex["predicted_intent"] for ex in examples):
        intent_examples = [ex for ex in examples if ex["predicted_intent"] == intent]
        if len(intent_examples) < 2:
            continue
        
        intent_embeddings = np.array([ex["embedding"] for ex in intent_examples])
        idx = faiss.IndexFlatIP(EMBEDDING_DIM)
        idx.add(intent_embeddings.astype(np.float32))
        intent_indices[intent] = {
            "index": idx,
            "examples": intent_examples,
            "embeddings": intent_embeddings
        }
        logger.info(f"  {intent}: {len(intent_examples)} vectors")
    
    return intent_indices


def main():
    logger.info("Starting Phase 4: Historical Resolution Retrieval")
    
    # Load resolved conversations
    conversations = load_resolved_conversations()
    
    # Load intent classifier
    clf = load_intent_classifier()
    
    # Prepare resolution examples
    examples = prepare_resolution_examples(conversations, clf)
    
    if not examples:
        logger.error("No resolution examples prepared!")
        sys.exit(1)
    
    # Extract customer queries for embedding
    query_texts = [ex["customer_query_clean"] for ex in examples]
    
    # Create embeddings
    logger.info("Creating embeddings for customer queries...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(query_texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    
    # Store embeddings in examples
    for i, ex in enumerate(examples):
        ex["embedding"] = embeddings[i].tolist()
    
    # Build main FAISS index
    index = build_faiss_index(embeddings)
    
    # Create intent-filtered indices
    intent_indices = create_intent_filtered_index(examples, embeddings)
    
    # Save everything
    save_index_and_metadata(index, examples, embeddings)
    
    # Save intent-filtered indices
    for intent, data in intent_indices.items():
        intent_idx_path = INDEX_DIR / f"index_{intent}.faiss"
        faiss.write_index(data["index"], str(intent_idx_path))
        
        meta_path = INDEX_DIR / f"metadata_{intent}.jsonl"
        with open(meta_path, "w") as f:
            for ex in data["examples"]:
                f.write(json.dumps(ex, default=str) + "\n")
    
    # Test retrieval
    test_retrieval(index, examples, clf, model)
    
    # Demo intent-filtered retrieval
    print("\n" + "=" * 80)
    print("INTENT-FILTERED RETRIEVAL DEMO")
    print("=" * 80)
    
    demo_query = "@AmazonHelp my package says delivered but I never received it!"
    query_clean = clean_text(demo_query)
    query_emb = model.encode([query_clean], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)
    pred_intent = clf.predict([query_clean])[0]
    
    print(f"\nQuery: {demo_query}")
    print(f"Predicted Intent: {pred_intent}")
    
    # General retrieval
    scores, indices = index.search(query_emb.astype(np.float32), 3)
    print(f"\nGeneral Top 3:")
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
        ex = examples[idx]
        print(f"  #{rank} [{ex['predicted_intent']}] score={score:.3f}: {ex['resolution'][:100]}")
    
    # Intent-filtered retrieval
    if pred_intent in intent_indices:
        filtered = filter_by_intent(index, examples, pred_intent, query_emb, 3)
        print(f"\nFiltered by intent '{pred_intent}' Top 3:")
        for rank, ex in enumerate(filtered, 1):
            print(f"  #{rank} score={ex['similarity_score']:.3f}: {ex['resolution'][:100]}")
    
    logger.info("Phase 4 complete!")


def clean_text(text: str) -> str:
    import re
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


if __name__ == "__main__":
    main()