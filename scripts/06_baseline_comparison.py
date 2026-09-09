#!/usr/bin/env python3
"""
Phase 6: Baseline Comparison
Compare RAG agent against at least 2 baselines:
1. Rule-based classifier (keyword matching)
2. Generic LLM (no retrieval, no intent classifier)
3. TF-IDF only (no retrieval, no intent)
"""

import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter
from dataclasses import dataclass

import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

REFINED_INTENTS = [
    "delivery_issue", "order_problem", "refund_return", "account_access",
    "billing_payment", "technical_device", "prime_subscription",
    "escalation_needed", "non_english", "resolution_confirmation"
]


def load_golden_test() -> List[Dict[str, Any]]:
    """Load golden test set."""
    test_path = ANNOTATIONS_DIR / "golden_test.jsonl"
    data = []
    with open(test_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_expanded_train() -> Tuple[List[str], List[str]]:
    """Load expanded training data for TF-IDF baseline."""
    train_path = ANNOTATIONS_DIR / "train_expanded.jsonl"
    texts = []
    labels = []
    with open(train_path, "r") as f:
        for line in f:
            ex = json.loads(line)
            texts.append(ex["text"])
            labels.append(ex["intent"])
    return texts, labels


# ============================================================
# BASELINE 1: Rule-based (Keyword Matching)
# ============================================================

RULE_KEYWORDS = {
    "delivery_issue": [
        "deliver", "delivered", "delivery", "tracking", "package", "parcel",
        "courier", "shipping", "shipped", "arrive", "late", "delay",
        "missing", "lost", "not received", "not arrived", "says delivered",
        "shows delivered", "handed to me", "handed over", "transit", "carrier"
    ],
    "order_problem": [
        "wrong item", "missing item", "damaged", "cancel order", "pre.order",
        "pre order", "order.*wrong", "item.*wrong", "received.*wrong",
        "got.*wrong", "wrong product", "item missing"
    ],
    "refund_return": [
        "refund", "return", "replacement", "replace", "return label",
        "money back", "send back", "returned", "reimburse", "return process"
    ],
    "account_access": [
        "login", "log in", "password", "locked", "verify", "verification",
        "email.*code", "not receiving", "not receive", "code.*email",
        "reset", "account.*locked", "can't log", "cannot log", "sign in",
        "access.*account"
    ],
    "billing_payment": [
        "charge", "charged", "billing", "bill", "payment", "pay",
        "money", "bank", "card", "unauthorized", "fraudulent"
    ],
    "technical_device": [
        "echo", "kindle", "fire tv", "firetv", "fire stick", "alexa",
        "device", "streaming", "video.*not working", "app.*crash",
        "app.*not working", "website.*error", "site.*down"
    ],
    "prime_subscription": [
        "prime", "membership", "subscription", "student", "renew",
        "cancel prime", "prime member", "prime shipping", "prime video"
    ],
    "escalation_needed": [
        "escalate", "specialist", "team", "fraud", "legal", "lawyer",
        "ignored", "no response", "never replied", "days requesting",
        "complaint", "unresolved"
    ],
    "non_english": [
        "que", "la", "el", "por", "lo", "en", "mi", "si", "pero", "una",
        "ingres", "correo", "codigo", "recibo", "nada", "necesito",
        "cuenta", "gracias", "por favor", "consulta", "seur",
        "je", "vous", "est", "pas", "le", "ai", "command", "livr",
        "probl", "merci", "bonsoir", "conteste", "preuve",
        "das", "ist", "und", "nicht", "die", "der", "für", "mit",
        "eine", "habe", "kein"
    ],
    "resolution_confirmation": [
        "^thank", "^thanks", "resolved", "fixed", "working", "sorted",
        "appreciate", "great", "awesome", "perfect"
    ]
}

RULE_PRIORITY = [
    "non_english", "resolution_confirmation", "escalation_needed",
    "delivery_issue", "order_problem", "refund_return",
    "account_access", "billing_payment", "prime_subscription", "technical_device"
]


def clean_text(text: str) -> str:
    import re
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def rule_based_predict(text: str) -> Tuple[str, float]:
    """Rule-based intent prediction."""
    text_lower = text.lower()
    matches = []
    
    for intent in RULE_PRIORITY:
        for pattern in RULE_KEYWORDS[intent]:
            import re
            if re.search(pattern, text_lower, re.IGNORECASE):
                matches.append(intent)
                break
    
    if not matches:
        return "general", 0.3
    
    # Confidence based on number of matching keywords
    intent = matches[0]
    confidence = min(0.5 + 0.1 * len(matches), 0.9)
    return intent, confidence


# ============================================================
# BASELINE 2: TF-IDF Only (No Retrieval)
# ============================================================

def train_tfidf_baseline(X_train: List[str], y_train: List[str]) -> Pipeline:
    """Train TF-IDF + Logistic Regression (no retrieval)."""
    logger.info("Training TF-IDF baseline...")
    
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=15000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
            stop_words="english"
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            C=1.0
        ))
    ])
    
    pipeline.fit(X_train, y_train)
    calibrated = CalibratedClassifierCV(pipeline, method="sigmoid", cv=3)
    calibrated.fit(X_train, y_train)
    
    return calibrated


# ============================================================
# BASELINE 3: Generic Response (No Intent, No Retrieval)
# ============================================================

GENERIC_RESPONSES = {
    "default": (
        "Thank you for contacting Amazon Help! I'd be happy to assist you. "
        "Could you please provide more details about your issue so I can help better?"
    ),
    "delivery": "I'm sorry for the delivery issue. Please check tracking and let us know.",
    "account": "For account issues, try the password reset page or contact support.",
    "billing": "For billing questions, please check your account or contact billing support.",
    "technical": "For technical issues, try restarting your device and checking connections.",
}

def generic_predict(text: str) -> Tuple[str, float]:
    """Generic baseline - always returns general intent with low confidence."""
    text_lower = text.lower()
    if any(w in text_lower for w in ["deliver", "package", "tracking"]):
        return "delivery_issue", 0.4
    elif any(w in text_lower for w in ["account", "login", "password"]):
        return "account_access", 0.4
    elif any(w in text_lower for w in ["charge", "bill", "payment"]):
        return "billing_payment", 0.4
    elif any(w in text_lower for w in ["prime", "membership"]):
        return "prime_subscription", 0.4
    elif any(w in text_lower for w in ["refund", "return"]):
        return "refund_return", 0.4
    return "general_inquiry", 0.3


# ============================================================
# RAG Agent (from Phase 5) - Simplified for Evaluation
# ============================================================

def load_rag_agent():
    """Load RAG agent from Phase 5."""
    import faiss
    from sentence_transformers import SentenceTransformer
    
    INDEX_DIR = PROJECT_ROOT / "index"
    MODELS_DIR = PROJECT_ROOT / "models"
    
    clf = joblib.load(MODELS_DIR / "intent_classifier_v2.joblib")
    index = faiss.read_index(str(INDEX_DIR / "resolution_index.faiss"))
    embeddings = np.load(INDEX_DIR / "resolution_embeddings.npy")
    
    examples = []
    with open(INDEX_DIR / "resolution_metadata.jsonl", "r") as f:
        for line in f:
            examples.append(json.loads(line))
    
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    return clf, index, embeddings, examples, embedder


def rag_predict(query: str, clf, index, embeddings, examples, embedder, top_k=3) -> Tuple[str, float, List]:
    """RAG prediction with retrieval."""
    def clean_text(text):
        import re
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    query_clean = clean_text(query)
    pred_intent = clf.predict([query_clean])[0]
    proba = clf.predict_proba([query_clean])[0]
    confidence = max(proba)
    
    # Retrieve
    query_emb = embedder.encode([query_clean], convert_to_numpy=True)
    import faiss
    faiss.normalize_L2(query_emb)
    scores, indices = index.search(query_emb.astype(np.float32), top_k)
    
    retrieved = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < len(examples):
            ex = examples[idx]
            if ex["predicted_intent"] == pred_intent:
                retrieved.append((ex, float(score)))
    
    return pred_intent, confidence, retrieved


# ============================================================
# Evaluation
# ============================================================

def evaluate_baseline(name: str, test_data: List[Dict], predict_fn, rag_components=None) -> Dict:
    """Evaluate a baseline on test data."""
    logger.info(f"Evaluating {name}...")
    
    results = []
    for ex in test_data:
        query = ex.get("first_customer_message", "")
        true_intent = ex.get("annotated_intent", "")
        
        if rag_components:
            pred_intent, confidence, retrieved = predict_fn(query, *rag_components)
        else:
            pred_intent, confidence = predict_fn(query)
        
        # Map to refined taxonomy
        REFINED_MAPPING = {
            "delivery_issue": "delivery_issue", "tracking_request": "delivery_issue",
            "order_problem": "order_problem", "refund_return": "refund_return",
            "account_access": "account_access", "billing_payment": "billing_payment",
            "technical_app": "technical_device", "prime_subscription": "prime_subscription",
            "escalation_needed": "escalation_needed", "non_english": "non_english",
            "resolution_confirmation": "resolution_confirmation",
            "general_inquiry": "general_inquiry", "seller_marketplace": "general_inquiry",
            "gift_card": "general_inquiry", "spam_irrelevant": "general_inquiry"
        }
        
        pred_refined = REFINED_MAPPING.get(pred_intent, "general_inquiry")
        
        results.append({
            "conversation_id": ex["conversation_id"],
            "query": query[:100],
            "true_intent": true_intent,
            "predicted_intent": pred_refined,
            "raw_predicted": pred_intent,
            "confidence": confidence,
            "correct": pred_refined == true_intent
        })
    
    # Compute metrics
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    by_intent = Counter(r["true_intent"] for r in results)
    correct_by_intent = Counter(r["true_intent"] for r in results if r["correct"])
    
    per_intent = {}
    for intent in REFINED_INTENTS:
        total_i = by_intent.get(intent, 0)
        correct_i = correct_by_intent.get(intent, 0)
        per_intent[intent] = {
            "total": total_i,
            "correct": correct_i,
            "accuracy": correct_i / total_i if total_i > 0 else 0
        }
    
    return {
        "name": name,
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total > 0 else 0,
        "avg_confidence": np.mean([r["confidence"] for r in results]),
        "per_intent": per_intent,
        "predictions": results
    }


def print_comparison(results: List[Dict]):
    """Print comparison table."""
    print("\n" + "=" * 100)
    print("BASELINE COMPARISON")
    print("=" * 100)
    
    print(f"\n{'Baseline':<25} {'Accuracy':>10} {'Avg Conf':>10} {'Auto-handle*':>12}")
    print("-" * 60)
    
    for r in results:
        auto_handle = sum(1 for p in r["predictions"] if p["confidence"] >= 0.7 and p["predicted_intent"] != "escalation_needed")
        auto_rate = auto_handle / r["total"] if r["total"] > 0 else 0
        print(f"{r['name']:<25} {r['accuracy']:>10.2%} {r['avg_confidence']:>10.2f} {auto_rate:>12.2%}")
    
    print("\nPer-Intent Accuracy:")
    print(f"{'Intent':<25}", end="")
    for r in results:
        print(f" {r['name'][:15]:>15}", end="")
    print()
    print("-" * (25 + 16 * len(results)))
    
    for intent in REFINED_INTENTS:
        print(f"{intent:<25}", end="")
        for r in results:
            acc = r["per_intent"].get(intent, {}).get("accuracy", 0)
            total = r["per_intent"].get(intent, {}).get("total", 0)
            print(f" {acc:>12.2%} ({total:>2})", end="")
        print()


def main():
    logger.info("Starting Phase 6: Baseline Comparison")
    
    # Load test data
    test_data = load_golden_test()
    logger.info(f"Loaded {len(test_data)} test samples")
    
    # Load training data for TF-IDF baseline
    X_train, y_train = load_expanded_train()
    logger.info(f"Loaded {len(X_train)} training samples for TF-IDF")
    
    results = []
    
    # Baseline 1: Rule-based
    logger.info("Running Rule-based baseline...")
    rule_results = evaluate_baseline("Rule-based (Keywords)", test_data, rule_based_predict)
    results.append(rule_results)
    
    # Baseline 2: TF-IDF only
    logger.info("Running TF-IDF baseline...")
    tfidf_clf = train_tfidf_baseline(X_train, y_train)
    def tfidf_predict(text):
        pred = tfidf_clf.predict([text])[0]
        proba = tfidf_clf.predict_proba([text])[0]
        return pred, max(proba)
    tfidf_results = evaluate_baseline("TF-IDF + LR", test_data, tfidf_predict)
    results.append(tfidf_results)
    
    # Baseline 3: Generic
    logger.info("Running Generic baseline...")
    generic_results = evaluate_baseline("Generic (Keyword Heuristics)", test_data, generic_predict)
    results.append(generic_results)
    
    # RAG Agent (Phase 5)
    logger.info("Running RAG Agent...")
    clf, index, embeddings, examples, embedder = load_rag_agent()
    def rag_predict_fn(text):
        pred_intent, confidence, retrieved = rag_predict(text, clf, index, embeddings, examples, embedder)
        return pred_intent, confidence
    rag_results = evaluate_baseline("RAG Agent (Phase 5)", test_data, rag_predict_fn)
    results.append(rag_results)
    
    # Print comparison
    print_comparison(results)
    
    # Save detailed results
    output_path = RESULTS_DIR / "baseline_comparison.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved comparison to {output_path}")
    
    logger.info("Phase 6 complete!")


if __name__ == "__main__":
    main()