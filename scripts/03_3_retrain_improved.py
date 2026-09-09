#!/usr/bin/env python3
"""
Phase 3.4: Improved Intent Classification
Uses sentence-transformer embeddings + better classifier + refined taxonomy
"""

import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sentence_transformers import SentenceTransformer
import xgboost as xgb

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Refined taxonomy - split merged intents
IMPROVED_INTENTS = [
    "delivery_issue",           # package not arrived, late, tracking, "says delivered"
    "order_problem",            # wrong item, missing item, damaged, cancel order
    "refund_return",            # refund request, return process, replacement, return label
    "account_access",           # login, password reset, account locked, verification
    "billing_payment",          # unauthorized charge, billing question, payment failed
    "prime_billing",            # prime membership charges, student price, prime subscription fees
    "technical_device",         # Echo, Kindle, Fire TV, Alexa, device issues
    "technical_app",            # app/website errors, not device-specific
    "escalation_needed",        # complex issues needing specialist
    "non_english",              # Spanish, French, German, etc.
    "resolution_confirmation",  # customer confirms resolved
    "general_inquiry"           # product questions, policy, how-to
]

# Mapping from old auto-labels to improved taxonomy
IMPROVED_MAPPING = {
    "delivery_issue": "delivery_issue",
    "order_problem": "order_problem",
    "refund_return": "refund_return",
    "account_access": "account_access",
    "billing_payment": "billing_payment",
    "prime_subscription": "prime_billing",  # Split: prime charges -> prime_billing
    "technical_device": "technical_device",
    "escalation_needed": "escalation_needed",
    "non_english": "non_english",
    "resolution_confirmation": "resolution_confirmation",
    "general_inquiry": "general_inquiry",
    "seller_marketplace": "general_inquiry",
    "gift_card": "general_inquiry",
    "spam_irrelevant": "general_inquiry"
}

# Further refine: separate technical_device vs technical_app
TECHNICAL_DEVICE_KEYWORDS = ["echo", "kindle", "fire tv", "firetv", "fire stick", "alexa", "device", "streaming"]
TECHNICAL_APP_KEYWORDS = ["app", "application", "website", "site", "error", "bug", "crash", "not working", "broken", "glitch"]


def load_expanded_data() -> Tuple[List[str], List[str]]:
    """Load expanded training data."""
    texts = []
    labels = []
    
    for split in ["train", "val"]:
        path = ANNOTATIONS_DIR / f"{split}_expanded.jsonl"
        with open(path, "r") as f:
            for line in f:
                ex = json.loads(line)
                # Map to improved taxonomy
                old_intent = ex["intent"]
                new_intent = IMPROVED_MAPPING.get(old_intent, "general_inquiry")
                
                # Further split technical
                if new_intent == "technical_device":
                    text_lower = ex["text"].lower()
                    if any(kw in text_lower for kw in TECHNICAL_APP_KEYWORDS) and not any(kw in text_lower for kw in TECHNICAL_DEVICE_KEYWORDS):
                        new_intent = "technical_app"
                
                # Split prime billing
                if old_intent == "prime_subscription":
                    text_lower = ex["text"].lower()
                    if any(kw in text_lower for kw in ["charge", "charged", "billing", "payment", "fee", "price", "cost"]):
                        new_intent = "prime_billing"
                    else:
                        new_intent = "prime_subscription"
                
                labels.append(new_intent)
                texts.append(ex["text"])
    
    logger.info(f"Loaded {len(texts)} training samples")
    return texts, labels


def load_test_data() -> Tuple[List[str], List[str]]:
    """Load test data."""
    texts = []
    labels = []
    
    path = ANNOTATIONS_DIR / "test_expanded.jsonl"
    with open(path, "r") as f:
        for line in f:
            ex = json.loads(line)
            old_intent = ex["intent"]
            new_intent = IMPROVED_MAPPING.get(old_intent, "general_inquiry")
            
            if new_intent == "technical_device":
                text_lower = ex["text"].lower()
                if any(kw in text_lower for kw in TECHNICAL_APP_KEYWORDS) and not any(kw in text_lower for kw in TECHNICAL_DEVICE_KEYWORDS):
                    new_intent = "technical_app"
            
            if old_intent == "prime_subscription":
                text_lower = ex["text"].lower()
                if any(kw in text_lower for kw in ["charge", "charged", "billing", "payment", "fee", "price", "cost"]):
                    new_intent = "prime_billing"
                else:
                    new_intent = "prime_subscription"
            
            labels.append(new_intent)
            texts.append(ex["text"])
    
    logger.info(f"Loaded {len(texts)} test samples")
    return texts, labels


def create_embeddings(texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    """Create sentence embeddings."""
    logger.info(f"Creating embeddings with {model_name}...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    logger.info(f"Created embeddings: {embeddings.shape}")
    return embeddings


def train_classifiers(
    X_train: np.ndarray, 
    y_train: List[str],
    X_test: np.ndarray, 
    y_test: List[str]
) -> Dict[str, Any]:
    """Train multiple classifiers and compare."""
    
    logger.info("Training multiple classifiers...")
    
    classifiers = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED, C=1.0
        ),
        "LogisticRegression_C2": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED, C=2.0
        ),
        "SVM_RBF": SVC(kernel="rbf", class_weight="balanced", random_state=RANDOM_SEED, probability=True),
        "SVM_Linear": SVC(kernel="linear", class_weight="balanced", random_state=RANDOM_SEED, probability=True),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1
        ),
    }
    
    results = {}
    
    for name, clf in classifiers.items():
        logger.info(f"Training {name}...")
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
        
        # Train on full data
        clf.fit(X_train, y_train)
        
        # Test evaluation
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None
        
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")
        
        # Calibration
        if y_proba is not None:
            calibrated = CalibratedClassifierCV(clf, method="sigmoid", cv=3)
            calibrated.fit(X_train, y_train)
            cal_proba = calibrated.predict_proba(X_test)
            avg_conf = np.mean(np.max(cal_proba, axis=1))
        else:
            calibrated = None
            avg_conf = 0
        
        results[name] = {
            "model": calibrated if calibrated else clf,
            "cv_f1_macro_mean": float(np.mean(cv_scores)),
            "cv_f1_macro_std": float(np.std(cv_scores)),
            "test_accuracy": float(acc),
            "test_f1_macro": float(f1_macro),
            "test_f1_weighted": float(f1_weighted),
            "avg_confidence": float(avg_conf),
            "predictions": y_pred.tolist(),
            "probabilities": cal_proba.tolist() if y_proba is not None else None
        }
        
        logger.info(f"  {name}: CV F1={np.mean(cv_scores):.4f}±{np.std(cv_scores):.4f}, Test Acc={acc:.4f}, F1 Macro={f1_macro:.4f}")
    
    return results


def evaluate_best_model(results: Dict, y_test: List[str], intents: List[str]):
    """Evaluate best model in detail."""
    
    # Find best by CV F1 macro
    best_name = max(results.keys(), key=lambda k: results[k]["cv_f1_macro_mean"])
    best = results[best_name]
    
    logger.info(f"\nBest model: {best_name}")
    logger.info(f"  CV F1 Macro: {best['cv_f1_macro_mean']:.4f}±{best['cv_f1_macro_std']:.4f}")
    logger.info(f"  Test Accuracy: {best['test_accuracy']:.4f}")
    logger.info(f"  Test F1 Macro: {best['test_f1_macro']:.4f}")
    logger.info(f"  Test F1 Weighted: {best['test_f1_weighted']:.4f}")
    logger.info(f"  Avg Confidence: {best['avg_confidence']:.4f}")
    
    # Detailed report
    y_pred = best["predictions"]
    print(f"\n{'='*70}")
    print(f"DETAILED EVALUATION: {best_name}")
    print(f"{'='*70}")
    print(classification_report(y_test, y_pred, target_names=sorted(set(y_test)), zero_division=0))
    
    return best_name, best


def save_model(best_name: str, best: Dict, X_train: np.ndarray, y_train: List[str]):
    """Save best model and metadata."""
    
    model_path = MODELS_DIR / f"intent_classifier_v3_{best_name}.joblib"
    joblib.dump(best["model"], model_path)
    logger.info(f"Saved model to {model_path}")
    
    # Save embeddings model reference
    meta = {
        "model_name": best_name,
        "model_type": f"SentenceTransformer Embeddings + {best_name}",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "intents": sorted(set(y_train)),
        "test_accuracy": best["test_accuracy"],
        "test_f1_macro": best["test_f1_macro"],
        "test_f1_weighted": best["test_f1_weighted"],
        "cv_f1_macro_mean": best["cv_f1_macro_mean"],
        "cv_f1_macro_std": best["cv_f1_macro_std"],
        "avg_confidence": best["avg_confidence"],
        "improved_taxonomy": True,
        "split_intents": ["prime_billing", "technical_app"],
        "merged_from": ["prime_subscription", "technical_device"]
    }
    
    meta_path = MODELS_DIR / f"intent_classifier_v3_{best_name}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info(f"Saved metadata to {meta_path}")
    
    # Also save as default v3
    joblib.dump(best["model"], MODELS_DIR / "intent_classifier_v3.joblib")
    with open(MODELS_DIR / "intent_classifier_v3_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)


def test_on_golden(best_model, embedder, test_path: Path):
    """Test on golden test set."""
    logger.info("Testing on golden test set...")
    
    test_data = []
    with open(test_path, "r") as f:
        for line in f:
            test_data.append(json.loads(line))
    
    results = []
    for ex in test_data:
        query = ex.get("first_customer_message", "")
        true_intent = ex.get("annotated_intent", "")
        
        # Embed
        query_clean = query
        # Clean
        import re
        query_clean = re.sub(r"https?://\S+", "", query_clean)
        query_clean = re.sub(r"@\w+", "", query_clean)
        query_clean = re.sub(r"\s+", " ", query_clean).strip()
        
        emb = embedder.encode([query_clean], convert_to_numpy=True)
        pred = best_model.predict(emb)[0]
        proba = best_model.predict_proba(emb)[0]
        confidence = max(proba)
        
        results.append({
            "conversation_id": ex["conversation_id"],
            "query": query[:100],
            "true_intent": true_intent,
            "predicted_intent": pred,
            "confidence": float(confidence),
            "correct": pred == true_intent
        })
    
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    
    print(f"\nGolden Test Set Results ({total} samples):")
    print(f"  Accuracy: {correct/total:.2%}")
    print(f"  Avg Confidence: {np.mean([r['confidence'] for r in results]):.4f}")
    
    for r in results:
        status = "✓" if r["correct"] else "✗"
        print(f"  {status} {r['true_intent']} → {r['predicted_intent']} (conf: {r['confidence']:.2f})")
    
    return results


def main():
    logger.info("Starting Phase 3.4: Improved Intent Classification")
    
    # Load data
    X_train_texts, y_train = load_expanded_data()
    X_test_texts, y_test = load_test_data()
    
    logger.info(f"Training: {len(X_train_texts)}, Test: {len(X_test_texts)}")
    print(f"\nTraining label distribution:")
    for intent, count in Counter(y_train).most_common():
        print(f"  {intent}: {count}")
    
    # Create embeddings
    logger.info("Creating sentence embeddings...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    X_train = create_embeddings(X_train_texts)
    X_test = embedder.encode(X_test_texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    
    # Train classifiers
    results = train_classifiers(X_train, y_train, X_test, y_test)
    
    # Evaluate best
    best_name, best = evaluate_best_model(results, y_test, sorted(set(y_train)))
    
    # Save best model
    save_model(best_name, best, X_train, y_train)
    
    # Test on golden set
    golden_path = ANNOTATIONS_DIR / "golden_test.jsonl"
    if golden_path.exists():
        golden_results = test_on_golden(best["model"], embedder, golden_path)
    
    # Save all results
    results_path = PROJECT_ROOT / "results" / "improved_classifier_results.json"
    with open(results_path, "w") as f:
        serializable = {}
        for k, v in results.items():
            serializable[k] = {k2: v2 for k2, v2 in v.items() if k2 != "model"}
        json.dump(serializable, f, indent=2, default=str)
    logger.info(f"Saved results to {results_path}")
    
    logger.info("Phase 3.4 complete!")


if __name__ == "__main__":
    main()