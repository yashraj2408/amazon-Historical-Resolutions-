#!/usr/bin/env python3
"""
Phase 3: Intent Classification
Train and evaluate intent classifier on AmazonHelp golden set.
"""

import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
import joblib

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Final intent taxonomy (merged from analysis)
FINAL_INTENTS = [
    "delivery_issue",      # delivery_issue + tracking_request
    "order_problem",       # order_problem + refund_return (often coupled)
    "account_access",      # account_access + billing_payment (charge on account)
    "technical_app",       # technical_app + prime_subscription (app issues)
    "escalation_needed",   # escalation_needed
    "non_english",         # non_english
    "resolution_confirmation",  # resolution_confirmation
    "general_inquiry"      # general_inquiry + seller_marketplace + gift_card + spam
]

# Mapping from original pre-labels to final taxonomy
INTENT_MAPPING = {
    "delivery_issue": "delivery_issue",
    "tracking_request": "delivery_issue",
    "order_problem": "order_problem",
    "refund_return": "order_problem",  # merged
    "account_access": "account_access",
    "billing_payment": "account_access",  # merged
    "technical_app": "technical_app",
    "prime_subscription": "technical_app",  # merged (app/prime issues)
    "escalation_needed": "escalation_needed",
    "non_english": "non_english",
    "resolution_confirmation": "resolution_confirmation",
    "general_inquiry": "general_inquiry",
    "seller_marketplace": "general_inquiry",  # merged
    "gift_card": "general_inquiry",  # merged
    "spam_irrelevant": "general_inquiry"  # merged
}


def load_golden_split(split: str) -> List[Dict[str, Any]]:
    """Load golden set split."""
    path = ANNOTATIONS_DIR / f"golden_{split}.jsonl"
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def prepare_training_data(data: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """Extract texts and labels for training."""
    texts = []
    labels = []
    
    for conv in data:
        # Use first_customer_message field (from golden set format)
        first_customer = conv.get("first_customer_message", "")
        
        if not first_customer:
            continue
        
        intent = conv.get("annotated_intent")
        if not intent:
            continue
        
        # Map to final taxonomy
        final_intent = INTENT_MAPPING.get(intent, "general_inquiry")
        
        texts.append(first_customer)
        labels.append(final_intent)
    
    return texts, labels


def clean_text(text: str) -> str:
    """Clean text for classification."""
    import re
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def train_classifier(X_train: List[str], y_train: List[str]) -> Pipeline:
    """Train TF-IDF + Logistic Regression classifier with calibration."""
    logger.info("Training classifier...")
    
    # Clean texts
    X_train_clean = [clean_text(t) for t in X_train]
    
    # Pipeline: TF-IDF + Logistic Regression
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
            stop_words="english"
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            C=1.0
        ))
    ])
    
    pipeline.fit(X_train_clean, y_train)
    
    # Calibrate for probability estimates
    calibrated = CalibratedClassifierCV(pipeline, method="sigmoid", cv=3)
    calibrated.fit(X_train_clean, y_train)
    
    return calibrated


def evaluate_classifier(
    clf: Pipeline, 
    X_test: List[str], 
    y_test: List[str],
    split_name: str
) -> Dict[str, Any]:
    """Evaluate classifier on test set."""
    logger.info(f"Evaluating on {split_name}...")
    
    X_test_clean = [clean_text(t) for t in X_test]
    y_pred = clf.predict(X_test_clean)
    y_proba = clf.predict_proba(X_test_clean)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    
    # Per-class report
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=FINAL_INTENTS)
    
    # Confidence analysis
    max_proba = np.max(y_proba, axis=1)
    avg_confidence = np.mean(max_proba)
    
    results = {
        "split": split_name,
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "avg_confidence": round(avg_confidence, 4),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "intent_distribution": dict(Counter(y_test)),
        "predictions_distribution": dict(Counter(y_pred))
    }
    
    return results, y_pred, y_proba


def print_evaluation_results(results: Dict[str, Any]):
    """Print formatted evaluation results."""
    print(f"\n{'='*60}")
    print(f"EVALUATION: {results['split'].upper()}")
    print(f"{'='*60}")
    print(f"Accuracy:       {results['accuracy']:.4f}")
    print(f"F1 Macro:       {results['f1_macro']:.4f}")
    print(f"F1 Weighted:    {results['f1_weighted']:.4f}")
    print(f"Avg Confidence: {results['avg_confidence']:.4f}")
    
    print("\nPer-Class Metrics:")
    print(f"{'Intent':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>8}")
    print("-" * 65)
    for intent in FINAL_INTENTS:
        if intent in results["classification_report"]:
            m = results["classification_report"][intent]
            print(f"{intent:<25} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1-score']:>10.3f} {int(m['support']):>8}")
        else:
            print(f"{intent:<25} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'0':>8}")
    
    print(f"\nConfusion Matrix (rows=true, cols=pred):")
    print(f"{'':<25}", end="")
    for intent in FINAL_INTENTS:
        print(f"{intent[:8]:>10}", end="")
    print()
    for i, intent in enumerate(FINAL_INTENTS):
        print(f"{intent:<25}", end="")
        for j in range(len(FINAL_INTENTS)):
            val = results["confusion_matrix"][i][j] if i < len(results["confusion_matrix"]) and j < len(results["confusion_matrix"][0]) else 0
            print(f"{val:>10}", end="")
        print()


def save_model(clf: Pipeline, results: Dict[str, Any]):
    """Save trained model and metadata."""
    model_path = MODELS_DIR / "intent_classifier.joblib"
    joblib.dump(clf, model_path)
    logger.info(f"Saved model to {model_path}")
    
    # Save metadata
    metadata = {
        "model_type": "TF-IDF + LogisticRegression (calibrated)",
        "intents": FINAL_INTENTS,
        "intent_mapping": INTENT_MAPPING,
        "train_results": results.get("train"),
        "val_results": results.get("val"),
        "test_results": results.get("test"),
        "random_seed": RANDOM_SEED
    }
    meta_path = MODELS_DIR / "intent_classifier_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Saved metadata to {meta_path}")


def predict_sample(clf: Pipeline, texts: List[str]) -> List[Dict[str, Any]]:
    """Predict intent for sample texts."""
    texts_clean = [clean_text(t) for t in texts]
    preds = clf.predict(texts_clean)
    probas = clf.predict_proba(texts_clean)
    
    results = []
    for i, text in enumerate(texts):
        pred = preds[i]
        proba = probas[i]
        intent_idx = np.where(clf.classes_ == pred)[0][0]
        confidence = proba[intent_idx]
        
        # Top 3 predictions
        top3_idx = np.argsort(proba)[-3:][::-1]
        top3 = [(clf.classes_[j], round(proba[j], 3)) for j in top3_idx]
        
        results.append({
            "text": text,
            "predicted_intent": pred,
            "confidence": round(confidence, 3),
            "top3": top3
        })
    
    return results


def main():
    logger.info("Starting Phase 3: Intent Classification")
    
    # Load splits
    train_data = load_golden_split("train")
    val_data = load_golden_split("val")
    test_data = load_golden_split("test")
    
    logger.info(f"Loaded: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    
    # Prepare data
    X_train, y_train = prepare_training_data(train_data)
    X_val, y_val = prepare_training_data(val_data)
    X_test, y_test = prepare_training_data(test_data)
    
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    logger.info(f"Test samples: {len(X_test)}")
    
    # Check class distribution
    print("\nTraining label distribution:")
    for intent, count in Counter(y_train).most_common():
        print(f"  {intent}: {count}")
    
    # Train
    clf = train_classifier(X_train, y_train)
    
    # Evaluate on all splits
    all_results = {}
    
    train_results, _, _ = evaluate_classifier(clf, X_train, y_train, "train")
    all_results["train"] = train_results
    print_evaluation_results(train_results)
    
    val_results, _, _ = evaluate_classifier(clf, X_val, y_val, "val")
    all_results["val"] = val_results
    print_evaluation_results(val_results)
    
    test_results, test_preds, test_probas = evaluate_classifier(clf, X_test, y_test, "test")
    all_results["test"] = test_results
    print_evaluation_results(test_results)
    
    # Save model
    save_model(clf, all_results)
    
    # Demo predictions
    print("\n" + "=" * 60)
    print("DEMO PREDICTIONS")
    print("=" * 60)
    
    demo_texts = [
        "@AmazonHelp my package says delivered but I never received it!",
        "@AmazonHelp I was charged twice for my Prime membership",
        "@AmazonHelp my Echo Dot won't connect to wifi",
        "@AmazonHelp I want to return this item, how do I get a refund?",
        "@AmazonHelp I can't log into my account, password reset not working",
        "@AmazonHelp gracias por la ayuda, ya todo resuelto",
        "@AmazonHelp why hasn't my order shipped yet? It's been 3 days",
        "@AmazonHelp the delivery driver marked it delivered but it's not here"
    ]
    
    predictions = predict_sample(clf, demo_texts)
    for p in predictions:
        print(f"\nText: {p['text'][:80]}...")
        print(f"  → {p['predicted_intent']} (confidence: {p['confidence']:.2f})")
        print(f"  Top 3: {p['top3']}")
    
    # Save test predictions for analysis
    test_predictions = []
    for i, conv in enumerate(test_data):
        if i < len(test_preds):
            conv_copy = conv.copy()
            conv_copy["predicted_intent"] = test_preds[i]
            conv_copy["prediction_confidence"] = round(np.max(test_probas[i]), 3)
            test_predictions.append(conv_copy)
    
    pred_path = RESULTS_DIR / "intent_test_predictions.jsonl"
    with open(pred_path, "w") as f:
        for p in test_predictions:
            f.write(json.dumps(p, default=str) + "\n")
    logger.info(f"Saved test predictions to {pred_path}")
    
    logger.info("Phase 3 complete!")


if __name__ == "__main__":
    main()