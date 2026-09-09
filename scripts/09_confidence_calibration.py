#!/usr/bin/env python3
"""
Phase 9: Confidence Calibration + Threshold Optimization
Calibrates classifier probabilities and optimizes per-intent auto-handle thresholds.
"""

import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import Counter, defaultdict

import numpy as np
import joblib
from scipy.optimize import minimize_scalar
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"
RESULTS_DIR = PROJECT_ROOT / "results"
INDEX_DIR = PROJECT_ROOT / "index"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_RETRIEVAL = 3

ESCALATE_INTENTS = ["escalation_needed"]


def load_calibration_data() -> Tuple[np.ndarray, List[str], np.ndarray, List[str]]:
    """Load validation data for calibration."""
    # Load expanded validation set
    val_path = Path(__file__).parent.parent / "data" / "annotations" / "val_expanded.jsonl"
    texts = []
    labels = []
    with open(val_path, "r") as f:
        for line in f:
            ex = json.loads(line)
            texts.append(ex["text"])
            labels.append(ex["intent"])
    
    logger.info(f"Loaded {len(texts)} validation samples")
    
    # Create embeddings
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embedder.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    
    return embeddings, labels, embedder, texts


def load_classifier_and_embedder():
    """Load the trained classifier and embedder."""
    clf = joblib.load(PROJECT_ROOT / "models" / "intent_classifier_v3.joblib")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return clf, embedder


def temperature_scaling(logits: np.ndarray, labels: List[str], intent_classes: List[str]) -> float:
    """Find optimal temperature for calibration using NLL minimization."""
    
    # Convert labels to indices
    label_to_idx = {c: i for i, c in enumerate(intent_classes)}
    y_true = np.array([label_to_idx[l] for l in labels])
    
    def nll(temp: float) -> float:
        if temp <= 0:
            return 1e10
        scaled = logits / temp
        # Softmax
        exp_scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        probs = exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)
        # NLL
        correct_probs = probs[np.arange(len(y_true)), y_true]
        return -np.mean(np.log(correct_probs + 1e-15))
    
    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    return result.x


def apply_temperature(logits: np.ndarray, temp: float) -> np.ndarray:
    """Apply temperature scaling to logits."""
    scaled = logits / temp
    exp_scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
    return exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)


def get_logits_from_svm(clf, X: np.ndarray) -> np.ndarray:
    """Get decision function outputs (logits) from calibrated SVM."""
    # For CalibratedClassifierCV with SVM base estimator
    base_estimator = clf.calibrated_classifiers_[0].estimator
    return base_estimator.decision_function(X)


def isotonic_calibration(probas: np.ndarray, labels: List[str], intent_classes: List[str]) -> List[IsotonicRegression]:
    """Fit isotonic regression per class for calibration."""
    label_to_idx = {c: i for i, c in enumerate(intent_classes)}
    y_true = np.array([label_to_idx[l] for l in labels])
    
    calibrators = []
    for i in range(len(intent_classes)):
        y_binary = (y_true == i).astype(float)
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(probas[:, i], y_binary)
        calibrators.append(ir)
    
    return calibrators


def apply_isotonic(probas: np.ndarray, calibrators: List[IsotonicRegression]) -> np.ndarray:
    """Apply isotonic calibration per class."""
    calibrated = np.zeros_like(probas)
    for i, cal in enumerate(calibrators):
        calibrated[:, i] = cal.predict(probas[:, i])
    # Renormalize
    calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)
    return calibrated


def optimize_thresholds(
    val_probas: np.ndarray,
    val_labels: List[str],
    intent_classes: List[str],
    min_precision: float = 0.85
) -> Dict[str, float]:
    """Optimize per-intent auto-handle thresholds based on precision target."""
    
    label_to_idx = {c: i for i, c in enumerate(intent_classes)}
    y_true = np.array([label_to_idx[l] for l in val_labels])
    y_pred = np.argmax(val_probas, axis=1)
    
    thresholds = {}
    
    for i, intent in enumerate(intent_classes):
        # Get confidence scores for this class
        confidences = val_probas[:, i]
        is_true = (y_true == i)
        
        if intent in ESCALATE_INTENTS:
            thresholds[intent] = 1.0  # Never auto-handle escalation intents
            continue
        
        # Find threshold that achieves target precision
        best_thresh = 0.5
        best_precision = 0
        
        for thresh in np.linspace(0.1, 0.99, 90):
            pred_positive = (confidences >= thresh)
            if pred_positive.sum() == 0:
                continue
            precision = (pred_positive & is_true).sum() / pred_positive.sum()
            recall = (pred_positive & is_true).sum() / is_true.sum() if is_true.sum() > 0 else 0
            
            if precision >= min_precision and recall > 0.3:
                if precision > best_precision:
                    best_precision = precision
                    best_thresh = thresh
        
        thresholds[intent] = best_thresh
        logger.info(f"  {intent}: threshold={best_thresh:.2f}, precision={best_precision:.2f}")
    
    return thresholds


def evaluate_with_thresholds(
    test_probas: np.ndarray,
    test_labels: List[str],
    intent_classes: List[str],
    thresholds: Dict[str, float],
    escalate_intents: List[str]
) -> Dict[str, Any]:
    """Evaluate system with optimized thresholds."""
    
    label_to_idx = {c: i for i, c in enumerate(intent_classes)}
    y_true = np.array([label_to_idx[l] for l in test_labels])
    y_pred = np.argmax(test_probas, axis=1)
    max_probas = np.max(test_probas, axis=1)
    
    results = []
    for i in range(len(test_labels)):
        true_label = test_labels[i]
        pred_label = intent_classes[y_pred[i]]
        max_conf = max_probas[i]
        
        # Apply threshold
        intent_threshold = thresholds.get(pred_label, 0.7)
        should_auto = (pred_label not in escalate_intents) and (max_conf >= intent_threshold)
        
        results.append({
            "true": true_label,
            "pred": pred_label,
            "conf": max_conf,
            "threshold": intent_threshold,
            "auto_handle": should_auto,
            "correct": true_label == pred_label
        })
    
    # Metrics
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    auto_handled = sum(1 for r in results if r["auto_handle"])
    auto_correct = sum(1 for r in results if r["auto_handle"] and r["correct"])
    escalated = sum(1 for r in results if not r["auto_handle"])
    
    # Per-intent metrics
    per_intent = {}
    for intent in set(test_labels):
        intent_results = [r for r in results if r["true"] == intent]
        if not intent_results:
            continue
        total_i = len(intent_results)
        correct_i = sum(1 for r in intent_results if r["correct"])
        auto_i = sum(1 for r in intent_results if r["auto_handle"])
        auto_correct_i = sum(1 for r in intent_results if r["auto_handle"] and r["correct"])
        
        per_intent[intent] = {
            "total": total_i,
            "accuracy": correct_i / total_i,
            "auto_handle_rate": auto_i / total_i,
            "auto_precision": auto_correct_i / auto_i if auto_i > 0 else 0
        }
    
    return {
        "total": total,
        "accuracy": correct / total,
        "auto_handle_rate": auto_handled / total,
        "auto_precision": auto_correct / auto_handled if auto_handled > 0 else 0,
        "escalate_rate": escalated / total,
        "per_intent": per_intent,
        "detailed_results": results
    }


def main():
    logger.info("Starting Phase 9: Confidence Calibration + Threshold Optimization")
    
    # Load classifier and embedder
    clf, embedder = load_classifier_and_embedder()
    
    # Load validation data for calibration
    logger.info("Loading validation data for calibration...")
    val_embeddings, val_labels, _, _ = load_calibration_data()
    
    # Get logits from SVM
    logger.info("Getting logits from SVM...")
    base_estimator = clf.calibrated_classifiers_[0].estimator
    val_logits = base_estimator.decision_function(val_embeddings)
    
    # Get intent classes from classifier (only those present in model)
    intent_classes = list(clf.classes_)
    logger.info(f"Intent classes: {intent_classes}")
    
    # Temperature scaling
    logger.info("Applying temperature scaling...")
    temp = temperature_scaling(val_logits, val_labels, intent_classes)
    logger.info(f"Optimal temperature: {temp:.4f}")
    
    # Apply temperature scaling
    val_probas_temp = apply_temperature(val_logits, temp)
    
    # Isotonic calibration
    logger.info("Applying isotonic calibration...")
    calibrators = isotonic_calibration(val_probas_temp, val_labels, intent_classes)
    val_probas_cal = apply_isotonic(val_probas_temp, calibrators)
    
    # Optimize thresholds
    logger.info("Optimizing per-intent thresholds...")
    thresholds = optimize_thresholds(val_probas_cal, val_labels, intent_classes, min_precision=0.85)
    
    # Save calibration artifacts
    calibration_artifacts = {
        "temperature": float(temp),
        "thresholds": thresholds,
        "intent_classes": intent_classes
    }
    
    calib_path = MODELS_DIR / "calibration_artifacts.json"
    with open(calib_path, "w") as f:
        json.dump(calibration_artifacts, f, indent=2, default=str)
    logger.info(f"Saved calibration artifacts to {calib_path}")
    
    # Test on golden set
    logger.info("Evaluating on golden test set...")
    test_path = Path(__file__).parent.parent / "data" / "annotations" / "golden_test.jsonl"
    test_data = []
    with open(test_path, "r") as f:
        for line in f:
            test_data.append(json.loads(line))
    
    # Embed test queries
    test_texts = [ex.get("first_customer_message", "") for ex in test_data]
    test_labels = [ex.get("annotated_intent", "") for ex in test_data]
    test_embeddings = embedder.encode(test_texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    
    # Get calibrated probabilities
    test_logits = base_estimator.decision_function(test_embeddings)
    test_probas_temp = apply_temperature(test_logits, temp)
    test_probas_cal = apply_isotonic(test_probas_temp, calibrators)
    
    # Map unknown test labels to known classes
    label_to_idx = {c: i for i, c in enumerate(intent_classes)}
    test_labels_mapped = [l if l in label_to_idx else "non_english" for l in test_labels]
    
    # Evaluate with thresholds
    eval_results = evaluate_with_thresholds(
        test_probas_cal, test_labels_mapped, intent_classes, thresholds, ESCALATE_INTENTS
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("PHASE 9: CALIBRATION + THRESHOLD OPTIMIZATION RESULTS")
    print("=" * 80)
    print(f"\nTemperature: {temp:.4f}")
    print(f"\nOptimized Thresholds:")
    for intent, thresh in thresholds.items():
        print(f"  {intent:25s}: {thresh:.2f}")
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {eval_results['accuracy']:.2%}")
    print(f"  Auto-handle Rate: {eval_results['auto_handle_rate']:.2%}")
    print(f"  Auto Precision: {eval_results['auto_precision']:.2%}")
    print(f"  Escalate Rate: {eval_results['escalate_rate']:.2%}")
    
    print(f"\nPer-Intent Metrics:")
    print(f"{'Intent':<25} {'Acc':>8} {'Auto%':>8} {'AutoPrec':>8} {'Thresh':>8}")
    print("-" * 60)
    for intent, metrics in sorted(eval_results["per_intent"].items()):
        thresh = thresholds.get(intent, 0.7)
        print(f"{intent:<25} {metrics['accuracy']:>7.2%} {metrics['auto_handle_rate']:>7.2%} "
              f"{metrics['auto_precision']:>7.2%} {thresh:>7.2f}")
    
    # Save results
    output = {
        "temperature": float(temp),
        "thresholds": thresholds,
        "eval_results": eval_results
    }
    
    output_path = RESULTS_DIR / "calibration_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"Saved calibration results to {output_path}")
    
    # Save calibrated classifier wrapper
    calibrated_clf = {
        "base_classifier": clf,
        "temperature": temp,
        "calibrators": calibrators,
        "thresholds": thresholds,
        "intent_classes": intent_classes,
        "embedder_name": "sentence-transformers/all-MiniLM-L6-v2"
    }
    
    cal_clf_path = MODELS_DIR / "intent_classifier_calibrated.joblib"
    joblib.dump(calibrated_clf, cal_clf_path)
    logger.info(f"Saved calibrated classifier to {cal_clf_path}")
    
    logger.info("Phase 9 complete!")


if __name__ == "__main__":
    main()