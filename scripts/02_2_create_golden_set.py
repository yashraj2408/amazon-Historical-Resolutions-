#!/usr/bin/env python3
"""
Phase 2.2: Create golden evaluation set from annotated conversations.
Computes annotator agreement, finalizes taxonomy, creates train/val/test splits.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter, defaultdict
import random

import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# Constants
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
GOLDEN_SET_SIZE = 200  # Target size
MIN_ANNOTATORS = 2  # Minimum annotators per conversation for agreement


PROPOSED_INTENTS = [
    "delivery_issue", "order_problem", "refund_return", "account_access",
    "technical_app", "billing_payment", "prime_subscription", "seller_marketplace",
    "gift_card", "general_inquiry", "escalation_needed", "non_english",
    "spam_irrelevant", "resolution_confirmation"
]


def load_annotated_data(annotation_file: Path) -> List[Dict[str, Any]]:
    """Load annotated conversations from JSONL."""
    conversations = []
    with open(annotation_file, "r") as f:
        for line in f:
            conversations.append(json.loads(line))
    return conversations


def check_annotation_completeness(annotations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Check how many conversations have been annotated."""
    total = len(annotations)
    annotated = [a for a in annotations if a.get("annotated_intent") is not None]
    multi_annotator = defaultdict(list)
    
    for a in annotated:
        conv_id = a["conversation_id"]
        multi_annotator[conv_id].append(a)
    
    with_agreement = {k: v for k, v in multi_annotator.items() if len(v) >= MIN_ANNOTATORS}
    
    stats = {
        "total_conversations": total,
        "annotated_count": len(annotated),
        "unique_conversations_annotated": len(multi_annotator),
        "conversations_with_agreement": len(with_agreement),
        "annotators_found": set(a.get("annotator_id") for a in annotated if a.get("annotator_id")),
        "intent_distribution": Counter(a["annotated_intent"] for a in annotated if a.get("annotated_intent"))
    }
    
    return stats, with_agreement


def compute_agreement(agreements: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Compute inter-annotator agreement (Cohen's Kappa)."""
    if not agreements:
        return {"error": "No conversations with multiple annotators"}
    
    # For each conversation with multiple annotators, collect labels
    y_true = []
    y_pred = []
    
    for conv_id, anns in agreements.items():
        if len(anns) >= 2:
            # Use first annotator as reference, others as predictions
            labels = [a["annotated_intent"] for a in anns]
            for i in range(1, len(labels)):
                y_true.append(labels[0])
                y_pred.append(labels[i])
    
    if not y_true:
        return {"error": "No paired annotations for agreement"}
    
    # Compute Cohen's Kappa
    try:
        kappa = cohen_kappa_score(y_true, y_pred, labels=PROPOSED_INTENTS)
    except:
        kappa = None
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=PROPOSED_INTENTS)
    
    # Per-intent agreement
    intent_agreement = {}
    for i, intent in enumerate(PROPOSED_INTENTS):
        tp = cm[i, i]
        total_true = cm[i, :].sum()
        total_pred = cm[:, i].sum()
        intent_agreement[intent] = {
            "true_positives": int(tp),
            "total_annotated_as": int(total_pred),
            "total_truth": int(total_true),
            "precision": round(tp / total_pred, 3) if total_pred > 0 else 0,
            "recall": round(tp / total_true, 3) if total_true > 0 else 0
        }
    
    return {
        "cohen_kappa": round(kappa, 3) if kappa else None,
        "total_pairs": len(y_true),
        "confusion_matrix": cm.tolist(),
        "per_intent": intent_agreement
    }


def resolve_annotations(agreements: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Resolve multi-annotator labels via majority vote."""
    resolved = []
    
    for conv_id, anns in agreements.items():
        if not anns:
            continue
            
        if len(anns) == 1:
            # Single annotator - use their label
            resolved.append(anns[0])
        else:
            # Majority vote
            labels = [a["annotated_intent"] for a in anns if a.get("annotated_intent")]
            if not labels:
                continue
            majority_label = Counter(labels).most_common(1)[0][0]
            
            # Use first annotation as base, update intent
            base = anns[0].copy()
            base["annotated_intent"] = majority_label
            base["annotator_notes"] = f"Resolved from {len(anns)} annotators: {dict(Counter(labels))}"
            base["annotator_id"] = "majority_vote"
            resolved.append(base)
    
    return resolved


def create_golden_set(resolved: List[Dict[str, Any]], target_size: int = GOLDEN_SET_SIZE) -> List[Dict[str, Any]]:
    """Create balanced golden set from resolved annotations."""
    logger.info(f"Creating golden set (target: {target_size})...")
    
    # Group by intent
    by_intent = defaultdict(list)
    for conv in resolved:
        intent = conv.get("annotated_intent")
        if intent and intent in PROPOSED_INTENTS:
            by_intent[intent].append(conv)
    
    # Calculate per-intent quota
    intents_with_data = {k: v for k, v in by_intent.items() if v}
    n_intents = len(intents_with_data)
    
    if n_intents == 0:
        logger.warning("No annotated intents found!")
        return []
    
    base_quota = target_size // n_intents
    remainder = target_size % n_intents
    
    golden = []
    for i, (intent, convs) in enumerate(intents_with_data.items()):
        quota = base_quota + (1 if i < remainder else 0)
        # Sample up to quota
        if len(convs) > quota:
            selected = random.sample(convs, quota)
        else:
            selected = convs
            logger.warning(f"Intent {intent}: only {len(convs)} samples, need {quota}")
        golden.extend(selected)
    
    # If still short, fill from largest intents
    if len(golden) < target_size:
        all_remaining = []
        for intent, convs in by_intent.items():
            all_remaining.extend([c for c in convs if c not in golden])
        random.shuffle(all_remaining)
        golden.extend(all_remaining[:target_size - len(golden)])
    
    logger.info(f"Golden set created: {len(golden)} conversations")
    return golden


def create_splits(golden: List[Dict[str, Any]]) -> Tuple[List, List, List]:
    """Create conversation-level train/val/test splits (70/15/15)."""
    # Group by intent for stratification
    intent_labels = [c["annotated_intent"] for c in golden]
    
    # First split: 70% train, 30% temp
    train_idx, temp_idx = train_test_split(
        range(len(golden)), test_size=0.3, random_state=RANDOM_SEED,
        stratify=intent_labels
    )
    
    # Second split: 50/50 of temp = 15% val, 15% test
    temp_labels = [intent_labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=RANDOM_SEED,
        stratify=temp_labels
    )
    
    train = [golden[i] for i in train_idx]
    val = [golden[i] for i in val_idx]
    test = [golden[i] for i in test_idx]
    
    logger.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


def save_golden_set(
    golden: List[Dict[str, Any]], 
    train: List[Dict[str, Any]], 
    val: List[Dict[str, Any]], 
    test: List[Dict[str, Any]],
    agreement_stats: Dict[str, Any],
    annotation_stats: Dict[str, Any]
):
    """Save golden set and splits."""
    
    # Full golden set
    golden_path = ANNOTATIONS_DIR / "golden_set.jsonl"
    with open(golden_path, "w") as f:
        for conv in golden:
            f.write(json.dumps(conv, default=str) + "\n")
    
    # Splits
    for name, split in [("train", train), ("val", val), ("test", test)]:
        split_path = ANNOTATIONS_DIR / f"golden_{name}.jsonl"
        with open(split_path, "w") as f:
            for conv in split:
                f.write(json.dumps(conv, default=str) + "\n")
    
    # Metadata
    metadata = {
        "golden_set_size": len(golden),
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "intent_distribution": Counter(c["annotated_intent"] for c in golden),
        "split_intent_distribution": {
            "train": Counter(c["annotated_intent"] for c in train),
            "val": Counter(c["annotated_intent"] for c in val),
            "test": Counter(c["annotated_intent"] for c in test)
        },
        "agreement": agreement_stats,
        "annotation_stats": annotation_stats,
        "taxonomy_version": "1.0",
        "proposed_intents": PROPOSED_INTENTS
    }
    
    meta_path = ANNOTATIONS_DIR / "golden_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f"Saved golden set to {golden_path}")
    logger.info(f"Saved splits to {ANNOTATIONS_DIR}/golden_{{train,val,test}}.jsonl")
    logger.info(f"Saved metadata to {meta_path}")


def print_taxonomy_report(
    resolved: List[Dict[str, Any]], 
    agreement_stats: Dict[str, Any],
    golden: List[Dict[str, Any]]
):
    """Print taxonomy validation report."""
    print("\n" + "=" * 50)
    print("PHASE 2.2 TAXONOMY VALIDATION REPORT")
    print("=" * 50)
    
    # Agreement
    print(f"\nInter-Annotator Agreement:")
    print(f"  Cohen's Kappa: {agreement_stats.get('cohen_kappa', 'N/A')}")
    print(f"  Total comparison pairs: {agreement_stats.get('total_pairs', 0)}")
    
    # Per-intent agreement
    if "per_intent" in agreement_stats:
        print("\nPer-Intent Agreement:")
        for intent, stats in agreement_stats["per_intent"].items():
            if stats["total_truth"] > 0:
                print(f"  {intent:25s}: P={stats['precision']:.2f} R={stats['recall']:.2f} (n={stats['total_truth']})")
    
    # Golden set distribution
    golden_dist = Counter(c["annotated_intent"] for c in golden)
    print(f"\nGolden Set ({len(golden)} conversations):")
    for intent, count in golden_dist.most_common():
        print(f"  {intent:25s}: {count}")
    
    # Check for missing intents
    all_intents = set(PROPOSED_INTENTS)
    found_intents = set(golden_dist.keys())
    missing = all_intents - found_intents
    if missing:
        print(f"\n⚠️  MISSING INTENTS (no annotated examples): {missing}")
    
    # Cluster mixing analysis (if we have cluster_id)
    cluster_intent_map = defaultdict(Counter)
    for conv in resolved:
        cid = conv.get("cluster_id")
        intent = conv.get("annotated_intent")
        if cid is not None and intent:
            cluster_intent_map[cid][intent] += 1
    
    print("\nCluster → Intent Mapping (detecting mixed clusters):")
    for cid in sorted(cluster_intent_map.keys()):
        dist = cluster_intent_map[cid]
        total = sum(dist.values())
        dominant = dist.most_common(1)[0]
        purity = dominant[1] / total if total > 0 else 0
        status = "✓ PURE" if purity >= 0.8 else "⚠ MIXED" if purity >= 0.5 else "✗ HIGHLY MIXED"
        print(f"  Cluster {cid:2d}: {dict(dist)} → {status} (purity={purity:.2f})")
    
    print(f"\nFiles created:")
    print(f"  {ANNOTATIONS_DIR}/golden_set.jsonl")
    print(f"  {ANNOTATIONS_DIR}/golden_train.jsonl")
    print(f"  {ANNOTATIONS_DIR}/golden_val.jsonl")
    print(f"  {ANNOTATIONS_DIR}/golden_test.jsonl")
    print(f"  {ANNOTATIONS_DIR}/golden_metadata.json")


def main():
    logger.info("Starting Phase 2.2: Golden Set Creation")
    
    # Load annotated data
    annotation_file = ANNOTATIONS_DIR / "annotation_master.jsonl"
    if not annotation_file.exists():
        logger.error(f"Annotation file not found: {annotation_file}")
        logger.error("Please run 02_1_sample_for_annotation.py first, then annotate the conversations.")
        sys.exit(1)
    
    annotations = load_annotated_data(annotation_file)
    logger.info(f"Loaded {len(annotations)} annotated samples")
    
    # Check completeness
    stats, agreements = check_annotation_completeness(annotations)
    
    print(f"\nAnnotation Status:")
    print(f"  Total samples: {stats['total_conversations']}")
    print(f"  Annotated: {stats['annotated_count']}")
    print(f"  Unique conversations: {stats['unique_conversations_annotated']}")
    print(f"  With ≥{MIN_ANNOTATORS} annotators: {stats['conversations_with_agreement']}")
    print(f"  Annotators: {stats['annotators_found']}")
    
    if stats["annotated_count"] == 0:
        logger.error("No annotations found! Please annotate the conversations first.")
        sys.exit(1)
    
    # Compute agreement
    agreement_stats = compute_agreement(agreements)
    
    # Resolve annotations
    resolved = resolve_annotations(agreements)
    logger.info(f"Resolved {len(resolved)} conversations via majority vote")
    
    # Create golden set
    golden = create_golden_set(resolved, GOLDEN_SET_SIZE)
    
    if len(golden) == 0:
        logger.error("Golden set is empty!")
        sys.exit(1)
    
    # Create splits
    train, val, test = create_splits(golden)
    
    # Save everything
    save_golden_set(golden, train, val, test, agreement_stats, stats)
    
    # Print report
    print_taxonomy_report(resolved, agreement_stats, golden)
    
    logger.info("Phase 2.2 complete! Taxonomy frozen.")


if __name__ == "__main__":
    main()