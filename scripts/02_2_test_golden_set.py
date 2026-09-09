#!/usr/bin/env python3
"""
Phase 2.2 Test: Create golden evaluation set from single-annotator data.
For testing pipeline without full multi-annotator agreement.
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
GOLDEN_SET_SIZE = 200

PROPOSED_INTENTS = [
    "delivery_issue", "order_problem", "refund_return", "account_access",
    "technical_app", "billing_payment", "prime_subscription", "seller_marketplace",
    "gift_card", "general_inquiry", "escalation_needed", "non_english",
    "spam_irrelevant", "resolution_confirmation"
]


def load_annotated_data(annotation_file: Path) -> List[Dict[str, Any]]:
    conversations = []
    with open(annotation_file, "r") as f:
        for line in f:
            conversations.append(json.loads(line))
    return conversations


def check_annotation_completeness(annotations: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(annotations)
    annotated = [a for a in annotations if a.get("annotated_intent") is not None]
    
    stats = {
        "total_conversations": total,
        "annotated_count": len(annotated),
        "unique_conversations_annotated": len(set(a["conversation_id"] for a in annotated)),
        "annotators_found": set(a.get("annotator_id") for a in annotated if a.get("annotator_id")),
        "intent_distribution": Counter(a["annotated_intent"] for a in annotated if a.get("annotated_intent"))
    }
    return stats, annotated


def create_golden_set(resolved: List[Dict[str, Any]], target_size: int = GOLDEN_SET_SIZE) -> List[Dict[str, Any]]:
    logger.info(f"Creating golden set (target: {target_size})...")
    
    by_intent = defaultdict(list)
    for conv in resolved:
        intent = conv.get("annotated_intent")
        if intent and intent in PROPOSED_INTENTS:
            by_intent[intent].append(conv)
    
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
        if len(convs) > quota:
            selected = random.sample(convs, quota)
        else:
            selected = convs
            logger.warning(f"Intent {intent}: only {len(convs)} samples, need {quota}")
        golden.extend(selected)
    
    if len(golden) < target_size:
        all_remaining = []
        for intent, convs in by_intent.items():
            all_remaining.extend([c for c in convs if c not in golden])
        random.shuffle(all_remaining)
        golden.extend(all_remaining[:target_size - len(golden)])
    
    logger.info(f"Golden set created: {len(golden)} conversations")
    return golden


def create_splits(golden: List[Dict[str, Any]]) -> Tuple[List, List, List]:
    intent_labels = [c["annotated_intent"] for c in golden]
    
    # Check if stratification is possible (each class needs at least 2 samples for 3-way split)
    label_counts = Counter(intent_labels)
    can_stratify = all(count >= 2 for count in label_counts.values())
    
    if can_stratify:
        train_idx, temp_idx = train_test_split(
            range(len(golden)), test_size=0.3, random_state=RANDOM_SEED,
            stratify=intent_labels
        )
        temp_labels = [intent_labels[i] for i in temp_idx]
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=RANDOM_SEED,
            stratify=temp_labels
        )
    else:
        logger.warning("Some classes have <2 samples - using random split without stratification")
        indices = list(range(len(golden)))
        random.shuffle(indices)
        n_train = int(0.7 * len(golden))
        n_val = int(0.15 * len(golden))
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
    
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
    annotation_stats: Dict[str, Any]
):
    for name, split in [("train", train), ("val", val), ("test", test)]:
        split_path = ANNOTATIONS_DIR / f"golden_{name}.jsonl"
        with open(split_path, "w") as f:
            for conv in split:
                f.write(json.dumps(conv, default=str) + "\n")
    
    golden_path = ANNOTATIONS_DIR / "golden_set.jsonl"
    with open(golden_path, "w") as f:
        for conv in golden:
            f.write(json.dumps(conv, default=str) + "\n")
    
    # Cluster purity analysis
    cluster_intent_map = defaultdict(Counter)
    for conv in golden:
        cid = conv.get("cluster_id")
        intent = conv.get("annotated_intent")
        if cid is not None and intent:
            cluster_intent_map[cid][intent] += 1
    
    metadata = {
        "golden_set_size": len(golden),
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "intent_distribution": dict(Counter(c["annotated_intent"] for c in golden)),
        "split_intent_distribution": {
            "train": dict(Counter(c["annotated_intent"] for c in train)),
            "val": dict(Counter(c["annotated_intent"] for c in val)),
            "test": dict(Counter(c["annotated_intent"] for c in test))
        },
        "cluster_purity": {
            str(cid): {
                "distribution": dict(dist),
                "dominant_intent": dist.most_common(1)[0][0],
                "purity": dist.most_common(1)[0][1] / sum(dist.values())
            } for cid, dist in cluster_intent_map.items()
        },
        "annotation_stats": annotation_stats,
        "taxonomy_version": "1.0-test",
        "proposed_intents": PROPOSED_INTENTS,
        "note": "Single-annotator test run - no inter-annotator agreement computed"
    }
    
    meta_path = ANNOTATIONS_DIR / "golden_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f"Saved golden set to {golden_path}")
    logger.info(f"Saved splits to {ANNOTATIONS_DIR}/golden_{{train,val,test}}.jsonl")
    logger.info(f"Saved metadata to {meta_path}")
    
    return metadata


def print_taxonomy_report(golden: List[Dict[str, Any]], metadata: Dict[str, Any]):
    print("\n" + "=" * 50)
    print("PHASE 2.2 TEST - GOLDEN SET CREATION REPORT")
    print("=" * 50)
    
    # Intent distribution
    golden_dist = Counter(c["annotated_intent"] for c in golden)
    print(f"\nGolden Set ({len(golden)} conversations):")
    for intent, count in golden_dist.most_common():
        print(f"  {intent:25s}: {count}")
    
    # Missing intents
    all_intents = set(PROPOSED_INTENTS)
    found_intents = set(golden_dist.keys())
    missing = all_intents - found_intents
    if missing:
        print(f"\n⚠️  MISSING INTENTS: {missing}")
    
    # Cluster purity
    print("\nCluster → Intent Purity (golden set):")
    for cid_str, info in sorted(metadata.get("cluster_purity", {}).items(), key=lambda x: int(x[0])):
        dist = info["distribution"]
        purity = info["purity"]
        dominant = info["dominant_intent"]
        status = "✓ PURE" if purity >= 0.8 else "⚠ MIXED" if purity >= 0.5 else "✗ HIGHLY MIXED"
        print(f"  Cluster {cid_str:>2s}: {dict(dist)} → {status} (purity={purity:.2f}, dominant={dominant})")
    
    print(f"\nSplits: train={metadata['train_size']}, val={metadata['val_size']}, test={metadata['test_size']}")
    print(f"\nFiles created:")
    print(f"  {ANNOTATIONS_DIR}/golden_set.jsonl")
    print(f"  {ANNOTATIONS_DIR}/golden_train.jsonl")
    print(f"  {ANNOTATIONS_DIR}/golden_val.jsonl")
    print(f"  {ANNOTATIONS_DIR}/golden_test.jsonl")
    print(f"  {ANNOTATIONS_DIR}/golden_metadata.json")


def main():
    logger.info("Starting Phase 2.2 Test: Golden Set from Single-Annotator Data")
    
    annotation_file = ANNOTATIONS_DIR / "annotation_master.jsonl"
    if not annotation_file.exists():
        logger.error(f"File not found: {annotation_file}")
        sys.exit(1)
    
    annotations = load_annotated_data(annotation_file)
    logger.info(f"Loaded {len(annotations)} annotated samples")
    
    stats, resolved = check_annotation_completeness(annotations)
    
    print(f"\nAnnotation Status:")
    print(f"  Total samples: {stats['total_conversations']}")
    print(f"  Annotated: {stats['annotated_count']}")
    print(f"  Unique conversations: {stats['unique_conversations_annotated']}")
    print(f"  Annotators: {stats['annotators_found']}")
    
    if stats["annotated_count"] == 0:
        logger.error("No annotations found!")
        sys.exit(1)
    
    # Create golden set
    golden = create_golden_set(resolved, GOLDEN_SET_SIZE)
    
    if len(golden) == 0:
        logger.error("Golden set is empty!")
        sys.exit(1)
    
    # Create splits
    train, val, test = create_splits(golden)
    
    # Save
    metadata = save_golden_set(golden, train, val, test, stats)
    
    # Print report
    print_taxonomy_report(golden, metadata)
    
    logger.info("Phase 2.2 Test complete!")


if __name__ == "__main__":
    main()