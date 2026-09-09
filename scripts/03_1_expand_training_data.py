#!/usr/bin/env python3
"""
Phase 3.1: Expand training data using resolved conversations + keyword rules.
Creates larger labeled dataset for better classifier training.
"""

import json
import logging
import sys
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter, defaultdict
import random

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Refined Taxonomy (10 intents - split merged ones, drop unusable)
REFINED_INTENTS = [
    "delivery_issue",          # package not arrived, late, tracking, "says delivered"
    "order_problem",           # wrong item, missing item, damaged, cancel order
    "refund_return",           # refund request, return process, replacement, return label
    "account_access",          # login, password reset, account locked, verification
    "billing_payment",         # unauthorized charge, billing question, payment failed
    "technical_device",        # Echo, Kindle, Fire TV, Alexa, device issues
    "prime_subscription",      # Prime membership, student price, subscription cancel
    "escalation_needed",       # complex issues needing specialist
    "non_english",             # Spanish, French, German, etc.
    "resolution_confirmation"  # customer confirms resolved
]

# High-precision keyword rules for auto-labeling
KEYWORD_RULES = {
    "non_english": [
        r"\b(que|la|el|por|lo|en|mi|si|pero|una|ingres|correo|codigo|recibo|nada|necesito|cuenta|gracias|por favor|consulta|seur|coño|puta|mierda)\b",
        r"\b(je|vous|est|pas|le|ai|command|livr|probl|merci|bonsoir|conteste|preuve|signature|colis|hier|reçu|parfait)\b",
        r"\b(das|ist|und|nicht|die|der|für|mit|eine|habe|kein|app|funktioniert|taugt|nicht|kommunizieren)\b"
    ],
    "resolution_confirmation": [
        r"^(thank|thanks|resolved|fixed|working|sorted|appreciate|great|awesome|perfect).*",
        r"^thank you.*",
        r"^thanks.*(resolved|fixed|working|sorted)",
        r"resolved.*thank"
    ],
    "escalation_needed": [
        r"\b(escalate|specialist|team|fraud|legal|lawyer|ignored|no response|never replied|days requesting|complaint|unresolved)\b"
    ],
    "delivery_issue": [
        r"\b(deliver|delivered|delivery|tracking|package|parcel|courier|shipping|shipped|arrive|late|delay|missing|lost|not received|not arrived|says delivered|shows delivered|handed to me|handed over|transit|carrier)\b"
    ],
    "order_problem": [
        r"\b(wrong item|missing item|damaged|cancel order|pre.order|pre order|order.*wrong|item.*wrong|received.*wrong|got.*wrong|wrong product|item missing)\b"
    ],
    "refund_return": [
        r"\b(refund|return|replacement|replace|return label|money back|send back|returned|reimburse|return process)\b"
    ],
    "account_access": [
        r"\b(login|log in|password|locked|verify|verification|email.*code|not receiving|not receive|code.*email|reset|account.*locked|can't log|cannot log|sign in|access.*account)\b"
    ],
    "billing_payment": [
        r"\b(charge|charged|billing|bill|payment|pay|money|bank|card|unauthorized|fraudulent|£|€)\b(?!.*(prime|membership|subscription))"
    ],
    "prime_subscription": [
        r"\b(prime|membership|subscription|student|renew|cancel prime|prime member|prime shipping|prime video)\b"
    ],
    "technical_device": [
        r"\b(echo|kindle|fire tv|firetv|fire stick|alexa|device|streaming|video.*not working|app.*crash|app.*not working|website.*error|site.*down)\b"
    ]
}

# Priority order (specific first)
RULE_PRIORITY = [
    "non_english", "resolution_confirmation", "escalation_needed",
    "delivery_issue", "order_problem", "refund_return",
    "account_access", "billing_payment", "prime_subscription", "technical_device"
]

# Conflict resolution: some phrases match multiple
CONFLICT_RESOLUTION = {
    ("billing_payment", "prime_subscription"): "prime_subscription",
    ("billing_payment", "account_access"): "account_access",
    ("delivery_issue", "order_problem"): "delivery_issue",
    ("refund_return", "order_problem"): "refund_return",
    ("technical_device", "prime_subscription"): "technical_device",
    ("technical_device", "account_access"): "technical_device",
}


def load_conversations() -> List[Dict[str, Any]]:
    """Load all reconstructed conversations."""
    conv_path = PROCESSED_DIR / "amazonhelp_conversations.jsonl"
    conversations = []
    with open(conv_path, "r") as f:
        for line in f:
            conversations.append(json.loads(line))
    logger.info(f"Loaded {len(conversations)} conversations")
    return conversations


def load_golden_labeled() -> List[Dict[str, Any]]:
    """Load human-labeled golden set."""
    golden_path = ANNOTATIONS_DIR / "golden_set.jsonl"
    if not golden_path.exists():
        logger.warning("Golden set not found")
        return []
    
    data = []
    with open(golden_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    logger.info(f"Loaded {len(data)} golden labeled conversations")
    return data


def keyword_match(text: str, patterns: list) -> bool:
    """Check if text matches any pattern."""
    text_lower = text.lower()
    for pattern in patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    return False


def predict_intent_rules(text: str) -> Tuple[str, str]:
    """Predict intent using keyword rules."""
    matches = []
    
    for intent in RULE_PRIORITY:
        if keyword_match(text, KEYWORD_RULES[intent]):
            matches.append(intent)
    
    if not matches:
        return "general", "none"
    
    # Resolve conflicts
    if len(matches) > 1:
        # Check explicit conflict rules
        for (a, b), winner in CONFLICT_RESOLUTION.items():
            if a in matches and b in matches:
                return winner, "rule_conflict"
        # Default: first in priority
        return matches[0], "rule_multi"
    
    return matches[0], "rule_single"


def extract_training_examples(
    conversations: List[Dict[str, Any]],
    golden_labeled: List[Dict[str, Any]],
    max_per_intent: int = 2000
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Extract training examples from resolved conversations + golden set."""
    
    # Build lookup for golden labels
    golden_labels = {}
    for conv in golden_labeled:
        cid = conv["conversation_id"]
        intent = conv.get("annotated_intent")
        if intent:
            # Map to refined taxonomy (copied from intent_classifier)
            REFINED_MAPPING = {
                "delivery_issue": "delivery_issue",
                "tracking_request": "delivery_issue",
                "order_problem": "order_problem",
                "refund_return": "refund_return",
                "account_access": "account_access",
                "billing_payment": "billing_payment",
                "technical_app": "technical_device",
                "prime_subscription": "prime_subscription",
                "escalation_needed": "escalation_needed",
                "non_english": "non_english",
                "resolution_confirmation": "resolution_confirmation",
                "general_inquiry": "general",
                "seller_marketplace": "general",
                "gift_card": "general",
                "spam_irrelevant": "general"
            }
            golden_labels[cid] = REFINED_MAPPING.get(intent, "general")
    
    logger.info(f"Golden labels available for {len(golden_labels)} conversations")
    
    # Collect training examples
    examples = []
    intent_counts = Counter()
    
    for conv in conversations:
        # Only use resolved candidates
        if not conv.get("resolved_candidate", False):
            continue
        
        # Find first customer message
        first_customer_msg = None
        for msg in conv["messages"]:
            if msg["speaker"] == "customer":
                first_customer_msg = msg["text"]
                break
        
        if not first_customer_msg:
            continue
        
        conv_id = conv["conversation_id"]
        
        # Use golden label if available
        if conv_id in golden_labels:
            intent = golden_labels[conv_id]
            method = "golden"
        else:
            intent, method = predict_intent_rules(first_customer_msg)
        
        if intent == "general":
            continue  # Skip general/unclassifiable
        
        if intent_counts[intent] >= max_per_intent:
            continue
        
        examples.append({
            "conversation_id": conv_id,
            "text": first_customer_msg,
            "intent": intent,
            "label_source": method,
            "resolved_candidate": True,
            "resolution_score": conv.get("resolution_score", 0)
        })
        intent_counts[intent] += 1
    
    logger.info(f"Extracted {len(examples)} training examples")
    return examples, dict(intent_counts)


def balance_dataset(examples: List[Dict[str, Any]], target_per_intent: int = 500) -> List[Dict[str, Any]]:
    """Balance dataset by sampling up to target_per_intent per class."""
    by_intent = defaultdict(list)
    for ex in examples:
        by_intent[ex["intent"]].append(ex)
    
    balanced = []
    for intent, exs in by_intent.items():
        if len(exs) > target_per_intent:
            # Prioritize golden labels
            golden_exs = [e for e in exs if e["label_source"] == "golden"]
            rule_exs = [e for e in exs if e["label_source"] != "golden"]
            
            # Keep all golden, sample from rules
            keep_golden = golden_exs
            need_from_rules = target_per_intent - len(keep_golden)
            if need_from_rules > 0:
                sampled_rules = random.sample(rule_exs, min(need_from_rules, len(rule_exs)))
                balanced.extend(keep_golden + sampled_rules)
            else:
                balanced.extend(keep_golden[:target_per_intent])
        else:
            balanced.extend(exs)
    
    random.shuffle(balanced)
    logger.info(f"Balanced dataset: {len(balanced)} examples")
    return balanced


def create_splits(examples: List[Dict[str, Any]]) -> Tuple[List, List, List]:
    """Create stratified train/val/test splits."""
    from sklearn.model_selection import train_test_split
    
    texts = [e["text"] for e in examples]
    labels = [e["intent"] for e in examples]
    
    # Check if stratification possible
    label_counts = Counter(labels)
    can_stratify = all(c >= 2 for c in label_counts.values())
    
    indices = list(range(len(examples)))
    
    if can_stratify:
        train_idx, temp_idx = train_test_split(
            indices, test_size=0.2, random_state=RANDOM_SEED, stratify=labels
        )
        temp_labels = [labels[i] for i in temp_idx]
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=RANDOM_SEED, stratify=temp_labels
        )
    else:
        random.shuffle(indices)
        n_train = int(0.8 * len(examples))
        n_val = int(0.1 * len(examples))
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
    
    train = [examples[i] for i in train_idx]
    val = [examples[i] for i in val_idx]
    test = [examples[i] for i in test_idx]
    
    return train, val, test


def save_datasets(train: List, val: List, test: List, intent_counts: Dict):
    """Save expanded datasets."""
    
    # Save as JSONL for easy loading
    for name, data in [("train_expanded", train), ("val_expanded", val), ("test_expanded", test)]:
        path = ANNOTATIONS_DIR / f"{name}.jsonl"
        with open(path, "w") as f:
            for ex in data:
                f.write(json.dumps(ex, default=str) + "\n")
        logger.info(f"Saved {len(data)} examples to {path}")
    
    # Save as CSV for inspection
    all_data = train + val + test
    for name, data in [("train_expanded", train), ("val_expanded", val), ("test_expanded", test)]:
        df = pd.DataFrame(data)
        path = ANNOTATIONS_DIR / f"{name}.csv"
        df.to_csv(path, index=False)
    
    # Save combined
    df_all = pd.DataFrame(all_data)
    df_all.to_csv(ANNOTATIONS_DIR / "expanded_training_data.csv", index=False)
    
    # Save metadata
    meta = {
        "total_examples": len(all_data),
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "intent_distribution": dict(Counter(e["intent"] for e in all_data)),
        "label_source_distribution": dict(Counter(e["label_source"] for e in all_data)),
        "intent_counts_before_balance": intent_counts,
        "refined_intents": REFINED_INTENTS,
        "note": "Auto-labeled from resolved conversations + golden human labels"
    }
    with open(ANNOTATIONS_DIR / "expanded_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)


def print_summary(intent_counts: Dict, train: List, val: List, test: List):
    print("\n" + "=" * 60)
    print("PHASE 3.1: EXPANDED TRAINING DATA SUMMARY")
    print("=" * 60)
    
    print(f"\nExtracted examples (before balancing):")
    for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
        print(f"  {intent:25s}: {count}")
    print(f"  Total: {sum(intent_counts.values())}")
    
    print(f"\nBalanced splits:")
    print(f"  Train: {len(train)}")
    print(f"  Val:   {len(val)}")
    print(f"  Test:  {len(test)}")
    print(f"  Total: {len(train) + len(val) + len(test)}")
    
    # Distribution
    all_data = train + val + test
    dist = Counter(e["intent"] for e in all_data)
    print(f"\nFinal distribution:")
    for intent, count in dist.most_common():
        print(f"  {intent:25s}: {count}")
    
    source_dist = Counter(e["label_source"] for e in all_data)
    print(f"\nLabel sources:")
    for source, count in source_dist.most_common():
        print(f"  {source:20s}: {count}")


def main():
    logger.info("Starting Phase 3.1: Expand Training Data")
    
    # Load data
    conversations = load_conversations()
    golden_labeled = load_golden_labeled()
    
    # Extract examples
    examples, intent_counts = extract_training_examples(
        conversations, golden_labeled, max_per_intent=2000
    )
    
    if not examples:
        logger.error("No examples extracted!")
        sys.exit(1)
    
    # Balance
    balanced = balance_dataset(examples, target_per_intent=500)
    
    # Create splits
    train, val, test = create_splits(balanced)
    
    # Save
    save_datasets(train, val, test, intent_counts)
    
    # Print summary
    print_summary(intent_counts, train, val, test)
    
    logger.info("Phase 3.1 complete!")


if __name__ == "__main__":
    main()