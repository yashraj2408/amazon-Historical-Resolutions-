#!/usr/bin/env python3
"""
Phase 2.1: Sample conversations per cluster for manual taxonomy validation.
Outputs annotation-ready JSONL files for human review.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
import random

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"
ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

# Constants
RANDOM_SEED = 42
SAMPLES_PER_CLUSTER = 25
N_CLUSTERS = 20
PROPOSED_INTENTS = [
    "delivery_issue",
    "order_problem", 
    "refund_return",
    "account_access",
    "technical_app",
    "billing_payment",
    "prime_subscription",
    "seller_marketplace",
    "gift_card",
    "general_inquiry",
    "escalation_needed",
    "non_english",
    "spam_irrelevant",
    "resolution_confirmation"
]

def load_conversations() -> List[Dict[str, Any]]:
    """Load reconstructed conversations."""
    conv_path = PROCESSED_DIR / "amazonhelp_conversations.jsonl"
    conversations = []
    with open(conv_path, "r") as f:
        for line in f:
            conversations.append(json.loads(line))
    logger.info(f"Loaded {len(conversations)} conversations")
    return conversations


def load_customer_sample() -> pd.DataFrame:
    """Load customer sample with cluster assignments."""
    sample_path = PROCESSED_DIR / "amazonhelp_customer_sample.csv"
    df = pd.read_csv(sample_path)
    logger.info(f"Loaded {len(df)} customer samples with cluster assignments")
    return df


def load_cluster_stats() -> pd.DataFrame:
    """Load cluster statistics."""
    stats_path = RESULTS_DIR / "intent_candidate_clusters.csv"
    df = pd.read_csv(stats_path)
    return df


def build_conversation_index(conversations: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Build lookup: conversation_id -> conversation."""
    return {c["conversation_id"]: c for c in conversations}


def sample_conversations_per_cluster(
    customer_sample: pd.DataFrame,
    conversation_index: Dict[str, Dict[str, Any]],
    samples_per_cluster: int = SAMPLES_PER_CLUSTER
) -> Dict[int, List[Dict[str, Any]]]:
    """Sample conversations from each cluster for annotation."""
    logger.info(f"Sampling {samples_per_cluster} conversations per cluster...")
    
    random.seed(RANDOM_SEED)
    cluster_samples = defaultdict(list)
    
    for cluster_id in range(N_CLUSTERS):
        cluster_convs = customer_sample[customer_sample["cluster_id"] == cluster_id]["conversation_id"].unique()
        
        # Filter to conversations that exist in our index
        valid_convs = [cid for cid in cluster_convs if cid in conversation_index]
        
        if len(valid_convs) == 0:
            logger.warning(f"Cluster {cluster_id}: no valid conversations found")
            continue
            
        # Sample
        n_samples = min(samples_per_cluster, len(valid_convs))
        sampled_ids = random.sample(valid_convs, n_samples)
        
        for conv_id in sampled_ids:
            conv = conversation_index[conv_id]
            cluster_samples[cluster_id].append(conv)
        
        logger.info(f"Cluster {cluster_id}: sampled {n_samples}/{len(valid_convs)} conversations")
    
    return cluster_samples


def format_conversation_for_annotation(conv: Dict[str, Any], cluster_id: int) -> Dict[str, Any]:
    """Format a conversation for human annotation."""
    # Build readable transcript
    transcript_lines = []
    for msg in conv["messages"]:
        speaker_label = "CUSTOMER" if msg["speaker"] == "customer" else "SUPPORT"
        timestamp = msg["timestamp"][:19] if msg["timestamp"] else "N/A"
        transcript_lines.append(f"[{timestamp}] {speaker_label}: {msg['text']}")
    
    transcript = "\n".join(transcript_lines)
    
    # Extract customer messages for intent classification
    customer_msgs = [m for m in conv["messages"] if m["speaker"] == "customer"]
    support_msgs = [m for m in conv["messages"] if m["speaker"] == "support"]
    
    # First customer message (the initial issue)
    first_customer_msg = customer_msgs[0]["text"] if customer_msgs else ""
    
    return {
        "conversation_id": conv["conversation_id"],
        "cluster_id": cluster_id,
        "message_count": len(conv["messages"]),
        "customer_message_count": len(customer_msgs),
        "support_message_count": len(support_msgs),
        "resolved_candidate": conv.get("resolved_candidate", False),
        "resolution_score": conv.get("resolution_score", 0),
        "first_customer_message": first_customer_msg,
        "full_transcript": transcript,
        # Annotation fields (to be filled by human)
        "annotated_intent": None,
        "annotated_entities": None,
        "annotated_resolution": None,
        "annotator_notes": None,
        "annotator_id": None
    }


def create_annotation_batches(cluster_samples: Dict[int, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Create annotation batches - one file per cluster."""
    all_samples = []
    
    for cluster_id, convs in cluster_samples.items():
        for conv in convs:
            annotated = format_conversation_for_annotation(conv, cluster_id)
            all_samples.append(annotated)
    
    # Shuffle for balanced annotation order
    random.seed(RANDOM_SEED)
    random.shuffle(all_samples)
    
    logger.info(f"Created {len(all_samples)} total annotation samples")
    return all_samples


def save_annotation_files(all_samples: List[Dict[str, Any]], cluster_samples: Dict[int, List[Dict[str, Any]]]):
    """Save annotation-ready files."""
    
    # 1. Master annotation file (all samples)
    master_path = ANNOTATIONS_DIR / "annotation_master.jsonl"
    with open(master_path, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, default=str) + "\n")
    logger.info(f"Saved master annotation file: {master_path}")
    
    # 2. Per-cluster files for focused annotation
    for cluster_id, convs in cluster_samples.items():
        cluster_file = ANNOTATIONS_DIR / f"cluster_{cluster_id:02d}_annotation.jsonl"
        with open(cluster_file, "w") as f:
            for conv in convs:
                annotated = format_conversation_for_annotation(conv, cluster_id)
                f.write(json.dumps(annotated, default=str) + "\n")
        logger.info(f"Saved cluster {cluster_id} annotation file: {cluster_file} ({len(convs)} samples)")
    
    # 3. Annotation guidelines
    guidelines_path = ANNOTATIONS_DIR / "ANNOTATION_GUIDELINES.md"
    create_guidelines(guidelines_path)
    
    # 4. Labeling schema reference
    schema_path = ANNOTATIONS_DIR / "LABELING_SCHEMA.json"
    create_schema(schema_path)


def create_guidelines(path: Path):
    """Create annotation guidelines document."""
    guidelines = f"""# Annotation Guidelines: AmazonHelp Intent Taxonomy Validation

## Objective
Validate and refine the intent taxonomy by manually labeling sampled conversations from each of the 20 KMeans clusters.

## Proposed Intent Categories (14)

| Intent | Description | Keywords / Signals |
|--------|-------------|-------------------|
| `delivery_issue` | Package not arrived, late, tracking problems, "says delivered but not received" | delivery, delivered, tracking, late, package, parcel, courier, shipping |
| `order_problem` | Wrong item, missing item, damaged, cancel order, pre-order issues | order, item, wrong, missing, damaged, cancel, pre-order |
| `refund_return` | Request refund, return process, return label, replacement | refund, return, replace, replacement, money back, label |
| `account_access` | Login issues, password reset, account locked, verification, email not received | login, password, account, locked, verify, verification, email, code |
| `technical_app` | App not working, website error, Alexa/device issues, Kindle | app, website, error, bug, alexa, echo, kindle, device, fire tv |
| `billing_payment` | Unauthorized charge, billing question, payment failed, charge on account | charge, charged, billing, payment, pay, money, bank, card |
| `prime_subscription` | Prime membership, student price, subscription cancel, benefits | prime, membership, subscription, student, cancel prime, renew |
| `seller_marketplace` | Third-party seller issues, marketplace orders | seller, third party, marketplace, vendor, merchant |
| `gift_card` | Gift card balance, redeem, not working | gift card, giftcard, balance, redeem, code |
| `general_inquiry` | Product questions, policy, how-to, non-specific | how to, question, policy, information, help |
| `escalation_needed` | Complex issues requiring specialist (fraud, legal, persistent unresolved) | escalate, specialist, team, fraud, legal, lawyer, ignored |
| `non_english` | Spanish, French, German, other non-English | que, la, el, por, je, vous, est, das, ist |
| `spam_irrelevant` | Promotional, quiz, winner, unrelated noise | quiz, winner, fame, promotion, marketing |
| `resolution_confirmation` | Customer confirms issue resolved, thanks support | thank, thanks, resolved, fixed, working, sorted, appreciate |

## Annotation Process

### For Each Conversation:
1. **Read the full transcript** - understand the full context
2. **Identify the PRIMARY intent** of the customer's initial issue (first customer message)
3. **Note any secondary intents** if the conversation spans multiple topics
4. **Extract key entities** - order IDs, product names, account details, etc.
5. **Assess resolution** - was this conversation resolved? (yes/no/partial/unclear)
6. **Add notes** - ambiguity, boundary cases, cluster mixing observations

### Labeling Rules:
- **Primary intent only** - pick the ONE most representative category
- If truly mixed (e.g., delivery + refund), pick the **initial** issue
- `non_english` takes precedence - label as non_english + note the likely intent
- `spam_irrelevant` for clearly non-support content
- `resolution_confirmation` only if the customer's PRIMARY message is "thanks, it's fixed"

### Cluster Mixing Detection:
- If a cluster contains conversations with **different primary intents**, note this
- This signals the cluster should be **split** in the final taxonomy
- If multiple clusters map to the **same intent**, they should be **merged**

## Output Format
Each annotated conversation adds these fields:
```json
{{
  "annotated_intent": "delivery_issue",
  "annotated_entities": {{"order_id": "403-7503264", "product": "Echo Show"}},
  "annotated_resolution": "yes",
  "annotator_notes": "Customer says delivered but not received. Support sent tracking link.",
  "annotator_id": "annotator_1"
}}
```

## Deliverables
1. Annotated master file with all samples
2. Intent distribution per cluster (confusion matrix)
3. Revised taxonomy proposal
4. Golden set: 150-250 conversations with agreed labels
"""
    with open(path, "w") as f:
        f.write(guidelines)
    logger.info(f"Created guidelines: {path}")


def create_schema(path: Path):
    """Create labeling schema reference."""
    schema = {
        "intents": PROPOSED_INTENTS,
        "intent_descriptions": {
            "delivery_issue": "Package delivery problems: late, missing, tracking, 'delivered but not received'",
            "order_problem": "Order content issues: wrong item, missing item, damaged, cancel order",
            "refund_return": "Refund requests, return process, replacements, return labels",
            "account_access": "Login, password reset, account locked, verification, email issues",
            "technical_app": "App/website errors, device issues (Alexa, Echo, Kindle, Fire TV)",
            "billing_payment": "Unauthorized charges, billing questions, payment failures",
            "prime_subscription": "Prime membership, student pricing, subscription management",
            "seller_marketplace": "Third-party seller issues, marketplace orders",
            "gift_card": "Gift card balance, redemption, issues",
            "general_inquiry": "Product questions, policy, how-to, general help",
            "escalation_needed": "Complex issues needing specialist: fraud, legal, persistent unresolved",
            "non_english": "Non-English conversations (Spanish, French, German, etc.)",
            "spam_irrelevant": "Promotional, quiz, marketing, non-support content",
            "resolution_confirmation": "Customer confirms resolution / thanks support"
        },
        "entity_types": [
            "order_id",
            "product_name", 
            "tracking_number",
            "account_email",
            "device_type",
            "seller_name",
            "amount",
            "date"
        ],
        "resolution_values": ["yes", "no", "partial", "unclear"],
        "annotation_fields": [
            "annotated_intent",
            "annotated_entities", 
            "annotated_resolution",
            "annotator_notes",
            "annotator_id"
        ]
    }
    with open(path, "w") as f:
        json.dump(schema, f, indent=2)
    logger.info(f"Created schema: {path}")


def print_summary(cluster_samples: Dict[int, List[Dict[str, Any]]], all_samples: List[Dict[str, Any]]):
    """Print sampling summary."""
    print("\n" + "=" * 50)
    print("PHASE 2.1 SAMPLING COMPLETE")
    print("=" * 50)
    print(f"\nTotal clusters sampled: {len(cluster_samples)}")
    print(f"Total annotation samples: {len(all_samples)}")
    print(f"Samples per cluster: {SAMPLES_PER_CLUSTER}")
    
    print("\nCluster breakdown:")
    for cluster_id in sorted(cluster_samples.keys()):
        count = len(cluster_samples[cluster_id])
        print(f"  Cluster {cluster_id:2d}: {count} conversations")
    
    print(f"\nFiles created in {ANNOTATIONS_DIR}:")
    print(f"  annotation_master.jsonl - all samples for annotation")
    for cluster_id in sorted(cluster_samples.keys()):
        print(f"  cluster_{cluster_id:02d}_annotation.jsonl - cluster-specific")
    print(f"  ANNOTATION_GUIDELINES.md - labeling instructions")
    print(f"  LABELING_SCHEMA.json - intent/entity schema reference")
    
    print("\nNEXT STEPS:")
    print("1. Open ANNOTATION_GUIDELINES.md")
    print("2. Annotate conversations in annotation_master.jsonl or per-cluster files")
    print("3. Run 02_2_create_golden_set.py after annotation complete")


def main():
    logger.info("Starting Phase 2.1: Sampling for Taxonomy Validation")
    
    # Load data
    conversations = load_conversations()
    customer_sample = load_customer_sample()
    cluster_stats = load_cluster_stats()
    
    # Build index
    conversation_index = build_conversation_index(conversations)
    
    # Sample per cluster
    cluster_samples = sample_conversations_per_cluster(
        customer_sample, conversation_index, SAMPLES_PER_CLUSTER
    )
    
    # Create annotation batches
    all_samples = create_annotation_batches(cluster_samples)
    
    # Save files
    save_annotation_files(all_samples, cluster_samples)
    
    # Print summary
    print_summary(cluster_samples, all_samples)
    
    logger.info("Phase 2.1 sampling complete!")


if __name__ == "__main__":
    main()