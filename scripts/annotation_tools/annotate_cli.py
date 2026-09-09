#!/usr/bin/env python3
"""
Simple CLI annotation tool for Phase 2.1 taxonomy validation.
Run: python scripts/annotation_tools/annotate_cli.py --annotator YOUR_NAME
"""

import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"

PROPOSED_INTENTS = [
    "delivery_issue", "order_problem", "refund_return", "account_access",
    "technical_app", "billing_payment", "prime_subscription", "seller_marketplace",
    "gift_card", "general_inquiry", "escalation_needed", "non_english",
    "spam_irrelevant", "resolution_confirmation"
]

INTENT_DESCRIPTIONS = {
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
}


def load_annotations(filepath: Path) -> List[Dict[str, Any]]:
    """Load annotation file."""
    data = []
    with open(filepath, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_annotations(filepath: Path, data: List[Dict[str, Any]]):
    """Save annotation file."""
    with open(filepath, "w") as f:
        for item in data:
            f.write(json.dumps(item, default=str) + "\n")


def print_intent_menu():
    """Print intent selection menu."""
    print("\n" + "=" * 60)
    print("INTENT TAXONOMY")
    print("=" * 60)
    for i, intent in enumerate(PROPOSED_INTENTS):
        print(f"  {i+1:2d}. {intent:25s} - {INTENT_DESCRIPTIONS[intent]}")
    print("  0.  Skip this conversation")
    print("=" * 60)


def annotate_conversation(conv: Dict[str, Any], annotator_id: str) -> Dict[str, Any]:
    """Interactive annotation for a single conversation."""
    print("\n" + "=" * 80)
    print(f"CONVERSATION: {conv['conversation_id']}  |  Cluster: {conv['cluster_id']}")
    print(f"Messages: {conv['message_count']} (Customer: {conv['customer_message_count']}, Support: {conv['support_message_count']})")
    print("=" * 80)
    print(conv['full_transcript'])
    print("-" * 80)
    print(f"FIRST CUSTOMER MESSAGE: {conv['first_customer_message'][:200]}")
    
    while True:
        print_intent_menu()
        choice = input(f"\nSelect intent (1-{len(PROPOSED_INTENTS)}) or 0 to skip: ").strip()
        
        if choice == "0":
            return None
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(PROPOSED_INTENTS):
                selected_intent = PROPOSED_INTENTS[idx]
                break
            else:
                print("Invalid choice. Try again.")
        except ValueError:
            print("Please enter a number.")
    
    # Optional: entities
    entities_input = input("Extract entities? (order_id, product, etc.) [press Enter to skip]: ").strip()
    entities = {}
    if entities_input:
        # Simple parsing: "order_id: 123, product: Echo"
        for pair in entities_input.split(","):
            if ":" in pair:
                k, v = pair.split(":", 1)
                entities[k.strip()] = v.strip()
    
    # Resolution
    resolution = input("Resolution? (yes/no/partial/unclear) [unclear]: ").strip() or "unclear"
    
    # Notes
    notes = input("Notes (ambiguity, boundary cases, etc.) [press Enter to skip]: ").strip()
    
    # Create annotated record
    annotated = conv.copy()
    annotated["annotated_intent"] = selected_intent
    annotated["annotated_entities"] = entities if entities else None
    annotated["annotated_resolution"] = resolution
    annotated["annotator_notes"] = notes if notes else None
    annotated["annotator_id"] = annotator_id
    
    print(f"✓ Labeled as: {selected_intent}")
    return annotated


def main():
    parser = argparse.ArgumentParser(description="CLI annotation tool for AmazonHelp taxonomy validation")
    parser.add_argument("--annotator", required=True, help="Your annotator ID (e.g., 'annotator_1')")
    parser.add_argument("--file", default="annotation_master.jsonl", help="Annotation file to work on")
    parser.add_argument("--cluster", type=int, help="Only annotate specific cluster (0-19)")
    parser.add_argument("--resume", action="store_true", help="Skip already annotated conversations")
    args = parser.parse_args()
    
    filepath = ANNOTATIONS_DIR / args.file
    if not filepath.exists():
        print(f"File not found: {filepath}")
        print("Run 02_1_sample_for_annotation.py first!")
        sys.exit(1)
    
    # Load data
    data = load_annotations(filepath)
    
    # Filter
    if args.cluster is not None:
        data = [d for d in data if d.get("cluster_id") == args.cluster]
    
    if args.resume:
        data = [d for d in data if d.get("annotated_intent") is None]
    
    print(f"Loaded {len(data)} conversations to annotate")
    print(f"Annotator: {args.annotator}")
    
    if len(data) == 0:
        print("Nothing to annotate!")
        return
    
    # Annotate
    annotated_count = 0
    for i, conv in enumerate(data):
        print(f"\nProgress: {i+1}/{len(data)}")
        result = annotate_conversation(conv, args.annotator)
        
        if result is None:
            print("Skipped.")
            continue
        
        # Update in original data list
        for j, orig in enumerate(data):
            if orig["conversation_id"] == conv["conversation_id"]:
                data[j] = result
                break
        
        annotated_count += 1
        
        # Save progress every 5 annotations
        if annotated_count % 5 == 0:
            save_annotations(filepath, data)
            print(f"Progress saved ({annotated_count} annotated)")
        
        # Continue prompt
        if i < len(data) - 1:
            cont = input("\nContinue? (y/n): ").strip().lower()
            if cont != "y":
                break
    
    # Final save
    save_annotations(filepath, data)
    print(f"\nDone! Annotated {annotated_count} conversations.")
    print(f"Saved to: {filepath}")


if __name__ == "__main__":
    main()