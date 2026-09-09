#!/usr/bin/env python3
"""
Semi-automated pre-labeling for Phase 2.1.
Uses cluster statistics + keyword rules to assign initial intent labels.
Human then reviews/corrects in a simple CSV.
"""

import json
import pandas as pd
import re
from pathlib import Path
from collections import Counter

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"
MASTER_FILE = ANNOTATIONS_DIR / "annotation_master.jsonl"
OUTPUT_CSV = ANNOTATIONS_DIR / "annotation_review.csv"
OUTPUT_JSONL = ANNOTATIONS_DIR / "annotation_master_labeled.jsonl"

# Intent keyword rules (high precision patterns)
INTENT_RULES = {
    "non_english": [
        r"\b(que|la|el|por|lo|en|mi|si|pero|una|ingres|correo|codigo|recibo|nada|necesito|cuenta|gracias|por favor|consulta|avor|seur|coño|puta|mierda)\b",
        r"\b(je|vous|est|pas|le|ai|command|livr|probl|merci|bonsoir|conteste|preuve|signature|colis|hier|reçu|parfait)\b",
        r"\b(das|ist|und|nicht|die|der|für|mit|eine|habe|kein|app|funktioniert|taugt|nicht|kommunizieren)\b"
    ],
    "spam_irrelevant": [
        r"\b(quiz|winner|fame|famous|promotion|marketing|advertis)\b"
    ],
    "resolution_confirmation": [
        r"^(thank|thanks|resolved|fixed|working|sorted|appreciate|great|awesome|perfect).*",
        r"^thank you.*",
        r"^thanks.*resolved"
    ],
    "delivery_issue": [
        r"\b(deliver|delivered|delivery|tracking|package|parcel|courier|shipping|shipped|arrive|late|delay|missing|lost|not received|not arrived|says delivered|shows delivered|handed to me|handed over)\b"
    ],
    "order_problem": [
        r"\b(wrong item|missing item|damaged|cancel order|pre.order|pre order|order.*wrong|item.*wrong|received.*wrong|got.*wrong)\b"
    ],
    "refund_return": [
        r"\b(refund|return|replacement|replace|return label|money back|send back|returned|reimburse)\b"
    ],
    "account_access": [
        r"\b(login|log in|password|locked|verify|verification|email.*code|not receiving|not receive|code.*email|reset|account.*locked|can't log|cannot log|sign in)\b"
    ],
    "technical_app": [
        r"\b(app|application|website|site|error|bug|crash|not working|broken|glitch|alexa|echo|kindle|fire tv|firetv|device|streaming|video|music)\b"
    ],
    "billing_payment": [
        r"\b(charge|charged|billing|bill|payment|pay|money|bank|card|unauthorized|fraudulent|£|$|€)\b"
    ],
    "prime_subscription": [
        r"\b(prime|membership|subscription|student|renew|cancel prime|prime member|prime shipping|prime video)\b"
    ],
    "seller_marketplace": [
        r"\b(seller|third party|marketplace|vendor|merchant)\b"
    ],
    "gift_card": [
        r"\b(gift card|giftcard|balance|redeem)\b"
    ],
    "general_inquiry": [
        r"\b(how to|question|policy|information|help|ask|inquire)\b"
    ],
    "escalation_needed": [
        r"\b(escalate|specialist|team|fraud|legal|lawyer|ignored|no response|never replied|days requesting|complaint)\b"
    ]
}

# Priority order (first match wins) - more specific first
PRIORITY_ORDER = [
    "non_english", "spam_irrelevant", "resolution_confirmation",
    "escalation_needed", "delivery_issue", "order_problem",
    "refund_return", "account_access", "technical_app",
    "billing_payment", "prime_subscription", "seller_marketplace",
    "gift_card", "general_inquiry"
]

# Cluster → likely intent mapping (from analysis)
CLUSTER_HINTS = {
    0: "delivery_issue",      # update, team, parcel, tracking
    1: "non_english",         # spanish/portuguese
    2: "resolution_confirmation",  # thanks, help, link
    3: "general_inquiry",     # help, need help, hi
    4: "delivery_issue",      # delivered, package, says delivered
    5: "prime_subscription",  # prime, membership, student
    6: "billing_payment",     # account, charge, bank, money
    7: "non_english",         # spanish
    8: "delivery_issue",      # reply, delivery, waiting
    9: "prime_subscription",  # prime, video, delivery
    10: "general_inquiry",    # noisy cluster - mixed
    11: "escalation_needed",  # customer service, worst
    12: "refund_return",      # refund, replacement, item
    13: "escalation_needed",  # day requesting response
    14: "delivery_issue",     # delivery, prime, guaranteed
    15: "resolution_confirmation",  # thank, resolved
    16: "account_access",     # email, verification
    17: "technical_app",      # app, kindle, alexa
    18: "order_problem",      # order, cancel, refund
    19: "non_english"         # french
}


def load_annotations() -> list:
    """Load annotation master file."""
    data = []
    with open(MASTER_FILE, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def keyword_match(text: str, patterns: list) -> bool:
    """Check if text matches any pattern."""
    text_lower = text.lower()
    for pattern in patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    return False


def predict_intent(conv: dict) -> tuple:
    """Predict intent using keyword rules + cluster hint."""
    first_msg = conv.get("first_customer_message", "").lower()
    full_transcript = conv.get("full_transcript", "").lower()
    cluster_id = conv.get("cluster_id")
    
    # Check first customer message first (most indicative)
    text_to_check = first_msg if first_msg else full_transcript
    
    # Apply keyword rules in priority order
    for intent in PRIORITY_ORDER:
        if keyword_match(text_to_check, INTENT_RULES[intent]):
            return intent, "keyword"
    
    # Fallback to cluster hint
    if cluster_id in CLUSTER_HINTS:
        return CLUSTER_HINTS[cluster_id], "cluster_hint"
    
    return "general_inquiry", "default"


def create_review_csv(data: list) -> pd.DataFrame:
    """Create CSV for human review with pre-filled labels."""
    rows = []
    
    for conv in data:
        pred_intent, method = predict_intent(conv)
        
        # Extract entities with simple regex
        entities = {}
        text = conv.get("full_transcript", "")
        
        # Order IDs
        order_matches = re.findall(r'\b\d{3}-\d{7}-\d{7}\b', text)
        if order_matches:
            entities["order_id"] = order_matches[0]
        
        # Products
        products = ["Echo", "Kindle", "Fire TV", "Fire Stick", "Alexa", "Prime"]
        for p in products:
            if p.lower() in text.lower():
                entities["product"] = p
                break
        
        row = {
            "conversation_id": conv["conversation_id"],
            "cluster_id": conv["cluster_id"],
            "message_count": conv["message_count"],
            "predicted_intent": pred_intent,
            "prediction_method": method,
            "first_customer_message": conv["first_customer_message"][:200],
            "transcript": conv["full_transcript"][:500],
            # Human review columns
            "corrected_intent": pred_intent,  # Pre-filled - human edits if wrong
            "entities": str(entities) if entities else "",
            "resolution": "unclear",
            "notes": "",
            "annotator_id": ""
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def main():
    print("Loading annotations...")
    data = load_annotations()
    print(f"Loaded {len(data)} conversations")
    
    # Generate predictions
    df = create_review_csv(data)
    
    # Show prediction distribution
    print("\nPredicted intent distribution:")
    print(df["predicted_intent"].value_counts().to_string())
    print(f"\nBy method: {df['prediction_method'].value_counts().to_dict()}")
    
    # Save CSV for review
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved review CSV: {OUTPUT_CSV}")
    print("Columns: conversation_id, cluster_id, predicted_intent, first_customer_message, transcript, corrected_intent, entities, resolution, notes, annotator_id")
    
    # Also save updated JSONL with predictions
    labeled_data = []
    for _, row in df.iterrows():
        # Find original
        orig = next(d for d in data if d["conversation_id"] == row["conversation_id"])
        labeled = orig.copy()
        labeled["annotated_intent"] = row["corrected_intent"]
        labeled["annotated_entities"] = eval(row["entities"]) if row["entities"] else None
        labeled["annotated_resolution"] = row["resolution"]
        labeled["annotator_notes"] = row["notes"]
        labeled["annotator_id"] = "prelabel_auto"
        labeled_data.append(labeled)
    
    with open(OUTPUT_JSONL, "w") as f:
        for item in labeled_data:
            f.write(json.dumps(item, default=str) + "\n")
    print(f"Saved pre-labeled JSONL: {OUTPUT_JSONL}")
    
    print("\n" + "=" * 50)
    print("NEXT STEPS:")
    print("=" * 50)
    print(f"1. Open {OUTPUT_CSV} in Excel/Sheets")
    print("2. Review 'predicted_intent' column")
    print("3. Correct in 'corrected_intent' column where needed")
    print("4. Fill in entities, resolution, notes, annotator_id")
    print("5. Save CSV")
    print("6. Run: python3 scripts/annotation_tools/csv_to_jsonl.py")
    print("7. Then run: python3 scripts/02_2_create_golden_set.py")


if __name__ == "__main__":
    main()