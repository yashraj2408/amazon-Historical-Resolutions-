#!/usr/bin/env python3
"""
Simple Flask backend for the RAG Agent frontend.
"""

import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import faiss
import joblib
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
INDEX_DIR = PROJECT_ROOT / "index"

# Constants
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_RETRIEVAL = 3
ESCALATE_INTENTS = ["escalation_needed"]

# Global components
classifier = None
index = None
examples = None
embeddings = None
embedder = None
calibration_artifacts = None


def load_components():
    """Load all model components."""
    global classifier, index, examples, embeddings, embedder, calibration_artifacts
    
    logger.info("Loading classifier...")
    classifier = joblib.load(PROJECT_ROOT / "models" / "intent_classifier_v3.joblib")
    
    logger.info("Loading FAISS index...")
    index = faiss.read_index(str(INDEX_DIR / "resolution_index.faiss"))
    
    logger.info("Loading metadata...")
    examples = []
    with open(INDEX_DIR / "resolution_metadata.jsonl", "r") as f:
        for line in f:
            examples.append(json.loads(line))
    
    logger.info("Loading embeddings...")
    embeddings = np.load(INDEX_DIR / "resolution_embeddings.npy")
    
    logger.info("Loading embedder...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    logger.info("Loading calibration...")
    with open(PROJECT_ROOT / "models" / "calibration_artifacts.json", "r") as f:
        calibration_artifacts = json.load(f)
    
    logger.info("All components loaded!")


def clean_text(text: str) -> str:
    import re
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_intent(query: str) -> Tuple[str, float]:
    query_clean = clean_text(query)
    query_emb = embedder.encode([query_clean], convert_to_numpy=True)
    pred = classifier.predict(query_emb)[0]
    proba = classifier.predict_proba(query_emb)[0]
    confidence = float(max(proba))
    return str(pred), confidence


def retrieve(query: str, intent: str = None, filter_by_intent: bool = True) -> List[Dict]:
    query_clean = clean_text(query)
    query_emb = embedder.encode([query_clean], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)
    
    if filter_by_intent and intent:
        intent_index_path = INDEX_DIR / f"index_{intent}.faiss"
        if intent_index_path.exists():
            intent_index = faiss.read_index(str(intent_index_path))
            meta_path = INDEX_DIR / f"metadata_{intent}.jsonl"
            if meta_path.exists():
                intent_examples = []
                with open(meta_path, "r") as f:
                    for line in f:
                        intent_examples.append(json.loads(line))
                
                scores, indices = intent_index.search(query_emb.astype(np.float32), TOP_K_RETRIEVAL)
                results = []
                for idx, score in zip(indices[0], scores[0]):
                    if idx < len(intent_examples):
                        ex = intent_examples[idx]
                        results.append({
                            "conversation_id": ex["conversation_id"],
                            "customer_query": ex["customer_query"][:300],
                            "resolution": ex["resolution"][:400],
                            "predicted_intent": ex["predicted_intent"],
                            "similarity_score": float(score)
                        })
                return results
    
    # Fallback
    search_k = min(TOP_K_RETRIEVAL * 5, index.ntotal)
    scores, indices = index.search(query_emb.astype(np.float32), search_k)
    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < len(examples):
            ex = examples[idx]
            if not filter_by_intent or ex["predicted_intent"] == intent:
                results.append({
                    "conversation_id": ex["conversation_id"],
                    "customer_query": ex["customer_query"][:300],
                    "resolution": ex["resolution"][:400],
                    "predicted_intent": ex["predicted_intent"],
                    "similarity_score": float(score)
                })
                if len(results) >= TOP_K_RETRIEVAL:
                    break
    return results


def make_decision(intent: str, confidence: float, retrieval_results: List) -> Tuple[str, str]:
    if intent in ESCALATE_INTENTS:
        return "ESCALATE", f"Intent '{intent}' requires specialist handling"
    if confidence < calibration_artifacts["thresholds"].get(intent, 0.7):
        return "ESCALATE", f"Low intent confidence ({confidence:.2f} < threshold)"
    if not retrieval_results:
        return "ESCALATE", "No similar resolved conversations found"
    max_score = max(r["similarity_score"] for r in retrieval_results)
    if max_score < 0.6:
        return "ESCALATE", f"Low retrieval similarity (max={max_score:.2f})"
    return "AUTO_HANDLE", f"High confidence ({confidence:.2f}), good retrieval match ({max_score:.2f})"


def generate_response(query: str, intent: str, retrieval_results: List, decision: str) -> str:
    if decision == "ESCALATE":
        return (
            "I understand you need help with this issue. Let me connect you with a specialist "
            "who can provide more detailed assistance. They'll be able to access your account "
            "and resolve this for you."
        )
    
    actions = []
    for ex in retrieval_results[:2]:
        res = ex["resolution"].lower()
        if "check" in res or "try" in res:
            actions.append("troubleshooting steps")
        if "contact" in res or "reach out" in res or "dm" in res:
            actions.append("direct contact/DM")
        if "refund" in res or "return" in res:
            actions.append("refund/return process")
        if "tracking" in res or "carrier" in res:
            actions.append("tracking investigation")
        if "reset" in res or "password" in res:
            actions.append("password/account reset")
        if "prime" in res and "membership" in res:
            actions.append("Prime membership review")
        if "escalat" in res or "specialist" in res:
            actions.append("specialist escalation")
    
    unique_actions = list(dict.fromkeys(actions))[:3]
    
    templates = {
        "delivery_issue": (
            "I'm sorry to hear about the delivery issue! Based on similar cases, here's what typically helps:\n\n"
            "{actions}\n\n"
            "Could you share your order details so I can investigate further?"
        ),
        "account_access": (
            "I understand you're having trouble accessing your account. Here are the standard steps:\n\n"
            "{actions}\n\n"
            "If these don't work, I'll connect you with account support for secure verification."
        ),
        "refund_return": (
            "I'd be happy to help with your refund/return! For similar cases:\n\n"
            "{actions}\n\n"
            "Could you provide the order number so I can check the specific status?"
        ),
        "billing_payment": (
            "I'm sorry for the billing concern! Let me help:\n\n"
            "{actions}\n\n"
            "If you need a detailed review, I can connect you with billing support."
        ),
        "prime_billing": (
            "I understand the concern about your Prime membership charges. Here's what we typically do:\n\n"
            "{actions}\n\n"
            "Could you confirm the email on the account so I can look into this?"
        ),
        "technical_device": (
            "I'm sorry for the technical trouble! For device issues, these steps usually help:\n\n"
            "{actions}\n\n"
            "If the issue persists, I can connect you with technical support for real-time troubleshooting."
        ),
        "prime_subscription": (
            "I understand you have questions about your Prime subscription. Here's what typically helps:\n\n"
            "{actions}\n\n"
            "Let me know if you need help with membership benefits or changes."
        ),
        "order_problem": (
            "I'm sorry about the issue with your order! Based on similar cases:\n\n"
            "{actions}\n\n"
            "Could you share the order number so I can look into this specifically?"
        ),
        "escalation_needed": (
            "I understand this needs specialist attention. Let me escalate this for you:\n\n"
            "{actions}\n\n"
            "A specialist will reach out to you shortly."
        ),
        "non_english": (
            "Gracias por contactarnos. Entendemos tu consulta y estamos aquí para ayudarte.\n\n"
            "{actions}\n\n"
            "¿Podrías proporcionar más detalles para poder asistirte mejor?"
        ),
        "resolution_confirmation": (
            "Thank you for letting us know! We're glad the issue is resolved.\n\n"
            "If you need any further assistance, don't hesitate to reach out."
        ),
        "general_inquiry": (
            "Thank you for reaching out to Amazon Help! I'd be happy to assist.\n\n"
            "{actions}\n\n"
            "Could you provide a bit more detail about what you need help with?"
        )
    }
    
    template = templates.get(intent, templates["general_inquiry"])
    actions_str = "\n".join([f"• {a}" for a in unique_actions]) if unique_actions else "• Our team will investigate and get back to you"
    return template.format(actions=actions_str)


# Initialize Flask app
app = Flask(__name__, static_folder="static")
CORS(app)


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/process", methods=["POST"])
def process_query():
    data = request.get_json()
    query = data.get("query", "").strip()
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    # Step 1: Intent classification
    intent, confidence = predict_intent(query)
    
    # Step 2: Retrieval
    retrieval_results = retrieve(query, intent, filter_by_intent=True)
    
    # Step 3: Decision
    decision, reason = make_decision(intent, confidence, retrieval_results)
    
    # Step 4: Response generation
    response_text = generate_response(query, intent, retrieval_results, decision)
    
    return jsonify({
        "query": query,
        "intent": intent,
        "confidence": round(confidence, 3),
        "decision": decision,
        "reason": reason,
        "response": response_text,
        "retrieval_count": len(retrieval_results),
        "retrieved_examples": retrieval_results,
        "threshold_used": calibration_artifacts["thresholds"].get(intent, 0.7) if calibration_artifacts else 0.7
    })


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model_loaded": classifier is not None})


if __name__ == "__main__":
    load_components()
    app.run(host="0.0.0.0", port=5001, debug=True)