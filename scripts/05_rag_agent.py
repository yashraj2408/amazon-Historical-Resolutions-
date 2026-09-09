#!/usr/bin/env python3
"""
Phase 5: RAG Response Generation
Complete pipeline: Intent Classification → Retrieval → Grounded Response Generation → AUTO_HANDLE/ESCALATE
"""

import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import joblib

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
INDEX_DIR = PROJECT_ROOT / "index"
MODELS_DIR = PROJECT_ROOT / "models"
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_DIM = 384
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_RETRIEVAL = 3


class Decision(Enum):
    AUTO_HANDLE = "AUTO_HANDLE"
    ESCALATE = "ESCALATE"


@dataclass
class RetrievalResult:
    conversation_id: str
    customer_query: str
    resolution: str
    predicted_intent: str
    similarity_score: float


@dataclass
class AgentResponse:
    decision: Decision
    reason: str
    response_text: str
    retrieved_examples: List[RetrievalResult]
    predicted_intent: str
    intent_confidence: float


class RAGAgent:
    def __init__(
        self,
        classifier_path: Path,
        index_path: Path,
        metadata_path: Path,
        embeddings_path: Path,
        embedding_model: str = EMBEDDING_MODEL,
        top_k: int = TOP_K_RETRIEVAL,
        auto_handle_threshold: float = 0.7,
        escalate_intents: List[str] = None
    ):
        self.top_k = top_k
        self.auto_handle_threshold = auto_handle_threshold
        self.escalate_intents = escalate_intents or ["escalation_needed"]
        
        # Load components
        logger.info("Loading intent classifier...")
        self.classifier = joblib.load(classifier_path)
        
        logger.info("Loading FAISS index...")
        self.index = faiss.read_index(str(index_path))
        
        logger.info("Loading metadata...")
        self.examples = []
        with open(metadata_path, "r") as f:
            for line in f:
                self.examples.append(json.loads(line))
        
        logger.info("Loading embeddings...")
        self.embeddings = np.load(embeddings_path)
        
        logger.info("Loading embedding model...")
        self.embedder = SentenceTransformer(embedding_model)
        
        logger.info("RAG Agent initialized")
    
    def clean_text(self, text: str) -> str:
        import re
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    def predict_intent(self, query: str) -> Tuple[str, float]:
        """Predict intent and confidence."""
        query_clean = self.clean_text(query)
        pred = self.classifier.predict([query_clean])[0]
        proba = self.classifier.predict_proba([query_clean])[0]
        confidence = max(proba)
        return pred, confidence
    
    def retrieve(
        self, 
        query: str, 
        intent: str = None, 
        filter_by_intent: bool = True
    ) -> List[RetrievalResult]:
        """Retrieve similar resolved conversations."""
        query_clean = self.clean_text(query)
        query_emb = self.embedder.encode([query_clean], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)
        
        if filter_by_intent and intent:
            # Use intent-filtered index if available
            intent_index_path = INDEX_DIR / f"index_{intent}.faiss"
            if intent_index_path.exists():
                intent_index = faiss.read_index(str(intent_index_path))
                # Need to load intent examples
                meta_path = INDEX_DIR / f"metadata_{intent}.jsonl"
                if meta_path.exists():
                    intent_examples = []
                    with open(meta_path, "r") as f:
                        for line in f:
                            intent_examples.append(json.loads(line))
                    
                    scores, indices = intent_index.search(query_emb.astype(np.float32), self.top_k)
                    
                    results = []
                    for idx, score in zip(indices[0], scores[0]):
                        if idx < len(intent_examples):
                            ex = intent_examples[idx]
                            results.append(RetrievalResult(
                                conversation_id=ex["conversation_id"],
                                customer_query=ex["customer_query"],
                                resolution=ex["resolution"],
                                predicted_intent=ex["predicted_intent"],
                                similarity_score=float(score)
                            ))
                    return results
        
        # Fallback: general index with post-filtering
        search_k = min(self.top_k * 5, self.index.ntotal)
        scores, indices = self.index.search(query_emb.astype(np.float32), search_k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.examples):
                ex = self.examples[idx]
                if not filter_by_intent or ex["predicted_intent"] == intent:
                    results.append(RetrievalResult(
                        conversation_id=ex["conversation_id"],
                        customer_query=ex["customer_query"],
                        resolution=ex["resolution"],
                        predicted_intent=ex["predicted_intent"],
                        similarity_score=float(score)
                    ))
                    if len(results) >= self.top_k:
                        break
        
        return results
    
    def make_decision(
        self, 
        intent: str, 
        intent_confidence: float,
        retrieval_results: List[RetrievalResult]
    ) -> Tuple[Decision, str]:
        """Decide AUTO_HANDLE vs ESCALATE."""
        
        # Escalate if intent is in escalation list
        if intent in self.escalate_intents:
            return Decision.ESCALATE, f"Intent '{intent}' requires specialist handling"
        
        # Escalate if low confidence
        if intent_confidence < self.auto_handle_threshold:
            return Decision.ESCALATE, f"Low intent confidence ({intent_confidence:.2f} < {self.auto_handle_threshold})"
        
        # Escalate if no relevant retrieval results
        if not retrieval_results:
            return Decision.ESCALATE, "No similar resolved conversations found"
        
        # Check retrieval quality
        max_score = max(r.similarity_score for r in retrieval_results)
        if max_score < 0.6:
            return Decision.ESCALATE, f"Low retrieval similarity (max={max_score:.2f})"
        
        return Decision.AUTO_HANDLE, f"High confidence ({intent_confidence:.2f}), good retrieval match ({max_score:.2f})"
    
    def generate_response(
        self, 
        query: str, 
        retrieval_results: List[RetrievalResult],
        decision: Decision
    ) -> str:
        """Generate grounded response using retrieved resolutions."""
        
        if decision == Decision.ESCALATE:
            return (
                "I understand you need help with this issue. Let me connect you with a specialist "
                "who can provide more detailed assistance. They'll be able to access your account "
                "and resolve this for you."
            )
        
        # Build context from retrieved resolutions
        context_parts = []
        for i, result in enumerate(retrieval_results, 1):
            context_parts.append(
                f"Example {i} (similarity: {result.similarity_score:.2f}):\n"
                f"Customer: {result.customer_query[:200]}\n"
                f"Support: {result.resolution[:300]}"
            )
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Template-based response generation (no LLM needed for demo)
        # In production, this would be an LLM prompt
        if retrieval_results[0].predicted_intent == "delivery_issue":
            return self._generate_delivery_response(query, retrieval_results)
        elif retrieval_results[0].predicted_intent == "account_access":
            return self._generate_account_response(query, retrieval_results)
        elif retrieval_results[0].predicted_intent == "refund_return":
            return self._generate_refund_response(query, retrieval_results)
        elif retrieval_results[0].predicted_intent == "prime_subscription":
            return self._generate_prime_response(query, retrieval_results)
        elif retrieval_results[0].predicted_intent == "technical_device":
            return self._generate_technical_response(query, retrieval_results)
        else:
            return self._generate_generic_response(query, retrieval_results)
    
    def _generate_delivery_response(self, query: str, results: List[RetrievalResult]) -> str:
        return (
            "I'm sorry to hear about the delivery issue! Based on similar cases, here's what I recommend:\n\n"
            "1. Check your mailbox and around the delivery area\n"
            "2. Verify with neighbors or building management\n"
            "3. If still missing, we can investigate with the carrier\n\n"
            "Would you like me to look into this further? If so, please provide your order details "
            "and I'll connect you with the right team to track this down."
        )
    
    def _generate_account_response(self, query: str, results: List[RetrievalResult]) -> str:
        return (
            "I understand you're having trouble accessing your account. Here are the standard steps:\n\n"
            "1. Try the password reset page directly\n"
            "2. Make sure you're using the correct email/phone\n"
            "3. Clear browser cache/cookies or try a different browser\n\n"
            "If these don't work, I'll need to connect you with account support who can "
            "verify your identity and help regain access."
        )
    
    def _generate_refund_response(self, query: str, results: List[RetrievalResult]) -> str:
        return (
            "I'd be happy to help with your refund request! For returns and refunds:\n\n"
            "1. Check the return status in your orders\n"
            "2. Refunds typically process within 3-5 business days after return is received\n"
            "3. If it's been longer, we can investigate\n\n"
            "Could you provide the order number so I can check the specific status?"
        )
    
    def _generate_prime_response(self, query: str, results: List[RetrievalResult]) -> str:
        return (
            "I'm sorry for the issue with your Prime membership! Let me help:\n\n"
            "1. Check your membership status in Your Account\n"
            "2. If you were charged incorrectly, we can review and refund\n"
            "3. Student Prime verification may need renewal\n\n"
            "Could you share the email on the account so I can look into this?"
        )
    
    def _generate_technical_response(self, query: str, results: List[RetrievalResult]) -> str:
        return (
            "I'm sorry for the technical trouble! For device issues:\n\n"
            "1. Try restarting the device\n"
            "2. Check Wi-Fi connection and router\n"
            "3. Make sure the app is updated\n\n"
            "If the issue persists, I can connect you with technical support for real-time troubleshooting."
        )
    
    def _generate_generic_response(self, query: str, results: List[RetrievalResult]) -> str:
        return (
            "Thank you for reaching out! I've found similar cases that were resolved. "
            "Let me connect you with the right team to help with your specific issue. "
            "Could you provide a bit more detail about what you need assistance with?"
        )
    
    def process(self, query: str) -> AgentResponse:
        """Full pipeline: classify → retrieve → decide → generate."""
        
        # Step 1: Intent classification
        intent, confidence = self.predict_intent(query)
        
        # Step 2: Retrieval
        retrieval_results = self.retrieve(query, intent, filter_by_intent=True)
        
        # Step 3: Decision
        decision, reason = self.make_decision(intent, confidence, retrieval_results)
        
        # Step 4: Response generation
        response_text = self.generate_response(query, retrieval_results, decision)
        
        return AgentResponse(
            decision=decision,
            reason=reason,
            response_text=response_text,
            retrieved_examples=retrieval_results,
            predicted_intent=intent,
            intent_confidence=confidence
        )


def evaluate_on_golden(agent: RAGAgent, test_path: Path) -> Dict[str, Any]:
    """Evaluate agent on golden test set."""
    logger.info(f"Evaluating on {test_path}")
    
    test_data = []
    with open(test_path, "r") as f:
        for line in f:
            test_data.append(json.loads(line))
    
    results = []
    for ex in test_data:
        query = ex.get("first_customer_message", "")
        true_intent = ex.get("annotated_intent", "")
        
        response = agent.process(query)
        
        results.append({
            "conversation_id": ex["conversation_id"],
            "query": query[:100],
            "true_intent": true_intent,
            "predicted_intent": response.predicted_intent,
            "intent_confidence": response.intent_confidence,
            "intent_correct": response.predicted_intent == true_intent,
            "decision": response.decision.value,
            "reason": response.reason,
            "response": response.response_text[:200],
            "retrieval_count": len(response.retrieved_examples),
            "max_similarity": max([r.similarity_score for r in response.retrieved_examples]) if response.retrieved_examples else 0
        })
    
    # Compute metrics
    total = len(results)
    intent_correct = sum(1 for r in results if r["intent_correct"])
    auto_handle = sum(1 for r in results if r["decision"] == "AUTO_HANDLE")
    escalate = sum(1 for r in results if r["decision"] == "ESCALATE")
    
    metrics = {
        "total_samples": total,
        "intent_accuracy": intent_correct / total if total > 0 else 0,
        "auto_handle_rate": auto_handle / total if total > 0 else 0,
        "escalate_rate": escalate / total if total > 0 else 0,
        "avg_intent_confidence": np.mean([r["intent_confidence"] for r in results]),
        "avg_max_similarity": np.mean([r["max_similarity"] for r in results]),
        "results": results
    }
    
    return metrics


def main():
    logger.info("Starting Phase 5: RAG Response Generation")
    
    # Paths
    classifier_path = MODELS_DIR / "intent_classifier_v2.joblib"
    index_path = INDEX_DIR / "resolution_index.faiss"
    metadata_path = INDEX_DIR / "resolution_metadata.jsonl"
    embeddings_path = INDEX_DIR / "resolution_embeddings.npy"
    test_path = ANNOTATIONS_DIR / "golden_test.jsonl"
    
    # Check for embeddings in checkpoints
    if not embeddings_path.exists():
        # Find in checkpoints
        cp_dir = INDEX_DIR / "checkpoints"
        emb_files = list(cp_dir.glob("embeddings_*.npy"))
        if emb_files:
            embeddings_path = max(emb_files, key=lambda p: int(p.stem.split("_")[1]))
            logger.info(f"Using checkpoint embeddings: {embeddings_path}")
    
    # Initialize agent
    agent = RAGAgent(
        classifier_path=classifier_path,
        index_path=index_path,
        metadata_path=metadata_path,
        embeddings_path=embeddings_path,
        top_k=3,
        auto_handle_threshold=0.7,
        escalate_intents=["escalation_needed"]
    )
    
    # Demo queries
    demo_queries = [
        "@AmazonHelp my package says delivered but I never received it!",
        "@AmazonHelp I was charged twice for my Prime membership",
        "@AmazonHelp my Echo Dot won't connect to wifi",
        "@AmazonHelp I want to return this item, how do I get a refund?",
        "@AmazonHelp I can't log into my account, password reset not working",
        "@AmazonHelp I've been waiting 3 weeks for a response from your team!",
        "@AmazonHelp gracias por la ayuda, ya todo resuelto"
    ]
    
    print("\n" + "=" * 80)
    print("RAG AGENT DEMO")
    print("=" * 80)
    
    for query in demo_queries:
        response = agent.process(query)
        print(f"\nQuery: {query}")
        print(f"Intent: {response.predicted_intent} (conf: {response.intent_confidence:.2f})")
        print(f"Decision: {response.decision.value} - {response.reason}")
        print(f"Retrieved: {len(response.retrieved_examples)} examples")
        print(f"Response: {response.response_text[:150]}...")
    
    # Evaluate on golden test set
    if test_path.exists():
        logger.info("Evaluating on golden test set...")
        metrics = evaluate_on_golden(agent, test_path)
        
        print("\n" + "=" * 80)
        print("GOLDEN SET EVALUATION")
        print("=" * 80)
        print(f"Total samples: {metrics['total_samples']}")
        print(f"Intent Accuracy: {metrics['intent_accuracy']:.2%}")
        print(f"Auto-handle Rate: {metrics['auto_handle_rate']:.2%}")
        print(f"Escalate Rate: {metrics['escalate_rate']:.2%}")
        print(f"Avg Intent Confidence: {metrics['avg_intent_confidence']:.2f}")
        print(f"Avg Max Similarity: {metrics['avg_max_similarity']:.2f}")
        
        # Save evaluation
        eval_path = RESULTS_DIR / "rag_evaluation.json"
        with open(eval_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info(f"Saved evaluation to {eval_path}")
    
    logger.info("Phase 5 complete!")


if __name__ == "__main__":
    main()