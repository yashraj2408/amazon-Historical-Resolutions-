#!/usr/bin/env python3
"""
Phase 8: LLM-based Response Generation
Replace template responses with LLM-generated responses grounded in retrieved historical conversations.
"""

import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

import numpy as np
import faiss
import joblib
from sentence_transformers import SentenceTransformer

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

# Try to import OpenAI, fallback to local LLM simulation
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available, using simulated LLM responses")

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
    generation_method: str  # "llm" or "template"

class LLMResponseGenerator:
    def __init__(self, use_openai: bool = False, model: str = "gpt-4o-mini"):
        self.use_openai = use_openai and OPENAI_AVAILABLE
        self.model = model
        
        if self.use_openai:
            logger.info(f"Using OpenAI {model} for response generation")
        else:
            logger.info("Using simulated LLM (template-based with context)")
    
    def generate_response(
        self, 
        query: str, 
        intent: str, 
        retrieved: List[RetrievalResult],
        decision: str
    ) -> str:
        """Generate grounded response using LLM or template."""
        
        if decision == "ESCALATE":
            return self._generate_escalation_response(query, intent)
        
        # Build context from retrieved examples
        context = self._build_context(retrieved)
        
        if self.use_openai:
            return self._generate_openai_response(query, intent, context, retrieved)
        else:
            return self._generate_simulated_response(query, intent, context, retrieved)
    
    def _build_context(self, retrieved: List[RetrievalResult]) -> str:
        """Build context string from retrieved examples."""
        if not retrieved:
            return "No similar historical conversations found."
        
        context_parts = []
        for i, ex in enumerate(retrieved, 1):
            context_parts.append(
                f"Example {i} (similarity: {ex.similarity_score:.2f}):\n"
                f"Customer: {ex.customer_query[:300]}\n"
                f"Support: {ex.resolution[:400]}"
            )
        return "\n\n---\n\n".join(context_parts)
    
    def _generate_openai_response(self, query: str, intent: str, context: str, retrieved: List) -> str:
        """Generate response using OpenAI API."""
        prompt = f"""You are AmazonHelp customer support agent. Generate a helpful, empathetic response.

Customer Issue: {query}
Detected Intent: {intent}

Historical Similar Cases (for grounding):
{context}

Guidelines:
- Be empathetic and professional
- Reference specific steps from historical cases when relevant
- Don't make up information not in context
- If issue needs account access, guide to secure channel
- Keep response concise but complete
- Match AmazonHelp tone: helpful, apologetic when appropriate, action-oriented

Response:"""
        
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are AmazonHelp, Amazon's customer support on Twitter. Be helpful, empathetic, and action-oriented."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            return self._generate_simulated_response("", "", "", [])
    
    def _generate_simulated_response(self, query: str, intent: str, context: str, retrieved: List) -> str:
        """Generate realistic simulated response based on retrieved context."""
        
        # Extract key actions from retrieved resolutions
        actions = []
        for ex in retrieved[:2]:
            res = ex.resolution.lower()
            # Extract actionable items
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
        
        unique_actions = list(dict.fromkeys(actions))[:3]  # Deduplicate, max 3
        
        # Build response based on intent
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
    
    def _generate_escalation_response(self, query: str, intent: str) -> str:
        return (
            "I understand you need help with this issue. Let me connect you with a specialist "
            "who can provide more detailed assistance. They'll be able to access your account "
            "and resolve this for you."
        )


class RAGAgentWithLLM:
    def __init__(
        self,
        classifier_path: Path,
        index_path: Path,
        metadata_path: Path,
        embeddings_path: Path,
        embedding_model: str = EMBEDDING_MODEL,
        top_k: int = TOP_K_RETRIEVAL,
        auto_handle_threshold: float = 0.7,
        escalate_intents: List[str] = None,
        use_openai: bool = False
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
        
        logger.info("Initializing LLM response generator...")
        self.llm = LLMResponseGenerator(use_openai=use_openai)
        
        logger.info("RAG Agent with LLM initialized")
    
    def clean_text(self, text: str) -> str:
        import re
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    
    def predict_intent(self, query: str) -> Tuple[str, float]:
        query_clean = self.clean_text(query)
        # Embed query first (classifier was trained on embeddings)
        query_emb = self.embedder.encode([query_clean], convert_to_numpy=True)
        pred = self.classifier.predict(query_emb)[0]
        proba = self.classifier.predict_proba(query_emb)[0]
        confidence = max(proba)
        return pred, confidence
    
    def retrieve(self, query: str, intent: str = None, filter_by_intent: bool = True) -> List[RetrievalResult]:
        query_clean = self.clean_text(query)
        query_emb = self.embedder.encode([query_clean], convert_to_numpy=True)
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
        
        # Fallback
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
    
    def make_decision(self, intent: str, confidence: float, retrieval_results: List) -> Tuple[Decision, str]:
        if intent in self.escalate_intents:
            return Decision.ESCALATE, f"Intent '{intent}' requires specialist handling"
        if confidence < self.auto_handle_threshold:
            return Decision.ESCALATE, f"Low intent confidence ({confidence:.2f} < {self.auto_handle_threshold})"
        if not retrieval_results:
            return Decision.ESCALATE, "No similar resolved conversations found"
        max_score = max(r.similarity_score for r in retrieval_results)
        if max_score < 0.6:
            return Decision.ESCALATE, f"Low retrieval similarity (max={max_score:.2f})"
        return Decision.AUTO_HANDLE, f"High confidence ({confidence:.2f}), good retrieval match ({max_score:.2f})"
    
    def process(self, query: str) -> AgentResponse:
        intent, confidence = self.predict_intent(query)
        retrieval_results = self.retrieve(query, intent, filter_by_intent=True)
        decision, reason = self.make_decision(intent, confidence, retrieval_results)
        
        response_text = self.llm.generate_response(query, intent, retrieval_results, decision.value)
        
        return AgentResponse(
            decision=decision,
            reason=reason,
            response_text=response_text,
            retrieved_examples=retrieval_results,
            predicted_intent=intent,
            intent_confidence=confidence,
            generation_method="llm" if self.llm.use_openai else "simulated_llm"
        )


def evaluate_on_golden(agent: RAGAgentWithLLM, test_path: Path) -> Dict[str, Any]:
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
            "response": response.response_text[:300],
            "generation_method": response.generation_method,
            "retrieval_count": len(response.retrieved_examples),
            "max_similarity": max([r.similarity_score for r in response.retrieved_examples]) if response.retrieved_examples else 0
        })
    
    total = len(results)
    intent_correct = sum(1 for r in results if r["intent_correct"])
    auto_handle = sum(1 for r in results if r["decision"] == "AUTO_HANDLE")
    
    return {
        "total_samples": total,
        "intent_accuracy": intent_correct / total if total > 0 else 0,
        "auto_handle_rate": auto_handle / total if total > 0 else 0,
        "avg_intent_confidence": np.mean([r["intent_confidence"] for r in results]),
        "avg_max_similarity": np.mean([r["max_similarity"] for r in results]),
        "results": results
    }


def main():
    logger.info("Starting Phase 8: LLM-based Response Generation")
    
    # Paths
    classifier_path = MODELS_DIR / "intent_classifier_v3.joblib"
    index_path = INDEX_DIR / "resolution_index.faiss"
    metadata_path = INDEX_DIR / "resolution_metadata.jsonl"
    embeddings_path = INDEX_DIR / "resolution_embeddings.npy"
    test_path = ANNOTATIONS_DIR / "golden_test.jsonl"
    
    # Initialize agent (use simulated LLM by default)
    agent = RAGAgentWithLLM(
        classifier_path=classifier_path,
        index_path=index_path,
        metadata_path=metadata_path,
        embeddings_path=embeddings_path,
        top_k=3,
        auto_handle_threshold=0.7,
        escalate_intents=["escalation_needed"],
        use_openai=False  # Set True if you have OpenAI API key
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
    print("PHASE 8: LLM RESPONSE GENERATION DEMO")
    print("=" * 80)
    
    for query in demo_queries:
        response = agent.process(query)
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"Intent: {response.predicted_intent} (conf: {response.intent_confidence:.2f})")
        print(f"Decision: {response.decision.value} - {response.reason}")
        print(f"Generation: {response.generation_method}")
        print(f"Retrieved: {len(response.retrieved_examples)} examples")
        print(f"Response:\n{response.response_text}")
    
    # Evaluate on golden test set
    if test_path.exists():
        logger.info("Evaluating on golden test set...")
        metrics = evaluate_on_golden(agent, test_path)
        
        print("\n" + "=" * 80)
        print("GOLDEN SET EVALUATION WITH LLM RESPONSES")
        print("=" * 80)
        print(f"Total samples: {metrics['total_samples']}")
        print(f"Intent Accuracy: {metrics['intent_accuracy']:.2%}")
        print(f"Auto-handle Rate: {metrics['auto_handle_rate']:.2%}")
        print(f"Avg Intent Confidence: {metrics['avg_intent_confidence']:.2f}")
        print(f"Avg Max Similarity: {metrics['avg_max_similarity']:.2f}")
        
        # Save evaluation
        eval_path = RESULTS_DIR / "llm_rag_evaluation.json"
        with open(eval_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        logger.info(f"Saved evaluation to {eval_path}")
    
    logger.info("Phase 8 complete!")


if __name__ == "__main__":
    main()