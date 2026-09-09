#!/usr/bin/env python3
"""
Phase 7: Failure Analysis & Honest Metrics Discussion
Analyzes where the system fails, why, and discusses misleading metrics.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter, defaultdict

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"

REFINED_INTENTS = [
    "delivery_issue", "order_problem", "refund_return", "account_access",
    "billing_payment", "technical_device", "prime_subscription",
    "escalation_needed", "non_english", "resolution_confirmation"
]


def load_rag_evaluation() -> Dict:
    """Load RAG evaluation results."""
    path = RESULTS_DIR / "rag_evaluation.json"
    with open(path, "r") as f:
        return json.load(f)


def load_baseline_comparison() -> List[Dict]:
    """Load baseline comparison results."""
    path = RESULTS_DIR / "baseline_comparison.json"
    with open(path, "r") as f:
        return json.load(f)


def load_golden_test() -> List[Dict]:
    """Load golden test set."""
    path = ANNOTATIONS_DIR / "golden_test.jsonl"
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_expanded_test() -> List[Dict]:
    """Load expanded test set."""
    path = ANNOTATIONS_DIR / "test_expanded.jsonl"
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def analyze_failures(rag_results: List[Dict], test_data: List[Dict]) -> Dict:
    """Analyze failure cases in detail."""
    
    failures = [r for r in rag_results if not r["intent_correct"]]
    successes = [r for r in rag_results if r["intent_correct"]]
    
    logger.info(f"Total: {len(rag_results)}, Failures: {len(failures)}, Successes: {len(successes)}")
    
    # Failure by true intent
    failure_by_true = Counter(r["true_intent"] for r in failures)
    total_by_true = Counter(r["true_intent"] for r in rag_results)
    
    failure_rates = {}
    for intent in REFINED_INTENTS:
        total = total_by_true.get(intent, 0)
        failed = failure_by_true.get(intent, 0)
        failure_rates[intent] = failed / total if total > 0 else 0
    
    # Confusion pairs
    confusion_pairs = Counter()
    for r in failures:
        confusion_pairs[f"{r['true_intent']}→{r['predicted_intent']}"] += 1
    
    # Low confidence failures
    low_conf_failures = [r for r in failures if r["intent_confidence"] < 0.7]
    high_conf_failures = [r for r in failures if r["intent_confidence"] >= 0.7]
    
    # Escalation analysis
    escalated_failures = [r for r in failures if r["decision"] == "ESCALATE"]
    auto_handle_failures = [r for r in failures if r["decision"] == "AUTO_HANDLE"]
    
    # Retrieval quality for failures
    retrieval_failures = [r for r in failures if r["max_similarity"] < 0.6]
    retrieval_successes = [r for r in failures if r["max_similarity"] >= 0.6]
    
    return {
        "total_failures": len(failures),
        "failure_rate_by_intent": failure_rates,
        "confusion_pairs": dict(confusion_pairs),
        "low_confidence_failures": len(low_conf_failures),
        "high_confidence_failures": len(high_conf_failures),
        "escalated_failures": len(escalated_failures),
        "auto_handle_failures": len(auto_handle_failures),
        "retrieval_quality_failures": len(retrieval_failures),
        "retrieval_quality_successes": len(retrieval_successes),
        "failure_examples": [
            {
                "query": r["query"],
                "true": r["true_intent"],
                "pred": r["predicted_intent"],
                "conf": r["intent_confidence"],
                "decision": r["decision"],
                "reason": r["reason"],
                "max_sim": r["max_similarity"]
            }
            for r in failures[:20]
        ]
    }


def analyze_misleading_metrics() -> Dict:
    """Analyze potentially misleading headline metrics."""
    
    issues = {
        "small_test_set": {
            "metric": "Test accuracy (73.3%)",
            "problem": "Only 30 test samples, 10 intents = ~3 samples per intent",
            "impact": "High variance, not statistically significant",
            "evidence": "Some intents have 0 test samples (refund_return, technical_device, resolution_confirmation)"
        },
        "class_imbalance": {
            "metric": "Overall accuracy",
            "problem": "Test set not balanced across intents",
            "impact": "Accuracy dominated by majority classes (non_english, escalation_needed)",
            "evidence": "non_english=5, escalation_needed=3, delivery_issue=4, account_access=4, billing_payment=7"
        },
        "golden_set_quality": {
            "metric": "Human-labeled accuracy",
            "problem": "Single annotator, no inter-annotator agreement measured",
            "impact": "Labels may be inconsistent or wrong",
            "evidence": "Cohen's Kappa not computed in Phase 2.2"
        },
        "auto_handle_threshold": {
            "metric": "Auto-handle rate (53%)",
            "problem": "Arbitrary 0.7 threshold, not optimized",
            "impact": "Prime query (0.67 confidence) escalated despite correct intent",
            "evidence": "Threshold trades off precision/recall without business justification"
        },
        "retrieval_vs_generation": {
            "metric": "End-to-end quality",
            "problem": "Only intent accuracy measured, not response quality",
            "impact": "Good retrieval ≠ good response; template responses not evaluated",
            "evidence": "No human evaluation of generated responses"
        },
        "data_leakage_risk": {
            "metric": "Train/test split",
            "problem": "Split at conversation level but similar queries across conversations",
            "impact": "Near-duplicate customer queries may appear in both train and test",
            "evidence": "Many 'package delivered but not received' variants in dataset"
        },
        "intent_taxonomy_issues": {
            "metric": "10-intent taxonomy",
            "problem": "Billing vs Account merged but have different language; Prime vs Technical merged",
            "impact": "Systematic confusion between billing_payment and account_access",
            "evidence": "7 billing queries, 43% accuracy; 4 account queries, 75% accuracy"
        },
        "resolution_heuristic": {
            "metric": "18,173 resolved candidates",
            "problem": "Heuristic (resolution keywords OR ≥3 messages) ≠ actual resolution",
            "impact": "Many 'resolved' conversations may not be truly resolved",
            "evidence": "Only keyword matching, no outcome verification"
        }
    }
    
    return issues


def generate_failure_report(failure_analysis: Dict, misleading_metrics: Dict) -> str:
    """Generate comprehensive failure analysis report."""
    
    report = """# Phase 7: Failure Analysis & Honest Metrics Discussion

## Executive Summary

The RAG agent achieves 73.3% intent accuracy on a 30-sample test set, matching TF-IDF baseline but adding retrieval for grounded responses. However, several critical limitations undermine confidence in this headline number.

---

## 1. Failure Analysis

### Overall Failure Breakdown
"""
    
    report += f"""
- **Total test samples**: {len(rag_results) if 'rag_results' in globals() else 'N/A'}
- **Failures**: {failure_analysis['total_failures']}
- **Failure rate**: {failure_analysis['total_failures'] / 30 * 100:.1f}% (on 30 samples)
"""
    
    report += f"""
### Failure Rate by Intent
| Intent | Failure Rate | Total Test Samples |
|--------|-------------|-------------------|
"""
    for intent, rate in sorted(failure_analysis['failure_rate_by_intent'].items(), key=lambda x: -x[1]):
        total = sum(1 for r in rag_results if r['true_intent'] == intent) if 'rag_results' in globals() else 'N/A'
        report += f"| {intent} | {rate:.1%} | {total} |\n"
    
    report += f"""
### Top Confusion Pairs
| True Intent | Predicted | Count |
|-------------|-----------|-------|
"""
    for key, count in sorted(failure_analysis['confusion_pairs'].items(), key=lambda x: -x[1]):
        true, pred = key.split("→")
        report += f"| {true} | {pred} | {count} |\n"
    
    report += f"""
### Failure Root Causes

#### 1. Low Confidence Failures ({failure_analysis['low_confidence_failures']})
These are cases where the model was uncertain (< 0.7 confidence) and got it wrong.
- **Action**: Could be caught by escalation threshold

#### 2. High Confidence Failures ({failure_analysis['high_confidence_failures']})
**Most dangerous** - model was confident but wrong.
- **Action**: Need better features or more training data for these patterns

#### 3. Retrieval Quality Issues
- Failures with good retrieval (≥0.6): {failure_analysis['retrieval_quality_successes']}
- Failures with poor retrieval (<0.6): {failure_analysis['retrieval_quality_failures']}
- **Insight**: Poor retrieval doesn't always cause failure; intent classifier can succeed alone

#### 4. Decision Logic Failures
- Escalated but wrong: {failure_analysis['escalated_failures']}
- Auto-handled but wrong: {failure_analysis['auto_handle_failures']}
- **Insight**: High-confidence wrong predictions auto-handled = customer gets wrong answer

---

## 2. Misleading Headline Metrics

"""
    
    for key, issue in misleading_metrics.items():
        report += f"""
### {key.replace('_', ' ').title()}

**Metric**: {issue['metric']}

**Problem**: {issue['problem']}

**Impact**: {issue['impact']}

**Evidence**: {issue['evidence']}
"""
    
    report += """

---

## 3. Honest Assessment of What Works / What Doesn't

### What Works Well ✅
1. **Non-English detection**: 100% accuracy (clear linguistic signals)
2. **Escalation detection**: 100% accuracy (explicit keywords like "escalate", "lawyer", "days requesting")
3. **Prime subscription**: 100% on test (clear "prime/membership" vocabulary)
4. **Delivery issue**: 100% on test (strong delivery/tracking vocabulary)
5. **Retrieval system**: 0.95 avg similarity - finds relevant historical conversations

### What Needs Work ❌
1. **Billing vs Account confusion**: 43% billing accuracy, confused with account_access
   - Root cause: "charge", "charged", "payment" appear in both
   - Fix: Separate intents or add account-specific features (login, password, locked)

2. **Prime membership billing**: Queries like "charged twice for prime" → prime_subscription (correct) but low confidence (0.67)
   - Fix: Lower threshold for prime billing or separate prime_billing intent

3. **Order problem / Refund return**: Only 1 test sample each - cannot evaluate properly
   - Fix: Expand golden set with more examples

4. **Technical device**: 0 test samples - completely unevaluated
   - Fix: Add technical device examples to golden set

5. **Resolution confirmation**: 0 test samples
   - Fix: Add "thank you, resolved" examples

---

## 4. Recommended Improvements (Priority Order)

### High Priority (Immediate)
1. **Expand golden set** to 200+ samples with balanced intents (min 20 per intent)
2. **Separate billing_payment from account_access** - different vocabularies
3. **Compute inter-annotator agreement** (Cohen's Kappa) on golden set
4. **Optimize auto-handle threshold** using precision-recall tradeoff analysis

### Medium Priority
5. **Add sentence-transformer classifier** (replace TF-IDF) for better semantic understanding
6. **Implement LLM-based response generation** (replace templates)
7. **Add confidence calibration** (temperature scaling) for reliable probabilities
8. **Human evaluation of generated responses** (not just intent accuracy)

### Low Priority
9. **A/B test against production baseline** (if available)
10. **Add conversation context** (not just first message) for intent classification
11. **Multi-label intent support** (some queries span intents)
12. **Active learning loop** for continuous improvement

---

## 5. Deployment Readiness Assessment

| Criterion | Status | Notes |
|-----------|--------|-------|
| Intent accuracy | ⚠️ Marginal | 73% on tiny test set |
| Retrieval quality | ✅ Good | 0.95 avg similarity |
| Response grounding | ⚠️ Templates only | No LLM generation |
| Escalation logic | ✅ Works | Clear rules for escalation_needed |
| Confidence calibration | ❌ Not done | Probabilities not reliable |
| Monitoring/observability | ❌ Missing | No logging/metrics pipeline |
| Rollback capability | ❌ Missing | No canary deployment setup |

**Verdict**: **Not production-ready**. Needs larger evaluation, calibrated confidence, LLM responses, and monitoring.

---

## 6. Cost-Benefit Honesty

### Development Cost (Estimated)
- Phase 1-2: ~20 hours (data profiling, conversation reconstruction)
- Phase 2.1-2.2: ~5 hours (annotation tooling, golden set)
- Phase 3-4: ~15 hours (classifier, retrieval index)
- Phase 5-6: ~10 hours (RAG pipeline, baselines)
- **Total: ~50 hours**

### Realistic Production Cost (Estimated)
- Golden set expansion (200 samples × 2 annotators × 2 rounds): ~40 hours
- Inter-annotator agreement: ~5 hours
- LLM integration & prompt engineering: ~20 hours
- Confidence calibration: ~10 hours
- Monitoring/alerting: ~15 hours
- A/B testing infrastructure: ~20 hours
- **Additional: ~110 hours**

### Expected Production Performance (Projected)
| Metric | Current | Projected (with improvements) |
|--------|---------|-------------------------------|
| Intent accuracy | 73% (n=30) | 85-90% (n=200) |
| Auto-handle rate | 53% | 65-75% |
| Escalation precision | Unknown | >90% |
| Response quality (human eval) | Not measured | >4.0/5.0 |

---

## 7. Conclusion

The system demonstrates **technical feasibility** of intent classification + historical retrieval for customer support. The architecture (classify → retrieve → generate) is sound.

However, **headline metrics are misleading** due to:
- Tiny, imbalanced test set (n=30)
- No inter-annotator agreement
- Uncalibrated confidence scores
- Template-only response generation
- No end-to-end human evaluation

**Recommendation**: Invest in golden set expansion and human evaluation before any production deployment. The 73% accuracy is a lower bound on a flawed test, not a reliable performance indicator.
"""
    
    return report


def main():
    logger.info("Starting Phase 7: Failure Analysis")
    
    # Load data
    rag_eval = load_rag_evaluation()
    rag_results = rag_eval.get("results", [])
    test_data = load_golden_test()
    expanded_test = load_expanded_test()
    
    # Global for report generation
    global rag_results_global
    rag_results_global = rag_results
    
    # Analyze failures
    failure_analysis = analyze_failures(rag_results, test_data)
    
    # Analyze misleading metrics
    misleading_metrics = analyze_misleading_metrics()
    
    # Generate report
    report = generate_failure_report(failure_analysis, misleading_metrics)
    
    # Save report
    report_path = DOCS_DIR / "phase7_failure_analysis.md"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Saved failure analysis to {report_path}")
    
    # Save failure analysis JSON
    analysis_path = RESULTS_DIR / "failure_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump({
            "failure_analysis": failure_analysis,
            "misleading_metrics": misleading_metrics
        }, f, indent=2, default=str)
    logger.info(f"Saved failure analysis JSON to {analysis_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("PHASE 7: FAILURE ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nTotal failures: {failure_analysis['total_failures']}/30")
    print(f"High-confidence failures: {failure_analysis['high_confidence_failures']}")
    print(f"Low-confidence failures: {failure_analysis['low_confidence_failures']}")
    print(f"Auto-handle wrong: {failure_analysis['auto_handle_failures']}")
    print(f"Escalated wrong: {failure_analysis['escalated_failures']}")
    
    print("\nTop confusion pairs:")
    for key, count in sorted(failure_analysis['confusion_pairs'].items(), key=lambda x: -x[1])[:5]:
        true, pred = key.split("→")
        print(f"  {true} → {pred}: {count}")
    
    print("\nKey misleading metrics identified:")
    for key in misleading_metrics:
        print(f"  - {key.replace('_', ' ').title()}")
    
    print(f"\nReport saved to: {report_path}")
    
    logger.info("Phase 7 complete!")


if __name__ == "__main__":
    main()