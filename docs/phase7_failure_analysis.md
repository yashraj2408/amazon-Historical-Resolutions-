# Phase 7: Failure Analysis & Honest Metrics Discussion

## Executive Summary

The RAG agent achieves 73.3% intent accuracy on a 30-sample test set, matching TF-IDF baseline but adding retrieval for grounded responses. However, several critical limitations undermine confidence in this headline number.

---

## 1. Failure Analysis

### Overall Failure Breakdown

- **Total test samples**: N/A
- **Failures**: 8
- **Failure rate**: 26.7% (on 30 samples)

### Failure Rate by Intent
| Intent | Failure Rate | Total Test Samples |
|--------|-------------|-------------------|
| billing_payment | 57.1% | N/A |
| account_access | 25.0% | N/A |
| delivery_issue | 0.0% | N/A |
| order_problem | 0.0% | N/A |
| refund_return | 0.0% | N/A |
| technical_device | 0.0% | N/A |
| prime_subscription | 0.0% | N/A |
| escalation_needed | 0.0% | N/A |
| non_english | 0.0% | N/A |
| resolution_confirmation | 0.0% | N/A |

### Top Confusion Pairs
| True Intent | Predicted | Count |
|-------------|-----------|-------|
| billing_payment | prime_subscription | 3 |
| billing_payment | technical_device | 1 |
| technical_app | technical_device | 1 |
| technical_app | delivery_issue | 1 |
| general_inquiry | billing_payment | 1 |
| account_access | billing_payment | 1 |

### Failure Root Causes

#### 1. Low Confidence Failures (6)
These are cases where the model was uncertain (< 0.7 confidence) and got it wrong.
- **Action**: Could be caught by escalation threshold

#### 2. High Confidence Failures (2)
**Most dangerous** - model was confident but wrong.
- **Action**: Need better features or more training data for these patterns

#### 3. Retrieval Quality Issues
- Failures with good retrieval (≥0.6): 7
- Failures with poor retrieval (<0.6): 1
- **Insight**: Poor retrieval doesn't always cause failure; intent classifier can succeed alone

#### 4. Decision Logic Failures
- Escalated but wrong: 6
- Auto-handled but wrong: 2
- **Insight**: High-confidence wrong predictions auto-handled = customer gets wrong answer

---

## 2. Misleading Headline Metrics


### Small Test Set

**Metric**: Test accuracy (73.3%)

**Problem**: Only 30 test samples, 10 intents = ~3 samples per intent

**Impact**: High variance, not statistically significant

**Evidence**: Some intents have 0 test samples (refund_return, technical_device, resolution_confirmation)

### Class Imbalance

**Metric**: Overall accuracy

**Problem**: Test set not balanced across intents

**Impact**: Accuracy dominated by majority classes (non_english, escalation_needed)

**Evidence**: non_english=5, escalation_needed=3, delivery_issue=4, account_access=4, billing_payment=7

### Golden Set Quality

**Metric**: Human-labeled accuracy

**Problem**: Single annotator, no inter-annotator agreement measured

**Impact**: Labels may be inconsistent or wrong

**Evidence**: Cohen's Kappa not computed in Phase 2.2

### Auto Handle Threshold

**Metric**: Auto-handle rate (53%)

**Problem**: Arbitrary 0.7 threshold, not optimized

**Impact**: Prime query (0.67 confidence) escalated despite correct intent

**Evidence**: Threshold trades off precision/recall without business justification

### Retrieval Vs Generation

**Metric**: End-to-end quality

**Problem**: Only intent accuracy measured, not response quality

**Impact**: Good retrieval ≠ good response; template responses not evaluated

**Evidence**: No human evaluation of generated responses

### Data Leakage Risk

**Metric**: Train/test split

**Problem**: Split at conversation level but similar queries across conversations

**Impact**: Near-duplicate customer queries may appear in both train and test

**Evidence**: Many 'package delivered but not received' variants in dataset

### Intent Taxonomy Issues

**Metric**: 10-intent taxonomy

**Problem**: Billing vs Account merged but have different language; Prime vs Technical merged

**Impact**: Systematic confusion between billing_payment and account_access

**Evidence**: 7 billing queries, 43% accuracy; 4 account queries, 75% accuracy

### Resolution Heuristic

**Metric**: 18,173 resolved candidates

**Problem**: Heuristic (resolution keywords OR ≥3 messages) ≠ actual resolution

**Impact**: Many 'resolved' conversations may not be truly resolved

**Evidence**: Only keyword matching, no outcome verification


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
