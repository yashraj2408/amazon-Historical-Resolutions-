# Annotation Guidelines: AmazonHelp Intent Taxonomy Validation

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
{
  "annotated_intent": "delivery_issue",
  "annotated_entities": {"order_id": "403-7503264", "product": "Echo Show"},
  "annotated_resolution": "yes",
  "annotator_notes": "Customer says delivered but not received. Support sent tracking link.",
  "annotator_id": "annotator_1"
}
```

## Deliverables
1. Annotated master file with all samples
2. Intent distribution per cluster (confusion matrix)
3. Revised taxonomy proposal
4. Golden set: 150-250 conversations with agreed labels
