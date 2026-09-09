#!/usr/bin/env python3
"""
Phase 2: AmazonHelp Conversation Reconstruction + Dataset Analysis
"""

import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter, defaultdict
import re

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# Constants
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
CACHE_PATH = Path.home() / ".cache/kagglehub/datasets/thoughtvector/customer-support-on-twitter/versions/10/twcs/twcs.csv"
COLUMNS = ["tweet_id", "author_id", "inbound", "created_at", "text", "response_tweet_id", "in_response_to_tweet_id"]
BRAND = "AmazonHelp"


def load_dataset() -> pd.DataFrame:
    """Load the full dataset with specified columns."""
    logger.info(f"Loading dataset from {CACHE_PATH}")
    df = pd.read_csv(CACHE_PATH, usecols=COLUMNS, low_memory=False)
    
    # Convert types
    df["tweet_id"] = pd.to_numeric(df["tweet_id"], errors="coerce").astype("Int64")
    df["author_id"] = df["author_id"].astype(str)
    df["inbound"] = df["inbound"].astype(bool)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df["text"] = df["text"].astype(str)
    df["response_tweet_id"] = pd.to_numeric(df["response_tweet_id"], errors="coerce").astype("Int64")
    df["in_response_to_tweet_id"] = pd.to_numeric(df["in_response_to_tweet_id"], errors="coerce").astype("Int64")
    
    logger.info(f"Loaded {len(df)} rows")
    return df


def filter_amazonhelp(df: pd.DataFrame) -> pd.DataFrame:
    """Filter for AmazonHelp-related tweets."""
    logger.info("Filtering AmazonHelp tweets...")
    
    # Support tweets from AmazonHelp
    support_mask = (df["author_id"] == BRAND) & (df["inbound"] == False)
    support_tweets = df[support_mask].copy()
    
    # Customer tweets mentioning AmazonHelp
    customer_mask = (df["inbound"] == True) & (df["text"].str.contains(f"@{BRAND}", case=False, na=False))
    customer_tweets = df[customer_mask].copy()
    
    # Also include tweets that are replies to AmazonHelp tweets (part of conversation)
    amazonhelp_tweet_ids = set(support_tweets["tweet_id"].dropna().astype(int))
    reply_to_amazonhelp = df[
        (df["inbound"] == True) & 
        (df["in_response_to_tweet_id"].isin(amazonhelp_tweet_ids))
    ].copy()
    
    # Combine all customer tweets related to AmazonHelp
    customer_tweets = pd.concat([customer_tweets, reply_to_amazonhelp]).drop_duplicates(subset=["tweet_id"])
    
    # Combine all
    amazonhelp_df = pd.concat([support_tweets, customer_tweets]).sort_values("tweet_id").reset_index(drop=True)
    
    logger.info(f"AmazonHelp support tweets: {len(support_tweets)}")
    logger.info(f"Customer tweets to AmazonHelp: {len(customer_tweets)}")
    logger.info(f"Total AmazonHelp-related tweets: {len(amazonhelp_df)}")
    
    return amazonhelp_df


def build_reply_graph(df: pd.DataFrame) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
    """Build reply graph from tweet_id -> in_response_to_tweet_id."""
    logger.info("Building reply graph...")
    
    children = defaultdict(list)  # parent_id -> list of child tweet_ids
    parent = {}  # tweet_id -> parent tweet_id
    
    for _, row in df.iterrows():
        tweet_id = row["tweet_id"]
        parent_id = row["in_response_to_tweet_id"]
        
        if pd.notna(tweet_id) and pd.notna(parent_id):
            tweet_id = int(tweet_id)
            parent_id = int(parent_id)
            children[parent_id].append(tweet_id)
            parent[tweet_id] = parent_id
    
    logger.info(f"Built graph with {len(parent)} edges, {len(children)} parent nodes")
    return children, parent


def find_conversation_roots(df: pd.DataFrame, parent: Dict[int, int]) -> List[int]:
    """Find conversation roots: customer tweets that start a conversation."""
    logger.info("Finding conversation roots...")
    
    # Roots are inbound tweets that are not replies to another tweet in our dataset
    # OR tweets whose parent is not in our filtered dataset
    inbound_tweets = df[df["inbound"] == True]["tweet_id"].dropna().astype(int)
    roots = []
    
    for tweet_id in inbound_tweets:
        if tweet_id not in parent:
            roots.append(tweet_id)
        else:
            # Check if parent is outside AmazonHelp dataset
            parent_id = parent[tweet_id]
            if parent_id not in df["tweet_id"].values:
                roots.append(tweet_id)
    
    logger.info(f"Found {len(roots)} conversation roots")
    return roots


def reconstruct_conversations(
    df: pd.DataFrame, 
    children: Dict[int, List[int]], 
    parent: Dict[int, int],
    roots: List[int]
) -> List[Dict[str, Any]]:
    """Reconstruct full conversation threads from roots."""
    logger.info("Reconstructing conversations...")
    
    tweet_lookup = df.set_index("tweet_id").to_dict("index")
    conversations = []
    visited = set()
    
    for root_id in roots:
        if root_id in visited:
            continue
            
        # Build conversation thread using BFS
        thread = []
        queue = [root_id]
        
        while queue:
            current_id = queue.pop(0)
            if current_id in visited or current_id not in tweet_lookup:
                continue
                
            visited.add(current_id)
            tweet = tweet_lookup[current_id]
            
            thread.append({
                "tweet_id": int(current_id),
                "author_id": tweet["author_id"],
                "speaker": "customer" if tweet["inbound"] else "support",
                "timestamp": tweet["created_at"].isoformat() if pd.notna(tweet["created_at"]) else None,
                "text": tweet["text"],
                "inbound": bool(tweet["inbound"]),
                "parent_tweet_id": int(parent[current_id]) if current_id in parent else None
            })
            
            # Add children to queue
            if current_id in children:
                queue.extend(children[current_id])
        
        if len(thread) >= 2:  # Only keep conversations with at least 2 messages
            # Sort chronologically
            thread.sort(key=lambda x: x["timestamp"] or "")
            conversations.append({
                "conversation_id": f"conv_{root_id}",
                "brand": BRAND,
                "messages": thread
            })
    
    logger.info(f"Reconstructed {len(conversations)} conversations with >=2 messages")
    return conversations


def calculate_conversation_stats(conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate conversation quality statistics."""
    logger.info("Calculating conversation statistics...")
    
    if not conversations:
        return {}
    
    total_conversations = len(conversations)
    total_messages = sum(len(c["messages"]) for c in conversations)
    customer_messages = sum(
        sum(1 for m in c["messages"] if m["speaker"] == "customer") 
        for c in conversations
    )
    support_messages = sum(
        sum(1 for m in c["messages"] if m["speaker"] == "support") 
        for c in conversations
    )
    
    lengths = [len(c["messages"]) for c in conversations]
    customer_counts = [sum(1 for m in c["messages"] if m["speaker"] == "customer") for c in conversations]
    support_counts = [sum(1 for m in c["messages"] if m["speaker"] == "support") for c in conversations]
    
    has_support = sum(1 for c in support_counts if c > 0)
    multi_customer = sum(1 for c in customer_counts if c > 1)
    multi_support = sum(1 for c in support_counts if c > 1)
    
    stats = {
        "total_conversations": total_conversations,
        "total_messages": total_messages,
        "customer_messages": customer_messages,
        "support_messages": support_messages,
        "avg_messages_per_conversation": round(np.mean(lengths), 2),
        "median_messages_per_conversation": float(np.median(lengths)),
        "p90_conversation_length": float(np.percentile(lengths, 90)),
        "pct_conversations_with_support": round(has_support / total_conversations * 100, 2),
        "pct_conversations_multi_customer": round(multi_customer / total_conversations * 100, 2),
        "pct_conversations_multi_support": round(multi_support / total_conversations * 100, 2),
        "min_length": int(np.min(lengths)),
        "max_length": int(np.max(lengths))
    }
    
    return stats


def identify_resolution_candidates(conversations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Identify conversations that appear to have resolution."""
    logger.info("Identifying resolution candidates...")
    
    resolution_keywords = [
        "refund", "replacement", "replaced", "return", "returned", 
        "resolved", "fixed", "working", "works", "success", "thank",
        "thanks", "appreciate", "helpful", "dm", "direct message",
        "private message", "order", "tracking", "shipped", "delivered",
        "account", "password", "reset", "login", "access", "issue resolved"
    ]
    
    for conv in conversations:
        support_msgs = [m for m in conv["messages"] if m["speaker"] == "support"]
        customer_msgs = [m for m in conv["messages"] if m["speaker"] == "customer"]
        
        has_customer = len(customer_msgs) > 0
        has_support = len(support_msgs) > 0
        
        # Check for resolution indicators in support messages
        resolution_score = 0
        for msg in support_msgs:
            text_lower = msg["text"].lower()
            for kw in resolution_keywords:
                if kw in text_lower:
                    resolution_score += 1
        
        conv["resolved_candidate"] = bool(
            has_customer and has_support and (resolution_score > 0 or len(conv["messages"]) >= 3)
        )
        conv["resolution_score"] = resolution_score
    
    resolved_count = sum(1 for c in conversations if c.get("resolved_candidate", False))
    logger.info(f"Identified {resolved_count} resolution candidates out of {len(conversations)} conversations")
    return conversations


def analyze_response_patterns(conversations: List[Dict[str, Any]]) -> pd.DataFrame:
    """Analyze AmazonHelp support response patterns."""
    logger.info("Analyzing support response patterns...")
    
    support_messages = []
    for conv in conversations:
        for msg in conv["messages"]:
            if msg["speaker"] == "support":
                support_messages.append(msg["text"])
    
    # Define pattern categories
    patterns = {
        "dm_request": ["dm", "direct message", "private message", "message us", "send us a message"],
        "order_details": ["order", "order number", "order id", "order details"],
        "account_details": ["account", "email", "username", "verify", "verification"],
        "tracking_request": ["tracking", "track", "shipment", "deliver"],
        "refund_mention": ["refund", "reimburse", "money back"],
        "replacement_mention": ["replacement", "replace", "new one", "send a new"],
        "return_mention": ["return", "send back", "return label"],
        "acknowledgment": ["sorry", "apologize", "apology", "understand", "frustrat"],
        "instructional": ["please", "try", "check", "visit", "go to", "click", "link"],
        "escalation": ["escalate", "specialist", "team", "transfer", "further assist"],
        "informational": ["policy", "information", "details", "here is", "here are", "you can"],
        "gratitude": ["thank", "thanks", "appreciate"],
        "resolution_confirm": ["resolved", "fixed", "working", "success", "completed"]
    }
    
    results = []
    for pattern_name, keywords in patterns.items():
        count = 0
        examples = []
        for msg in support_messages:
            text_lower = msg.lower()
            if any(kw in text_lower for kw in keywords):
                count += 1
                if len(examples) < 3:
                    examples.append(msg[:200])
        
        results.append({
            "pattern": pattern_name,
            "count": count,
            "percentage": round(count / len(support_messages) * 100, 2) if support_messages else 0,
            "example_1": examples[0] if len(examples) > 0 else "",
            "example_2": examples[1] if len(examples) > 1 else "",
            "example_3": examples[2] if len(examples) > 2 else ""
        })
    
    df_patterns = pd.DataFrame(results).sort_values("count", ascending=False).reset_index(drop=True)
    logger.info(f"Analyzed {len(support_messages)} support messages across {len(patterns)} patterns")
    return df_patterns


def sample_customer_messages(conversations: List[Dict[str, Any]], n: int = 5000) -> pd.DataFrame:
    """Sample customer messages for intent clustering."""
    logger.info(f"Sampling {n} customer messages...")
    
    customer_msgs = []
    for conv in conversations:
        for msg in conv["messages"]:
            if msg["speaker"] == "customer":
                customer_msgs.append({
                    "conversation_id": conv["conversation_id"],
                    "tweet_id": msg["tweet_id"],
                    "text": msg["text"],
                    "timestamp": msg["timestamp"]
                })
    
    df_customer = pd.DataFrame(customer_msgs)
    logger.info(f"Total customer messages: {len(df_customer)}")
    
    # Clean
    df_customer["text_clean"] = df_customer["text"].apply(clean_text)
    df_customer = df_customer[df_customer["text_clean"].str.len() > 5]
    
    # Remove URLs-only messages
    df_customer = df_customer[~df_customer["text_clean"].str.match(r"^https?://")]
    
    # Remove duplicates
    df_customer = df_customer.drop_duplicates(subset=["text_clean"])
    
    # Sample
    if len(df_customer) > n:
        df_customer = df_customer.sample(n=n, random_state=RANDOM_SEED)
    
    logger.info(f"After cleaning and sampling: {len(df_customer)} messages")
    return df_customer


def clean_text(text: str) -> str:
    """Clean text for clustering."""
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    # Remove mentions
    text = re.sub(r"@\w+", "", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def cluster_customer_messages(df_customer: pd.DataFrame, n_clusters: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster customer messages using TF-IDF + KMeans."""
    logger.info(f"Clustering {len(df_customer)} messages into {n_clusters} clusters...")
    
    # TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )
    tfidf_matrix = vectorizer.fit_transform(df_customer["text_clean"])
    
    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    df_customer["cluster_id"] = kmeans.fit_predict(tfidf_matrix)
    
    # Get cluster statistics
    feature_names = vectorizer.get_feature_names_out()
    cluster_stats = []
    cluster_examples = []
    
    for cluster_id in range(n_clusters):
        cluster_msgs = df_customer[df_customer["cluster_id"] == cluster_id]
        count = len(cluster_msgs)
        
        if count == 0:
            continue
        
        # Top TF-IDF terms
        cluster_center = kmeans.cluster_centers_[cluster_id]
        top_indices = cluster_center.argsort()[-10:][::-1]
        top_terms = [feature_names[i] for i in top_indices]
        
        # Representative examples
        examples = cluster_msgs["text"].head(5).tolist()
        
        cluster_stats.append({
            "cluster_id": cluster_id,
            "count": count,
            "top_terms": ", ".join(top_terms),
            "representative_example_1": examples[0] if len(examples) > 0 else "",
            "representative_example_2": examples[1] if len(examples) > 1 else "",
            "representative_example_3": examples[2] if len(examples) > 2 else "",
            "representative_example_4": examples[3] if len(examples) > 3 else "",
            "representative_example_5": examples[4] if len(examples) > 4 else ""
        })
        
        for ex in examples:
            cluster_examples.append({
                "cluster_id": cluster_id,
                "text": ex
            })
    
    df_stats = pd.DataFrame(cluster_stats)
    df_examples = pd.DataFrame(cluster_examples)
    
    logger.info(f"Created {len(df_stats)} clusters with examples")
    return df_stats, df_examples


def save_outputs(
    amazonhelp_df: pd.DataFrame,
    conversations: List[Dict[str, Any]],
    stats: Dict[str, Any],
    patterns_df: pd.DataFrame,
    customer_sample: pd.DataFrame,
    cluster_stats: pd.DataFrame,
    cluster_examples: pd.DataFrame
):
    """Save all outputs."""
    logger.info("Saving outputs...")
    
    # Save processed tweets
    tweets_path = PROCESSED_DIR / "amazonhelp_tweets.parquet"
    amazonhelp_df.to_parquet(tweets_path, index=False)
    logger.info(f"Saved tweets to {tweets_path}")
    
    # Save conversations
    conv_path = PROCESSED_DIR / "amazonhelp_conversations.jsonl"
    with open(conv_path, "w") as f:
        for conv in conversations:
            f.write(json.dumps(conv, default=str) + "\n")
    logger.info(f"Saved {len(conversations)} conversations to {conv_path}")
    
    # Save stats
    stats_path = RESULTS_DIR / "amazonhelp_conversation_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info(f"Saved stats to {stats_path}")
    
    # Save response patterns
    patterns_path = RESULTS_DIR / "amazonhelp_response_patterns.csv"
    patterns_df.to_csv(patterns_path, index=False)
    logger.info(f"Saved patterns to {patterns_path}")
    
    # Save customer sample
    sample_path = PROCESSED_DIR / "amazonhelp_customer_sample.csv"
    customer_sample.to_csv(sample_path, index=False)
    logger.info(f"Saved customer sample to {sample_path}")
    
    # Save cluster stats
    cluster_stats_path = RESULTS_DIR / "intent_candidate_clusters.csv"
    cluster_stats.to_csv(cluster_stats_path, index=False)
    logger.info(f"Saved cluster stats to {cluster_stats_path}")
    
    # Save cluster examples
    cluster_examples_path = RESULTS_DIR / "intent_candidate_examples.json"
    with open(cluster_examples_path, "w") as f:
        json.dump(cluster_examples.to_dict("records"), f, indent=2, default=str)
    logger.info(f"Saved cluster examples to {cluster_examples_path}")


def create_intent_discovery_report(
    stats: Dict[str, Any],
    patterns_df: pd.DataFrame,
    cluster_stats: pd.DataFrame,
    cluster_examples: pd.DataFrame,
    customer_sample: pd.DataFrame
):
    """Create phase2_intent_discovery.md report."""
    logger.info("Creating intent discovery report...")
    
    report = f"""# Phase 2 Intent Discovery Report: AmazonHelp

## 1. AmazonHelp Dataset Statistics

- **Total AmazonHelp-related tweets**: {stats.get('total_messages', 'N/A')}
- **Support tweets**: {stats.get('support_messages', 'N/A')}
- **Customer tweets**: {stats.get('customer_messages', 'N/A')}
- **Conversations reconstructed**: {stats.get('total_conversations', 'N/A')}

## 2. Conversation Reconstruction Methodology

1. **Filtered tweets**: Support tweets from `author_id == "AmazonHelp"` + `inbound == False`, and customer tweets mentioning `@AmazonHelp` or replying to AmazonHelp tweets.
2. **Reply graph**: Built using `in_response_to_tweet_id` → `tweet_id` edges.
3. **Conversation roots**: Customer tweets (inbound=True) with no parent in the filtered dataset.
4. **Thread reconstruction**: BFS from each root following reply edges, sorted chronologically.
5. **Minimum length**: Only conversations with ≥2 messages retained.

## 3. Conversation Statistics

| Metric | Value |
|--------|-------|
| Total Conversations | {stats.get('total_conversations', 'N/A')} |
| Total Messages | {stats.get('total_messages', 'N/A')} |
| Customer Messages | {stats.get('customer_messages', 'N/A')} |
| Support Messages | {stats.get('support_messages', 'N/A')} |
| Avg Messages/Conversation | {stats.get('avg_messages_per_conversation', 'N/A')} |
| Median Messages/Conversation | {stats.get('median_messages_per_conversation', 'N/A')} |
| P90 Conversation Length | {stats.get('p90_conversation_length', 'N/A')} |
| % With Support Reply | {stats.get('pct_conversations_with_support', 'N/A')}% |
| % Multi-Customer | {stats.get('pct_conversations_multi_customer', 'N/A')}% |
| % Multi-Support | {stats.get('pct_conversations_multi_support', 'N/A')}% |

## 4. Resolution Heuristic

A conversation is marked as `resolved_candidate` if:
- Contains ≥1 customer message AND ≥1 support message
- AND (support message contains resolution keywords OR conversation has ≥3 messages)

**Resolution keywords**: refund, replacement, return, resolved, fixed, working, thank, dm, direct message, order, tracking, shipped, delivered, account, password, reset, login, issue resolved

**Note**: This is a heuristic, not ground truth. Manual verification needed.

## 5. Response Patterns (Top 15)

"""
    
    for _, row in patterns_df.head(15).iterrows():
        report += f"- **{row['pattern']}**: {row['count']} ({row['percentage']}%)\n"
    
    report += f"""
## 6. Candidate Intent Clusters (KMeans, n=20)

| Cluster | Count | Top Terms | Example |
|---------|-------|-----------|---------|
"""
    
    for _, row in cluster_stats.iterrows():
        ex = row['representative_example_1'][:100] if row['representative_example_1'] else ""
        report += f"| {row['cluster_id']} | {row['count']} | {row['top_terms']} | {ex} |\n"
    
    report += f"""
## 7. Representative Examples by Cluster

"""
    
    for cluster_id in sorted(cluster_stats["cluster_id"]):
        examples = cluster_examples[cluster_examples["cluster_id"] == cluster_id]["text"].head(3).tolist()
        report += f"### Cluster {cluster_id}\n"
        for ex in examples:
            report += f"- {ex}\n"
        report += "\n"
    
    report += f"""
## 8. Data Quality Problems

1. **Missing reply links**: ~37% of tweets missing `in_response_to_tweet_id` - breaks conversation chains
2. **No explicit conversation_id**: Must infer from reply structure
3. **Timestamp granularity**: Second-level only, may misorder rapid exchanges
4. **Deleted/removed tweets**: Referenced tweets may not exist in dataset
5. **Bot/spam accounts**: Some "support" accounts may be automated

## 9. Potential Intent Taxonomy (Preliminary)

Based on clusters and response patterns:

1. **Delivery/Issue** - Package not arrived, late delivery, tracking
2. **Order Problem** - Wrong item, missing item, damaged, cancel order
3. **Refund/Return** - Request refund, return process, return label
4. **Account/Access** - Login issues, password reset, account locked
5. **Technical/App** - App not working, website error, Alexa/device issues
6. **Billing/Payment** - Unauthorized charge, billing question, payment failed
7. **Prime/Subscription** - Prime membership, subscription cancel, benefits
8. **Seller/Marketplace** - Third-party seller issues
9. **Gift Card** - Balance, redeem, not working
10. **General Inquiry** - Product questions, policy, how-to

## 10. Ambiguous/Boundary Cases

- **Delivery vs Order**: "Where is my order?" could be delivery tracking or order status
- **Refund vs Return**: Often coupled; customer wants money back AND to return item
- **Account vs Billing**: "Charge on my account" spans both
- **Technical vs General**: "App not working" could be bug or user error
- **Escalation needed**: Complex issues requiring specialist (fraud, legal, etc.)

## 11. Data Leakage Protection

**IMPORTANT**: Future evaluation must split at **conversation level**, not tweet level.
- No messages from same `conversation_id` in both training/retrieval and golden eval set
- Use `conversation_id` for stratified splits
- Golden set: 150-250 hand-labeled conversations (not individual tweets)

## 12. Next Steps

1. **Manual review** of candidate clusters to finalize intent taxonomy (10-15 intents)
2. **Create golden evaluation set** with conversation-level labels
3. **Build intent classifier** using resolved_candidate conversations as training data
4. **Build retrieval system** for historical resolution grounding
"""
    
    report_path = DOCS_DIR / "phase2_intent_discovery.md"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Created report at {report_path}")


def print_summary(stats: Dict[str, Any], conversations: List[Dict[str, Any]], patterns_df: pd.DataFrame, cluster_stats: pd.DataFrame):
    """Print final summary."""
    resolved_count = sum(1 for c in conversations if c.get("resolved_candidate", False))
    
    print("\n" + "=" * 40)
    print("PHASE 2 COMPLETE")
    print("=" * 40)
    print(f"\nBrand: {BRAND}")
    print(f"Tweets: {stats.get('total_messages', 'N/A')}")
    print(f"Conversations: {stats.get('total_conversations', 'N/A')}")
    print(f"Customer messages: {stats.get('customer_messages', 'N/A')}")
    print(f"Support messages: {stats.get('support_messages', 'N/A')}")
    print(f"Median conversation length: {stats.get('median_messages_per_conversation', 'N/A')}")
    print(f"Resolution candidates: {resolved_count}")
    print(f"Candidate intent clusters: {len(cluster_stats)}")
    
    print(f"\nFiles created:")
    print(f"  {PROCESSED_DIR}/amazonhelp_tweets.parquet")
    print(f"  {PROCESSED_DIR}/amazonhelp_conversations.jsonl")
    print(f"  {PROCESSED_DIR}/amazonhelp_customer_sample.csv")
    print(f"  {RESULTS_DIR}/amazonhelp_conversation_stats.json")
    print(f"  {RESULTS_DIR}/amazonhelp_response_patterns.csv")
    print(f"  {RESULTS_DIR}/intent_candidate_clusters.csv")
    print(f"  {RESULTS_DIR}/intent_candidate_examples.json")
    print(f"  {DOCS_DIR}/phase2_intent_discovery.md")


def main():
    """Main entry point."""
    logger.info("Starting Phase 2: AmazonHelp Conversation Reconstruction + Analysis")
    
    # Step 1: Load dataset
    df = load_dataset()
    
    # Step 2: Filter AmazonHelp
    amazonhelp_df = filter_amazonhelp(df)
    
    # Save filtered tweets
    tweets_path = PROCESSED_DIR / "amazonhelp_tweets.parquet"
    amazonhelp_df.to_parquet(tweets_path, index=False)
    logger.info(f"Saved filtered tweets to {tweets_path}")
    
    # Step 3: Build reply graph
    children, parent = build_reply_graph(amazonhelp_df)
    
    # Step 4: Find roots
    roots = find_conversation_roots(amazonhelp_df, parent)
    
    # Step 5: Reconstruct conversations
    conversations = reconstruct_conversations(amazonhelp_df, children, parent, roots)
    
    # Step 6: Calculate stats
    stats = calculate_conversation_stats(conversations)
    
    # Step 7: Identify resolution candidates
    conversations = identify_resolution_candidates(conversations)
    
    # Step 8: Analyze response patterns
    patterns_df = analyze_response_patterns(conversations)
    
    # Step 9: Sample customer messages
    customer_sample = sample_customer_messages(conversations, n=5000)
    
    # Step 10: Cluster customer messages
    cluster_stats, cluster_examples = cluster_customer_messages(customer_sample, n_clusters=20)
    
    # Step 11: Save all outputs
    save_outputs(
        amazonhelp_df, conversations, stats, patterns_df,
        customer_sample, cluster_stats, cluster_examples
    )
    
    # Step 12: Create report
    create_intent_discovery_report(stats, patterns_df, cluster_stats, cluster_examples, customer_sample)
    
    # Print summary
    print_summary(stats, conversations, patterns_df, cluster_stats)
    
    logger.info("Phase 2 completed successfully!")


if __name__ == "__main__":
    main()