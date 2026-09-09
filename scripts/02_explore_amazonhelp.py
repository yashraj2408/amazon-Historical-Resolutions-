#!/usr/bin/env python3
"""
Phase 2: Explore AmazonHelp conversations to define intents and understand patterns.
"""

import pandas as pd
import json
import logging
import sys
from pathlib import Path
from collections import Counter
import re

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def load_dataset():
    """Load the full dataset."""
    cache_path = Path.home() / ".cache/kagglehub/datasets/thoughtvector/customer-support-on-twitter/versions/10/twcs/twcs.csv"
    logger.info(f"Loading dataset from {cache_path}")
    df = pd.read_csv(cache_path, low_memory=False)
    logger.info(f"Loaded {len(df)} rows")
    return df

def filter_amazonhelp(df):
    """Filter for AmazonHelp conversations."""
    # Support tweets from AmazonHelp
    support_tweets = df[(df['author_id'] == 'AmazonHelp') & (df['inbound'] == False)]
    
    # Customer tweets mentioning AmazonHelp
    customer_tweets = df[(df['inbound'] == True) & (df['text'].str.contains('@AmazonHelp', case=False, na=False))]
    
    logger.info(f"AmazonHelp support tweets: {len(support_tweets)}")
    logger.info(f"Customer tweets to AmazonHelp: {len(customer_tweets)}")
    
    return support_tweets, customer_tweets

def analyze_conversation_patterns(support_tweets, customer_tweets):
    """Analyze conversation patterns to define intents."""
    
    # Sample customer tweets to understand intent categories
    sample_customer = customer_tweets.sample(n=min(500, len(customer_tweets)), random_state=42)
    
    print("\n=== SAMPLE CUSTOMER TWEETS ===")
    for _, row in sample_customer.head(20).iterrows():
        print(f"  @{row['author_id']}: {row['text'][:100]}")
    
    # Sample support replies
    sample_support = support_tweets.sample(n=min(200, len(support_tweets)), random_state=42)
    
    print("\n=== SAMPLE SUPPORT REPLIES ===")
    for _, row in sample_support.head(20).iterrows():
        print(f"  AmazonHelp: {row['text'][:100]}")
    
    # Analyze keywords in customer tweets
    all_customer_text = " ".join(customer_tweets['text'].astype(str).tolist())
    words = re.findall(r'\b\w+\b', all_customer_text.lower())
    word_freq = Counter(words)
    
    print("\n=== TOP KEYWORDS IN CUSTOMER TWEETS ===")
    for word, count in word_freq.most_common(50):
        if len(word) > 3:
            print(f"  {word}: {count}")
    
    # Look for intent-indicating phrases
    intent_patterns = {
        'delivery_issue': ['delivery', 'deliver', 'shipped', 'shipping', 'arrive', 'arrived', 'package', 'parcel', 'tracking', 'late', 'delay'],
        'order_issue': ['order', 'ordered', 'cancel', 'cancelled', 'refund', 'return', 'exchange', 'wrong', 'missing', 'item'],
        'account_issue': ['account', 'login', 'password', 'email', 'username', 'verify', 'verification', 'locked', 'hacked'],
        'billing_issue': ['charge', 'charged', 'billing', 'bill', 'payment', 'pay', 'refund', 'money', 'price', 'cost'],
        'technical_issue': ['app', 'website', 'site', 'error', 'bug', 'crash', 'not working', 'broken', 'issue', 'problem', 'glitch'],
        'prime_video': ['prime', 'video', 'movie', 'show', 'watch', 'stream', 'streaming'],
        'alexa_device': ['alexa', 'echo', 'device', 'kindle', 'fire', 'tablet', 'hardware'],
        'seller_issue': ['seller', 'third party', 'marketplace', 'vendor', 'merchant'],
        'subscription': ['subscribe', 'subscription', 'membership', 'prime', 'renew', 'cancel subscription'],
        'gift_card': ['gift card', 'giftcard', 'balance', 'redeem', 'code'],
    }
    
    print("\n=== INTENT KEYWORD MATCHES ===")
    for intent, keywords in intent_patterns.items():
        matches = sum(1 for kw in keywords if kw in all_customer_text.lower())
        print(f"  {intent}: {matches} keyword occurrences")
    
    return sample_customer, sample_support

def build_conversation_threads(df, support_tweets, customer_tweets):
    """Build conversation threads using reply structure."""
    # Create a mapping of tweet_id -> tweet for quick lookup
    tweet_map = df.set_index('tweet_id').to_dict('index')
    
    # Find conversation starters (customer tweets to AmazonHelp with no in_response_to)
    conv_starters = customer_tweets[customer_tweets['in_response_to_tweet_id'].isna()]
    logger.info(f"Conversation starters: {len(conv_starters)}")
    
    threads = []
    for _, starter in conv_starters.head(100).iterrows():
        thread = [starter]
        current_id = starter['tweet_id']
        
        # Follow the thread
        while True:
            replies = df[(df['in_response_to_tweet_id'] == current_id) & (df['author_id'] == 'AmazonHelp')]
            if len(replies) == 0:
                break
            reply = replies.iloc[0]
            thread.append(reply)
            current_id = reply['tweet_id']
            
            # Check for customer follow-up
            customer_replies = df[(df['in_response_to_tweet_id'] == current_id) & (df['inbound'] == True)]
            if len(customer_replies) > 0:
                thread.append(customer_replies.iloc[0])
                current_id = customer_replies.iloc[0]['tweet_id']
            else:
                break
        
        if len(thread) > 1:
            threads.append(thread)
    
    logger.info(f"Built {len(threads)} conversation threads")
    return threads

def analyze_threads(threads):
    """Analyze conversation threads for patterns."""
    print("\n=== SAMPLE CONVERSATION THREADS ===")
    for i, thread in enumerate(threads[:10]):
        print(f"\n--- Thread {i+1} (length: {len(thread)}) ---")
        for tweet in thread:
            role = "CUSTOMER" if tweet['inbound'] else "SUPPORT"
            print(f"  [{role}] {tweet['text'][:120]}")

def save_amazonhelp_data(support_tweets, customer_tweets, threads):
    """Save filtered AmazonHelp data for later use."""
    # Save support tweets
    support_path = DATA_DIR / "amazonhelp_support_tweets.csv"
    support_tweets.to_csv(support_path, index=False)
    logger.info(f"Saved support tweets to {support_path}")
    
    # Save customer tweets
    customer_path = DATA_DIR / "amazonhelp_customer_tweets.csv"
    customer_tweets.to_csv(customer_path, index=False)
    logger.info(f"Saved customer tweets to {customer_path}")
    
    # Save threads as JSON
    threads_data = []
    for thread in threads:
        threads_data.append([{
            'tweet_id': int(t['tweet_id']),
            'author_id': t['author_id'],
            'inbound': bool(t['inbound']),
            'created_at': t['created_at'],
            'text': t['text'],
            'in_response_to_tweet_id': float(t['in_response_to_tweet_id']) if pd.notna(t['in_response_to_tweet_id']) else None
        } for t in thread])
    
    threads_path = DATA_DIR / "amazonhelp_threads.json"
    with open(threads_path, 'w') as f:
        json.dump(threads_data, f, indent=2)
    logger.info(f"Saved {len(threads_data)} threads to {threads_path}")

def main():
    logger.info("Starting Phase 2: AmazonHelp Exploration")
    
    df = load_dataset()
    support_tweets, customer_tweets = filter_amazonhelp(df)
    
    sample_customer, sample_support = analyze_conversation_patterns(support_tweets, customer_tweets)
    threads = build_conversation_threads(df, support_tweets, customer_tweets)
    analyze_threads(threads)
    save_amazonhelp_data(support_tweets, customer_tweets, threads)
    
    logger.info("Phase 2 exploration complete!")

if __name__ == "__main__":
    main()