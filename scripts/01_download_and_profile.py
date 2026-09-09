#!/usr/bin/env python3
"""
Phase 1: Dataset Profiling and Brand Selection
Downloads and profiles the Kaggle customer-support-on-twitter dataset.
"""

import os
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import kagglehub

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
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)


def download_dataset() -> Path:
    """Download the Kaggle dataset and return the local path."""
    logger.info("Downloading dataset from Kaggle...")
    try:
        path = kagglehub.dataset_download("thoughtvector/customer-support-on-twitter")
        dataset_path = Path(path)
        logger.info(f"Dataset downloaded to: {dataset_path}")
        return dataset_path
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        raise


def discover_files(dataset_path: Path) -> List[Dict[str, Any]]:
    """Discover all files in the dataset directory."""
    logger.info(f"Discovering files in: {dataset_path}")
    files_info = []
    for file_path in dataset_path.rglob("*"):
        if file_path.is_file():
            stat = file_path.stat()
            files_info.append({
                "name": file_path.name,
                "path": str(file_path),
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "extension": file_path.suffix.lower()
            })
    logger.info(f"Found {len(files_info)} files")
    for f in files_info:
        logger.info(f"  - {f['name']} ({f['size_mb']} MB)")
    return files_info


def inspect_files(files_info: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Inspect CSV/TSV files to get row counts, columns, missing values, duplicates."""
    logger.info("Inspecting data files...")
    inspection_results = {}
    
    for file_info in files_info:
        if file_info["extension"] in [".csv", ".tsv", ".txt"]:
            path = file_info["path"]
            try:
                # Determine delimiter
                sep = "\t" if file_info["extension"] == ".tsv" else ","
                
                # Read first few rows to infer structure
                df_sample = pd.read_csv(path, sep=sep, nrows=5)
                
                # Read full file for stats (might be large, so we'll be careful)
                logger.info(f"Reading full file: {file_info['name']}...")
                df = pd.read_csv(path, sep=sep, low_memory=False)
                
                inspection_results[file_info["name"]] = {
                    "file_path": path,
                    "file_size_mb": file_info["size_mb"],
                    "rows": int(df.shape[0]),
                    "columns": list(df.columns),
                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                    "missing_values": df.isnull().sum().to_dict(),
                    "duplicate_rows": int(df.duplicated().sum()),
                    "sample_rows": df.head(5).to_dict(orient="records")
                }
                logger.info(f"  Rows: {df.shape[0]}, Columns: {df.shape[1]}")
                
            except Exception as e:
                logger.warning(f"Could not inspect {file_info['name']}: {e}")
                inspection_results[file_info["name"]] = {"error": str(e)}
    
    return inspection_results


def identify_main_tweet_file(inspection_results: Dict[str, Any]) -> Optional[str]:
    """Identify the main tweet data file based on column names and size."""
    tweet_keywords = ["tweet", "text", "author", "user", "id", "conversation", "in_reply_to"]
    
    best_match = None
    best_score = 0
    best_rows = 0
    
    for filename, info in inspection_results.items():
        if "error" in info:
            continue
        columns = [c.lower() for c in info.get("columns", [])]
        score = sum(1 for kw in tweet_keywords if any(kw in c for c in columns))
        rows = info.get("rows", 0)
        # Prefer higher score, then more rows (to pick main dataset over sample)
        if score > best_score or (score == best_score and rows > best_rows):
            best_score = score
            best_rows = rows
            best_match = filename
    
    return best_match


def load_tweets(main_file: str, inspection_results: Dict[str, Any]) -> pd.DataFrame:
    """Load the main tweet dataframe."""
    logger.info(f"Loading tweets from: {main_file}")
    info = inspection_results[main_file]
    path = info["file_path"]
    ext = Path(main_file).suffix.lower()
    sep = "\t" if ext == ".tsv" else ","
    
    df = pd.read_csv(path, sep=sep, low_memory=False)
    logger.info(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def profile_columns(df: pd.DataFrame) -> Dict[str, Any]:
    """Profile columns to infer their meanings."""
    logger.info("Profiling columns...")
    columns = df.columns.tolist()
    dtypes = {col: str(df[col].dtype) for col in columns}
    missing = df.isnull().sum().to_dict()
    unique_counts = {col: int(df[col].nunique()) for col in columns}
    
    # Infer column roles based on names and data
    inferred_roles = {}
    for col in columns:
        col_lower = col.lower()
        if any(kw in col_lower for kw in ["tweet_id", "id_str", "status_id"]) and "reply" not in col_lower:
            inferred_roles[col] = "tweet_id"
        elif any(kw in col_lower for kw in ["author", "user", "screen_name", "handle"]) and "in_reply" not in col_lower:
            inferred_roles[col] = "author/user"
        elif any(kw in col_lower for kw in ["text", "tweet_text", "content", "message"]):
            inferred_roles[col] = "tweet_text"
        elif any(kw in col_lower for kw in ["time", "date", "created", "timestamp"]):
            inferred_roles[col] = "timestamp"
        elif any(kw in col_lower for kw in ["reply_to", "in_reply_to", "parent"]):
            inferred_roles[col] = "reply_to_tweet_id"
        elif any(kw in col_lower for kw in ["conversation", "thread", "conversation_id"]):
            inferred_roles[col] = "conversation/thread"
        elif any(kw in col_lower for kw in ["brand", "company", "support", "account"]):
            inferred_roles[col] = "brand/support_account"
        else:
            inferred_roles[col] = "unknown"
    
    profile = {
        "columns": columns,
        "dtypes": dtypes,
        "missing_values": missing,
        "unique_counts": unique_counts,
        "inferred_roles": inferred_roles
    }
    
    logger.info("Column profiles:")
    for col in columns:
        logger.info(f"  {col}: dtype={dtypes[col]}, missing={missing[col]}, unique={unique_counts[col]}, role={inferred_roles[col]}")
    
    return profile


def analyze_brands(df: pd.DataFrame, column_profile: Dict[str, Any]) -> pd.DataFrame:
    """Analyze dataset by brand/support account."""
    logger.info("Analyzing brands...")
    
    # Find author column
    author_col = None
    for col, role in column_profile["inferred_roles"].items():
        if role == "author/user":
            author_col = col
            break
    
    # Find inbound column (True = customer->brand, False = brand->customer)
    inbound_col = "inbound" if "inbound" in df.columns else None
    
    if author_col is None:
        logger.warning("Could not identify author column")
        return pd.DataFrame()
    
    # Identify support accounts: authors who have outbound (inbound=False) tweets
    if inbound_col and inbound_col in df.columns:
        support_accounts = set(df[df[inbound_col] == False][author_col].unique())
        logger.info(f"Identified {len(support_accounts)} support accounts from inbound=False")
    else:
        # Fallback: accounts with support-like names
        support_keywords = ["support", "help", "care", "service", "assist"]
        support_accounts = set()
        for author in df[author_col].unique():
            author_lower = str(author).lower()
            if any(kw in author_lower for kw in support_keywords):
                support_accounts.add(author)
        logger.info(f"Identified {len(support_accounts)} support accounts from name keywords")
    
    # Pre-compute customer tweets mentioning each brand for efficiency
    if inbound_col and inbound_col in df.columns:
        customer_tweets = df[df[inbound_col] == True]
        # Build a mapping of brand -> count of customer tweets mentioning @brand
        brand_mentions = {}
        for brand in support_accounts:
            count = customer_tweets["text"].str.contains(f"@{brand}", case=False, na=False).sum()
            brand_mentions[brand] = int(count)
    else:
        brand_mentions = {brand: 0 for brand in support_accounts}
    
    results = []
    
    # Group by brand (support account)
    for brand in sorted(support_accounts):
        # Support tweets from this brand (outbound)
        support_tweets_df = df[(df[author_col] == brand) & (df[inbound_col] == False)] if inbound_col else df[df[author_col] == brand]
        support_tweets = len(support_tweets_df)
        
        # Customer tweets mentioning this brand
        customer_tweets = brand_mentions.get(brand, 0)
        
        total_tweets = support_tweets + customer_tweets
        
        # Conversations: each customer tweet mentioning the brand starts a conversation
        conversations = customer_tweets
        avg_conv_length = total_tweets / conversations if conversations > 0 else 0
        
        results.append({
            "brand": str(brand),
            "total_tweets": int(total_tweets),
            "customer_tweets": int(customer_tweets),
            "support_tweets": int(support_tweets),
            "conversations": int(conversations),
            "avg_conversation_length": round(avg_conv_length, 2)
        })
    
    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        results_df = results_df.sort_values("total_tweets", ascending=False).reset_index(drop=True)
    
    logger.info(f"Analyzed {len(results_df)} support brands")
    return results_df


def save_results(
    files_info: List[Dict],
    inspection_results: Dict,
    column_profile: Dict,
    brand_stats: pd.DataFrame
):
    """Save all results to JSON and CSV."""
    logger.info("Saving results...")
    
    # Dataset profile JSON
    profile = {
        "dataset_files": files_info,
        "file_inspection": inspection_results,
        "column_profile": column_profile,
        "brand_statistics_summary": {
            "total_brands": len(brand_stats),
            "top_brands": brand_stats.head(20).to_dict(orient="records")
        }
    }
    
    profile_path = RESULTS_DIR / "dataset_profile.json"
    with open(profile_path, "w") as f:
        json.dump(profile, f, indent=2, default=str)
    logger.info(f"Saved dataset profile to: {profile_path}")
    
    # Brand statistics CSV
    stats_path = RESULTS_DIR / "brand_statistics.csv"
    brand_stats.to_csv(stats_path, index=False)
    logger.info(f"Saved brand statistics to: {stats_path}")


def print_summary(
    files_info: List[Dict],
    inspection_results: Dict,
    main_file: str,
    df: pd.DataFrame,
    column_profile: Dict,
    brand_stats: pd.DataFrame
):
    """Print final summary."""
    main_info = inspection_results.get(main_file, {})
    
    print("\n" + "=" * 40)
    print("PHASE 1 COMPLETE")
    print("=" * 40)
    print(f"\nDataset: thoughtvector/customer-support-on-twitter")
    print(f"Main file: {main_file}")
    print(f"Rows: {main_info.get('rows', 'N/A')}")
    print(f"Columns: {main_info.get('columns', 'N/A')}")
    
    print("\nColumn roles inferred:")
    for col, role in column_profile["inferred_roles"].items():
        print(f"  {col}: {role}")
    
    print("\nTop candidate brands:")
    for i, row in brand_stats.head(20).iterrows():
        print(f"  {i+1}. {row['brand']} - {row['total_tweets']} tweets, {row['conversations']} conversations")
    
    print(f"\nResults saved to:")
    print(f"  {RESULTS_DIR}/brand_statistics.csv")
    print(f"  {RESULTS_DIR}/dataset_profile.json")


def create_phase1_notes(
    files_info: List[Dict],
    inspection_results: Dict,
    column_profile: Dict,
    brand_stats: pd.DataFrame,
    main_file: str
):
    """Create phase1_notes.md documentation."""
    logger.info("Creating phase1_notes.md...")
    
    main_info = inspection_results.get(main_file, {})
    
    notes = f"""# Phase 1 Notes: Dataset Profiling and Brand Selection

## Dataset Files Found

"""
    for f in files_info:
        notes += f"- **{f['name']}** ({f['size_mb']} MB, {f['extension']})\n"
    
    notes += f"""

## Main Data File: {main_file}

- **Rows**: {main_info.get('rows', 'N/A')}
- **Columns**: {len(main_info.get('columns', []))}

### Columns and Inferred Roles

| Column | Dtype | Missing | Unique | Inferred Role |
|--------|-------|---------|--------|---------------|
"""
    
    for col in main_info.get("columns", []):
        dtype = main_info.get("dtypes", {}).get(col, "unknown")
        missing = main_info.get("missing_values", {}).get(col, 0)
        unique = column_profile.get("unique_counts", {}).get(col, 0)
        role = column_profile.get("inferred_roles", {}).get(col, "unknown")
        notes += f"| {col} | {dtype} | {missing} | {unique} | {role} |\n"
    
    notes += f"""

## Brand/Support Account Identification

"""
    brand_col = None
    for col, role in column_profile.get("inferred_roles", {}).items():
        if role == "brand/support_account":
            brand_col = col
            break
    
    if brand_col:
        notes += f"Found explicit brand column: **{brand_col}**\n"
    else:
        author_col = None
        for col, role in column_profile.get("inferred_roles", {}).items():
            if role == "author/user":
                author_col = col
                break
        if author_col:
            notes += f"No explicit brand column found. Using author column: **{author_col}**\n"
            notes += "Brand accounts inferred from author names containing support/help/care/service keywords.\n"
        else:
            notes += "Could not identify brand or author column.\n"
    
    notes += f"""

## Dataset Quality Issues

"""
    total_missing = sum(main_info.get("missing_values", {}).values())
    total_duplicates = main_info.get("duplicate_rows", 0)
    notes += f"- **Missing values**: {total_missing} total across all columns\n"
    notes += f"- **Duplicate rows**: {total_duplicates}\n"
    
    for col, missing in main_info.get("missing_values", {}).items():
        if missing > 0:
            notes += f"  - {col}: {missing} missing\n"
    
    notes += f"""

## Assumptions Made

1. The main tweet data file was identified automatically based on column name keywords.
2. Brand/support accounts were identified from {'an explicit brand column' if brand_col else 'the author column'}.
3. Conversation/thread analysis depends on the presence of a conversation_id column.
4. Customer vs support tweet classification assumes the brand account tweets are from the brand itself.
5. Large dataset (3M+ tweets) - only basic profiling performed; deep analysis not done.

## Limitations

1. Column role inference is heuristic-based on column names only.
2. Brand identification may include non-support accounts if using author column.
3. Conversation threading requires conversation_id column which may not exist.
4. No semantic analysis of tweet content performed.
5. Time-series analysis not performed.

## Top Brands by Conversation Volume

"""
    for i, row in brand_stats.head(20).iterrows():
        notes += f"{i+1}. **{row['brand']}**: {row['total_tweets']} tweets, {row['conversations']} conversations, avg length {row['avg_conversation_length']}\n"
    
    notes_path = DOCS_DIR / "phase1_notes.md"
    with open(notes_path, "w") as f:
        f.write(notes)
    logger.info(f"Created {notes_path}")


def main():
    """Main entry point."""
    logger.info("Starting Phase 1: Dataset Profiling and Brand Selection")
    
    # Step 1: Download dataset
    dataset_path = download_dataset()
    
    # Step 2: Discover files
    files_info = discover_files(dataset_path)
    
    # Step 3: Inspect files
    inspection_results = inspect_files(files_info)
    
    # Step 4: Identify main tweet file
    main_file = identify_main_tweet_file(inspection_results)
    if not main_file:
        logger.error("Could not identify main tweet data file")
        sys.exit(1)
    logger.info(f"Main tweet file identified: {main_file}")
    
    # Step 5: Load tweets
    df = load_tweets(main_file, inspection_results)
    
    # Step 6: Print first 5 rows
    logger.info("First 5 rows:")
    print(df.head(5).to_string())
    
    # Step 7: Print dataframe shape
    logger.info(f"DataFrame shape: {df.shape}")
    print(f"\nDataFrame shape: {df.shape}")
    
    # Step 8: Print column names and dtypes
    logger.info("Column names and dtypes:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")
    
    # Step 9: Profile columns
    column_profile = profile_columns(df)
    
    # Step 10-11: Analyze brands
    brand_stats = analyze_brands(df, column_profile)
    
    # Step 12: Print top 20 brands
    print("\nTop 20 brands by conversation volume:")
    print(brand_stats.head(20).to_string(index=False))
    
    # Step 13-15: Save results
    save_results(files_info, inspection_results, column_profile, brand_stats)
    
    # Step 16: Create documentation
    create_phase1_notes(files_info, inspection_results, column_profile, brand_stats, main_file)
    
    # Step 17: Print summary
    print_summary(files_info, inspection_results, main_file, df, column_profile, brand_stats)
    
    logger.info("Phase 1 completed successfully!")


if __name__ == "__main__":
    main()