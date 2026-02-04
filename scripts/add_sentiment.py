"""
Add FinBERT sentiment to events_labeled.parquet

Usage:
    python -m scripts.add_sentiment
"""
import pandas as pd
from src.sentiment import add_sentiment_to_dataframe

INPUT_PATH = "data/intermediate/events_labeled.parquet"
OUTPUT_PATH = "data/intermediate/events_with_sentiment.parquet"


def main():
    print(f"Loading {INPUT_PATH}...")
    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df):,} rows")
    
    # Add sentiment
    df = add_sentiment_to_dataframe(df, text_column="headline", batch_size=64)
    
    # Save
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"Total rows: {len(df):,}")
    
    # Print sentiment distribution
    print("\nSentiment distribution:")
    print(df["sentiment"].value_counts())


if __name__ == "__main__":
    main()
