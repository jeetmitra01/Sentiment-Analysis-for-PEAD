# src/sentiment.py
"""
FinBERT sentiment analysis with batched GPU inference.
"""
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


# Label mapping for ProsusAI/finbert
LABELS = ["positive", "negative", "neutral"]


def load_finbert(device=None):
    """
    Load FinBERT model and tokenizer.
    
    Args:
        device: 'cuda', 'cpu', or None (auto-detect)
    
    Returns:
        tuple: (model, tokenizer, device)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading FinBERT on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model = model.to(device)
    model.eval()
    
    print("FinBERT loaded successfully!")
    return model, tokenizer, device


def predict_sentiment_batch(texts, model, tokenizer, device, max_length=128):
    """
    Predict sentiment for a batch of texts.
    
    Args:
        texts: List of strings
        model: FinBERT model
        tokenizer: FinBERT tokenizer
        device: 'cuda' or 'cpu'
        max_length: Max token length (128 is usually enough for headlines)
    
    Returns:
        list of tuples: [(sentiment, confidence), ...]
    """
    # Tokenize batch
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get probabilities and predictions
    probs = torch.softmax(outputs.logits, dim=1)
    pred_indices = probs.argmax(dim=1)
    confidences = probs.max(dim=1).values
    
    # Convert to labels
    results = [
        (LABELS[idx.item()], conf.item())
        for idx, conf in zip(pred_indices, confidences)
    ]
    
    return results


def add_sentiment_to_dataframe(
    df: pd.DataFrame,
    text_column: str = "headline",
    batch_size: int = 64,
    device: str = None
) -> pd.DataFrame:
    """
    Add sentiment columns to a DataFrame.
    
    Args:
        df: DataFrame with text column
        text_column: Name of the column containing text
        batch_size: Number of texts to process at once
        device: 'cuda', 'cpu', or None (auto-detect)
    
    Returns:
        DataFrame with 'sentiment' and 'confidence' columns added
    """
    # Load model
    model, tokenizer, device = load_finbert(device)
    
    # Deduplicate texts to avoid redundant inference
    unique_texts = df[text_column].unique()
    print(f"Processing {len(unique_texts):,} unique texts from {len(df):,} rows...")
    
    # Process in batches with progress bar
    text_to_sentiment = {}
    
    for i in tqdm(range(0, len(unique_texts), batch_size), desc="FinBERT inference"):
        batch_texts = unique_texts[i:i + batch_size].tolist()
        results = predict_sentiment_batch(batch_texts, model, tokenizer, device)
        
        for text, result in zip(batch_texts, results):
            text_to_sentiment[text] = result
        
        # Clear CUDA cache periodically to prevent OOM
        if device == "cuda" and i % (batch_size * 100) == 0:
            torch.cuda.empty_cache()
    
    # Map results back to dataframe
    df = df.copy()
    df["sentiment"] = df[text_column].map(lambda x: text_to_sentiment[x][0])
    df["confidence"] = df[text_column].map(lambda x: text_to_sentiment[x][1])
    
    return df
