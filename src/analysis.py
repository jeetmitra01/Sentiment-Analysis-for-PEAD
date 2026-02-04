# src/analysis.py
"""
Analysis functions for sentiment-returns relationship.
"""
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def returns_by_sentiment(df: pd.DataFrame, return_col: str = "excess_ret") -> pd.DataFrame:
    """
    Calculate return statistics grouped by sentiment.
    
    Args:
        df: DataFrame with 'sentiment' and return columns
        return_col: Which return column to analyze ('ret', 'excess_ret', 'spy_ret')
    
    Returns:
        DataFrame with mean, median, std, count per sentiment
    """
    stats_df = df.groupby("sentiment")[return_col].agg([
        ("mean", "mean"),
        ("median", "median"),
        ("std", "std"),
        ("count", "count"),
    ])
    
    # Add standard error
    stats_df["se"] = stats_df["std"] / np.sqrt(stats_df["count"])
    
    # Reorder: positive, neutral, negative
    order = ["positive", "neutral", "negative"]
    stats_df = stats_df.reindex(order)
    
    return stats_df


def statistical_tests(df: pd.DataFrame, return_col: str = "excess_ret") -> dict:
    """
    Perform t-tests comparing sentiment groups.
    
    Returns:
        dict with test results for each comparison
    """
    pos = df[df["sentiment"] == "positive"][return_col].dropna()
    neg = df[df["sentiment"] == "negative"][return_col].dropna()
    neu = df[df["sentiment"] == "neutral"][return_col].dropna()
    
    results = {}
    
    # Positive vs Negative
    t_stat, p_val = stats.ttest_ind(pos, neg)
    results["positive_vs_negative"] = {
        "t_statistic": t_stat,
        "p_value": p_val,
        "significant": p_val < 0.05,
        "mean_diff": pos.mean() - neg.mean()
    }
    
    # Positive vs Neutral
    t_stat, p_val = stats.ttest_ind(pos, neu)
    results["positive_vs_neutral"] = {
        "t_statistic": t_stat,
        "p_value": p_val,
        "significant": p_val < 0.05,
        "mean_diff": pos.mean() - neu.mean()
    }
    
    # Negative vs Neutral
    t_stat, p_val = stats.ttest_ind(neg, neu)
    results["negative_vs_neutral"] = {
        "t_statistic": t_stat,
        "p_value": p_val,
        "significant": p_val < 0.05,
        "mean_diff": neg.mean() - neu.mean()
    }
    
    return results


def cumulative_returns(
    df: pd.DataFrame,
    return_col: str = "excess_ret",
    long_sentiment: str = "positive",
    short_sentiment: str = "negative"
) -> pd.DataFrame:
    """
    Calculate cumulative returns for a long/short sentiment strategy.
    
    Strategy: Long stocks with positive sentiment, Short stocks with negative sentiment.
    Assumes equal-weight positions.
    
    Returns:
        DataFrame with daily strategy returns and cumulative returns
    """
    df = df.copy()
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    
    # Get long and short positions
    long_df = df[df["sentiment"] == long_sentiment].copy()
    short_df = df[df["sentiment"] == short_sentiment].copy()
    
    # Calculate daily average returns
    long_daily = long_df.groupby("entry_date")[return_col].mean()
    short_daily = short_df.groupby("entry_date")[return_col].mean()
    
    # Combine: long positive + short negative (flip sign for short)
    combined = pd.DataFrame({
        "long_ret": long_daily,
        "short_ret": -short_daily,  # Flip sign for short positions
    }).fillna(0)
    
    combined["strategy_ret"] = (combined["long_ret"] + combined["short_ret"]) / 2
    combined["cumulative_ret"] = (1 + combined["strategy_ret"]).cumprod() - 1
    
    return combined


def plot_returns_distribution(
    df: pd.DataFrame,
    return_col: str = "excess_ret",
    figsize: tuple = (12, 5)
) -> plt.Figure:
    """
    Plot histogram of returns by sentiment.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    
    colors = {"positive": "#2ecc71", "neutral": "#95a5a6", "negative": "#e74c3c"}
    order = ["positive", "neutral", "negative"]
    
    for ax, sentiment in zip(axes, order):
        data = df[df["sentiment"] == sentiment][return_col].clip(-0.2, 0.2)
        ax.hist(data, bins=50, color=colors[sentiment], alpha=0.7, edgecolor="white")
        ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)
        ax.axvline(x=data.mean(), color="red", linestyle="-", linewidth=2,
                   label=f"Mean: {data.mean()*100:.3f}%")
        ax.set_title(f"{sentiment.capitalize()}\n(n={len(data):,})")
        ax.set_xlabel(f"{return_col} (%)")
        ax.legend(loc="upper right")
    
    axes[0].set_ylabel("Frequency")
    fig.suptitle("Return Distribution by Sentiment", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    return fig


def plot_cumulative_returns(
    cum_df: pd.DataFrame,
    figsize: tuple = (12, 6)
) -> plt.Figure:
    """
    Plot cumulative returns of the sentiment strategy.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(cum_df.index, cum_df["cumulative_ret"] * 100, 
            color="#3498db", linewidth=2, label="Long Positive / Short Negative")
    
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax.fill_between(cum_df.index, 0, cum_df["cumulative_ret"] * 100,
                    where=cum_df["cumulative_ret"] > 0, color="#2ecc71", alpha=0.3)
    ax.fill_between(cum_df.index, 0, cum_df["cumulative_ret"] * 100,
                    where=cum_df["cumulative_ret"] < 0, color="#e74c3c", alpha=0.3)
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title("Sentiment Strategy Cumulative Returns\n(Long Positive, Short Negative)", 
                 fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add final return annotation
    final_ret = cum_df["cumulative_ret"].iloc[-1] * 100
    ax.annotate(f"Final: {final_ret:.1f}%", 
                xy=(cum_df.index[-1], final_ret),
                xytext=(10, 0), textcoords="offset points",
                fontsize=12, fontweight="bold")
    
    plt.tight_layout()
    return fig


def plot_returns_by_horizon(
    df: pd.DataFrame,
    return_col: str = "excess_ret",
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    Compare returns by sentiment across different holding periods.
    """
    summary = df.groupby(["horizon", "sentiment"])[return_col].mean().unstack()
    summary = summary[["positive", "neutral", "negative"]]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(summary.index))
    width = 0.25
    
    colors = {"positive": "#2ecc71", "neutral": "#95a5a6", "negative": "#e74c3c"}
    
    for i, sentiment in enumerate(["positive", "neutral", "negative"]):
        ax.bar(x + i * width, summary[sentiment] * 100, width, 
               label=sentiment.capitalize(), color=colors[sentiment])
    
    ax.set_xlabel("Holding Period (days)")
    ax.set_ylabel(f"Mean {return_col} (%)")
    ax.set_title("Returns by Sentiment and Holding Period", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{h} day" for h in summary.index])
    ax.legend()
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    return fig


def plot_returns_by_confidence(
    df: pd.DataFrame,
    return_col: str = "excess_ret",
    confidence_bins: list = [0, 0.6, 0.8, 0.9, 1.0],
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    Analyze if higher confidence predictions perform better.
    """
    df = df.copy()
    df["conf_bin"] = pd.cut(df["confidence"], bins=confidence_bins, 
                            labels=["0.0-0.6", "0.6-0.8", "0.8-0.9", "0.9-1.0"])
    
    summary = df.groupby(["conf_bin", "sentiment"])[return_col].mean().unstack()
    summary = summary[["positive", "neutral", "negative"]]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    summary.plot(kind="bar", ax=ax, color=["#2ecc71", "#95a5a6", "#e74c3c"])
    
    ax.set_xlabel("Confidence Range")
    ax.set_ylabel(f"Mean {return_col} (%)")
    ax.set_title("Returns by Sentiment and Model Confidence", fontsize=14, fontweight="bold")
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax.legend(title="Sentiment")
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    return fig
