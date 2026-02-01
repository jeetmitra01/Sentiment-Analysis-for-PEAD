# src/prices.py
import pandas as pd
import yfinance as yf


def fetch_daily_close(tickers, start, end, batch_size=50):
    """
    Fetch daily close prices for multiple tickers using batched yfinance downloads.
    
    Args:
        tickers: List of ticker symbols
        start: Start date
        end: End date  
        batch_size: Number of tickers to download per batch (default 50)
    
    Returns:
        DataFrame with dates as index and ticker symbols as columns
    """
    tickers = list(set(tickers))  # Remove duplicates
    all_prices = []

    # Process tickers in batches
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        print(f"Fetching batch {i // batch_size + 1}: {len(batch)} tickers...")
        
        try:
            # Download all tickers in batch at once
            data = yf.download(
                batch,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
                group_by="ticker",
            )

            if data.empty:
                continue

            # Handle the response based on number of tickers
            if len(batch) == 1:
                # Single ticker: columns are just price fields like 'Close'
                ticker = batch[0]
                if "Close" in data.columns:
                    close_data = data[["Close"]].rename(columns={"Close": ticker})
                    all_prices.append(close_data)
            else:
                # Multiple tickers: MultiIndex columns (ticker, field)
                # Extract Close prices for each ticker
                for ticker in batch:
                    try:
                        if ticker in data.columns.get_level_values(0):
                            close_series = data[ticker]["Close"]
                            if not close_series.dropna().empty:
                                close_df = close_series.to_frame(name=ticker)
                                all_prices.append(close_df)
                    except (KeyError, TypeError):
                        # Ticker not in response or structure issue
                        continue

        except Exception as exc:
            print(f"Batch download error: {exc}")
            continue

    if not all_prices:
        return pd.DataFrame()

    # Concatenate all price data
    prices = pd.concat(all_prices, axis=1)
    
    # Remove duplicate columns if any
    prices = prices.loc[:, ~prices.columns.duplicated()]
    
    prices.index = pd.to_datetime(prices.index)
    return prices
