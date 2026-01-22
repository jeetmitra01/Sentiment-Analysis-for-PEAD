
import pandas as pd
import yfinance as yf

def fetch_daily_close(tickers, start, end) -> pd.DataFrame:
    px = yf.download(
        tickers=list(tickers),
        start=str(start),
        end=str(end + pd.Timedelta(days=1)),
        auto_adjust=True,
        progress=False
    )["Close"]

    if isinstance(px, pd.Series):
        px = px.to_frame()

    px.index = pd.to_datetime(px.index)
    return px
