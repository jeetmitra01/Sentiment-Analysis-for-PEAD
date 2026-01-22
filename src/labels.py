
import numpy as np
import pandas as pd
from .prices import fetch_daily_close

def attach_returns(events: pd.DataFrame) -> pd.DataFrame:
    tickers = sorted(set(events["ticker"])) + ["SPY"]
    start = events["entry_date"].min() - pd.Timedelta(days=7)
    end = events["exit_date"].max() + pd.Timedelta(days=7)
    print("Fetching prices for", len(tickers), "tickers")

    closes = fetch_daily_close(tickers, start, end)

    def close_at(sym, d):
        try:
            return closes.loc[d, sym]
        except Exception:
            return np.nan

    out = events.copy()
    out["entry_close"] = [close_at(t, d) for t, d in zip(out["ticker"], out["entry_date"])]
    out["exit_close"]  = [close_at(t, d) for t, d in zip(out["ticker"], out["exit_date"])]
    out["spy_entry"]   = [close_at("SPY", d) for d in out["entry_date"]]
    out["spy_exit"]    = [close_at("SPY", d) for d in out["exit_date"]]

    out = out.dropna(subset=["entry_close","exit_close","spy_entry","spy_exit"]).copy()
    out["ret"] = out["exit_close"] / out["entry_close"] - 1.0
    out["spy_ret"] = out["spy_exit"] / out["spy_entry"] - 1.0
    out["excess_ret"] = out["ret"] - out["spy_ret"]
    return out
