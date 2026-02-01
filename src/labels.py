
import numpy as np
import pandas as pd
from .prices import fetch_daily_close

def attach_returns(events: pd.DataFrame) -> pd.DataFrame:
    # Drop events with obviously invalid dates
    events = events[
        (events["entry_date"] >= pd.Timestamp("2015-01-01")) &
        (events["exit_date"] <= pd.Timestamp("today"))
    ].copy()

    tickers = sorted(set(events["ticker"])) + ["SPY"]

    start = events["entry_date"].min().date()
    end = events["exit_date"].max().date()

    closes = fetch_daily_close(tickers, start, end)

    def close_at(sym, d):
        try:
            return closes.loc[pd.Timestamp(d), sym]
        except Exception:
            return np.nan

    out = events.copy()
    out["entry_close"] = [close_at(t, d) for t, d in zip(out["ticker"], out["entry_date"])]
    out["exit_close"]  = [close_at(t, d) for t, d in zip(out["ticker"], out["exit_date"])]
    out["spy_entry"]   = [close_at("SPY", d) for d in out["entry_date"]]
    out["spy_exit"]    = [close_at("SPY", d) for d in out["exit_date"]]

    mask = (
        out["entry_close"].notna()
        & out["exit_close"].notna()
        & out["spy_entry"].notna()
        & out["spy_exit"].notna()
    )

    return out[mask].assign(
        ret=lambda x: x["exit_close"] / x["entry_close"] - 1,
        spy_ret=lambda x: x["spy_exit"] / x["spy_entry"] - 1,
        excess_ret=lambda x: x["ret"] - x["spy_ret"],
    )
