import pandas as pd
import pandas_market_calendars as mcal

NYSE = mcal.get_calendar("NYSE")

def build_events(df_news: pd.DataFrame, horizons=(1, 5)) -> pd.DataFrame:
    df = df_news.copy()

    # ---- 1. Build NYSE trading calendar ONCE ----
    min_date = df["published_et"].dt.date.min()
    max_date = df["published_et"].dt.date.max() + pd.Timedelta(days=10)

    schedule = NYSE.schedule(start_date=min_date, end_date=max_date)
    trading_days = pd.Index(schedule.index.date)

    # ---- 2. Map each publish date -> next trading day ----
    publish_dates = df["published_et"].dt.date.values

    # searchsorted gives first trading day strictly AFTER publish date
    idx = trading_days.searchsorted(publish_dates, side="right")
    df["entry_date"] = trading_days[idx]

    # convert to Timestamp (timezone-naive for yfinance)
    df["entry_date"] = pd.to_datetime(df["entry_date"])

    # ---- 3. Build exit dates using trading-day offsets ----
    trading_days_ts = pd.to_datetime(trading_days)
    day_to_pos = {d: i for i, d in enumerate(trading_days_ts)}

    rows = []
    for h in horizons:
        tmp = df.copy()
        tmp["horizon"] = h

        tmp["exit_date"] = tmp["entry_date"].map(
            lambda d: trading_days_ts[day_to_pos[d] + h]
            if day_to_pos[d] + h < len(trading_days_ts)
            else pd.NaT
        )

        tmp = tmp.dropna(subset=["exit_date"])
        rows.append(tmp)

    return pd.concat(rows, ignore_index=True)
