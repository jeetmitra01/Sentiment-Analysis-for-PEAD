
import pandas as pd
import pandas_market_calendars as mcal

NYSE = mcal.get_calendar("NYSE")

# being conservative and calculating the next trading day after news was published
# TODO: if timestamp is close to market open consider that date instead of next date.
def next_trading_day_from_et(ts_et: pd.Timestamp) -> pd.Timestamp:
    d = ts_et.date()
    sched = NYSE.schedule(start_date=d, end_date=(d + pd.Timedelta(days=10)).date())
    trading_days = sched.index
    after = trading_days[trading_days.date > d]
    return after[0] if len(after) else pd.NaT


def build_events(df_news: pd.DataFrame, horizons=(1, 5)) -> pd.DataFrame:
    df = df_news.copy()
    df["entry_date"] = df["published_et"].apply(next_trading_day_from_et)
    df = df.dropna(subset=["entry_date"]).copy()
    df["entry_date"] = pd.to_datetime(df["entry_date"]).dt.tz_convert(None)

    min_d = df["entry_date"].min()
    max_d = df["entry_date"].max() + pd.Timedelta(days=40)

    sched = NYSE.schedule(start_date=min_d.date(), end_date=max_d.date())
    trading_days = pd.Index(sched.index.tz_localize(None))
    day_to_pos = {d: i for i, d in enumerate(trading_days)}

    rows = []
    for h in horizons:
        tmp = df.copy()
        tmp["horizon"] = h

        def exit_day(entry):
            i = day_to_pos.get(entry, None)
            if i is None:
                return pd.NaT
            j = i + h
            return trading_days[j] if j < len(trading_days) else pd.NaT

        tmp["exit_date"] = tmp["entry_date"].apply(exit_day)
        tmp = tmp.dropna(subset=["exit_date"])
        rows.append(tmp)

    return pd.concat(rows, ignore_index=True)
