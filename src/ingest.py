
import pandas as pd

def load_kaggle_analyst_ratings(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"title": "headline", "date": "published", "stock": "ticker"})
    df = df.dropna(subset=["headline", "published", "ticker"]).copy()

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    df["published_utc"] = pd.to_datetime(df["published"], utc=True, errors="coerce")
    df = df.dropna(subset=["published_utc"]).copy()
    df["published_et"] = df["published_utc"].dt.tz_convert("America/New_York")

    return df[["ticker", "headline", "published_utc", "published_et"]]
