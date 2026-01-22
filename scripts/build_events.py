
from src.ingest import load_kaggle_analyst_ratings
from src.events import build_events
from src.labels import attach_returns

NEWS_PATH = "data/raw/analyst_ratings_processed.csv"
OUT_PATH = "data/intermediate/events_labeled.parquet"

def main():
    news = load_kaggle_analyst_ratings(NEWS_PATH)
    print("Loaded news")
    print("Building events")
    events = build_events(news, horizons=(1, 5))
    events_labeled = attach_returns(events)
    events_labeled.to_parquet(OUT_PATH, index=False)
    print("Saved:", OUT_PATH, "rows:", len(events_labeled))

if __name__ == "__main__":
    main()
