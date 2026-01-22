import time
import pandas as pd
import yfinance as yf

def fetch_daily_close(tickers, start, end, batch_size=100, sleep=5):
    all_px = []
    tickers = list(set(tickers))

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        print(f"Downloading batch {i//batch_size + 1} / {(len(tickers)-1)//batch_size + 1}")

        success = False
        for attempt in range(3):
            try:
                px = yf.download(
                    tickers=batch,
                    start=str(start),
                    end=str(end + pd.Timedelta(days=1)),
                    auto_adjust=True,
                    progress=False,
                    threads=False,   # IMPORTANT
                )["Close"]

                if isinstance(px, pd.Series):
                    px = px.to_frame()

                all_px.append(px)
                success = True
                break

            except Exception as e:
                print(f"Attempt {attempt+1} failed: {e}")
                time.sleep(sleep * (attempt + 1))

        if not success:
            print("Skipping batch due to repeated failures.")

        time.sleep(sleep)  # cooldown to avoid rate limit

    out = pd.concat(all_px, axis=1)
    out.index = pd.to_datetime(out.index)
    return out
