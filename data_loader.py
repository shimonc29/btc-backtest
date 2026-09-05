from pathlib import Path
from urllib.request import urlretrieve
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/pplonski/datasets-for-start/master/bitcoin-historical-data/btc_4h_data_2018_to_2025.csv"
DATA_DIR = Path("data")
DATA_PATH = DATA_DIR / "btc_4h_data_2018_to_2025.csv"


def download_data(force=False):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if force or not DATA_PATH.exists():
        print(f"Downloading BTC 4H OHLCV data from {DATA_URL}")
        urlretrieve(DATA_URL, DATA_PATH)
    return DATA_PATH


def validate_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"open time", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["open time"] = pd.to_datetime(df["open time"], errors="raise")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    df = df.sort_values("open time").drop_duplicates("open time")
    if len(df) < 1000:
        raise ValueError(f"Unexpectedly small dataset: {len(df)} rows")

    print(f"Validated {len(df):,} candles")
    print(f"Range: {df['open time'].iloc[0]} -> {df['open time'].iloc[-1]}")
    return df


if __name__ == "__main__":
    path = download_data()
    validate_data(path)
