"""Self-contained version of bake_history.py for GitHub Actions.

Doesn't import the strategy package — config is inlined so this single
file can live in the GitHub Pages repo and be invoked by a workflow.

Output: historic_data.json (yfinance BTC-USD OHLCV, ~6 years).
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf


# Inline strategy config — must match strategy/core.py StrategyConfig
TICKER = "BTC-USD"
START_DATE = "2020-04-01"
LOOKBACK_DAYS = 200


def main() -> None:
    end = datetime.now().strftime("%Y-%m-%d")
    start = (pd.to_datetime(START_DATE) - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")

    print(f"Downloading {TICKER} from yfinance, {start} → {end}...")
    raw = yf.download(TICKER, start=start, end=end, progress=False, auto_adjust=False)
    if raw.empty:
        raise RuntimeError("yfinance returned no data")

    bars = []
    for ts, row in raw.iterrows():
        def col(name: str) -> float:
            v = row[name]
            return float(v.iloc[0] if hasattr(v, "iloc") else v)
        bars.append({
            "ts": int(ts.timestamp() * 1000),
            "date": ts.strftime("%Y-%m-%d"),
            "open": col("Open"),
            "high": col("High"),
            "low": col("Low"),
            "close": col("Close"),
            "volume": col("Volume"),
        })

    out = {
        "source": "yfinance BTC-USD",
        "baked_at": datetime.now().isoformat(timespec="seconds"),
        "start_date": START_DATE,
        "lookback_days": LOOKBACK_DAYS,
        "n_bars": len(bars),
        "bars": bars,
    }

    out_path = Path("historic_data.json")
    out_path.write_text(json.dumps(out))
    size_kb = out_path.stat().st_size / 1024
    print(f"Wrote: {out_path}  ({len(bars)} bars, {size_kb:.0f} KB)")
    print(f"Range: {bars[0]['date']} → {bars[-1]['date']}")
    print(f"Last close: ${bars[-1]['close']:,.2f}")


if __name__ == "__main__":
    main()
