"""Self-contained baker for overlay standalone dashboard.

Bakes BTC OHLCV (yfinance) + funding rate (Binance) + SOPR (bitcoin-data.com) +
pre-computed overlay signals into a single JSON file consumed by
`dashboard_overlay_standalone.html`. Designed for GitHub Actions.

No imports from strategy/ or overlay/ packages — config is inlined so this file
can live next to dashboard_overlay_standalone.html in the GitHub Pages repo.

Output: overlay_data.json
"""
from __future__ import annotations

import datetime as dt
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# ----- Inline config (must match overlay/core.py StrategyConfig) ----------------
TICKER = "BTC-USD"
START_DATE = "2020-04-01"
LOOKBACK_DAYS = 200

# Overlay signal params (final tuned defaults 2026-05-06)
OVERLAY_WINDOW_DAYS = 365
OVERLAY_TOP_PCT = 0.90
OVERLAY_BOTTOM_PCT = 0.10
OVERLAY_SMOOTH_DAYS = 7

BINANCE_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
SOPR_API_URL = "https://bitcoin-data.com/api/v1/sopr"


# ----- Funding fetcher (inline copy of overlay/overlay.py) ----------------------
def aggregate_funding_to_daily(raw: list[dict]) -> list[dict]:
    bucket: dict[str, list[float]] = {}
    for item in raw:
        ts = dt.datetime.fromtimestamp(int(item["fundingTime"]) / 1000, tz=dt.timezone.utc)
        bucket.setdefault(ts.date().isoformat(), []).append(float(item["fundingRate"]))
    return sorted(
        ({"date": d, "value": sum(vs) / len(vs)} for d, vs in bucket.items()),
        key=lambda r: r["date"],
    )


def fetch_funding_history(start_date: str, symbol: str = "BTCUSDT") -> list[dict]:
    start_ms = int(dt.datetime.fromisoformat(start_date)
                   .replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
    out: list[dict] = []
    cursor = start_ms
    while True:
        params = {"symbol": symbol, "startTime": cursor, "limit": 1000}
        resp = requests.get(BINANCE_FUNDING_URL, params=params, timeout=15)
        resp.raise_for_status()
        page = resp.json()
        if not page:
            break
        out.extend(page)
        cursor = int(page[-1]["fundingTime"]) + 1
        if len(page) < 1000:
            break
    return out


def fetch_sopr_history() -> list[dict]:
    resp = requests.get(SOPR_API_URL, headers={"Accept": "application/json"}, timeout=30)
    resp.raise_for_status()
    raw = resp.json()
    out = []
    for item in raw:
        d, v = item.get("d"), item.get("sopr")
        if d is None or v is None:
            continue
        try:
            out.append({"date": str(d), "value": float(v)})
        except (TypeError, ValueError):
            continue
    return sorted(out, key=lambda r: r["date"])


# ----- Signal computation (inline) ----------------------------------------------
def calculate_adx(df: pd.DataFrame, period: int = 27) -> pd.Series:
    """Mirrors strategy/indicators.py — SMA-based ADX (NOT Wilder EMA)."""
    high, low, close = df["high"], df["low"], df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    plus_dm = high - high.shift(1)
    minus_dm = low.shift(1) - low
    plus_dm = plus_dm.copy()
    minus_dm = minus_dm.copy()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0

    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.rolling(period).mean()


def calculate_choppiness(df: pd.DataFrame, period: int = 149) -> pd.Series:
    """Mirrors strategy/indicators.py."""
    high, low, close = df["high"], df["low"], df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_sum = tr.rolling(period).sum()
    high_max = high.rolling(period).max()
    low_min = low.rolling(period).min()
    hl_range = high_max - low_min
    return 100 * np.log10(atr_sum / hl_range) / np.log10(period)


def calculate_efficiency_ratio(df: pd.DataFrame, period: int = 2) -> pd.Series:
    """Mirrors strategy/indicators.py — fillna(0)."""
    close = df["close"]
    change = (close - close.shift(period)).abs()
    volatility = close.diff().abs().rolling(period).sum()
    er = change / volatility
    return er.fillna(0)


def compute_signals_inline(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    w, smooth = OVERLAY_WINDOW_DAYS, OVERLAY_SMOOTH_DAYS
    out["funding_pct"] = out["funding"].rolling(w, min_periods=w).rank(pct=True)
    out["sopr_pct"] = out["sopr"].rolling(w, min_periods=w).rank(pct=True)
    out["funding_top"] = (out["funding_pct"] >= OVERLAY_TOP_PCT).fillna(False)
    out["funding_bottom"] = ((out["funding_pct"] <= OVERLAY_BOTTOM_PCT).fillna(False)
                             & out["funding_pct"].notna())
    out["sopr_top"] = (out["sopr_pct"] >= OVERLAY_TOP_PCT).fillna(False)
    out["sopr_bottom"] = ((out["sopr_pct"] <= OVERLAY_BOTTOM_PCT).fillna(False)
                          & out["sopr_pct"].notna())
    for col in ("funding_top", "funding_bottom", "sopr_top", "sopr_bottom"):
        out[f"{col}_7d"] = out[col].rolling(smooth, min_periods=1).max().fillna(0).astype(bool)
    out["top_signal"] = out["funding_top_7d"] & out["sopr_top_7d"]
    out["bottom_signal"] = out["funding_bottom_7d"] & out["sopr_bottom_7d"]
    return out


# ----- Main bake ----------------------------------------------------------------
def main() -> None:
    end = dt.datetime.now().strftime("%Y-%m-%d")
    start = (pd.to_datetime(START_DATE) - timedelta(days=LOOKBACK_DAYS + OVERLAY_WINDOW_DAYS)).strftime("%Y-%m-%d")

    print(f"Downloading {TICKER} from yfinance, {start} → {end}...")
    raw = yf.download(TICKER, start=start, end=end, progress=False, auto_adjust=False)
    if raw.empty:
        raise RuntimeError("yfinance returned no data")

    df = pd.DataFrame({
        "timestamp": raw.index,
        "open": np.asarray(raw["Open"].values).flatten(),
        "high": np.asarray(raw["High"].values).flatten(),
        "low": np.asarray(raw["Low"].values).flatten(),
        "close": np.asarray(raw["Close"].values).flatten(),
        "volume": np.asarray(raw["Volume"].values).flatten(),
    }).reset_index(drop=True).sort_values("timestamp").reset_index(drop=True)

    print(f"Fetching funding rate from Binance...")
    funding_raw = fetch_funding_history(start_date=start)
    funding_daily = aggregate_funding_to_daily(funding_raw)
    f_map = {r["date"]: r["value"] for r in funding_daily}
    print(f"  {len(funding_daily)} daily funding entries, latest={funding_daily[-1]['date']}")

    print(f"Fetching SOPR from bitcoin-data.com...")
    sopr_daily = fetch_sopr_history()
    s_map = {r["date"]: r["value"] for r in sopr_daily}
    print(f"  {len(sopr_daily)} daily SOPR entries, latest={sopr_daily[-1]['date']}")

    df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    df["funding"] = df["date"].map(f_map).astype(float)
    df["sopr"] = df["date"].map(s_map).astype(float)

    print("Computing indicators (MA110, ADX, CHOP, ER)...")
    df["ma110"] = df["close"].rolling(110).mean()
    df["adx"] = calculate_adx(df, period=27)
    df["chop"] = calculate_choppiness(df, period=149)
    df["er"] = calculate_efficiency_ratio(df, period=2)

    print("Computing overlay signals...")
    df = compute_signals_inline(df)

    def _f(v):
        return None if pd.isna(v) else float(v)

    bars = []
    for _, row in df.iterrows():
        bars.append({
            "ts": int(pd.Timestamp(row["timestamp"]).timestamp() * 1000),
            "date": row["date"],
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
            "funding": _f(row["funding"]),
            "sopr": _f(row["sopr"]),
            "funding_pct": _f(row["funding_pct"]),
            "sopr_pct": _f(row["sopr_pct"]),
            "top_signal": bool(row["top_signal"]),
            "bottom_signal": bool(row["bottom_signal"]),
            "ma110": _f(row["ma110"]),
            "adx": _f(row["adx"]),
            "chop": _f(row["chop"]),
            "er": _f(row["er"]),
        })

    out = {
        "source": "yfinance BTC-USD + Binance funding + bitcoin-data.com SOPR",
        "baked_at": dt.datetime.now().isoformat(timespec="seconds"),
        "start_date": START_DATE,
        "lookback_days": LOOKBACK_DAYS,
        "overlay_config": {
            "window_days": OVERLAY_WINDOW_DAYS,
            "top_pct": OVERLAY_TOP_PCT,
            "bottom_pct": OVERLAY_BOTTOM_PCT,
            "smooth_days": OVERLAY_SMOOTH_DAYS,
        },
        "n_bars": len(bars),
        "bars": bars,
    }

    out_path = Path("overlay_data.json")
    out_path.write_text(json.dumps(out))
    size_kb = out_path.stat().st_size / 1024
    n_top = sum(1 for b in bars if b["top_signal"])
    n_bot = sum(1 for b in bars if b["bottom_signal"])
    print(f"\nWrote: {out_path}  ({len(bars)} bars, {size_kb:.0f} KB)")
    print(f"Range: {bars[0]['date']} → {bars[-1]['date']}")
    print(f"Last close: ${bars[-1]['close']:,.2f}")
    print(f"Top signals: {n_top} days | Bottom signals: {n_bot} days")


if __name__ == "__main__":
    main()
