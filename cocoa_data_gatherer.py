"""
=============================================================
  COCOA TRADING ASSISTANT — Data Gathering Module
=============================================================
  Collects and structures all data needed for daily analysis:
    1. Cocoa price & OHLCV data (CC=F via Yahoo Finance, USD/tonne)
    2. Technical indicators (pandas-ta)
    3. Related market data (USD index, soft commodities)
    4. News headlines & basic sentiment (NewsAPI / RSS)
    5. West Africa weather (Open-Meteo — free, no key needed)
    6. Saves a structured JSON snapshot for the AI analysis step

  SETUP:
    pip install pandas numpy requests python-dotenv yfinance

  Optional (for richer news):
    Get a free key at https://newsapi.org and add to .env:
    NEWS_API_KEY=your_key_here
=============================================================
"""

import os
import json
import logging
import requests

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False

from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

import numpy as np
import pandas as pd

# Optional — combined stress signal
try:
    from cocoa_stress_signal import compute_combined_stress_signal
    STRESS_SIGNAL_AVAILABLE = True
except ImportError:
    STRESS_SIGNAL_AVAILABLE = False

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# News intelligence — separate module
try:
    from cocoa_news_agent import fetch_news_intelligence
    NEWS_AGENT_AVAILABLE = True
except ImportError:
    NEWS_AGENT_AVAILABLE = False
    # log is now defined, safe to use
    log.warning("cocoa_news_agent.py not found — news intelligence disabled")

# Grinding release impact model
try:
    from cocoa_grinding_impact import compute_grinding_impact
    GRINDING_IMPACT_AVAILABLE = True
except ImportError:
    GRINDING_IMPACT_AVAILABLE = False

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")  # Optional but recommended
OUTPUT_FILE  = "cocoa_daily_snapshot.json"

INSTRUMENT_CURRENCY   = "USD"
INSTRUMENT_TICKER     = "CC=F"   # ICE New York Cocoa continuous, USD/tonne
INSTRUMENT_LABEL      = "ICE New York Cocoa (CC=F)"

# Related tickers for broader market context
RELATED_TICKERS = {
    "USD_Index":   "DX-Y.NYB",   # USD strength (inverse to cocoa)
    "GBPUSD":      "GBPUSD=X",   # Cross-reference for London contract context
    "Sugar":       "SB=F",        # Often correlated with cocoa
    "Coffee":      "KC=F",        # Fellow soft commodity
    "SP500":       "^GSPC",       # Global risk-on/off sentiment
}

# Key growing regions for weather monitoring
WEATHER_LOCATIONS = {
    "Abidjan_CoteIvoire": {"lat": 5.354, "lon": -4.005},   # Largest producer
    "Accra_Ghana":        {"lat": 5.600, "lon": -0.187},   # Second largest
    "Kumasi_Ghana":       {"lat": 6.688, "lon": -1.624},   # Ghana cocoa belt
    "Yamoussoukro_CI":    {"lat": 6.820, "lon": -5.275},   # CI interior belt
}

# ─────────────────────────────────────────────
#  1. PRICE DATA (CC=F from Yahoo Finance, USD)
# ─────────────────────────────────────────────

def fetch_price_data(period: str = "6mo") -> pd.DataFrame:
    """
    Fetch CC=F (ICE New York Cocoa, USD/tonne) from Yahoo Finance.

    No normalisation, no currency conversion — raw USD prices
    used directly for both display and technical analysis.

    Detects contract rollover gaps (where the open of day N differs
    from the close of day N-1 by more than 3%) and ratio-adjusts
    the historical series backwards so that indicators (EMAs, MACD,
    Bollinger Bands) are computed on a smooth series.  The most
    recent bar is always at the true market price.

    Returns a DataFrame with Open, High, Low, Close, Volume columns.
    """
    import yfinance as yf
    log.info(f"Fetching {INSTRUMENT_TICKER} from Yahoo Finance ({period})...")

    df = yf.download(INSTRUMENT_TICKER, period=period, interval="1d",
                     auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for {INSTRUMENT_TICKER}")

    # Flatten MultiIndex columns (yfinance 1.4+ returns ('Close', 'CC=F') etc.)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    elif any(isinstance(c, tuple) for c in df.columns):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.index = pd.to_datetime(df.index)
    df = df[["Open", "High", "Low", "Close", "Volume"]].sort_index()

    # Drop any rows where close is NaN or zero (bad data)
    df = df[df["Close"].notna() & (df["Close"] > 0)]

    # Save raw (unadjusted) closes BEFORE rollover repair.
    # Period changes (1w, 1m) should use raw prices — the user actually
    # experienced those prices.  The adjusted series is for indicators only.
    raw_close = df["Close"].copy()

    # ── Detect and repair contract rollover gaps ──────────────
    # CC=F is a continuous futures ticker.  When the front month
    # rolls, Yahoo stitches contracts together creating a price
    # gap that is NOT a real market move.  We detect these as
    # days where the open is >3% away from the prior close, then
    # ratio-adjust the entire history before the gap so the
    # series is smooth.  This preserves the true most-recent
    # price while giving indicators a clean series to work with.
    n_rolls = 0
    if len(df) >= 2:
        # Ensure float dtype so ratio multiplication works
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = df[col].astype(float)

        for i in range(len(df) - 1, 0, -1):
            prev_close = df["Close"].iloc[i - 1]
            curr_open  = df["Open"].iloc[i]
            if prev_close == 0:
                continue
            gap_pct = (curr_open - prev_close) / prev_close * 100
            if abs(gap_pct) > 3.0:
                # Ratio-adjust everything before this gap
                ratio = curr_open / prev_close
                for col in ["Open", "High", "Low", "Close"]:
                    df.iloc[:i, df.columns.get_loc(col)] *= ratio
                n_rolls += 1
                log.info(f"  Rollover detected at {df.index[i].date()}: "
                         f"gap {gap_pct:+.1f}%, adjusted {i} prior bars")

    if n_rolls:
        # Round after all adjustments
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = df[col].round(1)
        log.info(f"  {n_rolls} rollover gap(s) repaired in history")

    log.info(f"  ✅ {INSTRUMENT_TICKER}: {len(df)} bars "
             f"({df.index[0].date()} to {df.index[-1].date()}), "
             f"last close: {df['Close'].iloc[-1]:.1f} USD/t")
    return df, raw_close


def compute_indicators(df: pd.DataFrame, raw_close: pd.Series = None) -> dict:
    """
    Compute key technical indicators using pure pandas/numpy.
    No external TA library required — all indicators are standard
    calculations that only need pandas and numpy.
    """
    log.info("Computing technical indicators...")

    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]
    raw = raw_close if raw_close is not None else close

    # ── Helper: EMA ────────────────────────────────
    def ema(series, span):
        return series.ewm(span=span, adjust=False).mean()

    # ── Helper: RSI ────────────────────────────────
    def rsi(series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    # ── Helper: MACD ───────────────────────────────
    def macd(series, fast=12, slow=26, signal=9):
        ema_fast = ema(series, fast)
        ema_slow = ema(series, slow)
        macd_line = ema_fast - ema_slow
        signal_line = ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    # ── Helper: Bollinger Bands ────────────────────
    def bbands(series, period=20, std_dev=2):
        mid = series.rolling(period).mean()
        std = series.rolling(period).std()
        upper = mid + std_dev * std
        lower = mid - std_dev * std
        return upper, mid, lower

    # ── Helper: ATR ────────────────────────────────
    def atr(high_s, low_s, close_s, period=14):
        tr1 = high_s - low_s
        tr2 = (high_s - close_s.shift(1)).abs()
        tr3 = (low_s - close_s.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    # ── Helper: OBV ────────────────────────────────
    def obv(close_s, vol_s):
        direction = np.where(close_s > close_s.shift(1), 1,
                    np.where(close_s < close_s.shift(1), -1, 0))
        return (vol_s * direction).cumsum()

    # ── Helper: Stochastic ─────────────────────────
    def stochastic(high_s, low_s, close_s, k_period=14, d_period=3):
        lowest_low = low_s.rolling(k_period).min()
        highest_high = high_s.rolling(k_period).max()
        k = 100 * (close_s - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(d_period).mean()
        return k, d

    # ── Compute all indicators ─────────────────────
    ema_20  = ema(close, 20)
    ema_50  = ema(close, 50)
    ema_200 = ema(close, 200) if len(close) >= 200 else pd.Series(dtype=float)

    rsi_14 = rsi(close, 14)

    macd_line, macd_signal_line, macd_hist = macd(close)

    bb_upper, bb_mid, bb_lower = bbands(close)

    atr_14 = atr(high, low, close, 14)

    obv_series = pd.Series(obv(close, vol), index=close.index)

    stoch_k, stoch_d = stochastic(high, low, close)

    # ── Support / Resistance (simple rolling) ──
    resistance_20 = high.rolling(20).max()
    support_20    = low.rolling(20).min()
    resistance_50 = high.rolling(50).max()
    support_50    = low.rolling(50).min()

    # ── Extract latest values ──────────────────────
    def safe(series, idx=-1):
        try:
            v = series.iloc[idx]
            return round(float(v), 4) if pd.notna(v) else None
        except Exception:
            return None

    current_price = safe(close)
    ema20_val     = safe(ema_20)
    ema50_val     = safe(ema_50)
    ema200_val    = safe(ema_200) if len(ema_200) > 0 else None

    # Trend label
    if current_price and ema20_val and ema50_val and ema200_val:
        if current_price > ema20_val > ema50_val > ema200_val:
            trend = "Strong Uptrend (price above all EMAs)"
        elif current_price < ema20_val < ema50_val < ema200_val:
            trend = "Strong Downtrend (price below all EMAs)"
        elif current_price > ema50_val:
            trend = "Uptrend"
        elif current_price < ema50_val:
            trend = "Downtrend"
        else:
            trend = "Ranging / Mixed"
    elif current_price and ema20_val and ema50_val:
        if current_price < ema20_val and current_price < ema50_val and ema20_val < ema50_val:
            trend = "Strong Downtrend (price below EMA-20 & EMA-50)"
        elif current_price > ema20_val and current_price > ema50_val and ema20_val > ema50_val:
            trend = "Strong Uptrend (price above EMA-20 & EMA-50)"
        elif current_price > ema50_val:
            trend = "Uptrend (above EMA-50)"
        elif current_price < ema50_val:
            trend = "Downtrend (below EMA-50)"
        else:
            trend = "Ranging / Mixed"
    elif current_price and ema20_val:
        trend = "Above EMA-20" if current_price > ema20_val else "Below EMA-20"
    else:
        trend = "Insufficient data"

    macd_val    = safe(macd_line)
    macd_sig    = safe(macd_signal_line)
    macd_h      = safe(macd_hist)

    bb_u = safe(bb_upper)
    bb_m = safe(bb_mid)
    bb_l = safe(bb_lower)
    bb_width = round((bb_u - bb_l) / bb_m * 100, 2) if all([bb_u, bb_l, bb_m]) else None

    sk = safe(stoch_k)
    sd = safe(stoch_d)

    # Price change calculations
    today_open   = safe(df["Open"])
    prev_close   = safe(raw, -2)
    week_ago     = safe(raw, -6)  if len(raw) >= 6  else None
    month_ago    = safe(raw, -22) if len(raw) >= 22 else None

    def pct_change(current, previous):
        if current and previous:
            return round((current - previous) / previous * 100, 2)
        return None

    # Detect contract rollovers
    rollover_dates = []
    if len(df) >= 2:
        prev_closes = close.shift(1)
        opens       = df["Open"]
        gap_pct     = ((opens - prev_closes) / prev_closes * 100).dropna()
        for dt, gap in gap_pct.items():
            if abs(gap) > 3.0:
                rollover_dates.append(str(dt.date()))

    change_1d = pct_change(current_price, today_open)

    return {
        "price": {
            "current":        current_price,
            "open":           today_open,
            "high":           safe(high),
            "low":            safe(low),
            "prev_close":     prev_close,
            "volume":         safe(vol),
            "change_1d_pct":  change_1d,
            "change_1w_pct":  pct_change(current_price, week_ago),
            "change_1m_pct":  pct_change(current_price, month_ago),
            "week52_high":    round(float(high.max()), 1) if len(high) > 0 else None,
            "week52_low":     round(float(low.min()), 1)  if len(low) > 0 else None,
            "rollover_dates": rollover_dates[-5:] if rollover_dates else [],
            "rollover_note":  (
                f"{len(rollover_dates)} probable contract rollover(s) detected in history — "
                f"support/resistance levels near rollover dates may be distorted"
                if rollover_dates else None
            ),
        },
        "trend": {
            "label":     trend,
            "ema_20":    ema20_val,
            "ema_50":    ema50_val,
            "ema_200":   ema200_val,
            "price_vs_ema20":  round(current_price - ema20_val, 2)  if current_price and ema20_val  else None,
            "price_vs_ema200": round(current_price - ema200_val, 2) if current_price and ema200_val else None,
        },
        "momentum": {
            "rsi_14":       safe(rsi_14),
            "rsi_signal":   "Overbought" if safe(rsi_14) and safe(rsi_14) > 70
                            else "Oversold" if safe(rsi_14) and safe(rsi_14) < 30
                            else "Neutral",
            "macd":         macd_val,
            "macd_signal":  macd_sig,
            "macd_hist":    macd_h,
            "macd_cross":   "Bullish" if macd_val and macd_sig and macd_val > macd_sig
                            else "Bearish",
            "stoch_k":      sk,
            "stoch_d":      sd,
        },
        "volatility": {
            "atr_14":       safe(atr_14),
            "bb_upper":     bb_u,
            "bb_mid":       bb_m,
            "bb_lower":     bb_l,
            "bb_width_pct": bb_width,
            "bb_position":  "Above Upper Band" if current_price and bb_u and current_price > bb_u
                            else "Below Lower Band" if current_price and bb_l and current_price < bb_l
                            else "Within Bands",
        },
        "levels": {
            "resistance_20d": safe(resistance_20),
            "support_20d":    safe(support_20),
            "resistance_50d": safe(resistance_50),
            "support_50d":    safe(support_50),
        },
        "volume": {
            # OBV absolute value is not meaningful on its own.
            # Only the direction (accumulation vs distribution) matters.
            "obv_trend":        "Rising" if safe(obv_series) and safe(obv_series, -2) and safe(obv_series) > safe(obv_series, -2)
                                else "Falling",
            "obv_trend_label":  "Accumulation (OBV rising)" if safe(obv_series) and safe(obv_series, -2) and safe(obv_series) > safe(obv_series, -2)
                                else "Distribution (OBV falling)",
        },
    }


# ─────────────────────────────────────────────
#  2. RELATED MARKET CONTEXT
# ─────────────────────────────────────────────

# Stooq.com tickers — free, no API key, covers FX/indices/futures
# Format: https://stooq.com/q/l/?s=TICKER&f=sd2t2ohlcv&h&e=csv
STOOQ_TICKERS = {
    "USD_Index": ("dx.f",    "USD Index",       "Index"),
    "GBPUSD":    ("gbpusd",  "GBP/USD",         "FX"),
    "Sugar":     ("sb.f",    "Sugar #11",       "Commodity"),
    "Coffee":    ("kc.f",    "Coffee C",        "Commodity"),
    "SP500":     ("^spx",    "S&P 500",         "Index"),
}


def fetch_stooq_quote(ticker: str) -> dict:
    """
    Fetch a single quote from stooq.com CSV API.
    Returns dict with close, open, change_pct or raises on failure.
    """
    url = f"https://stooq.com/q/l/?s={ticker}&f=sd2t2ohlcv&h&e=csv"
    r = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    lines = r.text.strip().split("\n")
    if len(lines) < 2:
        raise ValueError(f"No data rows returned for {ticker}")
    header = lines[0].split(",")
    values = lines[1].split(",")
    row = dict(zip(header, values))
    close = float(row["Close"])
    open_ = float(row["Open"]) if row.get("Open") not in (None, "", "N/A") else None
    change_pct = round((close - open_) / open_ * 100, 2) if open_ else None
    return {"close": close, "open": open_, "change_pct": change_pct}


def fetch_related_markets(tickers: dict) -> dict:
    """
    Fetch related market data from stooq.com (no API key required).
    Covers FX pairs, indices, and soft commodity futures.
    Falls back to open.er-api.com for FX if stooq is unavailable.
    """
    log.info("Fetching related market data (stooq.com)...")
    results = {}

    for name in tickers:
        stooq_info = STOOQ_TICKERS.get(name)
        if not stooq_info:
            results[name] = {"price": None, "change_pct": None, "note": "no source configured"}
            continue

        ticker, label, category = stooq_info
        try:
            q = fetch_stooq_quote(ticker)
            results[name] = {
                "price":      round(q["close"], 4 if category == "FX" else 2),
                "change_pct": q["change_pct"],
                "label":      label,
                "source":     "stooq",
            }
            log.info(f"  ✅ {name:12s}: {results[name]['price']} "
                     f"({'+' if (q['change_pct'] or 0) >= 0 else ''}"
                     f"{q['change_pct']}%)")
        except Exception as e:
            log.warning(f"  ⚠️  {name} ({ticker}) failed: {e}")
            results[name] = {"price": None, "change_pct": None,
                             "note": f"fetch failed: {e}"}

    successes = sum(1 for v in results.values() if v.get("price") is not None)
    log.info(f"  → {successes}/{len(tickers)} markets fetched")
    return results


# ─────────────────────────────────────────────
#  3. CFTC COMMITMENTS OF TRADERS (COT) DATA
# ─────────────────────────────────────────────

# CFTC Disaggregated Futures report — ICE Cocoa contract code
COT_CONTRACT_CODE     = "073732"
COT_CURRENT_URL       = "https://www.cftc.gov/dea/newcot/f_disagg.txt"
COT_HISTORY_URL_FMT   = "https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip"
COT_CACHE_FILE        = "cot_cocoa_history.json"
COT_LOOKBACK_YEARS    = 3       # For percentile rankings


def fetch_cot_data() -> dict:
    """
    Fetch CFTC Commitments of Traders data for ICE Cocoa.

    Downloads the current week's disaggregated futures report,
    extracts managed money positioning, and computes:
      - Net position (longs - shorts)
      - Week-on-week change
      - Percentile rank vs last 3 years (for squeeze/liquidation risk)

    Returns a dict ready for the agent prompt.
    """
    log.info("Fetching CFTC COT data for ICE Cocoa...")

    try:
        result = _fetch_current_cot()
        if not result:
            return {"error": "No cocoa data in COT report"}

        # Load history for percentile calculation
        history = _load_cot_history()
        _append_and_save_cot(result, history)

        # Compute percentile rank of current net position
        net_pos = result.get("managed_money_net")
        if net_pos is not None and len(history) >= 10:
            all_nets = [h["managed_money_net"] for h in history
                        if h.get("managed_money_net") is not None]
            if all_nets:
                below = sum(1 for n in all_nets if n <= net_pos)
                result["net_position_percentile"] = round(below / len(all_nets) * 100, 1)
                result["percentile_sample_size"] = len(all_nets)

        # Add interpretive signals
        pct = result.get("net_position_percentile")
        if pct is not None:
            if pct >= 85:
                result["positioning_signal"] = "EXTREMELY_LONG"
                result["positioning_bias"] = "BEARISH"
                result["positioning_note"] = (
                    "Managed money net long in top 15% of 3yr range — "
                    "vulnerable to liquidation on any bearish catalyst"
                )
            elif pct >= 70:
                result["positioning_signal"] = "MODERATELY_LONG"
                result["positioning_bias"] = "WEAKLY_BEARISH"
                result["positioning_note"] = "Net long above average — some liquidation risk"
            elif pct <= 15:
                result["positioning_signal"] = "EXTREMELY_SHORT"
                result["positioning_bias"] = "BULLISH"
                result["positioning_note"] = (
                    "Managed money net short in bottom 15% of 3yr range — "
                    "high short squeeze risk on any bullish catalyst"
                )
            elif pct <= 30:
                result["positioning_signal"] = "MODERATELY_SHORT"
                result["positioning_bias"] = "WEAKLY_BULLISH"
                result["positioning_note"] = "Net short below average — some squeeze risk"
            else:
                result["positioning_signal"] = "NEUTRAL"
                result["positioning_bias"] = "NEUTRAL"
                result["positioning_note"] = "Positioning within normal range"

        log.info(
            f"  ✅ COT: Managed money net={result.get('managed_money_net'):+,} "
            f"(percentile: {result.get('net_position_percentile', 'N/A')}%, "
            f"signal: {result.get('positioning_signal', 'N/A')})"
        )
        return result

    except Exception as e:
        log.warning(f"  COT fetch failed: {e}")
        return {"error": str(e)}


def _fetch_current_cot() -> dict:
    """
    Download and parse the current week's CFTC disaggregated report.

    The f_disagg.txt file has NO HEADER ROW — columns are positional.
    Column layout (Disaggregated Futures-Only):
      0: Market_and_Exchange_Names
      1: As_of_Date (YYMMDD)
      2: Report_Date (YYYY-MM-DD)
      3: CFTC_Contract_Market_Code
      4: CFTC_Market_Code_Initials
      5: CFTC_Region_Code
      6: CFTC_Commodity_Code
      7: Open_Interest_All
      8: Prod_Merc_Long_All
      9: Prod_Merc_Short_All
     10: Swap_Dealer_Long_All
     11: Swap_Dealer_Short_All
     12: Swap_Dealer_Spread_All
     13: M_Money_Long_All
     14: M_Money_Short_All
     15: M_Money_Spread_All
    """
    import csv
    import io

    r = requests.get(COT_CURRENT_URL, timeout=30)
    r.raise_for_status()

    reader = csv.reader(io.StringIO(r.text))
    for row in reader:
        if len(row) < 16:
            continue

        market_name = row[0].strip()
        # Match ICE cocoa — the name contains "COCOA" and "ICE"
        if "COCOA" not in market_name.upper():
            continue
        if "ICE" not in market_name.upper():
            continue

        log.info(f"  COT: matched row: {market_name}")

        # Parse fields by position
        report_date = row[2].strip() if len(row) > 2 else ""
        cftc_code   = row[3].strip() if len(row) > 3 else ""
        oi          = _safe_int(row[7])
        pm_long     = _safe_int(row[8])
        pm_short    = _safe_int(row[9])
        mm_long     = _safe_int(row[13])
        mm_short    = _safe_int(row[14])
        mm_spread   = _safe_int(row[15])

        mm_net = (mm_long - mm_short) if mm_long is not None and mm_short is not None else None
        pm_net = (pm_long - pm_short) if pm_long is not None and pm_short is not None else None

        return {
            "report_date":          report_date,
            "source":               "CFTC Disaggregated Futures Report",
            "contract":             market_name,
            "cftc_code":            cftc_code,

            "managed_money_long":   mm_long,
            "managed_money_short":  mm_short,
            "managed_money_spread": mm_spread,
            "managed_money_net":    mm_net,

            "producer_merchant_long":  pm_long,
            "producer_merchant_short": pm_short,
            "producer_merchant_net":   pm_net,

            "total_open_interest":  oi,

            "mm_long_pct_oi":       round(mm_long / oi * 100, 1) if mm_long and oi else None,
            "mm_short_pct_oi":      round(mm_short / oi * 100, 1) if mm_short and oi else None,
        }

    return None


def _safe_int(val) -> int:
    """Parse a string to int, stripping commas and spaces."""
    if val is None:
        return None
    try:
        return int(str(val).strip().replace(",", ""))
    except (ValueError, TypeError):
        return None


def _load_cot_history() -> list:
    """Load cached COT history."""
    try:
        with open(COT_CACHE_FILE, "r") as f:
            history = json.load(f)
        # Keep only last N years
        cutoff = (datetime.now(timezone.utc) - timedelta(days=COT_LOOKBACK_YEARS * 365)
                  ).strftime("%Y-%m-%d")
        return [h for h in history if h.get("report_date", "") >= cutoff]
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _append_and_save_cot(current: dict, history: list):
    """Append current week's data to history if not already present."""
    report_date = current.get("report_date")
    if not report_date:
        return

    # Deduplicate by date
    existing_dates = {h.get("report_date") for h in history}
    if report_date not in existing_dates:
        history.append({
            "report_date":        report_date,
            "managed_money_net":  current.get("managed_money_net"),
            "managed_money_long": current.get("managed_money_long"),
            "managed_money_short": current.get("managed_money_short"),
            "total_open_interest": current.get("total_open_interest"),
        })
        # Keep last 3 years
        cutoff = (datetime.now(timezone.utc) - timedelta(days=COT_LOOKBACK_YEARS * 365)
                  ).strftime("%Y-%m-%d")
        history = [h for h in history if h.get("report_date", "") >= cutoff]
        try:
            with open(COT_CACHE_FILE, "w") as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            log.warning(f"Failed to save COT history: {e}")




# ─────────────────────────────────────────────
#  4. WEST AFRICA WEATHER (Open-Meteo)
# ─────────────────────────────────────────────

def fetch_weather(locations: dict) -> dict:
    """
    Fetch 7-day weather forecast for key cocoa growing regions.
    Uses Open-Meteo — completely free, no API key required.
    """
    log.info("Fetching West Africa weather data...")
    results = {}

    for location_name, coords in locations.items():
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude":           coords["lat"],
            "longitude":          coords["lon"],
            "daily":              [
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "rain_sum",
                "showers_sum",      # convective rainfall — dominant type in tropics
                "windspeed_10m_max",
                "et0_fao_evapotranspiration",   # evapotranspiration = moisture stress indicator
            ],
            "current_weather":    True,
            "timezone":           "auto",
            "forecast_days":      7,
        }
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            data    = r.json()
            daily   = data.get("daily", {})
            current = data.get("current_weather", {})

            # Calculate 7-day totals
            rain_frontal  = sum(v for v in daily.get("rain_sum", []) if v is not None)
            rain_convect  = sum(v for v in daily.get("showers_sum", []) if v is not None)
            rain_7d       = round(rain_frontal + rain_convect, 1)

            precip_7d = round(
                sum(v for v in daily.get("precipitation_sum", []) if v is not None),
                1,
            )

            et0_values = [
                v for v in daily.get("et0_fao_evapotranspiration", [])
                if v is not None
            ]
            et0_7d = round(sum(et0_values), 1) if et0_values else None

            max_temp_values = [
                v for v in daily.get("temperature_2m_max", [])
                if v is not None
            ]
            avg_max_t = (
                sum(max_temp_values) / len(max_temp_values)
                if max_temp_values
                else None
            )

            # Effective moisture: rain gets full credit; non-rain precipitation
            # (dew, fog drip, condensation) gets 35% credit — meaningful for
            # cocoa but not as effective as direct rainfall for soil recharge
            non_rain_precip  = max(0.0, precip_7d - rain_7d)
            effective_moisture_7d = round(rain_7d + non_rain_precip * 0.35, 1)

            # Build daily rainfall array combining both rain types
            daily_rain_frontal = daily.get("rain_sum", [])
            daily_rain_convect = daily.get("showers_sum", [])
            daily_rain_total = []
            for i in range(max(len(daily_rain_frontal), len(daily_rain_convect))):
                fr = daily_rain_frontal[i] if i < len(daily_rain_frontal) and daily_rain_frontal[i] is not None else 0
                cv = daily_rain_convect[i] if i < len(daily_rain_convect) and daily_rain_convect[i] is not None else 0
                daily_rain_total.append(round(fr + cv, 1))

            # Crop stress flags — use effective moisture for a more accurate picture
            flags = []
            if effective_moisture_7d < 10:
                if non_rain_precip > 0:
                    flags.append(
                        f"⚠️ Very low rainfall ({rain_7d}mm) — partial dew/fog relief "
                        f"({non_rain_precip:.1f}mm non-rain precip, "
                        f"effective moisture: {effective_moisture_7d}mm)"
                    )
                else:
                    flags.append("⚠️ Very low rainfall — potential drought stress")
            elif effective_moisture_7d > 100:
                flags.append("⚠️ Excessive precipitation — disease/fungal risk (black pod)")
            if avg_max_t is not None and avg_max_t > 35:
                flags.append("⚠️ High temperatures — heat stress possible")

            results[location_name] = {
                "current_temp_c":           current.get("temperature"),
                "current_windspeed":        current.get("windspeed"),
                "rain_7d_mm":               round(rain_7d, 1),
                "precip_7d_mm":             round(precip_7d, 1),
                "non_rain_precip_7d_mm":    round(non_rain_precip, 1),
                "effective_moisture_7d_mm": effective_moisture_7d,
                "et0_7d_mm":                et0_7d,
                "avg_max_temp_c":           round(avg_max_t, 1) if avg_max_t is not None else None,
                "crop_stress_flags":        flags if flags else ["No significant stress signals"],
                "daily_rain_mm":            daily_rain_total,
                "daily_et0_mm": [
                    round(v, 1) if v is not None else None
                    for v in daily.get("et0_fao_evapotranspiration", [])
                ],
                "daily_dates":              daily.get("time", []),
            }
            log.info(
                f"  → {location_name}: {rain_7d:.1f}mm rain + "
                f"{non_rain_precip:.1f}mm dew/other = "
                f"{effective_moisture_7d:.1f}mm effective moisture (7d)"
            )
        except Exception as e:
            log.warning(f"  Weather fetch failed for {location_name}: {e}")
            results[location_name] = {"error": str(e)}

    return results


def summarise_weather(weather: dict) -> dict:
    """Create a high-level weather summary across all locations."""
    all_flags        = []
    total_rain       = []
    total_effective  = []
    total_non_rain   = []

    for loc, data in weather.items():
        if "error" not in data:
            all_flags.extend(data.get("crop_stress_flags", []))
            rain = data.get("rain_7d_mm")
            eff  = data.get("effective_moisture_7d_mm")
            nr   = data.get("non_rain_precip_7d_mm", 0)
            if rain is not None:
                total_rain.append(rain)
            if eff is not None:
                total_effective.append(eff)
            if nr is not None:
                total_non_rain.append(nr)

    significant_alerts   = [f for f in all_flags if "⚠️" in f]
    avg_rain             = round(sum(total_rain)      / len(total_rain),      1) if total_rain      else None
    avg_effective        = round(sum(total_effective) / len(total_effective),  1) if total_effective else None
    avg_non_rain         = round(sum(total_non_rain)  / len(total_non_rain),   1) if total_non_rain  else None

    # Use effective moisture for the condition assessment
    moisture_for_condition = avg_effective if avg_effective is not None else avg_rain

    return {
        "avg_7d_rainfall_mm":        avg_rain,
        "avg_7d_non_rain_precip_mm": avg_non_rain,
        "avg_7d_effective_moisture_mm": avg_effective,
        "significant_alerts":        list(set(significant_alerts)),
        "overall_condition":         "Stressed" if significant_alerts else "Normal",
        "moisture_note": (
            f"Total rainfall: {avg_rain}mm (+ {avg_non_rain}mm other precip "
            f"= {avg_effective}mm effective moisture)"
            if avg_non_rain and avg_non_rain > 0.5
            else f"Total rainfall: {avg_rain}mm"
        ),
    }


# ─────────────────────────────────────────────
#  5. SEASONAL CONTEXT
# ─────────────────────────────────────────────

def get_seasonal_context() -> dict:
    """
    Cocoa has two harvest seasons. Flag which season we're currently in.
    Main crop: October – March (larger, ~75% of production)
    Mid crop:  April – September (smaller, ~25%)
    """
    month = datetime.now().month
    if 10 <= month <= 12 or 1 <= month <= 3:
        season = "Main Crop"
        notes  = ("Main crop season (Oct–Mar). Largest harvest of the year (~75% of annual "
                  "output). Supply data and quality reports are most market-moving now.")
    else:
        season = "Mid Crop"
        notes  = ("Mid crop season (Apr–Sep). Smaller harvest (~25% of annual output). "
                  "Market may be more sensitive to demand-side data and grinding figures.")

    # Flag key seasonal events
    seasonal_events = {
        1:  "Main crop winding down; watch Ghana Quality Control Board reports",
        3:  "Main crop ending; transition period — watch forward pricing",
        4:  "Mid crop beginning — watch early arrival estimates",
        9:  "Mid crop ending; anticipate main crop forecasts",
        10: "Main crop arriving — critical period for supply estimates",
    }

    return {
        "current_season":    season,
        "month":             datetime.now().strftime("%B"),
        "season_notes":      notes,
        "seasonal_alert":    seasonal_events.get(month, None),
    }


# ─────────────────────────────────────────────
#  6. MAIN ORCHESTRATOR
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
#  5b. ICE WAREHOUSE STOCKS & GRINDING DATA
# ─────────────────────────────────────────────

WAREHOUSE_HISTORY_FILE = os.getenv("WAREHOUSE_HISTORY_FILE", "ice_warehouse_history.json")
GRINDING_CACHE_FILE    = os.getenv("GRINDING_CACHE_FILE",    "grinding_data_cache.json")


def fetch_ice_warehouse_stocks() -> dict:
    """
    Scrape ICE certified cocoa warehouse stocks from Barchart.
    Updated daily by ICE — a key physical market indicator.
    Rising stocks = bearish (supply building); falling = bullish (demand absorbing).

    Returns current level, previous, change, and 30-day trend direction.
    """
    url = "https://www.barchart.com/cmdty/data/fundamental/explore/IC345DRW.CS"
    try:
        import re
        from bs4 import BeautifulSoup

        r = requests.get(url, timeout=30, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/121.0.0.0 Safari/537.36"
        })
        r.raise_for_status()

        soup = BeautifulSoup(r.text, "html.parser")
        text = soup.get_text("\n", strip=True)

        def parse_int(s):
            cleaned = re.sub(r"[^\d]", "", s) if s else ""
            return int(cleaned) if cleaned else None

        # ── Overview block ────────────────────────────────────────────────────
        m_ov = re.search(r"\bOverview\b(.*?)\bHistorical Data\b", text, flags=re.S)
        if not m_ov:
            raise ValueError("Could not find Overview section on Barchart page")
        ov = m_ov.group(1)

        def get_ov(key):
            m = re.search(rf"\b{re.escape(key)}\b\s*\n([^\n]+)", ov)
            return m.group(1).strip() if m else None

        current = parse_int(get_ov("Most Recent Value"))
        date    = get_ov("Most Recent Date") or datetime.now().strftime("%m-%d-%Y")
        prev    = parse_int(get_ov("Prior Value"))

        if not current:
            raise ValueError("Could not parse warehouse stocks from Barchart page")
        log.info(f"  Parsed overview: current={current:,}, prev={prev}, date={date}")

        # ── Historical block (7 rows shown on page) ───────────────────────────
        m_hist = re.search(
            r"\bHistorical Data\b(.*?)\bGet access to full historical data\b",
            text, flags=re.S
        )
        page_history = {}
        if m_hist:
            rows = re.findall(r"\b(\d{2}-\d{2}-\d{4})\b\s+([\d,]+)\b", m_hist.group(1))
            for d, v in rows[:7]:
                page_history[d] = int(v.replace(",", ""))
            log.info(f"  Historical table: {len(page_history)} days parsed")

        change  = current - prev if prev is not None else None
        chg_pct = round(change / prev * 100, 3) if prev and change is not None else None

        # ── Merge into rolling local history ──────────────────────────────────
        history = _load_warehouse_history()
        if page_history:
            history.update(page_history)   # 7 days from page, available immediately
        history[date] = current            # ensure today is recorded
        if len(history) > 60:
            for oldest in sorted(history.keys())[:-60]:
                del history[oldest]
        _save_warehouse_history(history)

        # Sort chronologically for trend calculations
        sorted_vals = [v for _, v in sorted(history.items())]
        trend_5d  = _stock_trend(sorted_vals, 5)
        trend_7d  = _stock_trend(sorted_vals, 7)
        trend_20d = _stock_trend(sorted_vals, 20)

        # Convert bags to tonnes (1 bag = ~62.5 kg for London cocoa)
        current_tonnes = round(current * 0.0625)
        primary_trend  = trend_5d if trend_5d != "Insufficient data" else (
                         "Rising" if change and change > 0 else
                         "Falling" if change and change < 0 else "Neutral"
        )
        signal = (
            "Bearish — stocks rising, supply building"   if primary_trend == "Rising"  else
            "Bullish — stocks falling, demand absorbing" if primary_trend == "Falling" else
            "Neutral"
        )

        log.info(f"  ✅ ICE warehouse stocks: {current:,} bags ({current_tonnes:,}t) | "
                 f"5d trend: {trend_5d} | signal: {signal.split(' —')[0]}")

        return {
            "source":          "Barchart / ICE",
            "as_of":           date,
            "current_bags":    current,
            "current_tonnes":  current_tonnes,
            "prev_bags":       prev,
            "change_bags":     change,
            "change_pct":      chg_pct,
            "trend_5d":        trend_5d  if trend_5d  != "Insufficient data" else None,
            "trend_7d":        trend_7d  if trend_7d  != "Insufficient data" else None,
            "trend_20d":       trend_20d if trend_20d != "Insufficient data" else None,
            "signal":          signal,
            "history_7d":      dict(sorted(history.items(), reverse=True)[:7]),
            "note":            "ICE-certified European warehouses (Amsterdam, Antwerp, Hamburg etc)",
        }

    except ImportError:
        log.warning("  BeautifulSoup not installed — pip install beautifulsoup4")
        return {"error": "beautifulsoup4 not installed"}
    except Exception as e:
        log.warning(f"  ICE warehouse fetch failed: {e}")
        return {"error": str(e)}


def _load_warehouse_history() -> dict:
    try:
        with open(WAREHOUSE_HISTORY_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def _save_warehouse_history(history: dict):
    with open(WAREHOUSE_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def _stock_trend(values: list, n: int) -> str:
    """Returns Rising/Falling/Flat based on last n values."""
    if len(values) < 2:
        return "Insufficient data"
    segment = values[-n:] if len(values) >= n else values
    if segment[-1] > segment[0]:
        return "Rising"
    elif segment[-1] < segment[0]:
        return "Falling"
    return "Flat"


def load_grinding_data() -> dict:
    """
    Load the most recent grinding data from local cache.
    Grinding figures are released quarterly by three regional associations:
      - ECA  (European Cocoa Association) — largest, most market-moving
      - NCA  (National Confectioners Association, North America)
      - CAA  (Cocoa Association of Asia)

    The cache is pre-seeded with the latest known figures and manually
    updated after each quarterly release. The system flags when the next
    release is due so the report can note it as a scheduled market event.
    """
    default_cache = {
        "_note": "Update this file after each quarterly grinding release",
        "_sources": {
            "ECA": "https://www.eurococoa.com/grind-stats/",
            "NCA": "https://candyusa.com/cocoa-grinds-report/",
            "CAA": "Cocoa Association of Asia press releases"
        },

        # ── Most recent actuals (Q1 2026, released 16 Apr 2026) ──────────────
        "latest_quarter": "Q1 2026",
        "release_date":   "2026-04-16",
        "ECA": {
            "quarter":         "Q1 2026",
            "volume_tonnes":   325895,
            "yoy_change_pct":  -7.8,
            "prior_quarter":   "Q4 2025",
            "prior_volume":    304470,
            "full_year_2025":  1327000,
            "full_year_yoy":   -5.9,
            "note":            "Lowest Q1 in 17 years. Bigger decline than consensus (-6%). Germany Q1 down 8.7% to 90,852t."
        },
        "NCA": {
            "quarter":         "Q1 2026",
            "volume_tonnes":   106000,
            "yoy_change_pct":  -3.8,
            "full_year_yoy":   -0.9,     # FY2025 estimate
            "note":            "North America Q1 2026 down 3.8% YoY, adds to bearish demand signal"
        },
        "CAA": {
            "quarter":         "Q1 2026",
            "volume_tonnes":   223503,
            "yoy_change_pct":  5.2,
            "trailing_12m_yoy": None,
            "note":            "Asia Q1 2026 surprise beat: +5.2% YoY vs expectations of -6.7%. Regional divergence widening."
        },

        # ── Next scheduled releases ──────────────────────────────────────────
        "next_releases": {
            "ECA_Q2_2026": "2026-07-16",
            "NCA_Q2_2026": "2026-07-16",
            "CAA_Q2_2026": "2026-07-18",   # typically 2 days after ECA
        },

        # ── Trend context ────────────────────────────────────────────────────
        "trend_summary": (
            "Q1 2026 grinding data confirms deepening regional divergence. "
            "Europe -7.8% YoY (lowest Q1 in 17 years), North America -3.8%. "
            "Asia surprised to the upside at +5.2% vs consensus -6.7%. "
            "Western demand destruction continues despite lower prices from 2024-25 highs. "
            "Asia providing partial offset — structurally stronger processing demand."
        ),
        "icco_balance": {
            "season":           "2024/25",
            "surplus_deficit":  "+75,000t surplus",
            "global_production": "4.70Mt",
            "global_grindings":  "4.60Mt",
            "source":           "ICCO Mar 2026 update (revised up from +49,000t in Nov 2025)",
            "note":             "StoneX forecasts 287,000t surplus for 2025/26 and 267,000t for 2026/27"
        }
    }

    try:
        with open(GRINDING_CACHE_FILE) as f:
            cache = json.load(f)
        log.info(f"  Loaded grinding cache: {cache.get('latest_quarter')} "
                 f"(released {cache.get('release_date')})")
        return cache
    except FileNotFoundError:
        # First run — write the default cache and return it
        with open(GRINDING_CACHE_FILE, "w") as f:
            json.dump(default_cache, f, indent=2)
        log.info("  Created grinding data cache with Q4 2025 figures")
        return default_cache


def get_grinding_release_alert(grinding: dict) -> str | None:
    """
    Returns a string alert if a grinding release is within 3 days,
    or None if no release is imminent.
    """
    today = datetime.now(timezone.utc).date()
    alerts = []
    for label, date_str in grinding.get("next_releases", {}).items():
        try:
            release_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            days_away = (release_date - today).days
            if 0 <= days_away <= 3:
                region = label.split("_")[0]
                quarter = "_".join(label.split("_")[1:])
                alerts.append(
                    f"⚠️  {region} {quarter} grinding data due "
                    f"{'TODAY' if days_away == 0 else f'in {days_away} day(s)'} "
                    f"({date_str}) — major scheduled market event"
                )
        except ValueError:
            pass
    return "\n".join(alerts) if alerts else None

def gather_all_data() -> dict:
    """
    Run all data gathering steps and return a single structured snapshot.
    This dict will later be passed to the AI analysis module.
    """
    log.info("=" * 55)
    log.info("  COCOA TRADING ASSISTANT — Daily Data Gather")
    log.info(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 55)

    snapshot = {
        "generated_at":          datetime.now(timezone.utc).isoformat(),
        "ticker":                INSTRUMENT_LABEL,
        "instrument_currency":   INSTRUMENT_CURRENCY,
    }

    # 1. Price data & technical indicators (CC=F, USD, no conversion)
    try:
        price_df, raw_close = fetch_price_data()
        snapshot["history_days"] = len(price_df)
        snapshot["ta_source"]   = f"{INSTRUMENT_TICKER} (Yahoo Finance, USD/tonne)"
        snapshot["technicals"]  = compute_indicators(price_df, raw_close)

        # Store recent OHLCV for reference
        recent = price_df.tail(30).copy()
        snapshot["ohlcv_30d"] = [
            {
                "date":  str(idx.date()),
                "open":  round(float(r["Open"]),  1),
                "high":  round(float(r["High"]),  1),
                "low":   round(float(r["Low"]),   1),
                "close": round(float(r["Close"]), 1),
            }
            for idx, r in recent.iterrows()
        ]
    except Exception as e:
        log.error(f"Price/TA step failed: {e}")
        snapshot["technicals"] = {"error": str(e)}

    # 2. Related markets
    try:
        snapshot["related_markets"] = fetch_related_markets(RELATED_TICKERS)
    except Exception as e:
        log.error(f"Related markets step failed: {e}")
        snapshot["related_markets"] = {"error": str(e)}

    # 2b. Extract GBP/USD for display conversion in agent output
    gbpusd = snapshot.get("related_markets", {}).get("GBPUSD", {}).get("price")
    if gbpusd:
        snapshot["gbpusd_rate"] = gbpusd
        # Add GBP equivalent to the price data
        px = snapshot.get("technicals", {}).get("price", {})
        if px.get("current"):
            snapshot["price_gbp"] = round(px["current"] / gbpusd, 1)
            log.info(f"  GBP equivalent: {snapshot['price_gbp']} GBP/t (GBPUSD={gbpusd})")
    else:
        snapshot["gbpusd_rate"] = None
        snapshot["price_gbp"] = None

    # 2c. CFTC Commitments of Traders (speculative positioning)
    try:
        snapshot["cot"] = fetch_cot_data()
    except Exception as e:
        log.warning(f"COT fetch failed (non-critical): {e}")
        snapshot["cot"] = {"error": str(e)}

    # 3. News intelligence (cocoa_news_agent.py)
    try:
        if NEWS_AGENT_AVAILABLE:
            log.info("Running news intelligence agent...")
            news_brief = fetch_news_intelligence()
            snapshot["news_intelligence"] = news_brief
            snapshot["news"] = {
                "sentiment": {
                    "overall":    news_brief.get("directional_signal", "Unknown"),
                    "confidence": news_brief.get("signal_confidence",  "Unknown"),
                },
                "article_counts": news_brief.get("article_counts", {}),
                "recent_summary": news_brief.get("recent_summary", ""),
            }
        else:
            log.warning("News agent unavailable — skipping news step")
            snapshot["news"] = {"error": "cocoa_news_agent.py not found"}
    except Exception as e:
        log.error(f"News intelligence step failed: {e}")
        snapshot["news"] = {"error": str(e)}

    # 4. Weather
    try:
        weather_raw = fetch_weather(WEATHER_LOCATIONS)
        snapshot["weather"] = {
            "locations": weather_raw,
            "summary":   summarise_weather(weather_raw),
        }
    except Exception as e:
        log.error(f"Weather step failed: {e}")
        snapshot["weather"] = {"error": str(e)}

    # 5. Seasonal context
    snapshot["seasonal"] = get_seasonal_context()

    # 5b. ICE warehouse stocks
    try:
        snapshot["ice_warehouse"] = fetch_ice_warehouse_stocks()
    except Exception as e:
        log.error(f"Warehouse stocks step failed: {e}")
        snapshot["ice_warehouse"] = {"error": str(e)}

    # 5c. Grinding data (quarterly cache)
    try:
        grinding = load_grinding_data()
        alert    = get_grinding_release_alert(grinding)
        snapshot["grinding_data"] = grinding
        if alert:
            snapshot["grinding_release_alert"] = alert
            log.info(f"  {alert}")
    except Exception as e:
        log.error(f"Grinding data step failed: {e}")
        snapshot["grinding_data"] = {"error": str(e)}

    # 5d. Grinding release impact model
    if GRINDING_IMPACT_AVAILABLE:
        try:
            # NOTE: must be the GBP price — passing USD here skews the
            # GBP/tonne impact estimates by ~25%.
            grinding_impact = compute_grinding_impact(
                current_price_gbp=snapshot.get("price_gbp"),
                fill_prices=True,
            )
            snapshot["grinding_impact"] = grinding_impact
            n_forecasts = len(grinding_impact.get("forecasts", []))
            log.info(f"  ✅ Grinding impact model: {n_forecasts} upcoming release(s) forecast")
        except Exception as e:
            log.warning(f"  Grinding impact model failed (non-critical): {e}")
            snapshot["grinding_impact"] = {"error": str(e)}

    # 6. Combined crop stress signal (NDWI x weather)
    if STRESS_SIGNAL_AVAILABLE:
        try:
            from cocoa_crop_monitor import load_crop_health_for_agent
            crop_data    = load_crop_health_for_agent()
            weather_data = snapshot.get("weather", {})
            if "error" not in crop_data:
                snapshot["combined_stress"] = compute_combined_stress_signal(
                    weather_data, crop_data
                )
                log.info("  ✅ Combined stress signal computed")
            else:
                log.info("  Skipping combined stress — crop health data not available")
                snapshot["combined_stress"] = {"error": crop_data.get("error")}
        except Exception as e:
            log.warning(f"Combined stress signal failed (non-critical): {e}")
            snapshot["combined_stress"] = {"error": str(e)}

    return snapshot


def save_snapshot(snapshot: dict, filepath: str = OUTPUT_FILE):
    """Save the data snapshot to a JSON file."""
    with open(filepath, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)
    log.info(f"\n✅ Snapshot saved to: {filepath}")


def print_summary(snapshot: dict):
    """Print a human-readable console summary."""
    ta   = snapshot.get("technicals", {})
    px   = ta.get("price", {})
    mom  = ta.get("momentum", {})
    sent = snapshot.get("news", {}).get("sentiment", {})
    wx   = snapshot.get("weather", {}).get("summary", {})
    seas = snapshot.get("seasonal", {})

    print("\n" + "=" * 55)
    print("  COCOA DAILY SNAPSHOT SUMMARY")
    print("=" * 55)
    print(f"  Generated : {snapshot.get('generated_at', 'N/A')}")
    print(f"  Ticker    : {snapshot.get('ticker', 'N/A')}")
    print()
    print("  ── Price ─────────────────────────────")
    print(f"  Current Price : {px.get('current')} USD/tonne")
    print(f"  1D Change     : {px.get('change_1d_pct')}%")
    print(f"  1W Change     : {px.get('change_1w_pct')}%")
    print(f"  1M Change     : {px.get('change_1m_pct')}%")
    print()
    print("  ── Technicals ────────────────────────")
    print(f"  Trend         : {ta.get('trend', {}).get('label')}")
    print(f"  RSI (14)      : {mom.get('rsi_14')} — {mom.get('rsi_signal')}")
    print(f"  MACD Cross    : {mom.get('macd_cross')}")
    print(f"  BB Position   : {ta.get('volatility', {}).get('bb_position')}")
    print()
    print("  ── News Sentiment ────────────────────")
    print(f"  Overall       : {sent.get('overall')}")
    article_counts = snapshot.get("news", {}).get("article_counts", {})
    if article_counts:
        total = sum(article_counts.values()) if isinstance(article_counts, dict) else 0
        print(f"  Articles      : {total} gathered")
    else:
        # Fallback for old-format sentiment with bullish/bearish/neutral
        print(f"  Articles      : {sent.get('total', 0)} found "
              f"(🟢 {sent.get('bullish', 0)} / 🔴 {sent.get('bearish', 0)} / ⚪ {sent.get('neutral', 0)})")
    if sent.get("confidence"):
        print(f"  Confidence    : {sent.get('confidence')}")
    print()
    print("  ── West Africa Weather ───────────────")
    print(f"  7d Avg Rain   : {wx.get('avg_7d_rainfall_mm')} mm")
    print(f"  Condition     : {wx.get('overall_condition')}")
    if wx.get("significant_alerts"):
        for alert in wx["significant_alerts"]:
            print(f"  {alert}")
    print()
    print("  ── Seasonal ──────────────────────────")
    print(f"  Season        : {seas.get('current_season')} ({seas.get('month')})")
    if seas.get("seasonal_alert"):
        print(f"  ⚠️  {seas.get('seasonal_alert')}")
    print("=" * 55 + "\n")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    snapshot = gather_all_data()
    save_snapshot(snapshot)
    print_summary(snapshot)
    print(f"📁 Full data saved to '{OUTPUT_FILE}' — ready for AI analysis step.")
