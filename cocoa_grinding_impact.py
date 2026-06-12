"""
=============================================================
  COCOA GRINDING RELEASE IMPACT MODEL
=============================================================
  Analyses historical grinding releases and their price impact
  to forecast the likely price reaction to upcoming releases.

  Methodology:
    1. Historical dataset: ECA/NCA/CAA releases with YoY% actuals
       and CC=F price T-1, T0, T+1, T+3, T+5 around each release
    2. Buckets releases by surprise magnitude vs prior trajectory
    3. Calculates average/median/range of price reactions per bucket
    4. For the next release: given current trajectory, estimates
       expected surprise bucket and likely price impact range

  Run standalone:
    python cocoa_grinding_impact.py

  Or imported:
    from cocoa_grinding_impact import compute_grinding_impact
=============================================================
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from statistics import mean, median, stdev

log = logging.getLogger(__name__)

IMPACT_CACHE_FILE = "grinding_impact_history.json"

# ─────────────────────────────────────────────────────────────────
#  HISTORICAL DATABASE
#  Format: release date, region, quarter, yoy_pct (actual),
#          consensus_yoy (market expectation going in),
#          price_t1 (day before), price_t0 (release day close),
#          price_t1p (1 day after), price_t3 (3 days after),
#          price_t5 (5 days after)
#
#  Prices are CC=F (ICE NY futures, USD/tonne) — raw, not normalised.
#  Price reactions are computed as % change from T-1.
#
#  consensus_yoy: analyst/market estimate going into the release.
#  If unknown, use the prior quarter's YoY as a rough proxy.
#  "surprise" = actual_yoy - consensus_yoy
# ─────────────────────────────────────────────────────────────────

HISTORICAL_RELEASES = [
    # ── 2026 ────────────────────────────────────────────────────────
    {
        "date":          "2026-04-16",
        "region":        "ECA",
        "quarter":       "Q1 2026",
        "actual_yoy":    -7.8,
        "consensus_yoy": -5.0,    # going-in consensus per UPCOMING_RELEASES at the time
        "volume":        325852,
        "price_t1":      None,    # auto-filled from CC=F on next run
        "price_t0":      None,
        "price_t1p":     None,
        "price_t3":      None,
        "price_t5":      None,
        "notes":         "ECA Q1 2026: 325,852t, -7.8% YoY. One of the softest Q1s "
                         "in a decade; no demand recovery despite ~70% price fall "
                         "from late-2024 peak. Rolling 12m total ~1.297Mt.",
        "price_filled":  False,
    },
    {
        "date":          "2026-04-16",
        "region":        "NCA",
        "quarter":       "Q1 2026",
        "actual_yoy":    -3.8,
        "consensus_yoy": -2.0,    # expectations ranged slight contraction to modest growth
        "volume":        106087,
        "price_t1":      None,
        "price_t0":      None,
        "price_t1p":     None,
        "price_t3":      None,
        "price_t5":      None,
        "notes":         "NCA Q1 2026: 106,087t, -3.8% YoY. Lower end of expectations "
                         "despite 15 reporting plants vs 14 prior year.",
        "price_filled":  False,
    },
    # ── 2025 ────────────────────────────────────────────────────────
    {
        "date":          "2026-01-15",
        "region":        "ECA",
        "quarter":       "Q4 2025",
        "actual_yoy":    -8.3,
        "consensus_yoy": -5.0,   # market expected ~-5% given prior trend
        "volume":        304470,
        "price_t1":      None,   # fill in from CC=F if needed
        "price_t0":      None,
        "price_t1p":     None,
        "price_t3":      None,
        "price_t5":      None,
        "notes":         "ECA Q4 2025. Worst annual grind since 2015. Germany -9.9%.",
        "price_filled":  False,
    },
    {
        "date":          "2025-10-16",
        "region":        "ECA",
        "quarter":       "Q3 2025",
        "actual_yoy":    -4.8,
        "consensus_yoy": -3.0,
        "volume":        None,
        "price_t1":      None,
        "price_t0":      None,
        "price_t1p":     None,
        "price_t3":      None,
        "price_t5":      None,
        "notes":         "ECA Q3 2025",
        "price_filled":  False,
    },
    {
        "date":          "2025-07-17",
        "region":        "ECA",
        "quarter":       "Q2 2025",
        "actual_yoy":    -3.2,
        "consensus_yoy": -2.0,
        "volume":        None,
        "price_t1":      None,
        "price_t0":      None,
        "price_t1p":     None,
        "price_t3":      None,
        "price_t5":      None,
        "notes":         "ECA Q2 2025",
        "price_filled":  False,
    },
    {
        "date":          "2025-04-10",
        "region":        "ECA",
        "quarter":       "Q1 2025",
        "actual_yoy":    -6.4,
        "consensus_yoy": -4.0,
        "volume":        None,
        "price_t1":      None,
        "price_t0":      None,
        "price_t1p":     None,
        "price_t3":      None,
        "price_t5":      None,
        "notes":         "ECA Q1 2025 — first quarter showing full demand destruction",
        "price_filled":  False,
    },
    # ── 2024 ────────────────────────────────────────────────────────
    {
        "date":          "2025-01-16",
        "region":        "ECA",
        "quarter":       "Q4 2024",
        "actual_yoy":    -7.2,
        "consensus_yoy": -3.5,
        "volume":        None,
        "price_t1":      None,
        "price_t0":      None,
        "price_t1p":     None,
        "price_t3":      None,
        "price_t5":      None,
        "notes":         "ECA Q4 2024 — significant demand destruction visible",
        "price_filled":  False,
    },
    {
        "date":          "2024-10-17",
        "region":        "ECA",
        "quarter":       "Q3 2024",
        "actual_yoy":    -5.1,
        "consensus_yoy": -2.0,
        "volume":        None,
        "price_t1":      None,
        "price_t0":      None,
        "price_t1p":     None,
        "price_t3":      None,
        "price_t5":      None,
        "notes":         "ECA Q3 2024 — price environment still elevated",
        "price_filled":  False,
    },
    {
        "date":          "2024-07-18",
        "region":        "ECA",
        "quarter":       "Q2 2024",
        "actual_yoy":    -1.8,
        "consensus_yoy": 0.5,
        "volume":        None,
        "price_t1":      None,
        "price_t0":      None,
        "price_t1p":     None,
        "price_t3":      None,
        "price_t5":      None,
        "notes":         "ECA Q2 2024",
        "price_filled":  False,
    },
    {
        "date":          "2024-04-11",
        "region":        "ECA",
        "quarter":       "Q1 2024",
        "actual_yoy":    2.1,
        "consensus_yoy": 1.0,
        "volume":        None,
        "price_t1":      None,
        "price_t0":      None,
        "price_t1p":     None,
        "price_t3":      None,
        "price_t5":      None,
        "notes":         "ECA Q1 2024 — still positive pre-demand destruction",
        "price_filled":  False,
    },
    {
        "date":          "2024-01-18",
        "region":        "ECA",
        "quarter":       "Q4 2023",
        "actual_yoy":    -0.9,
        "consensus_yoy": 1.5,
        "volume":        None,
        "price_t1":      None,
        "price_t0":      None,
        "price_t1p":     None,
        "price_t3":      None,
        "price_t5":      None,
        "notes":         "ECA Q4 2023 — first miss vs positive expectations",
        "price_filled":  False,
    },
    # ── 2023 ────────────────────────────────────────────────────────
    {
        "date":          "2023-10-19",
        "region":        "ECA",
        "quarter":       "Q3 2023",
        "actual_yoy":    3.4,
        "consensus_yoy": 2.0,
        "volume":        None,
        "price_t1":      None,
        "price_t0":      None,
        "price_t1p":     None,
        "price_t3":      None,
        "price_t5":      None,
        "notes":         "ECA Q3 2023 — positive grind, supply shortage narrative emerging",
        "price_filled":  False,
    },
    {
        "date":          "2023-07-20",
        "region":        "ECA",
        "quarter":       "Q2 2023",
        "actual_yoy":    1.2,
        "consensus_yoy": 1.5,
        "volume":        None,
        "price_t1":      None,
        "price_t0":      None,
        "price_t1p":     None,
        "price_t3":      None,
        "price_t5":      None,
        "notes":         "ECA Q2 2023 — slight miss",
        "price_filled":  False,
    },
    {
        "date":          "2023-04-13",
        "region":        "ECA",
        "quarter":       "Q1 2023",
        "actual_yoy":    2.8,
        "consensus_yoy": 2.0,
        "volume":        None,
        "price_t1":      None,
        "price_t0":      None,
        "price_t1p":     None,
        "price_t3":      None,
        "price_t5":      None,
        "notes":         "ECA Q1 2023",
        "price_filled":  False,
    },
]

# Upcoming releases to forecast
UPCOMING_RELEASES = [
    {
        "date":          "2026-07-16",   # ECA-confirmed (provisional) Q2 2026 date
        "region":        "ECA",
        "quarter":       "Q2 2026",
        "consensus_yoy": -7.8,    # PLACEHOLDER: prior quarter YoY as proxy — refine nearer the date
        "bull_case_yoy": -4.0,    # demand stabilisation at lower prices begins to show
        "bear_case_yoy": -11.0,   # demand destruction deepens despite price collapse
        "notes":         "Q2 2026 ECA. Key question: with prices ~70% off the 2024 peak "
                         "for two full quarters, does grind finally find a floor? "
                         "Q2 2025 comp was weak (-7.2%), so the YoY hurdle is low.",
    },
    {
        "date":          "2026-07-16",
        "region":        "NCA",
        "quarter":       "Q2 2026",
        "consensus_yoy": -3.8,    # PLACEHOLDER: prior quarter YoY as proxy — refine nearer the date
        "bull_case_yoy":  0.0,
        "bear_case_yoy": -7.0,
        "notes":         "Q2 2026 US grindings. Released same day as ECA. "
                         "Q2 2025 comp was -2.78%.",
    },
]


# ─────────────────────────────────────────────
#  PRICE FETCHING
# ─────────────────────────────────────────────

def fetch_prices_around_date(date_str: str, ticker: str = "CC=F") -> dict:
    """
    Fetch CC=F closing prices for T-1, T0, T+1, T+3, T+5 around a release date.
    Returns dict with keys: t1, t0, t1p, t3, t5 (all USD/tonne closes).
    """
    try:
        import yfinance as yf
        release_date = datetime.strptime(date_str, "%Y-%m-%d")
        start = (release_date - timedelta(days=10)).strftime("%Y-%m-%d")
        end   = (release_date + timedelta(days=10)).strftime("%Y-%m-%d")

        df = yf.download(ticker, start=start, end=end, interval="1d",
                         auto_adjust=True, progress=False)
        if df.empty:
            return {}

        closes = df["Close"].squeeze()
        dates  = [d.strftime("%Y-%m-%d") for d in closes.index]

        # Find the release date index (or nearest trading day)
        target = release_date.strftime("%Y-%m-%d")
        if target not in dates:
            # Find nearest date on or after
            future = [d for d in dates if d >= target]
            if not future:
                return {}
            target = future[0]

        idx = dates.index(target)

        def price_at(offset: int) -> float | None:
            i = idx + offset
            if 0 <= i < len(dates):
                return round(float(closes.iloc[i]), 2)
            return None

        return {
            "t1":  price_at(-1),  # day before release
            "t0":  price_at(0),   # release day close
            "t1p": price_at(1),   # 1 day after
            "t3":  price_at(3),   # 3 days after
            "t5":  price_at(5),   # 5 days after
        }

    except Exception as e:
        log.warning(f"  Could not fetch prices for {date_str}: {e}")
        return {}


def fill_price_history(releases: list, force: bool = False) -> list:
    """
    Fill in price data for historical releases that are missing it.
    Skips releases already filled unless force=True.
    Only fills releases older than 10 days (to have complete T+5 data).
    """
    today = datetime.now(timezone.utc).date()
    filled_count = 0

    for r in releases:
        if r.get("price_filled") and not force:
            continue

        release_date = datetime.strptime(r["date"], "%Y-%m-%d").date()
        if (today - release_date).days < 10:
            continue  # too recent for T+5

        prices = fetch_prices_around_date(r["date"])
        if prices:
            r.update(prices)
            r["price_filled"] = True
            filled_count += 1
            log.info(f"  Filled prices for {r['region']} {r['quarter']}: "
                     f"T-1={prices.get('t1')}, T0={prices.get('t0')}, "
                     f"T+5={prices.get('t5')}")

    if filled_count:
        log.info(f"  Filled price data for {filled_count} releases")
    return releases


# ─────────────────────────────────────────────
#  IMPACT CALCULATION
# ─────────────────────────────────────────────

def compute_price_reactions(releases: list) -> list:
    """
    For each release with price data, compute % price changes
    from T-1 to T0, T+1, T+3, T+5.
    Also computes the 'surprise' = actual_yoy - consensus_yoy.
    """
    enriched = []
    for r in releases:
        if not r.get("t1") or not r.get("t0"):
            continue

        t1  = r["t1"]
        surprise = (r.get("actual_yoy", 0) or 0) - (r.get("consensus_yoy", 0) or 0)

        def pct(t_price):
            if t_price is None or t1 == 0:
                return None
            return round((t_price - t1) / t1 * 100, 2)

        enriched.append({
            **r,
            "surprise":       round(surprise, 2),
            "reaction_t0":    pct(r.get("t0")),
            "reaction_t1p":   pct(r.get("t1p")),
            "reaction_t3":    pct(r.get("t3")),
            "reaction_t5":    pct(r.get("t5")),
        })
    return enriched


def bucket_label(surprise: float) -> str:
    """Classify a surprise value into a descriptive bucket."""
    if surprise <= -5:
        return "large_miss"       # much worse than expected (bearish)
    elif surprise <= -2:
        return "moderate_miss"    # worse than expected
    elif surprise < 2:
        return "in_line"          # roughly as expected
    elif surprise < 5:
        return "moderate_beat"    # better than expected (bullish)
    else:
        return "large_beat"       # much better than expected


BUCKET_DESCRIPTIONS = {
    "large_miss":     "Much worse than expected (actual YoY >5pp below consensus) — historically most bearish",
    "moderate_miss":  "Worse than expected (2–5pp below consensus)",
    "in_line":        "Roughly in line with consensus (within ±2pp)",
    "moderate_beat":  "Better than expected (2–5pp above consensus)",
    "large_beat":     "Much better than expected (>5pp above consensus) — historically most bullish",
}


def compute_bucket_stats(enriched: list) -> dict:
    """
    For each surprise bucket, compute average/median/range of
    price reactions at T0, T+1, T+3, T+5.
    """
    buckets = {}
    for r in enriched:
        b = bucket_label(r["surprise"])
        if b not in buckets:
            buckets[b] = {"releases": [], "t0": [], "t1p": [], "t3": [], "t5": []}
        buckets[b]["releases"].append(r)
        for horizon in ["t0", "t1p", "t3", "t5"]:
            val = r.get(f"reaction_{horizon}")
            if val is not None:
                buckets[b][horizon].append(val)

    stats = {}
    for b, data in buckets.items():
        s = {"count": len(data["releases"]), "description": BUCKET_DESCRIPTIONS[b]}
        for horizon in ["t0", "t1p", "t3", "t5"]:
            vals = data[horizon]
            if len(vals) >= 2:
                s[f"{horizon}_avg"]    = round(mean(vals), 2)
                s[f"{horizon}_median"] = round(median(vals), 2)
                s[f"{horizon}_stdev"]  = round(stdev(vals), 2)
                s[f"{horizon}_range"]  = [round(min(vals), 2), round(max(vals), 2)]
            elif len(vals) == 1:
                s[f"{horizon}_avg"]    = vals[0]
                s[f"{horizon}_median"] = vals[0]
                s[f"{horizon}_range"]  = [vals[0], vals[0]]
        stats[b] = s
    return stats


# ─────────────────────────────────────────────
#  FORECAST
# ─────────────────────────────────────────────

def forecast_impact(upcoming: dict, bucket_stats: dict,
                    current_price_gbp: float = None) -> dict:
    """
    Given an upcoming release and historical bucket stats,
    produce a structured forecast of likely price impact.
    """
    consensus = upcoming.get("consensus_yoy", 0)
    bull      = upcoming.get("bull_case_yoy", consensus + 3)
    bear      = upcoming.get("bear_case_yoy", consensus - 3)

    # Determine scenario buckets
    # Base = consensus — surprise vs itself is 0 = in_line
    # Bull case = beat, bear case = miss
    bull_surprise  = bull  - consensus
    bear_surprise  = bear  - consensus
    base_bucket    = bucket_label(0)                    # in_line
    bull_bucket    = bucket_label(bull_surprise)
    bear_bucket    = bucket_label(bear_surprise)

    def get_stats(bucket, horizon="t5"):
        s = bucket_stats.get(bucket, {})
        return {
            "avg":    s.get(f"{horizon}_avg"),
            "median": s.get(f"{horizon}_median"),
            "range":  s.get(f"{horizon}_range"),
            "stdev":  s.get(f"{horizon}_stdev"),
            "count":  s.get("count", 0),
        }

    forecast = {
        "release":        f"{upcoming['region']} {upcoming['quarter']}",
        "date":           upcoming["date"],
        "days_away":      (datetime.strptime(upcoming["date"], "%Y-%m-%d").date()
                          - datetime.now(timezone.utc).date()).days,
        "consensus_yoy":  consensus,
        "notes":          upcoming.get("notes", ""),
        "scenarios": {
            "base": {
                "label":       "Base case (in line with consensus)",
                "assumed_yoy": consensus,
                "surprise":    0.0,
                "bucket":      base_bucket,
                "impact_t0":   get_stats(base_bucket, "t0"),
                "impact_t5":   get_stats(base_bucket, "t5"),
            },
            "bull": {
                "label":       f"Bull case (better than expected: {bull:+.1f}% YoY)",
                "assumed_yoy": bull,
                "surprise":    round(bull_surprise, 1),
                "bucket":      bull_bucket,
                "impact_t0":   get_stats(bull_bucket, "t0"),
                "impact_t5":   get_stats(bull_bucket, "t5"),
            },
            "bear": {
                "label":       f"Bear case (worse than expected: {bear:+.1f}% YoY)",
                "assumed_yoy": bear,
                "surprise":    round(bear_surprise, 1),
                "bucket":      bear_bucket,
                "impact_t0":   get_stats(bear_bucket, "t0"),
                "impact_t5":   get_stats(bear_bucket, "t5"),
            },
        },
    }

    # Add GBP price impact estimate if current price available
    if current_price_gbp:
        for scenario_key, scenario in forecast["scenarios"].items():
            t5_avg = scenario["impact_t5"].get("avg")
            t5_range = scenario["impact_t5"].get("range")
            if t5_avg is not None:
                scenario["gbp_impact_t5_avg"] = round(current_price_gbp * t5_avg / 100, 0)
            if t5_range:
                scenario["gbp_impact_t5_range"] = [
                    round(current_price_gbp * t5_range[0] / 100, 0),
                    round(current_price_gbp * t5_range[1] / 100, 0),
                ]

    return forecast


# ─────────────────────────────────────────────
#  PERSISTENCE
# ─────────────────────────────────────────────

def load_impact_history() -> list:
    try:
        with open(IMPACT_CACHE_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def save_impact_history(releases: list):
    with open(IMPACT_CACHE_FILE, "w") as f:
        json.dump(releases, f, indent=2, default=str)


# ─────────────────────────────────────────────
#  MAIN ENTRY POINT
# ─────────────────────────────────────────────

def compute_grinding_impact(current_price_gbp: float = None,
                             fill_prices: bool = True) -> dict:
    """
    Full pipeline:
      1. Load / fill price history for historical releases
      2. Compute surprise and price reactions
      3. Build bucket stats
      4. Forecast upcoming releases
      5. Return structured dict for snapshot injection

    current_price_gbp: today's CMC mid price — used to convert
                       % impact estimates into GBP/tonne terms.
    """
    log.info("Computing grinding release impact model...")

    # Load any previously saved price data, merge with hardcoded releases
    cached = {r["date"] + r["region"]: r for r in load_impact_history()}
    releases = []
    for r in HISTORICAL_RELEASES:
        key = r["date"] + r["region"]
        if key in cached:
            releases.append(cached[key])   # use cached (may have prices filled)
        else:
            releases.append(dict(r))

    # Fill price data from yfinance for unfilled releases
    if fill_prices:
        releases = fill_price_history(releases)
        save_impact_history(releases)

    # Compute reactions and bucket stats
    enriched     = compute_price_reactions(releases)
    bucket_stats = compute_bucket_stats(enriched)

    filled_count = sum(1 for r in enriched if r.get("reaction_t5") is not None)
    log.info(f"  {filled_count}/{len(releases)} releases have complete price data")

    # Forecast each upcoming release — but only ones still in the future.
    # Past-dated entries mean UPCOMING_RELEASES needs a manual update
    # (move them to HISTORICAL_RELEASES with actuals, add next quarter).
    today_d = datetime.now(timezone.utc).date()
    future_releases, past_releases = [], []
    for upcoming in UPCOMING_RELEASES:
        try:
            rel_date = datetime.strptime(upcoming["date"], "%Y-%m-%d").date()
        except (KeyError, ValueError):
            continue
        (future_releases if rel_date >= today_d else past_releases).append(upcoming)

    if past_releases:
        log.warning(
            f"  ⚠️ {len(past_releases)} UPCOMING_RELEASES entries are past-dated "
            f"({', '.join(r['date'] for r in past_releases)}) — "
            "move them to HISTORICAL_RELEASES with actuals and add the next quarter."
        )

    forecasts = []
    for upcoming in future_releases:
        fc = forecast_impact(upcoming, bucket_stats, current_price_gbp)
        forecasts.append(fc)
        days = fc["days_away"]
        log.info(f"  ✅ Forecast: {fc['release']} in {days} days | "
                 f"consensus={upcoming['consensus_yoy']:+.1f}% YoY")

    result = {
        "generated_at":   datetime.now(timezone.utc).isoformat(),
        "data_note":      (
            f"{filled_count} of {len(releases)} historical ECA releases have "
            "price data. Bucket stats improve as more data accumulates."
        ),
        "bucket_stats":    bucket_stats,
        "forecasts":       forecasts,
        "historical_count": len(enriched),
        "stale_upcoming":  len(past_releases),
        "needs_schedule_update": bool(past_releases) or not future_releases,
    }

    if result["needs_schedule_update"]:
        result["maintenance_note"] = (
            "UPCOMING_RELEASES in cocoa_grinding_impact.py is out of date — "
            "no future release is configured. Forecasts are disabled until "
            "the next ECA/NCA release dates and consensus are added."
        )

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    result = compute_grinding_impact(current_price_gbp=2200.0, fill_prices=True)

    print("\n" + "=" * 60)
    print("  GRINDING RELEASE IMPACT MODEL")
    print("=" * 60)
    print(f"  Data: {result['data_note']}")

    print("\n  BUCKET STATISTICS (historical price reactions from T-1):")
    for bucket, stats in result["bucket_stats"].items():
        if stats["count"] == 0:
            continue
        print(f"\n  [{bucket.upper()}] n={stats['count']} — {stats['description']}")
        for h in ["t0", "t1p", "t5"]:
            avg   = stats.get(f"{h}_avg")
            rng   = stats.get(f"{h}_range")
            if avg is not None:
                print(f"    T+{'0' if h=='t0' else h[1:]:3s}: avg={avg:+.1f}%  range=[{rng[0]:+.1f}%, {rng[1]:+.1f}%]")

    print("\n  UPCOMING RELEASE FORECASTS:")
    for fc in result["forecasts"]:
        print(f"\n  {fc['release']} — {fc['date']} ({fc['days_away']} days away)")
        print(f"  Consensus: {fc['consensus_yoy']:+.1f}% YoY")
        print(f"  {fc['notes'][:120]}")
        for skey, s in fc["scenarios"].items():
            t5 = s["impact_t5"]
            gbp = s.get("gbp_impact_t5_avg", "N/A")
            avg = t5.get("avg", "N/A")
            rng = t5.get("range")
            rng_str = f"[{rng[0]:+.1f}%, {rng[1]:+.1f}%]" if rng else "insufficient data"
            print(f"    {skey.upper():5s}: assumed {s['assumed_yoy']:+.1f}% YoY → "
                  f"T+5 avg={avg if avg=='N/A' else f'{avg:+.1f}%':>7s}  "
                  f"range={rng_str}  "
                  f"(~£{gbp}/t)")
