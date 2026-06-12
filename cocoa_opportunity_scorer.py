"""
=============================================================
  COCOA TRADING ASSISTANT — Opportunity Scoring Framework
=============================================================
  Evaluates whether current market conditions represent an
  actionable trading opportunity, scoring across five dimensions:

    1. Valuation Gap     — how far is price from estimated fair value?
    2. Positioning        — does COT data amplify or conflict?
    3. Catalyst Proximity — is there an identifiable trigger?
    4. Satellite Signal   — does crop health data confirm the thesis?
    5. Track Record       — has the agent been accurate in similar setups?

  Output:
    - Opportunity score (0–100)
    - Alert level: SILENT / MONITOR / WATCHLIST / OPPORTUNITY
    - Recommended action and rationale

  Usage:
    from cocoa_opportunity_scorer import score_opportunity
    result = score_opportunity(snapshot, agent_recommendation, feedback_stats)

=============================================================
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  ALERT THRESHOLDS
# ─────────────────────────────────────────────

ALERT_LEVELS = {
    "SILENT":       (0,  39),    # Log internally, no notification
    "MONITOR":      (40, 64),    # Daily log entry, no notification
    "WATCHLIST":    (65, 79),    # Logged + surfaced in weekly review, NO Telegram
    "OPPORTUNITY":  (80, 100),   # Telegram full report (subject to cooldown)
}

# Component weights (must sum to 100)
WEIGHTS = {
    "valuation_gap":     30,
    "positioning":       25,
    "catalyst":          15,
    "satellite":         15,
    "track_record":      15,
}


# ─────────────────────────────────────────────
#  MAIN SCORING FUNCTION
# ─────────────────────────────────────────────

def score_opportunity(
    snapshot: dict,
    agent_rec: dict,
    feedback_stats: dict = None,
) -> dict:
    """
    Score the current market setup across all five dimensions.

    Args:
        snapshot:       The daily data snapshot (from cocoa_data_gatherer)
        agent_rec:      The parsed recommendation from Claude's analysis
                        (valuation_bias + timing_signal from extract_recommendation)
        feedback_stats: Optional feedback accuracy data (from build_feedback_prompt)

    Returns:
        dict with:
          - total_score (0-100)
          - alert_level (SILENT/MONITOR/WATCHLIST/OPPORTUNITY)
          - components (per-dimension scores and rationale)
          - direction (LONG/SHORT/NONE)
          - summary (one-line description)
    """
    components = {}

    # ── 1. Valuation Gap ─────────────────────────────────────
    components["valuation_gap"] = _score_valuation_gap(snapshot, agent_rec)

    # ── 2. Speculative Positioning (COT) ─────────────────────
    components["positioning"] = _score_positioning(snapshot, agent_rec)

    # ── 3. Catalyst Proximity ────────────────────────────────
    components["catalyst"] = _score_catalyst(snapshot)

    # ── 4. Satellite / Crop Health ───────────────────────────
    components["satellite"] = _score_satellite(snapshot, agent_rec)

    # ── 5. Track Record ──────────────────────────────────────
    components["track_record"] = _score_track_record(feedback_stats, agent_rec)

    # ── Compute weighted total ───────────────────────────────
    total = 0
    for key, weight in WEIGHTS.items():
        component_score = components[key]["score"]
        total += component_score * (weight / 100)

    total = round(min(100, max(0, total)), 1)

    # ── Determine direction ──────────────────────────────────
    assessment = (agent_rec.get("valuation_assessment") or "").lower()
    if "overvalued" in assessment:
        direction = "SHORT"
    elif "undervalued" in assessment:
        direction = "LONG"
    else:
        direction = "NONE"

    # ── Determine alert level ────────────────────────────────
    alert_level = "SILENT"
    for level, (lo, hi) in ALERT_LEVELS.items():
        if lo <= total <= hi:
            alert_level = level
            break

    # ── Build summary ────────────────────────────────────────
    summary = _build_summary(total, alert_level, direction, components, snapshot)

    result = {
        "total_score":    total,
        "alert_level":    alert_level,
        "direction":      direction,
        "components":     components,
        "summary":        summary,
        "scored_at":      datetime.now(timezone.utc).isoformat(),
    }

    log.info(
        f"  🎯 Opportunity score: {total}/100 → {alert_level} "
        f"({direction}) — {summary}"
    )

    return result


# ─────────────────────────────────────────────
#  COMPONENT 1: VALUATION GAP
# ─────────────────────────────────────────────

def _score_valuation_gap(snapshot: dict, rec: dict) -> dict:
    """
    Score based on the gap between current price and estimated fair value.

    Scoring:
      0-3% gap   → 0-10   (noise, not tradeable)
      3-7% gap   → 10-40  (mild mispricing)
      7-12% gap  → 40-70  (meaningful gap)
      12-20% gap → 70-90  (significant opportunity)
      >20% gap   → 90-100 (extreme dislocation)
    """
    current_price = _get_price(snapshot)
    fair_low = rec.get("fair_range_low")
    fair_high = rec.get("fair_range_high")

    if not current_price or fair_low is None or fair_high is None:
        return {"score": 0, "rationale": "Insufficient data to compute valuation gap"}

    fair_mid = (fair_low + fair_high) / 2
    gap_pct = abs(current_price - fair_mid) / fair_mid * 100

    # Is price above or below fair value?
    if current_price > fair_high:
        gap_direction = "above"
    elif current_price < fair_low:
        gap_direction = "below"
    else:
        gap_direction = "within"

    # Score the gap magnitude
    if gap_pct < 3:
        score = gap_pct * (10 / 3)
    elif gap_pct < 7:
        score = 10 + (gap_pct - 3) * (30 / 4)
    elif gap_pct < 12:
        score = 40 + (gap_pct - 7) * (30 / 5)
    elif gap_pct < 20:
        score = 70 + (gap_pct - 12) * (20 / 8)
    else:
        score = min(100, 90 + (gap_pct - 20) * (10 / 10))

    # Confidence adjustment — scale down if agent confidence is LOW
    confidence = (rec.get("valuation_confidence") or "").upper()
    if confidence == "LOW":
        score *= 0.6
    elif confidence == "MEDIUM":
        score *= 0.85

    score = round(min(100, score), 1)

    return {
        "score": score,
        "gap_pct": round(gap_pct, 1),
        "gap_direction": gap_direction,
        "fair_mid": round(fair_mid, 0),
        "current_price": current_price,
        "confidence": confidence,
        "rationale": (
            f"Price {current_price:.0f} is {gap_pct:.1f}% {gap_direction} "
            f"fair value ({fair_low:.0f}–{fair_high:.0f}). "
            f"Confidence: {confidence}."
        ),
    }


# ─────────────────────────────────────────────
#  COMPONENT 2: SPECULATIVE POSITIONING
# ─────────────────────────────────────────────

def _score_positioning(snapshot: dict, rec: dict) -> dict:
    """
    Score based on COT positioning data and whether it aligns with
    or conflicts with the valuation thesis.

    Key insight: positioning doesn't just add conviction —
    MISALIGNED positioning is the highest-value signal because
    it means the market is set up for a violent move when the
    thesis plays out.

    Scoring:
      Thesis is SHORT + specs extremely LONG  → 90-100 (liquidation fuel)
      Thesis is LONG  + specs extremely SHORT → 90-100 (squeeze fuel)
      Thesis is SHORT + specs extremely SHORT → 10-20  (who's left to sell?)
      Thesis is LONG  + specs extremely LONG  → 10-20  (who's left to buy?)
      Neutral positioning                     → 30-40  (no amplifier)
    """
    cot = snapshot.get("cot", {})
    if "error" in cot or not cot:
        return {"score": 30, "rationale": "COT data unavailable — neutral score"}

    percentile = cot.get("net_position_percentile")
    net_pos = cot.get("managed_money_net")

    if percentile is None:
        return {
            "score": 30,
            "net_position": net_pos,
            "rationale": "COT percentile not yet available (building history)"
        }

    # Score based on alignment/misalignment
    specs_long = percentile > 70
    specs_short = percentile < 30
    specs_extreme_long = percentile > 85
    specs_extreme_short = percentile < 15

    # Determine thesis direction
    assessment = (rec.get("valuation_assessment") or "").lower()
    if "overvalued" in assessment:
        thesis = "SHORT"
    elif "undervalued" in assessment:
        thesis = "LONG"
    else:
        # No directional thesis, but extreme positioning is still noteworthy
        if specs_extreme_short:
            score = 55
            rationale = (
                f"No directional thesis, but COT at {percentile:.0f}th percentile "
                f"is EXTREMELY SHORT — any bullish catalyst could trigger a violent "
                f"short squeeze. Monitor closely for a thesis to form."
            )
        elif specs_extreme_long:
            score = 55
            rationale = (
                f"No directional thesis, but COT at {percentile:.0f}th percentile "
                f"is EXTREMELY LONG — any bearish catalyst could trigger liquidation. "
                f"Monitor closely for a thesis to form."
            )
        else:
            score = 30
            rationale = f"No directional thesis — COT at {percentile:.0f}th percentile"

        return {
            "score": score,
            "percentile": percentile,
            "net_position": net_pos,
            "rationale": rationale,
        }

    if thesis == "SHORT":
        if specs_extreme_long:
            # Best setup: overvalued + everyone is long = liquidation bomb
            score = 90 + (percentile - 85) * (10 / 15)
            rationale = (
                f"IDEAL SETUP: Market overvalued + specs extremely long "
                f"({percentile:.0f}th pctile). Liquidation risk is severe — "
                f"any bearish catalyst triggers cascading selling."
            )
        elif specs_long:
            score = 65 + (percentile - 70) * (25 / 15)
            rationale = f"Specs moderately long ({percentile:.0f}th pctile) — supports short thesis."
        elif specs_extreme_short:
            # Worst setup: overvalued but everyone already short
            score = 10
            rationale = (
                f"CONFLICTING: Market overvalued but specs already extremely short "
                f"({percentile:.0f}th pctile). Short squeeze risk high — limited "
                f"downside even if thesis is correct."
            )
        elif specs_short:
            score = 20
            rationale = f"Specs already short ({percentile:.0f}th pctile) — limited fuel for further downside."
        else:
            score = 40
            rationale = f"Specs neutral ({percentile:.0f}th pctile) — no positioning amplifier."

    else:  # thesis == "LONG"
        if specs_extreme_short:
            # Best setup: undervalued + everyone is short = squeeze bomb
            score = 90 + (15 - percentile) * (10 / 15)
            rationale = (
                f"IDEAL SETUP: Market undervalued + specs extremely short "
                f"({percentile:.0f}th pctile). Short squeeze risk is severe — "
                f"any bullish catalyst triggers violent covering."
            )
        elif specs_short:
            score = 65 + (30 - percentile) * (25 / 15)
            rationale = f"Specs moderately short ({percentile:.0f}th pctile) — supports long thesis."
        elif specs_extreme_long:
            score = 10
            rationale = (
                f"CONFLICTING: Market undervalued but specs already extremely long "
                f"({percentile:.0f}th pctile). Who's left to buy?"
            )
        elif specs_long:
            score = 20
            rationale = f"Specs already long ({percentile:.0f}th pctile) — limited fuel for further upside."
        else:
            score = 40
            rationale = f"Specs neutral ({percentile:.0f}th pctile) — no positioning amplifier."

    return {
        "score": round(min(100, max(0, score)), 1),
        "percentile": percentile,
        "net_position": net_pos,
        "thesis": thesis,
        "rationale": rationale,
    }


# ─────────────────────────────────────────────
#  COMPONENT 3: CATALYST PROXIMITY
# ─────────────────────────────────────────────

def _score_catalyst(snapshot: dict) -> dict:
    """
    Score based on whether an identifiable catalyst is approaching.

    A mispricing without a catalyst can persist indefinitely.
    A mispricing WITH an imminent catalyst is actionable.

    Catalysts scored:
      - Grinding data release (ECA/NCA) — highest impact
      - Crop survey season (mid-crop or main-crop assessment window)
      - ICCO quarterly bulletin
      - Weather events during sensitive crop phases
    """
    now = datetime.now(timezone.utc)
    catalysts = []
    max_score = 0

    # Check grinding releases
    grinding = snapshot.get("grinding_data", {})
    for label, date_str in grinding.get("next_releases", {}).items():
        try:
            release_date = datetime.strptime(date_str, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
            days_away = (release_date - now).days
            if -3 <= days_away <= 30:
                # Score: highest when 3-10 days away (enough time to position)
                if days_away < 0:
                    # Just released — could still be moving price
                    cat_score = 70
                elif days_away <= 3:
                    cat_score = 85  # Imminent — high urgency
                elif days_away <= 10:
                    cat_score = 95  # Sweet spot — time to position
                elif days_away <= 20:
                    cat_score = 60
                else:
                    cat_score = 40

                catalysts.append({
                    "name": label,
                    "days_away": days_away,
                    "score": cat_score,
                    "type": "grinding_release",
                })
                max_score = max(max_score, cat_score)
        except (ValueError, TypeError):
            continue

    # Check crop season sensitivity
    seasonal = snapshot.get("seasonal", {})
    season = (seasonal.get("current_season") or "").lower()

    if "mid crop" in season:
        # Mid-crop flowering/pod development — surveys imminent
        catalysts.append({
            "name": "Mid-crop survey window",
            "days_away": None,
            "score": 55,
            "type": "crop_survey",
        })
        max_score = max(max_score, 55)

    # Check for weather stress as a catalyst
    stress = snapshot.get("combined_stress", {})
    stress_score = stress.get("stress_score")
    if stress_score is not None and stress_score > 60:
        catalysts.append({
            "name": f"Crop stress signal ({stress_score}/100)",
            "days_away": None,
            "score": 65,
            "type": "weather_stress",
        })
        max_score = max(max_score, 65)

    if not catalysts:
        return {
            "score": 15,
            "catalysts": [],
            "rationale": "No identifiable catalyst within 30 days. Mispricing may persist."
        }

    # Use the strongest catalyst as the score
    best = max(catalysts, key=lambda c: c["score"])
    return {
        "score": round(max_score, 1),
        "catalysts": catalysts,
        "best_catalyst": best["name"],
        "rationale": (
            f"Primary catalyst: {best['name']}"
            + (f" in {best['days_away']} days" if best.get("days_away") is not None else "")
            + f". {len(catalysts)} catalyst(s) identified."
        ),
    }


# ─────────────────────────────────────────────
#  COMPONENT 4: SATELLITE / CROP HEALTH
# ─────────────────────────────────────────────

def _score_satellite(snapshot: dict, rec: dict) -> dict:
    """
    Score based on whether satellite data confirms or contradicts
    the valuation thesis.

    The satellite signal is most valuable when it DIVERGES from
    the market's current pricing — that's an information edge.
    """
    crop = snapshot.get("crop_health", {})
    stress = snapshot.get("combined_stress", {})

    if not crop and not stress:
        return {"score": 30, "rationale": "No satellite data available — neutral score"}

    # Get the stress score and bias
    stress_score = stress.get("stress_score")  # 0-100, higher = more stress
    stress_bias = (stress.get("bias") or "").upper()

    assessment = (rec.get("valuation_assessment") or "").lower()

    if stress_score is None:
        return {"score": 30, "rationale": "Stress signal insufficient — neutral score"}

    # Check for LAI anomalies (strongest early warning)
    lai_warning = False
    regions = crop.get("regions", [])
    if isinstance(regions, list):
        for r in regions:
            lai = r.get("lai")
            if lai is not None and lai < 2.5:
                lai_warning = True
                break

    # Check for rainfall anomaly
    rain_warning = False
    if isinstance(regions, list):
        for r in regions:
            rain_anom = r.get("rainfall_anomaly_pct")
            if rain_anom is not None and rain_anom < -30:
                rain_warning = True
                break

    # Check for LST anomaly
    lst_warning = False
    if isinstance(regions, list):
        for r in regions:
            lst_anom = r.get("lst_anomaly_c")
            if lst_anom is not None and lst_anom > 2.0:
                lst_warning = True
                break

    # Score based on alignment with thesis
    # BULLISH thesis (undervalued) needs crop STRESS to confirm
    # BEARISH thesis (overvalued) needs crop HEALTH to confirm
    if "undervalued" in assessment:
        # We think price should be higher — does satellite agree?
        if stress_score > 60:
            score = 70 + min(30, (stress_score - 60) * 0.75)
            rationale = (
                f"CONFIRMS BULLISH: Crop stress at {stress_score}/100 supports "
                f"higher prices."
            )
        elif stress_score > 40:
            score = 40
            rationale = f"Mixed crop signal ({stress_score}/100) — inconclusive."
        else:
            score = 15
            rationale = (
                f"CONTRADICTS BULLISH: Crop health is good ({stress_score}/100). "
                f"No supply concern to support higher prices."
            )
    elif "overvalued" in assessment:
        # We think price should be lower — crop health confirms oversupply
        if stress_score < 30:
            score = 70
            rationale = (
                f"CONFIRMS BEARISH: Healthy crop ({stress_score}/100) supports "
                f"surplus narrative and lower prices."
            )
        elif stress_score < 50:
            score = 45
            rationale = f"Mostly healthy crop ({stress_score}/100) — mild support for bearish view."
        else:
            score = 15
            rationale = (
                f"CONTRADICTS BEARISH: Crop stress at {stress_score}/100 — "
                f"supply risk undermines the overvalued thesis."
            )
    else:
        score = 30
        rationale = f"No directional thesis to confirm. Stress score: {stress_score}/100."

    # Bonus for early warning signals (LAI, rainfall, LST anomalies)
    warnings = []
    if lai_warning:
        score = min(100, score + 15)
        warnings.append("LAI below 2.5 (canopy structural stress)")
    if rain_warning:
        score = min(100, score + 10)
        warnings.append("Rainfall >30% below normal")
    if lst_warning:
        score = min(100, score + 10)
        warnings.append("LST >2°C above normal (heat stress)")

    if warnings:
        rationale += " Early warnings: " + "; ".join(warnings) + "."

    return {
        "score": round(min(100, score), 1),
        "stress_score": stress_score,
        "lai_warning": lai_warning,
        "rain_warning": rain_warning,
        "lst_warning": lst_warning,
        "rationale": rationale,
    }


# ─────────────────────────────────────────────
#  COMPONENT 5: TRACK RECORD
# ─────────────────────────────────────────────

def _score_track_record(feedback_stats: dict, rec: dict) -> dict:
    """
    Score based on how accurate the agent has been in similar setups.

    This is the self-correcting mechanism: if the agent has been
    consistently wrong, the opportunity score is suppressed even
    if other components look strong. Conversely, if the agent has
    a good track record in this type of setup, conviction increases.
    """
    if not feedback_stats:
        return {
            "score": 50,  # Neutral — no track record yet
            "rationale": "No feedback data yet — using neutral score."
        }

    # Extract accuracy at different horizons
    dir_7d = feedback_stats.get("direction_accuracy_7d")
    dir_14d = feedback_stats.get("direction_accuracy_14d")
    dir_21d = feedback_stats.get("direction_accuracy_21d")

    # Use the best horizon accuracy (the agent may be right but slow)
    accuracies = [a for a in [dir_7d, dir_14d, dir_21d] if a is not None]
    if not accuracies:
        return {"score": 50, "rationale": "No scored predictions yet."}

    best_accuracy = max(accuracies)
    best_horizon = (
        "21d" if dir_21d == best_accuracy else
        "14d" if dir_14d == best_accuracy else "7d"
    )

    # Score: 50% accuracy = coin flip = 30 points
    # 65%+ = good = 70+ points
    # 80%+ = excellent = 90+ points
    # <40% = worse than random = 10 points
    if best_accuracy >= 80:
        score = 90 + (best_accuracy - 80) * 0.5
    elif best_accuracy >= 65:
        score = 70 + (best_accuracy - 65) * (20 / 15)
    elif best_accuracy >= 50:
        score = 30 + (best_accuracy - 50) * (40 / 15)
    elif best_accuracy >= 40:
        score = 15 + (best_accuracy - 40) * (15 / 10)
    else:
        score = max(5, best_accuracy * 0.375)

    # Confidence adjustment: if the agent states HIGH confidence
    # but its HIGH confidence calls are less accurate, penalise
    confidence = (rec.get("valuation_confidence") or "").upper()
    if confidence == "HIGH" and best_accuracy < 55:
        score *= 0.7
        rationale_suffix = " ⚠️ HIGH confidence stated but track record is poor — downgrading."
    else:
        rationale_suffix = ""

    return {
        "score": round(min(100, max(0, score)), 1),
        "best_accuracy": best_accuracy,
        "best_horizon": best_horizon,
        "rationale": (
            f"Best accuracy: {best_accuracy:.0f}% at {best_horizon} horizon. "
            f"Based on {feedback_stats.get('predictions_scored', '?')} scored predictions."
            + rationale_suffix
        ),
    }


# ─────────────────────────────────────────────
#  SUMMARY BUILDER
# ─────────────────────────────────────────────

def _build_summary(
    total: float,
    alert_level: str,
    direction: str,
    components: dict,
    snapshot: dict,
) -> str:
    """Build a one-line summary for logging and alerts."""
    price = _get_price(snapshot)
    price_str = f" at {price:.0f}" if price else ""

    vg = components["valuation_gap"]
    pos = components["positioning"]

    if alert_level == "SILENT":
        return f"No actionable opportunity{price_str}. Score {total}/100."

    elif alert_level == "MONITOR":
        return (
            f"Monitoring: {direction} bias{price_str}, "
            f"{vg.get('gap_pct', '?')}% gap. "
            f"COT {pos.get('percentile', '?')}th pctile. Score {total}/100."
        )

    elif alert_level == "WATCHLIST":
        best_cat = components["catalyst"].get("best_catalyst", "no specific catalyst")
        return (
            f"⚠️ WATCHLIST: {direction}{price_str}, "
            f"{vg.get('gap_pct', '?')}% gap, "
            f"COT {pos.get('percentile', '?')}th pctile. "
            f"Catalyst: {best_cat}. Score {total}/100."
        )

    else:  # OPPORTUNITY
        best_cat = components["catalyst"].get("best_catalyst", "imminent catalyst")
        return (
            f"🔔 OPPORTUNITY: {direction}{price_str}, "
            f"{vg.get('gap_pct', '?')}% gap, "
            f"COT {pos.get('percentile', '?')}th pctile, "
            f"{best_cat}. Score {total}/100 — FULL REPORT."
        )


# ─────────────────────────────────────────────
#  ALERT MESSAGE BUILDERS
# ─────────────────────────────────────────────

def format_watchlist_alert(result: dict, snapshot: dict) -> str:
    """Format a concise watchlist alert for Telegram."""
    c = result["components"]
    price = _get_price(snapshot)
    gbp = snapshot.get("price_gbp")
    gbp_str = f" (≈{gbp:.0f} GBP)" if gbp else ""

    msg = f"""🟡 COCOA WATCHLIST ALERT

Score: {result['total_score']}/100 — Conditions building
Direction: {result['direction']}
Price: {price:.0f} USD/t{gbp_str}

Valuation: {c['valuation_gap'].get('gap_pct', '?')}% gap ({c['valuation_gap'].get('gap_direction', '?')} fair value)
COT: {c['positioning'].get('percentile', '?')}th percentile (MM net: {c['positioning'].get('net_position', '?'):+,})
Catalyst: {c['catalyst'].get('best_catalyst', 'None identified')}
Crop: {c['satellite'].get('rationale', 'N/A')[:100]}

Not yet actionable — monitoring for trigger."""

    return msg


def format_opportunity_alert(result: dict, report: str, snapshot: dict) -> str:
    """Format a full opportunity alert for Telegram."""
    price = _get_price(snapshot)
    gbp = snapshot.get("price_gbp")
    gbp_str = f" (≈{gbp:.0f} GBP)" if gbp else ""

    header = f"""🔴 COCOA OPPORTUNITY ALERT

Score: {result['total_score']}/100 — High conviction setup
Direction: {result['direction']}
Price: {price:.0f} USD/t{gbp_str}

"""
    return header + report


# ─────────────────────────────────────────────
#  OPPORTUNITY LOG (persisted between runs)
# ─────────────────────────────────────────────

OPPORTUNITY_LOG_FILE = "cocoa_opportunity_log.json"


def log_opportunity(result: dict, snapshot: dict):
    """Append the opportunity score to a persistent log."""
    entry = {
        "scored_at":    result["scored_at"],
        "total_score":  result["total_score"],
        "alert_level":  result["alert_level"],
        "direction":    result["direction"],
        "summary":      result["summary"],
        "price":        _get_price(snapshot),
        "components": {
            k: {"score": v["score"], "rationale": v.get("rationale", "")}
            for k, v in result["components"].items()
        },
    }

    try:
        with open(OPPORTUNITY_LOG_FILE, "r") as f:
            log_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        log_data = []

    log_data.append(entry)

    # Keep last 180 days
    cutoff = (datetime.now(timezone.utc) - timedelta(days=180)).isoformat()
    log_data = [e for e in log_data if e.get("scored_at", "") >= cutoff]

    with open(OPPORTUNITY_LOG_FILE, "w") as f:
        json.dump(log_data, f, indent=2)


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def _get_price(snapshot: dict) -> Optional[float]:
    """Extract current price from snapshot."""
    px = snapshot.get("technicals", {}).get("price", {})
    if px.get("current"):
        return float(px["current"])
    return None


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cocoa Opportunity Scorer")
    parser.add_argument("--log", action="store_true", help="Show opportunity log")
    args = parser.parse_args()

    if args.log:
        try:
            with open(OPPORTUNITY_LOG_FILE, "r") as f:
                entries = json.load(f)
            print(f"\n{'Date':<22} {'Score':>5} {'Level':<12} {'Dir':<6} Summary")
            print("-" * 90)
            for e in entries[-20:]:
                print(
                    f"{e['scored_at'][:19]:<22} "
                    f"{e['total_score']:>5.1f} "
                    f"{e['alert_level']:<12} "
                    f"{e['direction']:<6} "
                    f"{e.get('summary', '')[:50]}"
                )
        except FileNotFoundError:
            print("No opportunity log yet.")
