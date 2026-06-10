"""
=============================================================
  COCOA TRADING ASSISTANT — Continuous Learning & Weekly Review
=============================================================
  Three learning mechanisms that run independently of alerts:

  1. SHADOW PREDICTIONS (daily)
     Every daily assessment is recorded and scored at 7/14/21 days,
     regardless of alert level. Gives ~250 scored predictions/year.

  2. BIG-MISS DETECTION (daily)
     If price moved >5% in the past 7 days and the agent's prior
     assessment didn't anticipate it, generates a post-mortem.

  3. WEEKLY REPORT (weekly)
     Structured self-review: accuracy, what changed, what was
     learned, current watchlist status.

  Usage:
    from cocoa_weekly_review import (
        record_shadow_prediction,
        check_big_misses,
        generate_weekly_report,
    )
=============================================================
"""

import os
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

log = logging.getLogger(__name__)

SHADOW_LEDGER_FILE  = os.getenv("SHADOW_LEDGER_FILE", "cocoa_shadow_ledger.json")
POSTMORTEM_FILE     = os.getenv("POSTMORTEM_FILE",     "cocoa_postmortems.json")
WEEKLY_REPORT_FILE  = os.getenv("WEEKLY_REPORT_FILE",  "cocoa_weekly_report.md")
WEEKLY_HISTORY_FILE = os.getenv("WEEKLY_HISTORY_FILE",  "cocoa_weekly_history.json")

BIG_MISS_THRESHOLD_PCT = 5.0   # Price move that triggers post-mortem
WEEKLY_REPORT_DAY      = 6     # Sunday (0=Monday, 6=Sunday)


# ─────────────────────────────────────────────
#  1. SHADOW PREDICTIONS — record everything
# ─────────────────────────────────────────────

def record_shadow_prediction(
    snapshot: dict,
    agent_rec: dict,
    opp_result: dict = None,
) -> dict:
    """
    Record a daily shadow prediction — happens every run regardless
    of alert level.  This is the foundation of continuous learning.

    Includes the full opportunity score breakdown so we can track
    which components are predictive over time.
    """
    price = _get_price(snapshot)
    now = datetime.now(timezone.utc)

    record = {
        "id":           now.strftime("%Y%m%d"),
        "date":         now.strftime("%Y-%m-%d"),
        "price":        price,

        # Agent's assessment
        "assessment":   agent_rec.get("valuation_assessment", "N/A"),
        "fair_low":     agent_rec.get("fair_range_low"),
        "fair_high":    agent_rec.get("fair_range_high"),
        "confidence":   agent_rec.get("valuation_confidence", "N/A"),
        "direction":    agent_rec.get("implied_direction", "NEUTRAL"),
        "timing":       agent_rec.get("timing_action", "N/A"),
        "driver":       agent_rec.get("valuation_driver", ""),
        "factors":      agent_rec.get("factors_cited", []),

        # Opportunity score breakdown
        "opp_score":         opp_result.get("total_score") if opp_result else None,
        "opp_alert_level":   opp_result.get("alert_level") if opp_result else None,
        "opp_components": {
            k: v.get("score") for k, v in opp_result.get("components", {}).items()
        } if opp_result else {},

        # Snapshot context (for post-mortem analysis)
        "cot_percentile":    snapshot.get("cot", {}).get("net_position_percentile"),
        "cot_net":           snapshot.get("cot", {}).get("managed_money_net"),
        "stress_score":      snapshot.get("combined_stress", {}).get("stress_score"),
        "warehouse_signal":  snapshot.get("warehouse_stocks", {}).get("signal"),

        # Evaluation (filled in later by score_shadow_predictions)
        "eval_7d":   {"status": "pending", "price": None, "move_pct": None, "direction_correct": None},
        "eval_14d":  {"status": "pending", "price": None, "move_pct": None, "direction_correct": None},
        "eval_21d":  {"status": "pending", "price": None, "move_pct": None, "direction_correct": None},
    }

    ledger = _load_shadow_ledger()

    # Don't duplicate if already ran today
    existing_ids = {r["id"] for r in ledger}
    if record["id"] in existing_ids:
        # Update today's entry
        ledger = [r for r in ledger if r["id"] != record["id"]]

    ledger.append(record)

    # Keep last 365 days
    cutoff = (now - timedelta(days=365)).strftime("%Y-%m-%d")
    ledger = [r for r in ledger if r.get("date", "") >= cutoff]

    _save_shadow_ledger(ledger)
    log.info(f"  📝 Shadow prediction recorded: {record['assessment']} at {price}")
    return record


def score_shadow_predictions(current_price: float) -> int:
    """
    Score any shadow predictions where enough time has elapsed.
    Called at the start of each daily run.

    Returns number of evaluations scored.
    """
    ledger = _load_shadow_ledger()
    now = datetime.now(timezone.utc)
    scored = 0

    for record in ledger:
        pred_date = record.get("date")
        pred_price = record.get("price")
        if not pred_date or not pred_price:
            continue

        try:
            pred_dt = datetime.strptime(pred_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            continue

        days_elapsed = (now - pred_dt).days

        for horizon, key in [(7, "eval_7d"), (14, "eval_14d"), (21, "eval_21d")]:
            ev = record.get(key, {})
            if ev.get("status") != "pending":
                continue
            if days_elapsed < horizon:
                continue

            move_pct = (current_price - pred_price) / pred_price * 100
            assessment = (record.get("assessment") or "").lower()

            if "overvalued" in assessment:
                direction_correct = move_pct < 0
            elif "undervalued" in assessment:
                direction_correct = move_pct > 0
            elif "fairly" in assessment or "fair" in assessment:
                direction_correct = abs(move_pct) < 3.0
            else:
                direction_correct = None

            # Check if price entered the fair range
            fair_low = record.get("fair_low")
            fair_high = record.get("fair_high")
            in_range = None
            if fair_low is not None and fair_high is not None:
                in_range = fair_low <= current_price <= fair_high

            record[key] = {
                "status": "scored",
                "price": current_price,
                "move_pct": round(move_pct, 2),
                "direction_correct": direction_correct,
                "in_fair_range": in_range,
                "scored_at": now.isoformat(),
            }
            scored += 1

    if scored > 0:
        _save_shadow_ledger(ledger)
        log.info(f"  📊 Scored {scored} shadow prediction horizon(s)")

    return scored


def get_shadow_accuracy(days: int = 30) -> dict:
    """
    Compute accuracy stats from the shadow ledger for the last N days.
    Used to feed into the weekly report and the agent prompt.
    """
    ledger = _load_shadow_ledger()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    recent = [r for r in ledger if r.get("date", "") >= cutoff]

    stats = {}
    for horizon in ["7d", "14d", "21d"]:
        key = f"eval_{horizon}"
        results = []
        for r in recent:
            ev = r.get(key, {})
            if ev.get("status") == "scored" and ev.get("direction_correct") is not None:
                results.append(ev["direction_correct"])

        if results:
            stats[horizon] = {
                "correct": sum(results),
                "total": len(results),
                "accuracy": round(sum(results) / len(results) * 100, 1),
            }

    # Component-level accuracy (which signals actually predict?)
    component_accuracy = {}
    for r in recent:
        ev = r.get("eval_14d", {})
        if ev.get("status") != "scored" or ev.get("direction_correct") is None:
            continue
        correct = ev["direction_correct"]

        # Score by COT regime
        pct = r.get("cot_percentile")
        if pct is not None:
            if pct < 15:
                regime = "cot_extreme_short"
            elif pct < 30:
                regime = "cot_moderate_short"
            elif pct > 85:
                regime = "cot_extreme_long"
            elif pct > 70:
                regime = "cot_moderate_long"
            else:
                regime = "cot_neutral"
            component_accuracy.setdefault(regime, []).append(correct)

        # Score by stress regime
        stress = r.get("stress_score")
        if stress is not None:
            if stress > 60:
                regime = "stress_high"
            elif stress > 40:
                regime = "stress_mixed"
            else:
                regime = "stress_benign"
            component_accuracy.setdefault(regime, []).append(correct)

        # Score by confidence level
        conf = (r.get("confidence") or "").upper()
        if conf in ("HIGH", "MEDIUM", "LOW"):
            component_accuracy.setdefault(f"conf_{conf.lower()}", []).append(correct)

    stats["components"] = {
        k: {
            "correct": sum(v),
            "total": len(v),
            "accuracy": round(sum(v) / len(v) * 100, 1),
        }
        for k, v in component_accuracy.items()
        if len(v) >= 3
    }

    return stats


# ─────────────────────────────────────────────
#  2. BIG-MISS DETECTION
# ─────────────────────────────────────────────

def check_big_misses(current_price: float) -> list:
    """
    Check if the market moved >5% in the past 7 days and the agent's
    assessment from 7 days ago didn't anticipate it.

    Returns a list of post-mortem dicts (empty if no misses).
    """
    ledger = _load_shadow_ledger()
    now = datetime.now(timezone.utc)
    postmortems = []

    # Look at predictions from 5-9 days ago (some flexibility on timing)
    for record in ledger:
        pred_date = record.get("date")
        pred_price = record.get("price")
        if not pred_date or not pred_price:
            continue

        try:
            pred_dt = datetime.strptime(pred_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            continue

        days_ago = (now - pred_dt).days
        if days_ago < 5 or days_ago > 9:
            continue

        move_pct = (current_price - pred_price) / pred_price * 100

        if abs(move_pct) < BIG_MISS_THRESHOLD_PCT:
            continue

        # Was the move anticipated?
        assessment = (record.get("assessment") or "").lower()
        direction = record.get("direction", "NEUTRAL")
        alert_level = record.get("opp_alert_level", "SILENT")

        anticipated = False
        if move_pct > 0 and ("undervalued" in assessment or direction == "BULLISH"):
            anticipated = True
        elif move_pct < 0 and ("overvalued" in assessment or direction == "SHORT"):
            anticipated = True

        if anticipated and alert_level in ("WATCHLIST", "OPPORTUNITY"):
            # Agent got it right and alerted — not a miss
            continue

        # This is a miss — generate post-mortem
        postmortem = {
            "date": now.strftime("%Y-%m-%d"),
            "prediction_date": pred_date,
            "prediction_price": pred_price,
            "current_price": current_price,
            "move_pct": round(move_pct, 2),
            "assessment_was": record.get("assessment"),
            "direction_was": direction,
            "alert_level_was": alert_level,
            "opp_score_was": record.get("opp_score"),
            "components_were": record.get("opp_components", {}),
            "cot_was": record.get("cot_percentile"),
            "stress_was": record.get("stress_score"),
            "driver_cited": record.get("driver"),
            "factors_cited": record.get("factors", []),
            "anticipated": anticipated,
            "type": "anticipated_but_silent" if anticipated else "directional_miss",
        }

        # Build the narrative
        if not anticipated:
            postmortem["narrative"] = (
                f"BIG MISS: Called {record.get('assessment')} at {pred_price:.0f} on {pred_date}, "
                f"but price moved {move_pct:+.1f}% to {current_price:.0f}. "
                f"Direction was wrong. "
                f"COT was at {record.get('cot_percentile', '?')}th percentile, "
                f"stress score was {record.get('stress_score', '?')}/100. "
                f"Driver cited: {record.get('driver', 'N/A')}. "
                f"Which signal was missed or overweighted?"
            )
        else:
            postmortem["narrative"] = (
                f"MISSED ALERT: Correctly assessed {record.get('assessment')} at {pred_price:.0f}, "
                f"price moved {move_pct:+.1f}% to {current_price:.0f} as expected, "
                f"but alert level was only {alert_level} (score: {record.get('opp_score', '?')}). "
                f"Should have alerted. "
                f"Which scoring component was too conservative?"
            )

        postmortems.append(postmortem)
        log.warning(f"  ⚠️ BIG MISS detected: {postmortem['narrative'][:100]}...")

    # Save postmortems
    if postmortems:
        _save_postmortems(postmortems)

    return postmortems


def get_recent_postmortems(days: int = 30) -> list:
    """Load post-mortems from the last N days for the agent prompt."""
    try:
        with open(POSTMORTEM_FILE, "r") as f:
            all_pm = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    return [pm for pm in all_pm if pm.get("date", "") >= cutoff]


# ─────────────────────────────────────────────
#  3. WEEKLY REPORT
# ─────────────────────────────────────────────

def generate_weekly_report(
    current_price: float,
    snapshot: dict,
    opp_result: dict = None,
) -> str:
    """
    Generate the structured weekly self-review report with
    improvement tracking, signal trust, hypothetical performance,
    and readiness assessment.
    """
    now = datetime.now(timezone.utc)
    week_start = now - timedelta(days=7)
    ledger = _load_shadow_ledger()

    # This week's predictions
    week_preds = [
        r for r in ledger
        if r.get("date", "") >= week_start.strftime("%Y-%m-%d")
    ]

    # ── Market this week ──────────────────────────
    first_price = week_preds[0]["price"] if week_preds else None
    last_price = current_price
    week_move = (
        round((last_price - first_price) / first_price * 100, 1)
        if first_price and last_price else None
    )

    cot_start = week_preds[0].get("cot_percentile") if week_preds else None
    cot_end = week_preds[-1].get("cot_percentile") if week_preds else None
    stress_start = week_preds[0].get("stress_score") if week_preds else None
    stress_end = week_preds[-1].get("stress_score") if week_preds else None
    warehouse = snapshot.get("warehouse_stocks", {}).get("signal", "N/A")

    lines = []
    lines.append(f"# 🍫 COCOA WEEKLY SURVEILLANCE REPORT — Week {now.isocalendar()[1]}, {now.year}")
    lines.append(f"**Generated:** {now.strftime('%Y-%m-%d %H:%M')} UTC")
    lines.append("")

    # ── Section 1: Readiness Assessment (traffic light at the top) ──
    readiness = _compute_readiness(ledger)
    light = {"GREEN": "🟢", "AMBER": "🟡", "RED": "🔴"}.get(readiness["level"], "⚪")
    lines.append(f"## {light} AGENT READINESS: {readiness['level']}")
    for note in readiness["notes"]:
        lines.append(f"  - {note}")
    if readiness.get("weeks_to_green"):
        lines.append(f"  - Estimated weeks to GREEN: {readiness['weeks_to_green']}")
    lines.append("")

    # ── Section 2: Market this week ───────────────
    lines.append("## 📈 MARKET THIS WEEK")
    if week_move is not None:
        direction = "📈" if week_move > 0 else "📉" if week_move < 0 else "➡️"
        lines.append(f"Price: {first_price:.0f} → {last_price:.0f} ({week_move:+.1f}%) {direction}")
    if cot_start is not None and cot_end is not None:
        cot_dir = "↑ shorts covering" if cot_end > cot_start else "↓ more shorting" if cot_end < cot_start else "→ stable"
        lines.append(f"COT: {cot_start:.0f}th → {cot_end:.0f}th percentile ({cot_dir})")
    if stress_start is not None and stress_end is not None:
        lines.append(f"Crop stress: {stress_start}/100 → {stress_end}/100")
    lines.append(f"Warehouse: {warehouse}")
    lines.append("")

    # ── Section 3: Rolling Accuracy Trend ─────────
    lines.append("## 📈 ACCURACY TREND")
    acc_1w  = get_shadow_accuracy(days=7)
    acc_4w  = get_shadow_accuracy(days=28)
    acc_12w = get_shadow_accuracy(days=84)

    # Format the comparison table
    lines.append("```")
    lines.append(f"{'':>16} {'This week':>12} {'4-week avg':>12} {'12-week avg':>12}")
    for horizon in ["7d", "14d", "21d"]:
        w1  = acc_1w.get(horizon, {}).get("accuracy", "—")
        w4  = acc_4w.get(horizon, {}).get("accuracy", "—")
        w12 = acc_12w.get(horizon, {}).get("accuracy", "—")
        w1_str  = f"{w1}%" if isinstance(w1, (int, float)) else w1
        w4_str  = f"{w4}%" if isinstance(w4, (int, float)) else w4
        w12_str = f"{w12}%" if isinstance(w12, (int, float)) else w12
        n1 = acc_1w.get(horizon, {}).get("total", 0)
        n4 = acc_4w.get(horizon, {}).get("total", 0)
        n12 = acc_12w.get(horizon, {}).get("total", 0)
        lines.append(f"Direction {horizon}:  {w1_str:>10} (n={n1}) {w4_str:>6} (n={n4}) {w12_str:>6} (n={n12})")
    lines.append("```")

    # Detect improvement or decline
    for horizon in ["14d", "21d"]:
        a4 = acc_4w.get(horizon, {}).get("accuracy")
        a12 = acc_12w.get(horizon, {}).get("accuracy")
        if a4 is not None and a12 is not None:
            diff = a4 - a12
            if diff > 8:
                lines.append(f"Trend: IMPROVING ↑ — {horizon} accuracy up {diff:.0f}pts over 12 weeks")
            elif diff < -8:
                lines.append(f"Trend: DECLINING ↓ — {horizon} accuracy down {abs(diff):.0f}pts over 12 weeks")
            else:
                lines.append(f"Trend: STABLE → — {horizon} accuracy within ±8pts")
            break
    lines.append("")

    # ── Section 4: Signal Trust Scores ────────────
    lines.append("## 🔬 SIGNAL TRUST (14d accuracy by regime)")
    comps = acc_12w.get("components", {})
    if comps:
        sorted_comps = sorted(comps.items(), key=lambda x: x[1]["accuracy"], reverse=True)
        for regime, stats in sorted_comps:
            emoji = "✅" if stats["accuracy"] >= 60 else "⚠️" if stats["accuracy"] >= 45 else "❌"
            label = regime.replace("_", " ").title()
            lines.append(f"  {emoji} {label}: {stats['accuracy']}% ({stats['correct']}/{stats['total']})")

        # Key insight
        if sorted_comps:
            best = sorted_comps[0]
            worst = sorted_comps[-1]
            lines.append("")
            lines.append(
                f"Key: {best[0].replace('_', ' ').title()} is most reliable "
                f"({best[1]['accuracy']}%). "
                f"{worst[0].replace('_', ' ').title()} is least reliable "
                f"({worst[1]['accuracy']}%)."
            )
    else:
        lines.append("  Not enough data yet — need 3+ scored predictions per regime.")
    lines.append("")

    # ── Section 5: Hypothetical Performance ───────
    lines.append("## 📊 HYPOTHETICAL PERFORMANCE")
    hypo = _compute_hypothetical_performance(ledger)
    if hypo["total_alerts"] > 0:
        lines.append(f"Total alerts sent: {hypo['total_alerts']} "
                     f"({hypo['watchlist_count']} WATCHLIST, {hypo['opportunity_count']} OPPORTUNITY)")
        lines.append(f"Direction correct: {hypo['correct']}/{hypo['total_alerts']} "
                     f"({hypo['win_rate']:.0f}%)")
        if hypo["avg_move"] is not None:
            lines.append(f"Avg move in alert direction at 14d: {hypo['avg_move']:+.1f}%")
        if hypo["best_trade"]:
            lines.append(f"Best trade: {hypo['best_trade']}")
        if hypo["worst_trade"]:
            lines.append(f"Worst trade: {hypo['worst_trade']}")
        if hypo["hypothetical_return"] is not None:
            lines.append(f"Hypothetical return (equal weight per alert): {hypo['hypothetical_return']:+.1f}%")
        lines.append(f"Alerts in last 7 days: {hypo['alerts_this_week']}")
    else:
        lines.append("No WATCHLIST or OPPORTUNITY alerts sent yet.")
        lines.append("Hypothetical tracking will activate after the first alert.")
    lines.append("")

    # ── Section 6: Big Misses ─────────────────────
    postmortems = get_recent_postmortems(days=7)
    if postmortems:
        lines.append("## ⚠️ BIG MISSES THIS WEEK")
        for pm in postmortems:
            lines.append(f"- {pm.get('narrative', 'N/A')}")
        lines.append("")
    else:
        lines.append("## ✅ NO BIG MISSES THIS WEEK")
        lines.append("")

    # ── Section 7: What I Learned ─────────────────
    lines.append("## 🔍 WHAT I LEARNED")

    moves = []
    for r in week_preds:
        ev = r.get("eval_7d", {})
        if ev.get("status") == "scored" and ev.get("move_pct") is not None:
            moves.append(ev["move_pct"])
    if moves:
        avg_move = sum(moves) / len(moves)
        if avg_move > 2:
            lines.append(f"- Market had a bullish bias this week (avg {avg_move:+.1f}%)")
        elif avg_move < -2:
            lines.append(f"- Market had a bearish bias this week (avg {avg_move:+.1f}%)")

    assessments = [r.get("assessment") for r in week_preds if r.get("assessment")]
    if len(set(assessments)) > 1:
        lines.append(f"- My view shifted this week: {' → '.join(assessments)}")
    elif assessments:
        lines.append(f"- Consistent view all week: {assessments[0]}")

    fair_ranges = [(r.get("fair_low"), r.get("fair_high")) for r in week_preds
                   if r.get("fair_low") is not None]
    if fair_ranges:
        avg_low = sum(f[0] for f in fair_ranges) / len(fair_ranges)
        avg_high = sum(f[1] for f in fair_ranges) / len(fair_ranges)
        lines.append(f"- Average fair range this week: {avg_low:.0f}–{avg_high:.0f}")

    if postmortems:
        for pm in postmortems:
            if pm.get("type") == "directional_miss":
                lines.append(f"- LESSON: {pm.get('driver_cited', 'Unknown')} — this signal led me wrong")
            elif pm.get("type") == "anticipated_but_silent":
                lines.append(f"- LESSON: Had the right view but score too low to alert — review thresholds")

    if not moves and not assessments and not fair_ranges and not postmortems:
        lines.append("- Not enough scored predictions yet to extract lessons.")
    lines.append("")

    # ── Section 8: Current Watchlist ──────────────
    lines.append("## 📡 WATCHLIST STATUS")
    if opp_result:
        lines.append(f"Score: {opp_result['total_score']}/100 ({opp_result['alert_level']})")
        lines.append(f"Direction: {opp_result.get('direction', 'NONE')}")
        for comp_name, comp_data in opp_result.get("components", {}).items():
            lines.append(f"  {comp_name}: {comp_data.get('score', '?')}/100 — "
                         f"{comp_data.get('rationale', '')[:80]}")
    else:
        lines.append("No current opportunity assessment available.")
    lines.append("")

    lines.append("---")
    lines.append("*Automated weekly self-review. Not financial advice.*")

    report = "\n".join(lines)

    # Save
    with open(WEEKLY_REPORT_FILE, "w") as f:
        f.write(report)

    _save_weekly_history(now, acc_4w, postmortems, opp_result, week_move, readiness, hypo)

    log.info(f"  📋 Weekly report generated: {WEEKLY_REPORT_FILE}")
    return report


# ─────────────────────────────────────────────
#  READINESS ASSESSMENT
# ─────────────────────────────────────────────

GREEN_THRESHOLD_14D  = 62   # 14d accuracy needed for GREEN
AMBER_THRESHOLD_14D  = 48   # Below this = RED
MIN_PREDICTIONS      = 15   # Need at least this many scored predictions

def _compute_readiness(ledger: list) -> dict:
    """
    Traffic light readiness assessment.
    GREEN = trust the alerts, act on OPPORTUNITY signals.
    AMBER = improving but not proven enough — observe but don't commit fully.
    RED = accuracy declining or too low — do not act, investigate.
    """
    acc_4w  = get_shadow_accuracy(days=28)
    acc_12w = get_shadow_accuracy(days=84)

    notes = []
    level = "AMBER"  # default

    # Check if we have enough data
    total_scored = sum(
        acc_12w.get(h, {}).get("total", 0) for h in ["7d", "14d", "21d"]
    )
    if total_scored < MIN_PREDICTIONS:
        notes.append(f"Only {total_scored} scored predictions — need {MIN_PREDICTIONS}+ for reliable assessment")
        return {"level": "AMBER", "notes": notes, "weeks_to_green": None}

    # Primary metric: 14d accuracy (most meaningful horizon)
    acc_14d_4w = acc_4w.get("14d", {}).get("accuracy")
    acc_14d_12w = acc_12w.get("14d", {}).get("accuracy")

    # Fall back to 7d if 14d not available
    if acc_14d_4w is None:
        acc_14d_4w = acc_4w.get("7d", {}).get("accuracy")
        acc_14d_12w = acc_12w.get("7d", {}).get("accuracy")

    if acc_14d_4w is None:
        notes.append("No accuracy data available yet")
        return {"level": "AMBER", "notes": notes, "weeks_to_green": None}

    # Determine level
    if acc_14d_4w >= GREEN_THRESHOLD_14D:
        level = "GREEN"
        notes.append(f"14d accuracy at {acc_14d_4w:.0f}% (target: {GREEN_THRESHOLD_14D}%+)")
        notes.append("Agent has demonstrated sufficient accuracy — act on OPPORTUNITY alerts")
    elif acc_14d_4w < AMBER_THRESHOLD_14D:
        level = "RED"
        notes.append(f"14d accuracy at {acc_14d_4w:.0f}% (below {AMBER_THRESHOLD_14D}% threshold)")
        notes.append("Accuracy too low — do NOT act on alerts. Investigate what's going wrong.")
    else:
        level = "AMBER"
        notes.append(f"14d accuracy at {acc_14d_4w:.0f}% (target: {GREEN_THRESHOLD_14D}% for GREEN)")
        gap = GREEN_THRESHOLD_14D - acc_14d_4w
        notes.append(f"Need {gap:.0f}pts improvement to reach GREEN")

    # Check trend
    if acc_14d_12w is not None and acc_14d_4w is not None:
        trend = acc_14d_4w - acc_14d_12w
        if trend > 5:
            notes.append(f"Improving: +{trend:.0f}pts over 12 weeks")
        elif trend < -5:
            notes.append(f"Declining: {trend:.0f}pts over 12 weeks")
            if level == "GREEN":
                level = "AMBER"
                notes.append("Downgraded to AMBER due to declining trend")

    # Check component reliability
    comps = acc_12w.get("components", {})
    strong = [k for k, v in comps.items() if v["accuracy"] >= 65 and v["total"] >= 5]
    weak = [k for k, v in comps.items() if v["accuracy"] < 40 and v["total"] >= 5]
    if strong:
        notes.append(f"Strong signals: {', '.join(s.replace('_', ' ').title() for s in strong)}")
    if weak:
        notes.append(f"Weak signals: {', '.join(w.replace('_', ' ').title() for w in weak)}")

    # Estimate weeks to green
    weeks_to_green = None
    if level != "GREEN" and acc_14d_12w is not None and acc_14d_4w is not None:
        improvement_rate = (acc_14d_4w - acc_14d_12w) / 12  # pts per week
        if improvement_rate > 0.5:
            gap = GREEN_THRESHOLD_14D - acc_14d_4w
            weeks_to_green = max(1, round(gap / improvement_rate))

    return {"level": level, "notes": notes, "weeks_to_green": weeks_to_green}


# ─────────────────────────────────────────────
#  HYPOTHETICAL PERFORMANCE
# ─────────────────────────────────────────────

def _compute_hypothetical_performance(ledger: list) -> dict:
    """
    Track what would have happened if you'd acted on every
    WATCHLIST and OPPORTUNITY alert.
    """
    now = datetime.now(timezone.utc)
    week_ago = (now - timedelta(days=7)).strftime("%Y-%m-%d")

    alerts = [
        r for r in ledger
        if r.get("opp_alert_level") in ("WATCHLIST", "OPPORTUNITY")
    ]

    if not alerts:
        return {
            "total_alerts": 0, "watchlist_count": 0, "opportunity_count": 0,
            "correct": 0, "win_rate": 0, "avg_move": None,
            "best_trade": None, "worst_trade": None,
            "hypothetical_return": None, "alerts_this_week": 0,
        }

    watchlist_count = sum(1 for a in alerts if a.get("opp_alert_level") == "WATCHLIST")
    opportunity_count = sum(1 for a in alerts if a.get("opp_alert_level") == "OPPORTUNITY")

    # Score each alert — use 14d evaluation
    scored_alerts = []
    for a in alerts:
        ev = a.get("eval_14d", {})
        if ev.get("status") != "scored":
            ev = a.get("eval_7d", {})
        if ev.get("status") != "scored":
            continue

        direction = a.get("direction", "NEUTRAL")
        move_pct = ev.get("move_pct", 0)

        # Was the trade profitable in the predicted direction?
        if direction in ("LONG", "BULLISH") and "undervalued" in (a.get("assessment") or "").lower():
            trade_pnl = move_pct  # long: profit if price went up
        elif direction in ("SHORT",) and "overvalued" in (a.get("assessment") or "").lower():
            trade_pnl = -move_pct  # short: profit if price went down
        else:
            trade_pnl = 0

        scored_alerts.append({
            "date": a.get("date"),
            "direction": direction,
            "move_pct": move_pct,
            "trade_pnl": round(trade_pnl, 2),
            "correct": ev.get("direction_correct", False),
            "alert_level": a.get("opp_alert_level"),
        })

    if not scored_alerts:
        alerts_this_week = sum(1 for a in alerts if a.get("date", "") >= week_ago)
        return {
            "total_alerts": len(alerts), "watchlist_count": watchlist_count,
            "opportunity_count": opportunity_count,
            "correct": 0, "win_rate": 0, "avg_move": None,
            "best_trade": None, "worst_trade": None,
            "hypothetical_return": None, "alerts_this_week": alerts_this_week,
        }

    correct = sum(1 for s in scored_alerts if s["correct"])
    win_rate = correct / len(scored_alerts) * 100
    avg_pnl = sum(s["trade_pnl"] for s in scored_alerts) / len(scored_alerts)

    # Cumulative return (compounded)
    cumulative = 1.0
    for s in scored_alerts:
        cumulative *= (1 + s["trade_pnl"] / 100)
    hypothetical_return = round((cumulative - 1) * 100, 1)

    # Best and worst
    best = max(scored_alerts, key=lambda s: s["trade_pnl"])
    worst = min(scored_alerts, key=lambda s: s["trade_pnl"])

    best_str = f"{best['trade_pnl']:+.1f}% ({best['date']} — {best['direction'].lower()})"
    worst_str = f"{worst['trade_pnl']:+.1f}% ({worst['date']} — {worst['direction'].lower()})"

    alerts_this_week = sum(1 for a in alerts if a.get("date", "") >= week_ago)

    return {
        "total_alerts": len(alerts),
        "watchlist_count": watchlist_count,
        "opportunity_count": opportunity_count,
        "correct": correct,
        "win_rate": win_rate,
        "avg_move": round(avg_pnl, 1),
        "best_trade": best_str,
        "worst_trade": worst_str,
        "hypothetical_return": hypothetical_return,
        "alerts_this_week": alerts_this_week,
    }


def should_generate_weekly_report() -> bool:
    """Check if today is the designated weekly report day."""
    now = datetime.now(timezone.utc)
    if now.weekday() != WEEKLY_REPORT_DAY:
        return False

    # Check we haven't already generated one today
    try:
        with open(WEEKLY_HISTORY_FILE, "r") as f:
            history = json.load(f)
        if history:
            last_date = history[-1].get("date", "")
            if last_date == now.strftime("%Y-%m-%d"):
                return False
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    return True


# ─────────────────────────────────────────────
#  BUILD LEARNING PROMPT (injected daily)
# ─────────────────────────────────────────────

def build_learning_prompt() -> str:
    """
    Build a prompt block with continuous learning insights.
    Injected into the daily analysis prompt alongside the
    existing feedback block.
    """
    lines = []

    # Shadow accuracy (last 30 days)
    accuracy = get_shadow_accuracy(days=30)
    if accuracy:
        lines.append("## CONTINUOUS LEARNING (shadow prediction accuracy, last 30 days)")
        for horizon in ["7d", "14d", "21d"]:
            if horizon in accuracy:
                a = accuracy[horizon]
                lines.append(f"  {horizon}: {a['correct']}/{a['total']} ({a['accuracy']}%)")

        comps = accuracy.get("components", {})
        if comps:
            lines.append("")
            lines.append("Signal reliability by regime:")
            for regime, stats in sorted(comps.items(), key=lambda x: x[1]["accuracy"], reverse=True):
                if stats["total"] >= 3:
                    emoji = "✅" if stats["accuracy"] >= 60 else "⚠️" if stats["accuracy"] >= 45 else "❌"
                    label = regime.replace("_", " ").title()
                    lines.append(f"  {emoji} {label}: {stats['accuracy']}% ({stats['correct']}/{stats['total']})")
        lines.append("")

    # Recent post-mortems (last 14 days)
    postmortems = get_recent_postmortems(days=14)
    if postmortems:
        lines.append("RECENT BIG MISSES (learn from these):")
        for pm in postmortems[-3:]:
            lines.append(f"  - {pm.get('narrative', '')[:150]}")
        lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────
#  PERSISTENCE
# ─────────────────────────────────────────────

def _load_shadow_ledger() -> list:
    try:
        with open(SHADOW_LEDGER_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save_shadow_ledger(ledger: list):
    tmp = SHADOW_LEDGER_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ledger, f, indent=2, default=str)
    os.replace(tmp, SHADOW_LEDGER_FILE)


def _save_postmortems(new_pms: list):
    try:
        with open(POSTMORTEM_FILE, "r") as f:
            existing = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing = []

    existing.extend(new_pms)
    # Keep last 180 days
    cutoff = (datetime.now(timezone.utc) - timedelta(days=180)).strftime("%Y-%m-%d")
    existing = [pm for pm in existing if pm.get("date", "") >= cutoff]

    with open(POSTMORTEM_FILE, "w") as f:
        json.dump(existing, f, indent=2, default=str)


def _save_weekly_history(now, accuracy, postmortems, opp_result, week_move,
                        readiness=None, hypo=None):
    try:
        with open(WEEKLY_HISTORY_FILE, "r") as f:
            history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        history = []

    entry = {
        "date": now.strftime("%Y-%m-%d"),
        "week": now.isocalendar()[1],
        "accuracy": accuracy,
        "big_misses": len(postmortems),
        "opp_score": opp_result.get("total_score") if opp_result else None,
        "week_move_pct": week_move,
        "readiness": readiness.get("level") if readiness else None,
        "hypothetical_return": hypo.get("hypothetical_return") if hypo else None,
    }

    history.append(entry)
    history = history[-52:]  # Keep 1 year

    with open(WEEKLY_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def _get_price(snapshot: dict) -> Optional[float]:
    px = snapshot.get("technicals", {}).get("price", {})
    if px.get("current"):
        return float(px["current"])
    return None
