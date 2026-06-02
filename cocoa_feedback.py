"""
=============================================================
  COCOA TRADING ASSISTANT — Feedback & Learning Module
=============================================================
  Tracks predictions, scores them against actual outcomes,
  and generates accuracy summaries that feed back into the
  agent prompt so it can self-correct over time.

  Components:
    1. Prediction Ledger  — stores each recommendation
    2. Evaluation Engine   — scores past predictions once
                             enough time has elapsed
    3. Accuracy Summariser — builds a prompt-ready feedback
                             block from scored predictions
    4. Factor Tracker      — tracks which analytical factors
                             the model cited and their hit rate

  Storage:
    - cocoa_prediction_ledger.json  (all predictions + scores)
    - cocoa_feedback_summary.json   (latest prompt-ready summary)

  Usage:
    # After generating a report — record the prediction
    from cocoa_feedback import record_prediction, evaluate_pending, build_feedback_prompt

    record_prediction(report_text, snapshot, parsed_rec)
    evaluate_pending()  # scores any predictions that are now due
    feedback_block = build_feedback_prompt()  # string for the agent prompt

  SETUP:
    pip install pandas  (only dependency beyond stdlib)
=============================================================
"""

import os
import json
import re
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

LEDGER_FILE           = os.getenv("PREDICTION_LEDGER_FILE", "cocoa_prediction_ledger.json")
FEEDBACK_SUMMARY_FILE = os.getenv("FEEDBACK_SUMMARY_FILE",  "cocoa_feedback_summary.json")

# How many days to wait before evaluating at each horizon
EVAL_HORIZONS = [7, 14, 21]

# How many recent scored predictions to include in the feedback prompt
FEEDBACK_WINDOW_DAYS = 90
MAX_FEEDBACK_PREDICTIONS = 30


# ─────────────────────────────────────────────
#  LEDGER: LOAD / SAVE
# ─────────────────────────────────────────────

def _load_ledger() -> list:
    """Load the prediction ledger from disk."""
    try:
        with open(LEDGER_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError as e:
        log.error(f"Ledger file corrupted: {e} — starting fresh backup")
        # Back up the corrupted file
        backup = LEDGER_FILE + f".corrupt.{datetime.now().strftime('%Y%m%d%H%M%S')}"
        try:
            os.rename(LEDGER_FILE, backup)
        except OSError:
            pass
        return []


def _save_ledger(ledger: list):
    """Persist the ledger to disk atomically."""
    tmp = LEDGER_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ledger, f, indent=2, default=str)
    os.replace(tmp, LEDGER_FILE)


# ─────────────────────────────────────────────
#  RECORD A NEW PREDICTION
# ─────────────────────────────────────────────

def record_prediction(report_text: str, snapshot: dict, parsed_rec: dict = None) -> dict:
    """
    Extract the prediction from today's report and store it in the ledger.

    Args:
        report_text:  The full markdown report from Claude
        snapshot:     The data snapshot used to generate the report
        parsed_rec:   Optional pre-parsed recommendation dict
                      (from extract_recommendation below, or cocoa_agent's extractor)

    Returns:
        The prediction record that was stored.
    """
    if parsed_rec is None:
        parsed_rec = extract_recommendation(report_text)

    # Get the price at time of prediction
    price_now = _get_current_price(snapshot)

    record = {
        "id":                 datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        "generated_at":       snapshot.get("generated_at", datetime.now(timezone.utc).isoformat()),
        "price_at_prediction": price_now,

        # ── Valuation Bias ──────────────────────────────────
        "valuation_bias": {
            "assessment":     parsed_rec.get("valuation_assessment", "N/A"),
            "magnitude":      parsed_rec.get("valuation_magnitude", "N/A"),
            "fair_range_low": parsed_rec.get("fair_range_low"),
            "fair_range_high": parsed_rec.get("fair_range_high"),
            "primary_driver": parsed_rec.get("valuation_driver", ""),
            "confidence":     parsed_rec.get("valuation_confidence", "N/A"),
        },

        # ── Timing Signal ──────────────────────────────────
        "timing_signal": {
            "action":         parsed_rec.get("timing_action", "N/A"),
            "catalyst":       parsed_rec.get("timing_catalyst", ""),
            "entry_level":    parsed_rec.get("timing_entry_level"),
            "invalidation":   parsed_rec.get("timing_invalidation", ""),
        },

        # ── Factors cited (for factor attribution tracking) ─
        "factors_cited":      parsed_rec.get("factors_cited", []),

        # ── Multi-horizon evaluations (filled in later) ──────
        # Each horizon (7d, 14d, 21d) is scored independently.
        # This captures whether the directional thesis was right
        # but the timing was off (correct at 21d but wrong at 7d).
        "evaluations": {
            f"{h}d": {
                "status":         "pending",
                "scored_at":      None,
                "price_at_eval":  None,
                "bias_direction_correct":  None,
                "bias_magnitude_correct":  None,
                "bias_error_pct":          None,
                "timing_useful":           None,
                "timing_entry_hit":        None,
            }
            for h in EVAL_HORIZONS
        },
    }

    ledger = _load_ledger()
    ledger.append(record)
    _save_ledger(ledger)

    log.info(
        f"  📝 Prediction recorded: {record['valuation_bias']['assessment']} "
        f"({record['valuation_bias']['confidence']}) at {price_now} USD/t"
    )
    return record


def extract_recommendation(report_text: str) -> dict:
    """
    Parse the new-format report (Valuation Bias + Timing Signal)
    to extract structured prediction fields.

    Handles both the new format and graceful degradation if
    the report structure is slightly different.
    """
    result = {}

    def extract(pattern, text, default=None):
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else default

    # ── Valuation Bias section ─────────────────────────────
    vb_section = extract(
        r"(?:###?\s*)?(?:✅\s*)?VALUATION BIAS(.*?)(?=###?\s*(?:⏱|TIMING)|$)",
        report_text, ""
    )

    assessment = extract(r"\*\*Assessment:\*\*\s*(.+?)(?:\n|$)", vb_section)
    if assessment:
        # Parse "Overvalued", "Undervalued", "Fairly valued"
        assessment_clean = assessment.strip().rstrip(".")
        result["valuation_assessment"] = assessment_clean

        # Infer direction from assessment
        if "overvalued" in assessment_clean.lower():
            result["implied_direction"] = "BEARISH"
        elif "undervalued" in assessment_clean.lower():
            result["implied_direction"] = "BULLISH"
        else:
            result["implied_direction"] = "NEUTRAL"

    result["valuation_magnitude"] = extract(
        r"\*\*Magnitude:\*\*\s*(.+?)(?:\n|$)", vb_section
    )
    result["valuation_confidence"] = extract(
        r"\*\*Confidence:\*\*\s*(.+?)(?:\n|$)", vb_section
    )
    result["valuation_driver"] = extract(
        r"\*\*Primary [Dd]river.*?:\*\*\s*(.+?)(?:\n|$)", vb_section
    )

    # Parse fair range — handles "£2,100–2,300", "2100-2300", etc.
    fair_range = extract(r"\*\*(?:Estimated )?[Ff]air [Rr]ange:\*\*\s*(.+?)(?:\n|$)", vb_section)
    if fair_range:
        nums = re.findall(r"[\d,]+\.?\d*", fair_range.replace(",", ""))
        if len(nums) >= 2:
            result["fair_range_low"] = float(nums[0])
            result["fair_range_high"] = float(nums[1])
        elif len(nums) == 1:
            val = float(nums[0])
            result["fair_range_low"] = val * 0.97
            result["fair_range_high"] = val * 1.03

    # ── Timing Signal section ──────────────────────────────
    ts_section = extract(
        r"(?:###?\s*)?(?:⏱\s*)?TIMING SIGNAL(.*?)(?=###?\s*(?:⚠|KEY RISKS|WATCH)|$)",
        report_text, ""
    )

    timing_action = extract(r"\*\*Timing:\*\*\s*(.+?)(?:\n|$)", ts_section)
    if timing_action:
        result["timing_action"] = timing_action.strip().rstrip(".")

    result["timing_catalyst"] = extract(
        r"\*\*(?:Wait for )?[Cc]atalyst.*?:\*\*\s*(.+?)(?:\n|$)", ts_section
    )
    result["timing_invalidation"] = extract(
        r"\*\*(?:Key )?[Ii]nvalidation:\*\*\s*(.+?)(?:\n|$)", ts_section
    )

    # Entry level — parse a number from the timing section
    entry_text = extract(r"\*\*(?:Entry|Level|Wait for level).*?:\*\*\s*(.+?)(?:\n|$)", ts_section)
    if entry_text:
        nums = re.findall(r"[\d,]+\.?\d*", entry_text.replace(",", ""))
        if nums:
            result["timing_entry_level"] = float(nums[0])

    # ── Factor attribution ─────────────────────────────────
    factors = []
    factor_keywords = {
        "supply_stress":    [r"supply\s+stress", r"crop\s+stress", r"satellite", r"NDWI", r"NDMI"],
        "technical_macd":   [r"MACD\s+cross", r"MACD\s+bull", r"MACD\s+bear"],
        "technical_ema":    [r"EMA[\s-]*\d+", r"above\s+EMA", r"below\s+EMA"],
        "technical_rsi":    [r"RSI", r"overbought", r"oversold"],
        "technical_bb":     [r"Bollinger", r"BB\s+"],
        "warehouse_stocks": [r"warehouse\s+stock", r"ICE\s+stock", r"certified\s+stock"],
        "grinding_data":    [r"grinding", r"ECA", r"NCA", r"demand\s+destruct"],
        "news_sentiment":   [r"news\s+sentiment", r"sentiment\s+is", r"headline"],
        "weather":          [r"rainfall", r"drought", r"weather\s+deter"],
        "macro_fx":         [r"GBP/USD", r"sterling", r"dollar\s+weak", r"dollar\s+strong"],
        "seasonal":         [r"mid[\s-]*crop", r"main[\s-]*crop", r"seasonal"],
        "surplus_deficit":  [r"surplus", r"deficit", r"ICCO\s+balance"],
    }

    # Check which factors appear in the Valuation Bias rationale + Timing Signal
    analysis_text = vb_section + " " + ts_section
    for factor_name, patterns in factor_keywords.items():
        for pat in patterns:
            if re.search(pat, analysis_text, re.IGNORECASE):
                factors.append(factor_name)
                break  # one match per factor is enough

    result["factors_cited"] = factors

    return result


# ─────────────────────────────────────────────
#  EVALUATE PENDING PREDICTIONS
# ─────────────────────────────────────────────

def evaluate_pending(current_price: float = None, price_fetcher=None) -> int:
    """
    Score predictions at each horizon (7d, 14d, 21d) once enough time
    has elapsed.  Each horizon is scored independently — a prediction
    can be wrong at 7d but correct at 21d (thesis right, timing off).

    Returns:
        Number of individual horizon scores written in this run.
    """
    ledger = _load_ledger()
    now = datetime.now(timezone.utc)
    scored_count = 0

    if current_price is None and price_fetcher:
        try:
            current_price = price_fetcher()
        except Exception as e:
            log.warning(f"Price fetcher failed: {e} — skipping evaluation")
            return 0

    if current_price is None:
        log.info("  No current price available — skipping prediction evaluation")
        return 0

    for record in ledger:
        # Migrate old single-evaluation format to multi-horizon
        if "evaluation" in record and "evaluations" not in record:
            old_ev = record.pop("evaluation")
            record["evaluations"] = {}
            if old_ev.get("status") == "scored":
                record["evaluations"]["7d"] = old_ev
            else:
                for h in EVAL_HORIZONS:
                    record["evaluations"][f"{h}d"] = {
                        "status": "pending", "scored_at": None,
                        "price_at_eval": None,
                        "bias_direction_correct": None,
                        "bias_magnitude_correct": None,
                        "bias_error_pct": None,
                        "timing_useful": None, "timing_entry_hit": None,
                    }

        evaluations = record.get("evaluations", {})
        gen_at = record.get("generated_at", "")
        try:
            pred_time = datetime.fromisoformat(gen_at)
            if pred_time.tzinfo is None:
                pred_time = pred_time.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue

        days_elapsed = (now - pred_time).days
        price_at_pred = record.get("price_at_prediction")
        if not price_at_pred:
            continue

        vb = record["valuation_bias"]
        ts = record["timing_signal"]
        assessment = (vb.get("assessment") or "").lower()

        for horizon in EVAL_HORIZONS:
            key = f"{horizon}d"
            ev = evaluations.get(key)
            if ev is None:
                ev = {"status": "pending", "scored_at": None,
                      "price_at_eval": None,
                      "bias_direction_correct": None,
                      "bias_magnitude_correct": None,
                      "bias_error_pct": None,
                      "timing_useful": None, "timing_entry_hit": None}
                evaluations[key] = ev

            if ev.get("status") != "pending":
                continue
            if days_elapsed < horizon:
                continue

            # ── Score this horizon ────────────────────────
            actual_move_pct = (current_price - price_at_pred) / price_at_pred * 100

            if "overvalued" in assessment:
                ev["bias_direction_correct"] = actual_move_pct < 0
            elif "undervalued" in assessment:
                ev["bias_direction_correct"] = actual_move_pct > 0
            elif "fairly" in assessment or "fair" in assessment:
                ev["bias_direction_correct"] = abs(actual_move_pct) < 3.0
            else:
                ev["bias_direction_correct"] = None

            fair_low = vb.get("fair_range_low")
            fair_high = vb.get("fair_range_high")
            if fair_low is not None and fair_high is not None:
                ev["bias_magnitude_correct"] = fair_low <= current_price <= fair_high
            else:
                ev["bias_magnitude_correct"] = None

            ev["bias_error_pct"] = round(actual_move_pct, 2)

            entry_level = ts.get("entry_level")
            if entry_level is not None:
                if "overvalued" in assessment:
                    ev["timing_entry_hit"] = current_price <= entry_level
                else:
                    ev["timing_entry_hit"] = current_price >= entry_level

            action = (ts.get("action") or "").lower()
            if "act now" in action:
                ev["timing_useful"] = ev["bias_direction_correct"]
            elif "wait" in action:
                ev["timing_useful"] = ev["bias_direction_correct"]

            ev["status"] = "scored"
            ev["scored_at"] = now.isoformat()
            ev["price_at_eval"] = current_price

            scored_count += 1
            log.info(
                f"  📊 Scored {record['id']} @{key}: "
                f"direction={'✅' if ev['bias_direction_correct'] else '❌'}, "
                f"target={'✅' if ev['bias_magnitude_correct'] else '❌'}, "
                f"move={actual_move_pct:+.1f}%"
            )

        record["evaluations"] = evaluations

    if scored_count > 0:
        _save_ledger(ledger)
        log.info(f"  → {scored_count} horizon score(s) written")

    return scored_count


# ─────────────────────────────────────────────
#  BUILD FEEDBACK PROMPT
# ─────────────────────────────────────────────

def build_feedback_prompt() -> str:
    """
    Generate a prompt-ready feedback block summarising recent
    prediction accuracy across all evaluation horizons (7d, 14d, 21d).
    """
    ledger = _load_ledger()
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=FEEDBACK_WINDOW_DAYS)

    # Collect predictions with at least one scored horizon
    scored = []
    for r in ledger:
        evals = r.get("evaluations", {})
        # Backward compat: old "evaluation" key
        if not evals and r.get("evaluation", {}).get("status") == "scored":
            evals = {"7d": r["evaluation"]}
        has_any_scored = any(
            e.get("status") == "scored" for e in evals.values()
        )
        if not has_any_scored:
            continue
        try:
            gen = datetime.fromisoformat(r["generated_at"])
            if gen.tzinfo is None:
                gen = gen.replace(tzinfo=timezone.utc)
            if gen >= cutoff:
                scored.append(r)
        except (ValueError, TypeError):
            continue

    if not scored:
        return ""

    scored = sorted(scored, key=lambda r: r["generated_at"], reverse=True)[:MAX_FEEDBACK_PREDICTIONS]

    # ── Aggregate per-horizon stats ───────────────────────
    horizon_stats = {}
    for h in EVAL_HORIZONS:
        key = f"{h}d"
        direction_results = []
        magnitude_results = []
        error_values = []
        confidence_buckets = {"HIGH": [], "MEDIUM": [], "LOW": []}
        factor_hits = {}

        for r in scored:
            evals = r.get("evaluations", {})
            if not evals and r.get("evaluation"):
                evals = {"7d": r["evaluation"]}
            ev = evals.get(key, {})
            if ev.get("status") != "scored":
                continue

            vb = r["valuation_bias"]

            if ev.get("bias_direction_correct") is not None:
                direction_results.append(ev["bias_direction_correct"])
            if ev.get("bias_magnitude_correct") is not None:
                magnitude_results.append(ev["bias_magnitude_correct"])
            if ev.get("bias_error_pct") is not None:
                error_values.append(ev["bias_error_pct"])

            conf = (vb.get("confidence") or "").upper().strip()
            if conf in confidence_buckets and ev.get("bias_direction_correct") is not None:
                confidence_buckets[conf].append(ev["bias_direction_correct"])

            dir_correct = ev.get("bias_direction_correct")
            if dir_correct is not None:
                for factor in r.get("factors_cited", []):
                    factor_hits.setdefault(factor, []).append(dir_correct)

        horizon_stats[key] = {
            "direction": direction_results,
            "magnitude": magnitude_results,
            "errors": error_values,
            "confidence": confidence_buckets,
            "factors": factor_hits,
        }

    # ── Format the feedback block ─────────────────────────
    lines = []
    lines.append("## PREDICTION ACCURACY FEEDBACK (last {} days, {} predictions)".format(
        FEEDBACK_WINDOW_DAYS, len(scored)
    ))
    lines.append("")

    # Direction accuracy comparison across horizons
    horizon_dir_line = []
    for h in EVAL_HORIZONS:
        key = f"{h}d"
        dr = horizon_stats[key]["direction"]
        if dr:
            pct = sum(dr) / len(dr) * 100
            horizon_dir_line.append(f"{h}d: {sum(dr)}/{len(dr)} ({pct:.0f}%)")
    if horizon_dir_line:
        lines.append("Direction accuracy by horizon: " + " | ".join(horizon_dir_line))
        # Check if accuracy improves with time (thesis right, timing off)
        accs = []
        for h in EVAL_HORIZONS:
            dr = horizon_stats[f"{h}d"]["direction"]
            if dr:
                accs.append((h, sum(dr) / len(dr) * 100))
        if len(accs) >= 2 and accs[-1][1] > accs[0][1] + 10:
            lines.append(
                f"  ⚠️ TIMING LAG — accuracy improves from {accs[0][1]:.0f}% at {accs[0][0]}d "
                f"to {accs[-1][1]:.0f}% at {accs[-1][0]}d. Your directional thesis is often "
                f"right but takes longer to play out. Consider wider timeframes or "
                f"later entry triggers."
            )

    # Fair range accuracy (use longest horizon for this)
    for h in reversed(EVAL_HORIZONS):
        mr = horizon_stats[f"{h}d"]["magnitude"]
        if mr:
            pct = sum(mr) / len(mr) * 100
            lines.append(f"Fair range accuracy ({h}d): {sum(mr)}/{len(mr)} ({pct:.0f}%)")
            break

    # Signed bias (use 7d — most responsive)
    errs_7d = horizon_stats.get("7d", {}).get("errors", [])
    if errs_7d:
        avg_err = sum(abs(e) for e in errs_7d) / len(errs_7d)
        avg_signed = sum(errs_7d) / len(errs_7d)
        lines.append(f"Avg absolute error (7d): {avg_err:.1f}% | Signed bias: {avg_signed:+.1f}%")
        if avg_signed > 2.0:
            lines.append("  ⚠️ SYSTEMATIC BULLISH BIAS — price consistently moves lower than predicted")
        elif avg_signed < -2.0:
            lines.append("  ⚠️ SYSTEMATIC BEARISH BIAS — price consistently moves higher than predicted")
    lines.append("")

    # Confidence calibration (use 14d as the primary reference)
    cal_key = "14d" if horizon_stats.get("14d", {}).get("direction") else "7d"
    conf_buckets = horizon_stats.get(cal_key, {}).get("confidence", {})
    conf_lines = []
    for level in ["HIGH", "MEDIUM", "LOW"]:
        results = conf_buckets.get(level, [])
        if results:
            correct = sum(results)
            total = len(results)
            conf_lines.append(f"  {level}: {correct}/{total} correct ({correct/total*100:.0f}%)")
    if conf_lines:
        lines.append(f"Confidence calibration ({cal_key}):")
        lines.extend(conf_lines)
        high_r = conf_buckets.get("HIGH", [])
        low_r = conf_buckets.get("LOW", [])
        if len(high_r) >= 2 and len(low_r) >= 2:
            if sum(high_r)/len(high_r) < sum(low_r)/len(low_r):
                lines.append("  ⚠️ CONFIDENCE MISCALIBRATED — reduce stated confidence levels.")
        lines.append("")

    # Factor attribution (use 14d horizon, more meaningful)
    factors_14 = horizon_stats.get("14d", {}).get("factors", {})
    if not factors_14:
        factors_14 = horizon_stats.get("7d", {}).get("factors", {})
    factor_lines = []
    for factor, results in sorted(factors_14.items()):
        if len(results) >= 3:
            correct = sum(results)
            total = len(results)
            pct = correct / total * 100
            emoji = "✅" if pct >= 60 else "⚠️" if pct >= 40 else "❌"
            label = factor.replace("_", " ").title()
            factor_lines.append(f"  {emoji} {label}: {pct:.0f}% ({correct}/{total})")
    if factor_lines:
        lines.append("Factor attribution (14d accuracy when cited):")
        lines.extend(sorted(factor_lines, reverse=True))
        lines.append("")

    # Recent misses (use 14d — captures the April-type scenario)
    recent_misses = []
    for r in scored:
        evals = r.get("evaluations", {})
        if not evals and r.get("evaluation"):
            evals = {"7d": r["evaluation"]}
        # Check 14d first, fall back to 7d
        ev_14 = evals.get("14d", {})
        ev_7 = evals.get("7d", {})
        ev = ev_14 if ev_14.get("status") == "scored" else ev_7
        if ev.get("bias_direction_correct") is False:
            recent_misses.append((r, ev))
    recent_misses = recent_misses[:3]

    if recent_misses:
        lines.append("Recent misses to learn from:")
        for r, ev in recent_misses:
            vb = r["valuation_bias"]
            date = r["generated_at"][:10]
            err = ev.get("bias_error_pct", 0)
            lines.append(
                f"  - {date}: Called {vb.get('assessment', '?')} at "
                f"{r.get('price_at_prediction', '?')} USD/t, "
                f"actual moved {err:+.1f}% to "
                f"{ev.get('price_at_eval', '?')} USD/t. "
                f"Driver: {vb.get('primary_driver', 'N/A')}"
            )
        lines.append("")

    # Save summary
    summary = {
        "generated_at": now.isoformat(),
        "predictions_scored": len(scored),
        "feedback_text": "\n".join(lines),
    }
    for h in EVAL_HORIZONS:
        dr = horizon_stats[f"{h}d"]["direction"]
        if dr:
            summary[f"direction_accuracy_{h}d"] = round(sum(dr)/len(dr)*100, 1)
    try:
        with open(FEEDBACK_SUMMARY_FILE, "w") as f:
            json.dump(summary, f, indent=2)
    except Exception as e:
        log.warning(f"Failed to save feedback summary: {e}")

    return "\n".join(lines)


# ─────────────────────────────────────────────
#  PRICE SANITY CHECK
# ─────────────────────────────────────────────

SCALE_HISTORY_FILE = os.getenv("SCALE_HISTORY_FILE", "cocoa_scale_history.json")

def validate_price(
    live_mid: float,
    cc_last_close: float,
    scale: float,
    threshold_pct: float = 8.0
) -> dict:
    """
    Validate the normalisation scale factor against recent history.
    Returns a dict with 'valid', 'scale_to_use', and 'warning'.

    If today's scale diverges from the recent average by more than
    threshold_pct, falls back to the last known good scale.
    """
    history = _load_scale_history()

    result = {
        "valid": True,
        "scale_to_use": scale,
        "warning": None,
        "raw_scale": scale,
    }

    if len(history) >= 3:
        recent_scales = [h["scale"] for h in history[-10:]]
        avg_scale = sum(recent_scales) / len(recent_scales)
        deviation_pct = abs(scale - avg_scale) / avg_scale * 100

        if deviation_pct > threshold_pct:
            last_good = history[-1]["scale"]
            result["valid"] = False
            result["scale_to_use"] = last_good
            result["warning"] = (
                f"Scale factor {scale:.6f} deviates {deviation_pct:.1f}% from "
                f"recent average {avg_scale:.6f}. Using last known good: {last_good:.6f}. "
                f"Check CMC/YF price feeds for stale data."
            )
            log.warning(f"  ⚠️ {result['warning']}")
            return result

    # Scale looks OK — record it
    history.append({
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "scale": round(scale, 8),
        "live_mid": live_mid,
        "cc_close": cc_last_close,
    })
    # Keep last 60 entries
    if len(history) > 60:
        history = history[-60:]
    _save_scale_history(history)

    return result


def cross_check_prices(yf_price: dict, cmc_price: dict, threshold_pct: float = 5.0) -> dict:
    """
    Compare Yahoo Finance derived price with CMC Markets price.
    Returns a dict with the recommended price to use and any warnings.
    """
    yf_mid = yf_price.get("mid") if yf_price and "error" not in yf_price else None
    cmc_mid = cmc_price.get("mid") if cmc_price and "error" not in cmc_price else None

    result = {
        "recommended_source": None,
        "recommended_mid": None,
        "warning": None,
        "yf_mid": yf_mid,
        "cmc_mid": cmc_mid,
    }

    if yf_mid and cmc_mid:
        divergence_pct = abs(yf_mid - cmc_mid) / max(yf_mid, cmc_mid) * 100
        if divergence_pct > threshold_pct:
            result["warning"] = (
                f"YF ({yf_mid:.1f}) and CMC ({cmc_mid:.1f}) diverge by "
                f"{divergence_pct:.1f}% — one source may be stale"
            )
            log.warning(f"  ⚠️ Price cross-check: {result['warning']}")
            # Prefer YF as it's more reliable (no expiring session keys)
            result["recommended_source"] = "Yahoo Finance (CMC divergent)"
            result["recommended_mid"] = yf_mid
        else:
            # Both agree — use CMC as it's the actual instrument
            result["recommended_source"] = "CMC Markets (cross-checked)"
            result["recommended_mid"] = cmc_mid
    elif cmc_mid:
        result["recommended_source"] = "CMC Markets (YF unavailable)"
        result["recommended_mid"] = cmc_mid
    elif yf_mid:
        result["recommended_source"] = "Yahoo Finance (CMC unavailable)"
        result["recommended_mid"] = yf_mid

    return result


def _load_scale_history() -> list:
    try:
        with open(SCALE_HISTORY_FILE, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save_scale_history(history: list):
    with open(SCALE_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def _get_current_price(snapshot: dict) -> Optional[float]:
    """Extract the current price from a snapshot."""
    px = snapshot.get("technicals", {}).get("price", {})
    if px.get("current"):
        return float(px["current"])
    return None


def get_ledger_stats() -> dict:
    """Return a quick summary of the ledger state."""
    ledger = _load_ledger()
    pending = 0
    scored = 0
    for r in ledger:
        evals = r.get("evaluations", {})
        if not evals and r.get("evaluation"):
            evals = {"7d": r["evaluation"]}
        any_scored = any(e.get("status") == "scored" for e in evals.values())
        all_scored = all(e.get("status") == "scored" for e in evals.values()) if evals else False
        if all_scored:
            scored += 1
        elif any_scored:
            scored += 1  # partially scored counts as scored
        else:
            pending += 1
    return {"total": len(ledger), "pending": pending, "scored": scored}


# ─────────────────────────────────────────────
#  CLI ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cocoa Prediction Feedback System")
    parser.add_argument("--stats", action="store_true", help="Show ledger statistics")
    parser.add_argument("--summary", action="store_true", help="Print feedback summary")
    parser.add_argument("--evaluate", type=float, default=None,
                        help="Evaluate pending predictions at this price (USD/t)")
    args = parser.parse_args()

    if args.stats:
        stats = get_ledger_stats()
        print(f"\nPrediction Ledger: {stats['total']} total "
              f"({stats['pending']} pending, {stats['scored']} scored)\n")

    if args.evaluate is not None:
        n = evaluate_pending(current_price=args.evaluate)
        print(f"\nEvaluated {n} prediction(s) at {args.evaluate} USD/t\n")

    if args.summary or (not args.stats and args.evaluate is None):
        feedback = build_feedback_prompt()
        if feedback:
            print("\n" + feedback)
        else:
            print("\nNo scored predictions yet — feedback loop will activate "
                  "after predictions have been evaluated.\n")
