"""
=============================================================
  COCOA PIPELINE — Resilient Orchestrator
=============================================================
  Replaces running three separate scripts (data_gatherer,
  crop_monitor, cocoa_agent) with a single pipeline that:

  1. Runs each data step independently with error isolation
  2. Reports exactly what worked and what failed
  3. Produces an opportunity score even with partial data
  4. Does NOT require the 'anthropic' package — the analysis
     is done by the Managed Agent itself, not a nested API call

  The Managed Agent runs this script, reads the output, and
  then does the analysis directly (Phase 3 architecture).

  Usage:
    python3 cocoa_pipeline.py
=============================================================
"""

import os
import sys
import json
import logging
import traceback
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

def load_cached_crop_health_for_agent(filepath: str = "cocoa_crop_health.json") -> dict | None:
    """
    Load cached crop health without importing cocoa_crop_monitor.py.

    This avoids importing `ee` at pipeline runtime. The full GEE crop monitor
    can still be run separately to refresh cocoa_crop_health.json.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None

    overall = data.get("overall_signal", {})
    regions = data.get("regions", {})
    generated_at = data.get("generated_at", "")

    age_days = None
    try:
        generated = datetime.fromisoformat(generated_at)
        age_days = (datetime.now(timezone.utc) - generated).days
    except Exception:
        pass

    region_summaries = []

    for region_name, rdata in regions.items():
        weekly = rdata.get("weekly", rdata.get("monthly", {}))
        if not weekly:
            continue

        latest_key = None
        latest = None

        for wk in sorted(weekly.keys(), reverse=True):
            candidate = weekly[wk]
            if (
                candidate.get("evi") is not None
                or candidate.get("soil_moisture") is not None
                or candidate.get("ndmi") is not None
            ):
                latest_key = wk
                latest = candidate
                break

        if latest is None:
            latest_key = sorted(weekly.keys())[-1]
            latest = weekly[latest_key]

        flags = [f["signal"] for f in latest.get("stress_flags", [])]
        rel_flags = [f["signal"] for f in latest.get("seasonal_flags", [])]
        chirps_ctx = rdata.get("chirps_context", {})

        region_summaries.append({
            "region": region_name,
            "country": rdata.get("country"),
            "week": latest_key,
            "evi": latest.get("evi"),
            "ndvi": latest.get("ndvi"),
            "ndmi": latest.get("ndmi"),
            "lai": latest.get("lai"),
            "lst_c": latest.get("lst_celsius"),
            "lst_anomaly_c": latest.get("lst_anomaly_c"),
            "lst_clim_mean_c": latest.get("lst_clim_mean_c"),
            "soil_moisture": latest.get("soil_moisture"),
            "soil_moisture_rootzone": latest.get("soil_moisture_rootzone"),
            "smap_source": latest.get("smap_source", ""),
            "chirps_rainfall_mm": latest.get("chirps_rainfall_mm"),
            "rainfall_anomaly_pct": latest.get("rainfall_anomaly_pct"),
            "rainfall_clim_mean_mm": latest.get("rainfall_clim_mean_mm"),
            "stress_fraction_ndmi": latest.get("stress_fraction_ndmi"),
            "stress_fraction_evi": latest.get("stress_fraction_evi"),
            "sar_vh": latest.get("sar_vh"),
            "sar_vv": latest.get("sar_vv"),
            "sar_vh_vv_ratio": latest.get("sar_vh_vv_ratio"),
            "optical_gap": latest.get("optical_gap", False),
            "chirps_30d_mm": chirps_ctx.get("chirps_30d_mm"),
            "chirps_60d_mm": chirps_ctx.get("chirps_60d_mm"),
            "chirps_90d_mm": chirps_ctx.get("chirps_90d_mm"),
            "flags": flags,
            "ndmi_seasonal_mean": latest.get("ndmi_seasonal_mean"),
            "ndmi_anomaly": latest.get("ndmi_anomaly"),
            "ndmi_anomaly_pct": latest.get("ndmi_anomaly_pct"),
            "ndmi_zscore": latest.get("ndmi_zscore"),
            "evi_seasonal_mean": latest.get("evi_seasonal_mean"),
            "evi_anomaly": latest.get("evi_anomaly"),
            "evi_zscore": latest.get("evi_zscore"),
            "seasonal_label": latest.get("seasonal_label"),
            "seasonal_n": latest.get("seasonal_n", 0),
            "seasonal_flags": rel_flags,
        })

    diff_summary = None
    last_diff = data.get("last_diff")
    if last_diff:
        diff_summary = {
            "since": last_diff.get("old_generated_at"),
            "overall_score_change": last_diff.get("overall_score_change"),
            "old_signal": last_diff.get("old_named_signal"),
            "new_signal": last_diff.get("new_named_signal"),
            "regions": {},
        }

        for rname, rdiff in last_diff.get("regions", {}).items():
            changes = {}
            for field in ["evi", "ndmi", "soil_moisture"]:
                if field in rdiff:
                    changes[field] = rdiff[field]
            if changes:
                diff_summary["regions"][rname] = changes

    return {
        "version": data.get("version", "1.0"),
        "granularity": data.get("granularity", "monthly"),
        "data_age_days": age_days,
        "stale": age_days > 14 if age_days is not None else None,
        "overall_score": overall.get("score"),
        "named_signal": overall.get("named_signal"),
        "overall_bias": overall.get("bias"),
        "overall_signal": overall.get("signal"),
        "avg_evi": overall.get("avg_evi"),
        "avg_ndmi": overall.get("avg_ndmi"),
        "avg_soil_moisture": overall.get("avg_soil_moisture"),
        "critical_flags": overall.get("critical_flags"),
        "warning_flags": overall.get("warning_flags"),
        "regions": region_summaries,
        "diff": diff_summary,
    }

def run_pipeline():
    """Run the full data gathering pipeline with error isolation."""

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "steps": {},
        "health": {},
        "snapshot_file": None,
        "opportunity_score": None,
    }

    # ══════════════════════════════════════════════
    #  STEP 1: Price Data & Technicals
    # ══════════════════════════════════════════════
    price_ok = False
    snapshot = {}
    try:
        log.info("Step 1: Fetching price data...")
        from cocoa_data_gatherer import fetch_price_data, compute_indicators

        df, raw_close = fetch_price_data()
        technicals = compute_indicators(df, raw_close)
        snapshot["technicals"] = technicals
        snapshot["history_days"] = len(df)

        price = technicals.get("price", {}).get("current")
        if price:
            price_ok = True
            results["steps"]["price"] = {"status": "OK", "price": price, "bars": len(df)}
            log.info(f"  ✅ Price: {price} USD/t ({len(df)} bars)")
        else:
            results["steps"]["price"] = {"status": "FAIL", "error": "Price returned None"}
            log.warning("  ❌ Price: returned None despite data")
    except Exception as e:
        results["steps"]["price"] = {"status": "FAIL", "error": str(e)}
        log.warning(f"  ❌ Price failed: {e}")

    # ══════════════════════════════════════════════
    #  STEP 2: Related Markets
    # ══════════════════════════════════════════════
    try:
        log.info("Step 2: Fetching related markets...")
        from cocoa_data_gatherer import fetch_related_markets, RELATED_TICKERS

        related = fetch_related_markets(RELATED_TICKERS)
        snapshot["related_markets"] = related

        # Extract GBPUSD for GBP display
        gbpusd = related.get("GBPUSD", {}).get("price")
        if gbpusd and price_ok:
            snapshot["gbpusd_rate"] = gbpusd
            snapshot["price_gbp"] = round(
                snapshot["technicals"]["price"]["current"] / gbpusd, 1
            )

        successes = sum(1 for v in related.values() if v.get("price"))
        results["steps"]["related_markets"] = {
            "status": "OK", "fetched": successes, "total": len(RELATED_TICKERS)
        }
        log.info(f"  ✅ Related markets: {successes}/{len(RELATED_TICKERS)}")
    except Exception as e:
        results["steps"]["related_markets"] = {"status": "FAIL", "error": str(e)}
        log.warning(f"  ❌ Related markets failed: {e}")

    # ══════════════════════════════════════════════
    #  STEP 3: COT Positioning
    # ══════════════════════════════════════════════
    try:
        log.info("Step 3: Fetching COT data...")
        from cocoa_data_gatherer import fetch_cot_data

        cot = fetch_cot_data()
        snapshot["cot"] = cot

        if "error" not in cot:
            results["steps"]["cot"] = {
                "status": "OK",
                "net": cot.get("managed_money_net"),
                "percentile": cot.get("net_position_percentile"),
                "signal": cot.get("positioning_signal"),
            }
            log.info(f"  ✅ COT: net={cot.get('managed_money_net')}, "
                     f"pctile={cot.get('net_position_percentile')}")
        else:
            results["steps"]["cot"] = {"status": "FAIL", "error": cot["error"]}
            log.warning(f"  ❌ COT: {cot['error']}")
    except Exception as e:
        results["steps"]["cot"] = {"status": "FAIL", "error": str(e)}
        snapshot["cot"] = {"error": str(e)}
        log.warning(f"  ❌ COT failed: {e}")

    # ══════════════════════════════════════════════
    #  STEP 4: Weather
    # ══════════════════════════════════════════════
    try:
        log.info("Step 4: Fetching weather...")
        from cocoa_data_gatherer import fetch_weather, WEATHER_LOCATIONS

        weather = fetch_weather(WEATHER_LOCATIONS)
        snapshot["weather"] = weather
        results["steps"]["weather"] = {"status": "OK"}
        log.info("  ✅ Weather fetched")
    except Exception as e:
        results["steps"]["weather"] = {"status": "FAIL", "error": str(e)}
        log.warning(f"  ❌ Weather failed: {e}")

    # ══════════════════════════════════════════════
    #  STEP 5: News
    # ══════════════════════════════════════════════
    try:
        log.info("Step 5: Fetching news...")
        from cocoa_news_agent import fetch_news_intelligence

        news = fetch_news_intelligence()
        snapshot["news_intelligence"] = news
        snapshot["news"] = {
            "sentiment": {
                "overall": news.get("directional_signal", "Unknown"),
                "confidence": news.get("signal_confidence", "Unknown"),
            },
        }
        results["steps"]["news"] = {
            "status": "OK",
            "signal": news.get("directional_signal"),
        }
        log.info(f"  ✅ News: {news.get('directional_signal')}")
    except ImportError as e:
        results["steps"]["news"] = {
            "status": "SKIP",
            "error": f"cocoa_news_agent import failed: {e}",
        }
        log.info(f"  ⏭️ News: import failed, skipping: {e}")
    except Exception as e:
        results["steps"]["news"] = {"status": "FAIL", "error": str(e)}
        log.warning(f"  ❌ News failed: {e}")

    # ══════════════════════════════════════════════
    #  STEP 6: Warehouse Stocks
    # ══════════════════════════════════════════════
    try:
        log.info("Step 6: Fetching warehouse stocks...")
        from cocoa_data_gatherer import fetch_ice_warehouse_stocks

        stocks = fetch_ice_warehouse_stocks()
        snapshot["warehouse_stocks"] = stocks
        results["steps"]["warehouse"] = {
            "status": "OK",
            "bags": stocks.get("current_bags"),
            "signal": stocks.get("signal"),
        }
        log.info(f"  ✅ Warehouse: {stocks.get('current_bags')} bags, {stocks.get('signal')}")
    except Exception as e:
        results["steps"]["warehouse"] = {"status": "FAIL", "error": str(e)}
        log.warning(f"  ❌ Warehouse failed: {e}")

    # ══════════════════════════════════════════════
    #  STEP 7: Crop Monitor (GEE)
    # ══════════════════════════════════════════════
    try:
        log.info("Step 7: Loading cached crop health...")

        crop = load_cached_crop_health_for_agent()

        if crop:
            snapshot["crop_health"] = crop
            results["steps"]["crop_monitor"] = {
                "status": "OK",
                "source": "cached",
                "data_age_days": crop.get("data_age_days"),
                "stale": crop.get("stale"),
                "signal": crop.get("named_signal") or crop.get("overall_signal"),
            }
            log.info("  ✅ Crop health loaded from cached cocoa_crop_health.json")
        else:
            results["steps"]["crop_monitor"] = {
                "status": "SKIP",
                "error": "No cached cocoa_crop_health.json found",
            }
            log.info("  ⏭️ Crop monitor: no cached cocoa_crop_health.json found")
    except Exception as e:
        results["steps"]["crop_monitor"] = {"status": "FAIL", "error": str(e)}
        log.warning(f"  ❌ Crop monitor failed: {e}")

    # ══════════════════════════════════════════════
    #  STEP 8: Combined Stress Signal
    # ══════════════════════════════════════════════
    try:
        log.info("Step 8: Computing combined stress signal...")
        from cocoa_stress_signal import compute_combined_stress_signal

        weather = snapshot.get("weather")
        crop = snapshot.get("crop_health")

        valid_weather_locations = [
            name
            for name, data in (weather or {}).items()
            if isinstance(data, dict) and "error" not in data
        ]

        if not weather or not valid_weather_locations:
            results["steps"]["stress_signal"] = {
                "status": "SKIP",
                "error": "No valid weather data",
            }
            log.info("  ⏭️ Stress signal: no valid weather data")

        elif not crop:
            results["steps"]["stress_signal"] = {
                "status": "SKIP",
                "error": "No crop health data",
            }
            log.info("  ⏭️ Stress signal: no crop health data")

        else:
            stress = compute_combined_stress_signal(
                weather_data={"locations": weather},
                crop_data=crop,
            )

            snapshot["combined_stress"] = stress
            results["steps"]["stress_signal"] = {
                "status": "OK",
                "score": stress.get("combined_stress_score"),
                "signal": stress.get("signal"),
                "bias": stress.get("bias"),
            }
            log.info(
                f"  ✅ Stress signal: {stress.get('combined_stress_score')}/100 "
                f"({stress.get('signal')})"
            )

    except ImportError:
        results["steps"]["stress_signal"] = {
            "status": "SKIP",
            "error": "Module not available",
        }
    except Exception as e:
        results["steps"]["stress_signal"] = {"status": "FAIL", "error": str(e)}
        log.warning(f"  ❌ Stress signal failed: {e}")

    # ══════════════════════════════════════════════
    #  STEP 9: Seasonal Context
    # ══════════════════════════════════════════════
    try:
        from cocoa_data_gatherer import get_seasonal_context

        seasonal = get_seasonal_context()
        snapshot["seasonal"] = seasonal
    except Exception:
        pass

    # ══════════════════════════════════════════════
    #  STEP 10: Grinding Data
    # ══════════════════════════════════════════════
    try:
        from cocoa_data_gatherer import load_grinding_data

        grinding = load_grinding_data()
        snapshot["grinding_data"] = grinding
    except Exception:
        pass

    # ══════════════════════════════════════════════
    #  SAVE SNAPSHOT
    # ══════════════════════════════════════════════
    snapshot["generated_at"] = datetime.now(timezone.utc).isoformat()
    try:
        with open("cocoa_daily_snapshot.json", "w") as f:
            json.dump(snapshot, f, indent=2, default=str)
        results["snapshot_file"] = "cocoa_daily_snapshot.json"
        log.info("  ✅ Snapshot saved")
    except Exception as e:
        log.warning(f"  ❌ Snapshot save failed: {e}")

    # ══════════════════════════════════════════════
    #  STEP 11: Feedback Loop
    # ══════════════════════════════════════════════
    try:
        from cocoa_feedback import evaluate_pending, build_feedback_prompt
        from cocoa_weekly_review import (
            score_shadow_predictions, check_big_misses
        )

        current_price = snapshot.get("technicals", {}).get("price", {}).get("current")
        if current_price:
            evaluate_pending(current_price=float(current_price))
            score_shadow_predictions(float(current_price))
            misses = check_big_misses(float(current_price))
            if misses:
                results["big_misses"] = len(misses)
                log.warning(f"  ⚠️ {len(misses)} big miss(es) detected")

        results["steps"]["feedback"] = {"status": "OK"}
    except Exception as e:
        results["steps"]["feedback"] = {"status": "FAIL", "error": str(e)}
        log.warning(f"  ❌ Feedback loop failed: {e}")

    # ══════════════════════════════════════════════
    #  HEALTH SUMMARY
    # ══════════════════════════════════════════════
    ok_count = sum(1 for s in results["steps"].values() if s.get("status") == "OK")
    fail_count = sum(1 for s in results["steps"].values() if s.get("status") == "FAIL")
    skip_count = sum(1 for s in results["steps"].values() if s.get("status") == "SKIP")
    total = len(results["steps"])

    results["health"] = {
        "ok": ok_count,
        "fail": fail_count,
        "skip": skip_count,
        "total": total,
        "price_available": price_ok,
        "cot_available": "error" not in snapshot.get("cot", {"error": True}),
        "satellite_available": bool(snapshot.get("crop_health") or snapshot.get("combined_stress")),
    }

    log.info(f"\n{'='*55}")
    log.info(f"  PIPELINE HEALTH: {ok_count}/{total} OK, {fail_count} FAIL, {skip_count} SKIP")
    log.info(f"  Price: {'✅' if price_ok else '❌'}  COT: {'✅' if results['health']['cot_available'] else '❌'}  "
             f"Satellite: {'✅' if results['health']['satellite_available'] else '❌'}")
    log.info(f"{'='*55}")

    # Print failed steps clearly
    for step_name, step_result in results["steps"].items():
        if step_result.get("status") == "FAIL":
            log.warning(f"  FAILED: {step_name} — {step_result.get('error', 'unknown')}")

    # ══════════════════════════════════════════════
    #  SAVE RESULTS
    # ══════════════════════════════════════════════
    with open("cocoa_pipeline_health.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # ══════════════════════════════════════════════
    #  OUTPUT FOR MANAGED AGENT
    # ══════════════════════════════════════════════
    # Print a structured summary that the managed agent can read
    # and use for its analysis
    print("\n" + "=" * 60)
    print("  COCOA PIPELINE RESULTS")
    print("=" * 60)
    print(f"\nHealth: {ok_count}/{total} steps OK")

    if price_ok:
        px = snapshot["technicals"]["price"]
        print(f"\nPrice: {px['current']} USD/t")
        print(f"  1D: {px.get('change_1d_pct')}% | 1W: {px.get('change_1w_pct')}% | 1M: {px.get('change_1m_pct')}%")
        trend = snapshot["technicals"].get("trend", {})
        print(f"  Trend: {trend.get('label')}")
        print(f"  RSI: {snapshot['technicals'].get('momentum', {}).get('rsi_14')}")
        gbp = snapshot.get("price_gbp")
        if gbp:
            print(f"  GBP equivalent: {gbp} GBP/t")

    cot = snapshot.get("cot", {})
    if "error" not in cot:
        print(f"\nCOT: MM net {cot.get('managed_money_net'):+,} | "
              f"Percentile: {cot.get('net_position_percentile')}% | "
              f"Signal: {cot.get('positioning_signal')}")

    stress = snapshot.get("combined_stress", {})
    if stress.get("stress_score") is not None:
        print(f"\nCrop stress: {stress['stress_score']}/100 | {stress.get('signal')}")

    warehouse = snapshot.get("warehouse_stocks", {})
    if warehouse.get("current_bags"):
        print(f"\nWarehouse: {warehouse['current_bags']:,} bags | {warehouse.get('signal')}")

    news = snapshot.get("news", {}).get("sentiment", {})
    if news.get("overall"):
        print(f"\nNews: {news['overall']} ({news.get('confidence')})")

    print(f"\nSnapshot saved to: cocoa_daily_snapshot.json")
    print(f"Health report saved to: cocoa_pipeline_health.json")
    print("=" * 60)

    return results, snapshot


if __name__ == "__main__":
    results, snapshot = run_pipeline()

    # Exit with error code if critical steps failed
    if not results["health"]["price_available"]:
        log.error("CRITICAL: Price data unavailable — analysis will be limited")
        sys.exit(1)
