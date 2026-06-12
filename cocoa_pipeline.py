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
    except ImportError:
        results["steps"]["news"] = {"status": "SKIP", "error": "cocoa_news_agent not available"}
        log.info("  ⏭️ News: module not available, skipping")
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
        log.info("Step 7: Running crop monitor...")
        from cocoa_crop_monitor import load_crop_health_for_agent

        crop = load_crop_health_for_agent()
        if crop:
            snapshot["crop_health"] = crop
            results["steps"]["crop_monitor"] = {"status": "OK"}
            log.info("  ✅ Crop health loaded from cache")
        else:
            results["steps"]["crop_monitor"] = {"status": "SKIP", "error": "No cached data"}
            log.info("  ⏭️ Crop monitor: no cached data")
    except Exception as e:
        results["steps"]["crop_monitor"] = {"status": "FAIL", "error": str(e)}
        log.warning(f"  ❌ Crop monitor failed: {e}")

    # ══════════════════════════════════════════════
    #  STEP 8: Combined Stress Signal
    # ══════════════════════════════════════════════
    try:
        from cocoa_stress_signal import compute_combined_stress_signal

        stress = compute_combined_stress_signal()
        if stress:
            snapshot["combined_stress"] = stress
            results["steps"]["stress_signal"] = {
                "status": "OK",
                "score": stress.get("stress_score"),
                "signal": stress.get("signal"),
            }
            log.info(f"  ✅ Stress signal: {stress.get('stress_score')}/100")
        else:
            results["steps"]["stress_signal"] = {"status": "SKIP", "error": "No data"}
    except ImportError:
        results["steps"]["stress_signal"] = {"status": "SKIP", "error": "Module not available"}
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
