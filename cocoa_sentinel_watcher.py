"""
=============================================================
  COCOA TRADING ASSISTANT — Satellite Data Watcher  v2.0
=============================================================
  Runs daily (lightweight). Checks GEE for new data across
  ALL satellite sources — not just Sentinel-2.

  v2.0: Now checks:
    - Sentinel-2 optical imagery (5-day revisit)
    - SMAP soil moisture (2-3 day revisit)
    - CHIRPS rainfall (daily)

  Triggers the crop monitor when ANY key dataset has updated.
  SMAP updates every 2-3 days and CHIRPS daily, so the crop
  monitor now runs more frequently than the S2-only watcher.

  CRON:
    30 6 * * 1-5 /path/to/run_watcher.sh >> cocoa.log 2>&1
=============================================================
"""

import os
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import ee

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

GEE_PROJECT        = os.getenv("GEE_PROJECT", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

STATE_FILE          = "watcher_state.json"
CROP_MONITOR_SCRIPT = os.path.join(os.path.dirname(__file__), "cocoa_crop_monitor.py")

MAX_CLOUD_PCT   = 30
MIN_NEW_IMAGES  = 2
GEE_LAG_DAYS    = 3

COCOA_REGIONS = {
    "CoteIvoire_West":    [-8.0, 4.5, -5.5, 7.5],
    "CoteIvoire_Central": [-5.5, 5.0, -3.5, 8.0],
    "Ghana_Ashanti":      [-2.5, 5.8, -0.5, 7.5],
    "Ghana_Western":      [-3.2, 4.8, -1.5, 6.5],
}

# Representative bounding box for SMAP/CHIRPS checks (whole cocoa belt)
COCOA_BELT_BOUNDS = [-8.0, 4.5, -0.5, 8.0]


# ─────────────────────────────────────────────
#  STATE MANAGEMENT
# ─────────────────────────────────────────────

def load_state():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "last_processed_date": None,
            "last_smap_date": None,
            "last_chirps_date": None,
            "last_check": None,
            "last_trigger": None,
            "run_count": 0,
            "history": [],
        }


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)


# ─────────────────────────────────────────────
#  GEE INITIALISATION
# ─────────────────────────────────────────────

def initialise_gee():
    import google.oauth2.service_account as sa
    if not GEE_PROJECT:
        log.error("GEE_PROJECT not set")
        raise SystemExit(1)
    key_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    try:
        if key_file and os.path.exists(key_file):
            creds = sa.Credentials.from_service_account_file(
                key_file,
                scopes=["https://www.googleapis.com/auth/earthengine"])
            ee.Initialize(credentials=creds, project=GEE_PROJECT)
        else:
            ee.Initialize(project=GEE_PROJECT)
        log.info(f"GEE initialised (project: {GEE_PROJECT})")
    except Exception as e:
        log.error(f"GEE init failed: {e}")
        raise SystemExit(1)


# ─────────────────────────────────────────────
#  SENTINEL-2 CHECK
# ─────────────────────────────────────────────

def get_latest_s2_acquisition(region_name, bounds, since_date):
    """Check for new usable Sentinel-2 imagery over a region."""
    geometry = ee.Geometry.Rectangle(bounds)
    cutoff = (datetime.now(timezone.utc)
              - timedelta(days=GEE_LAG_DAYS)).strftime("%Y-%m-%d")

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geometry)
        .filterDate(since_date, cutoff)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD_PCT))
        .sort("system:time_start", False)
    )

    count = collection.size().getInfo()
    if count == 0:
        return {"region": region_name, "new_images": 0,
                "latest_date": None, "cloud_pct": None}

    latest = collection.first()
    props = latest.getInfo().get("properties", {})
    ts_ms = props.get("system:time_start", 0)
    acq_dt = (datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
              if ts_ms else None)

    return {
        "region":      region_name,
        "new_images":  count,
        "latest_date": acq_dt.strftime("%Y-%m-%d") if acq_dt else None,
        "cloud_pct":   round(props.get("CLOUDY_PIXEL_PERCENTAGE", 0), 1),
    }


# ─────────────────────────────────────────────
#  SMAP CHECK (new in v2)
# ─────────────────────────────────────────────

def get_latest_smap_date(since_date):
    """
    Check for new SMAP Level-4 soil moisture data since a given date.
    Uses SPL4SMGP/008 (continuous, 3-hourly, surface + root zone).
    Falls back to SPL3SMP_E/006 if L4 returns nothing.
    """
    geometry = ee.Geometry.Rectangle(COCOA_BELT_BOUNDS)
    cutoff = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Try L4 first (preferred — continuous data)
    collection = (
        ee.ImageCollection("NASA/SMAP/SPL4SMGP/008")
        .filterBounds(geometry)
        .filterDate(since_date, cutoff)
        .sort("system:time_start", False)
    )

    count = collection.size().getInfo()
    if count == 0:
        # Fallback to L3 v006
        log.info("  SPL4SMGP empty — trying SPL3SMP_E/006")
        collection = (
            ee.ImageCollection("NASA/SMAP/SPL3SMP_E/006")
            .filterBounds(geometry)
            .filterDate(since_date, cutoff)
            .sort("system:time_start", False)
        )
        count = collection.size().getInfo()

    if count == 0:
        return {"source": "SMAP", "new_images": 0, "latest_date": None}

    latest = collection.first()
    ts_ms = latest.getInfo().get("properties", {}).get("system:time_start", 0)
    dt = (datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
          if ts_ms else None)

    return {
        "source":      "SMAP",
        "new_images":  count,
        "latest_date": dt.strftime("%Y-%m-%d") if dt else None,
    }


# ─────────────────────────────────────────────
#  CHIRPS CHECK (new in v2)
# ─────────────────────────────────────────────

def get_latest_chirps_date(since_date):
    """
    Check for new CHIRPS daily rainfall data.
    CHIRPS updates daily with ~5-day latency.
    """
    geometry = ee.Geometry.Rectangle(COCOA_BELT_BOUNDS)
    cutoff = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    collection = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterBounds(geometry)
        .filterDate(since_date, cutoff)
        .sort("system:time_start", False)
    )

    count = collection.size().getInfo()
    if count == 0:
        return {"source": "CHIRPS", "new_days": 0, "latest_date": None}

    latest = collection.first()
    ts_ms = latest.getInfo().get("properties", {}).get("system:time_start", 0)
    dt = (datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
          if ts_ms else None)

    return {
        "source":      "CHIRPS",
        "new_days":    count,
        "latest_date": dt.strftime("%Y-%m-%d") if dt else None,
    }


# ─────────────────────────────────────────────
#  CHECK ALL SOURCES
# ─────────────────────────────────────────────

def check_all_sources(state):
    """
    Check ALL satellite sources for new data.
    Returns a summary with per-source results and trigger decision.
    """
    now = datetime.now(timezone.utc)

    # ── Sentinel-2 ────────────────────────────────────────
    s2_since = state.get("last_processed_date") or (
        now - timedelta(days=10)).strftime("%Y-%m-%d")
    log.info(f"Checking Sentinel-2 since {s2_since}...")

    s2_results = {}
    s2_total = 0
    for rn, bounds in COCOA_REGIONS.items():
        log.info(f"  → {rn}")
        try:
            r = get_latest_s2_acquisition(rn, bounds, s2_since)
            s2_results[rn] = r
            s2_total += r["new_images"]
            if r["new_images"] > 0:
                log.info(f"     {r['new_images']} images, latest: {r['latest_date']}")
            else:
                log.info(f"     No new imagery")
        except Exception as e:
            log.warning(f"     Failed: {e}")
            s2_results[rn] = {"region": rn, "new_images": 0, "error": str(e)}

    s2_dates = [r["latest_date"] for r in s2_results.values() if r.get("latest_date")]
    s2_latest = max(s2_dates) if s2_dates else None
    s2_regions = sum(1 for r in s2_results.values() if r.get("new_images", 0) > 0)
    s2_trigger = s2_total >= MIN_NEW_IMAGES and s2_regions >= 2

    # ── SMAP ──────────────────────────────────────────────
    smap_since = state.get("last_smap_date") or (
        now - timedelta(days=5)).strftime("%Y-%m-%d")
    log.info(f"Checking SMAP since {smap_since}...")
    try:
        smap = get_latest_smap_date(smap_since)
        smap_new = smap.get("new_images", 0) > 0
        log.info(f"  SMAP: {smap.get('new_images', 0)} new, "
                 f"latest: {smap.get('latest_date')}")
    except Exception as e:
        log.warning(f"  SMAP check failed: {e}")
        smap = {"source": "SMAP", "new_images": 0, "latest_date": None}
        smap_new = False

    # ── CHIRPS ────────────────────────────────────────────
    # CHIRPS final daily on GEE (`UCSB-CHG/CHIRPS/DAILY`) has ~3 week
    # processing latency.  The fallback lookback must be long enough to
    # span that gap, otherwise a fresh state or reset always returns 0.
    # 35 days covers the worst-case lag with margin.
    chirps_fallback = (now - timedelta(days=35)).strftime("%Y-%m-%d")
    chirps_stored = state.get("last_chirps_date")
    if chirps_stored and chirps_stored < chirps_fallback:
        # Stored date is older than 35 days — cap it to avoid huge queries
        chirps_since = chirps_fallback
    else:
        chirps_since = chirps_stored or chirps_fallback
    log.info(f"Checking CHIRPS since {chirps_since}...")
    try:
        chirps = get_latest_chirps_date(chirps_since)
        chirps_new = chirps.get("new_days", 0) > 0
        log.info(f"  CHIRPS: {chirps.get('new_days', 0)} new days, "
                 f"latest: {chirps.get('latest_date')}")
    except Exception as e:
        log.warning(f"  CHIRPS check failed: {e}")
        chirps = {"source": "CHIRPS", "new_days": 0, "latest_date": None}
        chirps_new = False

    # ── Trigger Decision ──────────────────────────────────
    # Trigger if:
    #   S2 has new usable imagery across 2+ regions, OR
    #   SMAP has new data (soil moisture is a key leading indicator), OR
    #   CHIRPS has 3+ new days of rainfall data
    chirps_days = chirps.get("new_days", 0) or 0
    should_trigger = (s2_trigger
                      or smap_new
                      or chirps_days >= 3)

    trigger_reasons = []
    if s2_trigger:
        trigger_reasons.append(f"Sentinel-2: {s2_total} images across {s2_regions} regions")
    if smap_new:
        trigger_reasons.append(f"SMAP: new soil moisture (latest: {smap.get('latest_date')})")
    if chirps_days >= 3:
        trigger_reasons.append(f"CHIRPS: {chirps_days} new rainfall days")

    return {
        "s2_results":      s2_results,
        "s2_total":        s2_total,
        "s2_latest":       s2_latest,
        "s2_regions":      s2_regions,
        "smap":            smap,
        "chirps":          chirps,
        "should_trigger":  should_trigger,
        "trigger_reasons": trigger_reasons,
    }


# ─────────────────────────────────────────────
#  TRIGGER CROP MONITOR
# ─────────────────────────────────────────────

def trigger_crop_monitor():
    """Run the crop monitor in incremental mode."""
    log.info("=" * 50)
    log.info("  Triggering incremental crop monitor update...")
    log.info("=" * 50)

    if not os.path.exists(CROP_MONITOR_SCRIPT):
        log.error(f"Script not found: {CROP_MONITOR_SCRIPT}")
        return False

    try:
        result = subprocess.run(
            [sys.executable, CROP_MONITOR_SCRIPT, "--incremental"],
            capture_output=False,
            timeout=300,
        )
        if result.returncode == 0:
            log.info("Crop monitor update completed successfully")
            return True
        else:
            log.error(f"Crop monitor exited with code {result.returncode}")
            return False
    except subprocess.TimeoutExpired:
        log.error("Crop monitor timed out (5 min)")
        return False
    except Exception as e:
        log.error(f"Failed: {e}")
        return False


# ─────────────────────────────────────────────
#  TELEGRAM NOTIFICATION
# ─────────────────────────────────────────────

def send_telegram_notification(check, triggered, success):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    import urllib.request

    if not triggered:
        return

    reasons = "\n".join(f"  • {r}" for r in check.get("trigger_reasons", []))

    if success:
        msg = (
            f"🛰️ New satellite data detected\n\n"
            f"Trigger reasons:\n{reasons}\n\n"
            f"✅ Crop health analysis updated.\n"
            f"Today's report will include fresh data."
        )
    else:
        msg = (
            f"🛰️ New satellite data detected\n\n"
            f"Trigger reasons:\n{reasons}\n\n"
            f"⚠️ Crop monitor encountered an error.\n"
            f"Check logs."
        )

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = json.dumps({
            "chat_id": TELEGRAM_CHAT_ID,
            "text":    msg,
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST")
        with urllib.request.urlopen(req, timeout=10):
            pass
        log.info("Telegram notification sent")
    except Exception as e:
        log.warning(f"Telegram failed: {e}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def run_watcher(force=False):
    log.info("=" * 50)
    log.info("  SATELLITE DATA WATCHER v2")
    log.info(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 50)

    state = load_state()
    initialise_gee()

    now = datetime.now(timezone.utc).isoformat()
    check = check_all_sources(state)
    state["last_check"] = now

    log.info(f"\nCheck summary:")
    log.info(f"  S2 images   : {check['s2_total']} across {check['s2_regions']} regions")
    log.info(f"  SMAP        : {check['smap'].get('new_images', 0)} new")
    log.info(f"  CHIRPS      : {check['chirps'].get('new_days', 0)} new days")
    log.info(f"  Should trigger: {check['should_trigger']}")
    if check["trigger_reasons"]:
        for r in check["trigger_reasons"]:
            log.info(f"    → {r}")

    triggered = False
    success = False

    if force:
        log.info("Force flag — triggering regardless")
        triggered = True
    elif check["should_trigger"]:
        log.info("New data available — triggering crop monitor")
        triggered = True
    else:
        log.info("No significant new data — not triggered")

    if triggered:
        success = trigger_crop_monitor()
        if success:
            if check["s2_latest"]:
                state["last_processed_date"] = check["s2_latest"]
            if check["smap"].get("latest_date"):
                state["last_smap_date"] = check["smap"]["latest_date"]
            if check["chirps"].get("latest_date"):
                state["last_chirps_date"] = check["chirps"]["latest_date"]
            state["last_trigger"] = now
            state["run_count"] = state.get("run_count", 0) + 1

    # History (keep last 30)
    state.setdefault("history", []).append({
        "checked_at": now,
        "s2_images":  check["s2_total"],
        "smap_new":   check["smap"].get("new_images", 0),
        "chirps_new": check["chirps"].get("new_days", 0),
        "triggered":  triggered,
        "success":    success if triggered else None,
        "reasons":    check.get("trigger_reasons", []),
    })
    state["history"] = state["history"][-30:]

    save_state(state)
    send_telegram_notification(check, triggered, success)

    print("\n" + "=" * 50)
    print("  WATCHER v2 SUMMARY")
    print("=" * 50)
    print(f"  Sentinel-2   : {check['s2_total']} images, "
          f"{check['s2_regions']} regions, latest: {check['s2_latest']}")
    print(f"  SMAP         : {check['smap'].get('new_images', 0)} new, "
          f"latest: {check['smap'].get('latest_date')}")
    print(f"  CHIRPS       : {check['chirps'].get('new_days', 0)} days, "
          f"latest: {check['chirps'].get('latest_date')}")
    print(f"  Triggered    : {'Yes ✅' if triggered and success else 'Yes ⚠️' if triggered else 'No'}")
    if check["trigger_reasons"]:
        for r in check["trigger_reasons"]:
            print(f"    → {r}")
    print(f"  Total runs   : {state.get('run_count', 0)}")
    print("=" * 50 + "\n")

    return triggered and success


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Satellite Data Watcher v2")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Force-trigger crop monitor regardless of new data")
    parser.add_argument("--check-only", "-c", action="store_true",
                        help="Check for data but don't trigger crop monitor")
    parser.add_argument("--since", "-s", type=str, default=None,
                        help="Override all since-dates (YYYY-MM-DD)")
    args = parser.parse_args()

    if args.since:
        state = load_state()
        state["last_processed_date"] = args.since
        state["last_smap_date"] = args.since
        state["last_chirps_date"] = args.since
        save_state(state)
        log.info(f"All since-dates overridden to: {args.since}")

    if args.check_only:
        initialise_gee()
        state = load_state()
        check = check_all_sources(state)
        print(json.dumps(check, indent=2, default=str))
    else:
        run_watcher(force=args.force)
