"""
=============================================================
  COCOA TRADING ASSISTANT — Crop Health Monitor  v2.0
=============================================================
  Uses Google Earth Engine + multi-sensor satellite data to
  monitor vegetation health and moisture conditions across
  key cocoa growing regions of Côte d'Ivoire and Ghana.

  v2.0 CHANGES (from v1.0):
  ─────────────────────────────────────────────────────────
  1.  NDMI replaces NDWI  — (NIR−SWIR)/(NIR+SWIR) using
      B8/B11 (Gao moisture index) instead of McFeeters
      water index. Correctly targets leaf/canopy moisture.
  2.  Weekly composites replace monthly — catches rapid
      deterioration during critical growth windows.
  3.  EVI integrated — primary vegetation index for dense
      tropical canopy (NDVI saturates >0.8). NDVI kept as
      secondary.
  4.  SMAP soil moisture — leading indicator. Soil dries
      2–4 weeks before vegetation responds.
  5.  ET₀ evapotranspiration — atmospheric moisture demand.
      Wired into stress signal via moisture deficit ratio.
  6.  CHIRPS gridded rainfall — replaces sparse station
      data for backward-looking rainfall history.
  7.  ESA WorldCover land use mask — restricts analysis to
      tree cover pixels, excluding cities, savanna, water.
  8.  Sentinel-1 SAR fallback — cloud-penetrating radar
      provides moisture proxy when optical imagery fails.
  9.  Spatial stress fraction — % of pixels below threshold,
      not just regional mean. Catches localised drought.
  10. Cache and diff — stores run-to-run deltas for
      immediate change detection.
  11. Scoring decoupled from signal — named interaction
      signal leads, score calibrates severity within it.

  Indices computed:
    - EVI   (Enhanced Vegetation Index)       → Primary canopy health
    - NDVI  (Normalised Difference Veg Index) → Secondary canopy health
    - NDMI  (Normalised Difference Moisture)  → Canopy/leaf moisture
    - LST   (Land Surface Temperature, MODIS) → Heat stress
    - SMAP  (Soil Moisture Active Passive)    → Soil moisture (leading)
    - CHIRPS (Gridded Rainfall Estimate)      → Historical rainfall
    - S1 VV/VH (Sentinel-1 SAR backscatter)  → Cloud-free moisture proxy

  Output:
    - cocoa_crop_health.json  → structured data for the agent
    - cocoa_ndvi_chart.png    → EVI/NDMI trend chart
    - cocoa_crop_diff.json    → run-to-run change cache

  SETUP:
  ──────
  Same as v1.0 — requires Google Earth Engine account and
  earthengine-api. No additional dependencies.
=============================================================
"""

import os
import json
import logging
from datetime import datetime, timezone, timedelta
from dateutil.relativedelta import relativedelta
from statistics import mean as smean, stdev as sstdev
from copy import deepcopy

import ee
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

GEE_PROJECT = os.getenv("GEE_PROJECT", "")

OUTPUT_JSON  = "cocoa_crop_health.json"
OUTPUT_CHART = "cocoa_ndvi_chart.png"
DIFF_JSON    = "cocoa_crop_diff.json"

# How many months of history to maintain (weekly granularity within)
LOOKBACK_MONTHS = 12

# Cloud cover threshold for Sentinel-2 image filtering (%)
MAX_CLOUD_COVER = 20

# Minimum Sentinel-2 images per week to consider valid
MIN_IMAGES_PER_WEEK = 1

# SMAP / CHIRPS / SAR availability flags
ENABLE_SMAP    = True
ENABLE_CHIRPS  = True
ENABLE_SAR     = True


# ─────────────────────────────────────────────
#  COCOA GROWING REGIONS
# ─────────────────────────────────────────────

COCOA_REGIONS = {
    "CoteIvoire_West": {
        "bounds":      [-8.0, 4.5, -5.5, 7.5],
        "country":     "Côte d'Ivoire",
        "description": "Main western cocoa belt — highest density farms",
        "weight":      0.40,
    },
    "CoteIvoire_Central": {
        "bounds":      [-5.5, 5.0, -3.5, 8.0],
        "country":     "Côte d'Ivoire",
        "description": "Central belt including Yamoussoukro region",
        "weight":      0.25,
    },
    "Ghana_Ashanti": {
        "bounds":      [-2.5, 5.8, -0.5, 7.5],
        "country":     "Ghana",
        "description": "Ashanti region — core Ghana cocoa belt",
        "weight":      0.20,
    },
    "Ghana_Western": {
        "bounds":      [-3.2, 4.8, -1.5, 6.5],
        "country":     "Ghana",
        "description": "Western region — second major Ghana belt",
        "weight":      0.15,
    },
}


# ─────────────────────────────────────────────
#  GEE INITIALISATION
# ─────────────────────────────────────────────

def initialise_gee():
    """
    Initialise Google Earth Engine.
    Supports service account or OAuth credentials.
    """
    import google.oauth2.service_account as sa

    log.info("Initialising Google Earth Engine...")

    if not GEE_PROJECT:
        log.error("GEE_PROJECT is not set in your .env file.")
        log.error("Add this line:  GEE_PROJECT=your-project-id-here")
        raise SystemExit(1)

    key_file = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

    try:
        if key_file and os.path.exists(key_file):
            log.info(f"  Using service account credentials: {key_file}")
            credentials = sa.Credentials.from_service_account_file(
                key_file,
                scopes=["https://www.googleapis.com/auth/earthengine"]
            )
            ee.Initialize(credentials=credentials, project=GEE_PROJECT)
        else:
            log.info("  Using OAuth credentials (from ee.Authenticate())")
            ee.Initialize(project=GEE_PROJECT)

        log.info(f"  GEE initialised (project: {GEE_PROJECT})")

    except ee.EEException as e:
        err = str(e).lower()
        log.error(f"GEE initialisation failed: {e}")
        if "not authenticated" in err or "credentials" in err or "permission" in err:
            log.error("")
            log.error("Try one of these fixes:")
            log.error("  Option A: python -c \"import ee; ee.Authenticate()\"")
            log.error("  Option B: Set GOOGLE_APPLICATION_CREDENTIALS in .env")
        raise SystemExit(1)

    except Exception as e:
        log.error(f"Unexpected error during GEE init: {e}")
        raise SystemExit(1)


# ─────────────────────────────────────────────
#  ESA WORLDCOVER LAND USE MASK
# ─────────────────────────────────────────────

def get_tree_cover_mask(geometry: ee.Geometry) -> ee.Image:
    """
    Create a binary mask from ESA WorldCover 10m restricting
    analysis to tree cover pixels (class 10).

    This excludes cities, bare soil, water, grassland, cropland
    etc. that would dilute vegetation indices. In West African
    cocoa regions, tree cover class 10 is overwhelmingly cocoa
    agroforestry and tropical forest.

    Falls back to an all-ones mask if WorldCover unavailable.
    """
    try:
        worldcover = ee.ImageCollection(
            "ESA/WorldCover/v200"
        ).first().clip(geometry)

        # Class 10 = Tree cover
        tree_mask = worldcover.eq(10)
        return tree_mask
    except Exception as e:
        log.warning(f"  WorldCover mask unavailable ({e}) — using full region")
        return ee.Image.constant(1).clip(geometry)


# ─────────────────────────────────────────────
#  SENTINEL-2 VEGETATION & MOISTURE INDICES
# ─────────────────────────────────────────────

def mask_s2_clouds(image):
    """Mask clouds using Sentinel-2 QA60 band."""
    qa = image.select("QA60")
    cloud_bit_mask  = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
           qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    return image.updateMask(mask).divide(10000)


def add_indices(image):
    """
    Compute EVI, NDVI, and NDMI from Sentinel-2 bands.

    Key change from v1: NDMI replaces NDWI.
      - NDMI = (NIR − SWIR1) / (NIR + SWIR1) = (B8 − B11) / (B8 + B11)
        → Sensitive to leaf water content and canopy moisture
        → The correct index for drought/moisture stress detection
      - Old NDWI was (Green − NIR) / (Green + NIR) = McFeeters index
        → Designed for open water body detection, NOT vegetation moisture

    EVI is now the primary vegetation health index because NDVI
    saturates in dense tropical canopy (>0.8). EVI corrects for
    atmospheric and soil background effects.
    """
    # NDVI = (NIR - Red) / (NIR + Red)
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")

    # EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
    evi = image.expression(
        "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
        {
            "NIR":  image.select("B8"),
            "RED":  image.select("B4"),
            "BLUE": image.select("B2"),
        }
    ).rename("EVI")

    # NDMI = (NIR - SWIR1) / (NIR + SWIR1)  — Gao (1996)
    # This is the vegetation moisture index, NOT the McFeeters water index
    ndmi = image.normalizedDifference(["B8", "B11"]).rename("NDMI")

    return image.addBands([ndvi, evi, ndmi])


def fetch_weekly_indices(region_name: str, bounds: list,
                         start_date: str, end_date: str,
                         tree_mask: ee.Image = None) -> dict:
    """
    Compute mean EVI, NDVI, NDMI for a region over a ~7 day window.

    If tree_mask is provided, analysis is restricted to tree cover
    pixels only (ESA WorldCover class 10).

    Also computes spatial stress fraction — the percentage of pixels
    below the stress threshold. This catches localised drought that
    the regional mean would mask.

    Returns dict with mean values, image count, and stress fractions.
    """
    geometry = ee.Geometry.Rectangle(bounds)

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(geometry)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD_COVER))
        .map(mask_s2_clouds)
        .map(add_indices)
    )

    count = collection.size().getInfo()
    if count == 0:
        return {
            "evi": None, "ndvi": None, "ndmi": None,
            "image_count": 0,
            "stress_fraction_ndmi": None,
            "stress_fraction_evi": None,
        }

    # Take median composite to reduce remaining cloud/shadow noise
    composite = collection.median()

    # Apply land use mask if available
    if tree_mask is not None:
        composite = composite.updateMask(tree_mask)

    # ── Regional mean statistics ───────────────────────
    stats = composite.select(["EVI", "NDVI", "NDMI"]).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geometry,
        scale=250,          # v2: finer scale than v1's 500m
        maxPixels=1e9,
        bestEffort=True,
    ).getInfo()

    # ── Spatial stress fractions ────────────────────────
    # What % of tree-cover pixels are below stress thresholds?
    # This catches localised drought pockets the mean would miss.
    stress_fracs = {}
    for band, threshold, key in [
        ("NDMI", -0.10, "stress_fraction_ndmi"),   # NDMI < -0.10 = moisture stress
        ("EVI",   0.30, "stress_fraction_evi"),     # EVI < 0.30 = vegetation stress
    ]:
        try:
            band_img = composite.select(band)
            stressed = band_img.lt(threshold)
            # Mean of binary (0/1) image = fraction of stressed pixels
            frac = stressed.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=250,
                maxPixels=1e9,
                bestEffort=True,
            ).getInfo()
            stress_fracs[key] = round((frac.get(band, 0) or 0) * 100, 1)
        except Exception:
            stress_fracs[key] = None

    return {
        "evi":                 round(stats.get("EVI", 0) or 0, 4),
        "ndvi":                round(stats.get("NDVI", 0) or 0, 4),
        "ndmi":                round(stats.get("NDMI", 0) or 0, 4),
        "image_count":         count,
        "stress_fraction_ndmi": stress_fracs.get("stress_fraction_ndmi"),
        "stress_fraction_evi":  stress_fracs.get("stress_fraction_evi"),
    }


# ─────────────────────────────────────────────
#  MODIS LAND SURFACE TEMPERATURE
# ─────────────────────────────────────────────

def fetch_weekly_lst(bounds: list, start_date: str, end_date: str) -> dict:
    """
    Fetch mean daytime Land Surface Temperature from MODIS MOD11A2.
    Returns temperature in Celsius.

    MOD11A2 is an 8-day composite — query window extended ±4 days
    to reliably catch composites that overlap the target week.
    """
    geometry = ee.Geometry.Rectangle(bounds)

    # Extend window to catch 8-day composites
    start_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=4)
    end_dt   = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=4)
    wide_start = start_dt.strftime("%Y-%m-%d")
    wide_end   = end_dt.strftime("%Y-%m-%d")

    collection = (
        ee.ImageCollection("MODIS/061/MOD11A2")
        .filterBounds(geometry)
        .filterDate(wide_start, wide_end)
        .select("LST_Day_1km")
    )

    count = collection.size().getInfo()
    if count == 0:
        log.info(f"    LST: no MODIS MOD11A2 images ({wide_start} to {wide_end})")
        return {"lst_celsius": None, "image_count": 0}

    def kelvin_to_celsius(image):
        return image.multiply(0.02).subtract(273.15)

    mean_lst = (
        collection
        .map(kelvin_to_celsius)
        .mean()
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=1000,
            maxPixels=1e9,
            bestEffort=True,
        )
        .getInfo()
    )

    lst_val = mean_lst.get("LST_Day_1km")
    if lst_val is not None:
        log.info(f"    LST: {lst_val:.1f}°C ({count} images)")
    return {
        "lst_celsius":  round(lst_val, 2) if lst_val else None,
        "image_count":  count,
    }


# ─────────────────────────────────────────────
#  SMAP SOIL MOISTURE (LEADING INDICATOR)
# ─────────────────────────────────────────────

def fetch_weekly_smap(bounds: list, start_date: str, end_date: str) -> dict:
    """
    Fetch NASA SMAP Level-4 soil moisture (surface + root zone).

    Soil moisture is a LEADING INDICATOR for vegetation stress:
    soil dries 2-4 weeks before canopy NDMI responds. A divergence
    where SMAP drops but NDMI holds is an early warning.

    Uses SPL4SMGP v008 (Level-4) instead of SPL3SMP_E (Level-3):
      - L4 provides CONTINUOUS data (no gaps during instrument outages)
      - L4 includes root-zone moisture (0-100cm) — more relevant for
        cocoa trees with deep root systems than surface-only (0-5cm)
      - L4 is model-assimilated, so it fills spatial/temporal gaps
      - L3 v005 stopped at 2023-12-03; v006 has data from 2023-12-04

    Resolution: ~9km, 3-hourly composites.
    Dataset: NASA/SMAP/SPL4SMGP/008
    Bands:
      sm_surface    — surface soil moisture (0-5cm), m³/m³
      sm_rootzone   — root-zone soil moisture (0-100cm), m³/m³
    """
    if not ENABLE_SMAP:
        return {"soil_moisture": None, "soil_moisture_rootzone": None,
                "smap_image_count": 0}

    geometry = ee.Geometry.Rectangle(bounds)

    try:
        collection = (
            ee.ImageCollection("NASA/SMAP/SPL4SMGP/008")
            .filterBounds(geometry)
            .filterDate(start_date, end_date)
            .select(["sm_surface", "sm_rootzone"])
        )

        count = collection.size().getInfo()
        if count == 0:
            # Fallback: try L3 v006 (has data from 2023-12-04)
            log.info("    SPL4SMGP empty — trying SPL3SMP_E/006 fallback")
            collection = (
                ee.ImageCollection("NASA/SMAP/SPL3SMP_E/006")
                .filterBounds(geometry)
                .filterDate(start_date, end_date)
                .select("soil_moisture_am")
            )
            count = collection.size().getInfo()
            if count == 0:
                return {"soil_moisture": None, "soil_moisture_rootzone": None,
                        "smap_image_count": 0}

            # L3 fallback — surface only, with quality masking
            # Mask poor-quality retrievals (retrieval_qual_flag_am: 0 = good)
            def mask_l3_quality(img):
                return img.updateMask(
                    img.select("soil_moisture_am").gt(0).And(
                        img.select("soil_moisture_am").lt(0.6)))
            collection = collection.map(mask_l3_quality)

            mean_sm = (
                collection.mean()
                .reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=geometry,
                    scale=9000,
                    maxPixels=1e9,
                    bestEffort=True,
                ).getInfo()
            )
            sm_val = mean_sm.get("soil_moisture_am")
            return {
                "soil_moisture":          round(sm_val, 4) if sm_val else None,
                "soil_moisture_rootzone": None,
                "smap_image_count":       count,
                "smap_source":            "SPL3SMP_E/006",
            }

        # L4 primary path — has both surface and root-zone
        mean_sm = (
            collection.mean()
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=9000,
                maxPixels=1e9,
                bestEffort=True,
            )
            .getInfo()
        )

        sm_surface  = mean_sm.get("sm_surface")
        sm_rootzone = mean_sm.get("sm_rootzone")

        return {
            "soil_moisture":          round(sm_surface, 4) if sm_surface else None,
            "soil_moisture_rootzone": round(sm_rootzone, 4) if sm_rootzone else None,
            "smap_image_count":       count,
            "smap_source":            "SPL4SMGP/008",
        }

    except Exception as e:
        log.warning(f"  SMAP fetch failed ({e}) — continuing without soil moisture")
        return {"soil_moisture": None, "soil_moisture_rootzone": None,
                "smap_image_count": 0}


# ─────────────────────────────────────────────
#  CHIRPS GRIDDED RAINFALL
# ─────────────────────────────────────────────

def fetch_weekly_chirps(bounds: list, start_date: str, end_date: str) -> dict:
    """
    Fetch CHIRPS daily rainfall estimates, summed over the period.

    CHIRPS (Climate Hazards group InfraRed Precipitation with Station)
    provides gridded daily rainfall at 5km resolution across Africa.
    Far more spatially complete than station data.

    This gives actual historical rainfall over each region (not a
    forecast), anchoring the backward-looking precipitation assessment.

    Dataset: UCSB-CHG/CHIRPS/DAILY
    Band: precipitation (mm/day)
    """
    if not ENABLE_CHIRPS:
        return {"chirps_rainfall_mm": None, "chirps_days": 0}

    geometry = ee.Geometry.Rectangle(bounds)

    try:
        collection = (
            ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
            .filterBounds(geometry)
            .filterDate(start_date, end_date)
            .select("precipitation")
        )

        count = collection.size().getInfo()
        if count == 0:
            return {"chirps_rainfall_mm": None, "chirps_days": 0}

        # Sum daily rainfall over the period
        total_rain = (
            collection.sum()
            .reduceRegion(
                reducer=ee.Reducer.mean(),    # spatial mean of the sum
                geometry=geometry,
                scale=5000,
                maxPixels=1e9,
                bestEffort=True,
            )
            .getInfo()
        )

        rain_val = total_rain.get("precipitation")

        # Also compute daily mean for the period
        daily_mean = round(rain_val / count, 2) if rain_val and count > 0 else None

        return {
            "chirps_rainfall_mm":  round(rain_val, 1) if rain_val else None,
            "chirps_daily_mean_mm": daily_mean,
            "chirps_days":          count,
        }

    except Exception as e:
        log.warning(f"  CHIRPS fetch failed ({e}) — continuing without gridded rainfall")
        return {"chirps_rainfall_mm": None, "chirps_days": 0}


# ─────────────────────────────────────────────
#  MODIS LAI (Leaf Area Index)
# ─────────────────────────────────────────────

def fetch_weekly_lai(bounds: list, start_date: str, end_date: str) -> dict:
    """
    Fetch MODIS Leaf Area Index (MCD15A2H v061, 8-day composite, 500m).

    Uses the 8-day product (more reliably available on GEE than the 4-day
    MCD15A3H). Query window extended ±4 days to ensure we catch composites
    that overlap our target week.

    LAI measures canopy structure, not just colour. Healthy cocoa LAI is
    3.0-5.0. A drop below 2.5 signals structural problems (poor pod set,
    defoliation, disease) that standard NDVI/EVI will only show weeks later.
    """
    geometry = ee.Geometry.Rectangle(bounds)

    try:
        # Extend the query window by 4 days each side to catch 8-day
        # composites that overlap with our target week
        start_dt = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=4)
        end_dt   = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=4)
        wide_start = start_dt.strftime("%Y-%m-%d")
        wide_end   = end_dt.strftime("%Y-%m-%d")

        collection = (
            ee.ImageCollection("MODIS/061/MCD15A2H")
            .filterBounds(geometry)
            .filterDate(wide_start, wide_end)
            .select("Lai")
        )

        count = collection.size().getInfo()
        if count == 0:
            log.info(f"    LAI: no MODIS MCD15A2H images ({wide_start} to {wide_end})")
            return {"lai": None, "lai_image_count": 0}

        # MODIS LAI values are stored as 0-100, scale factor 0.1 → real LAI 0-10
        def scale_lai(img):
            return img.multiply(0.1)

        scaled = collection.map(scale_lai)
        mean_lai = (
            scaled.mean()
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=500,
                maxPixels=1e9,
                bestEffort=True,
            ).getInfo()
        )

        lai_val = mean_lai.get("Lai")
        if lai_val is not None:
            log.info(f"    LAI: {lai_val:.2f} ({count} images)")
        return {
            "lai":              round(lai_val, 2) if lai_val else None,
            "lai_image_count":  count,
        }

    except Exception as e:
        log.warning(f"  LAI fetch failed ({e})")
        return {"lai": None, "lai_image_count": 0}


# ─────────────────────────────────────────────
#  CHIRPS RAINFALL ANOMALY vs CLIMATOLOGY
# ─────────────────────────────────────────────

CHIRPS_CLIM_YEARS = 10   # Compare to 10-year average

def fetch_rainfall_anomaly(bounds: list, start_date: str, end_date: str) -> dict:
    """
    Compare current week's rainfall to the climatological normal
    for the same calendar week, using CHIRPS historical data.

    Returns the current week's total, the historical average for
    the same week-of-year over the last 10 years, and the anomaly
    as a percentage.

    This tells the agent whether "58mm this week" is normal or
    worryingly below average for this time of year.
    """
    geometry = ee.Geometry.Rectangle(bounds)

    try:
        # Parse the start date to get day-of-year range
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt   = datetime.strptime(end_date, "%Y-%m-%d")
        doy_start = start_dt.timetuple().tm_yday
        doy_end   = end_dt.timetuple().tm_yday
        current_year = start_dt.year

        # Current week's rainfall
        current_rain = (
            ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
            .filterBounds(geometry)
            .filterDate(start_date, end_date)
            .select("precipitation")
        )
        current_count = current_rain.size().getInfo()
        current_sum = None
        if current_count > 0:
            current_total = current_rain.sum().reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=5000,
                maxPixels=1e9,
                bestEffort=True,
            ).getInfo()
            current_sum = current_total.get("precipitation")

        # Historical same-week average over CHIRPS_CLIM_YEARS
        historical_totals = []
        for year_offset in range(1, CHIRPS_CLIM_YEARS + 1):
            hist_year = current_year - year_offset
            try:
                hist_start = f"{hist_year}-{start_dt.strftime('%m-%d')}"
                hist_end   = f"{hist_year}-{end_dt.strftime('%m-%d')}"
                # Validate dates (catches Feb 29 in non-leap years)
                datetime.strptime(hist_start, "%Y-%m-%d")
            except ValueError:
                continue
            try:
                hist_rain = (
                    ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
                    .filterBounds(geometry)
                    .filterDate(hist_start, hist_end)
                    .select("precipitation")
                )
                if hist_rain.size().getInfo() > 0:
                    hist_total = hist_rain.sum().reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=geometry,
                        scale=5000,
                        maxPixels=1e9,
                        bestEffort=True,
                    ).getInfo()
                    val = hist_total.get("precipitation")
                    if val is not None:
                        historical_totals.append(val)
            except Exception:
                continue

        clim_mean = None
        anomaly_pct = None
        if historical_totals:
            clim_mean = round(sum(historical_totals) / len(historical_totals), 1)
            if clim_mean > 0 and current_sum is not None:
                anomaly_pct = round((current_sum - clim_mean) / clim_mean * 100, 1)

        return {
            "rainfall_current_mm":   round(current_sum, 1) if current_sum else None,
            "rainfall_clim_mean_mm": clim_mean,
            "rainfall_anomaly_pct":  anomaly_pct,
            "rainfall_clim_years":   len(historical_totals),
        }

    except Exception as e:
        log.warning(f"  Rainfall anomaly fetch failed ({e})")
        return {"rainfall_current_mm": None, "rainfall_anomaly_pct": None}


# ─────────────────────────────────────────────
#  LST ANOMALY vs HISTORICAL NORMAL
# ─────────────────────────────────────────────

LST_CLIM_YEARS = 10

def fetch_lst_anomaly(bounds: list, start_date: str, end_date: str) -> dict:
    """
    Compare current week's Land Surface Temperature to the historical
    normal for the same calendar week.

    Heat stress during flowering directly reduces cherelle survival.
    An LST anomaly of +2°C during mid-crop flowering is a significant
    signal for reduced pod set.
    """
    geometry = ee.Geometry.Rectangle(bounds)

    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt   = datetime.strptime(end_date, "%Y-%m-%d")
        current_year = start_dt.year

        # Extend window ±4 days for 8-day composites
        wide_start_dt = start_dt - timedelta(days=4)
        wide_end_dt   = end_dt + timedelta(days=4)

        # Current week's LST
        def kelvin_to_celsius(img):
            return img.multiply(0.02).subtract(273.15)

        current_col = (
            ee.ImageCollection("MODIS/061/MOD11A2")
            .filterBounds(geometry)
            .filterDate(wide_start_dt.strftime("%Y-%m-%d"),
                        wide_end_dt.strftime("%Y-%m-%d"))
            .select("LST_Day_1km")
            .map(kelvin_to_celsius)
        )
        current_count = current_col.size().getInfo()
        current_lst = None
        if current_count > 0:
            mean_vals = current_col.mean().reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=1000,
                maxPixels=1e9,
                bestEffort=True,
            ).getInfo()
            current_lst = mean_vals.get("LST_Day_1km")

        # Historical same-week average (also with ±4 day window)
        historical_lsts = []
        for year_offset in range(1, LST_CLIM_YEARS + 1):
            hist_year = current_year - year_offset
            try:
                hist_start = (wide_start_dt.replace(year=hist_year)).strftime("%Y-%m-%d")
                hist_end   = (wide_end_dt.replace(year=hist_year)).strftime("%Y-%m-%d")
            except ValueError:
                # Leap year edge case (Feb 29 in a non-leap year)
                continue
            try:
                hist_col = (
                    ee.ImageCollection("MODIS/061/MOD11A2")
                    .filterBounds(geometry)
                    .filterDate(hist_start, hist_end)
                    .select("LST_Day_1km")
                    .map(kelvin_to_celsius)
                )
                if hist_col.size().getInfo() > 0:
                    hist_vals = hist_col.mean().reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=geometry,
                        scale=1000,
                        maxPixels=1e9,
                        bestEffort=True,
                    ).getInfo()
                    val = hist_vals.get("LST_Day_1km")
                    if val is not None:
                        historical_lsts.append(val)
            except Exception:
                continue

        clim_mean = None
        anomaly_c = None
        if historical_lsts:
            clim_mean = round(sum(historical_lsts) / len(historical_lsts), 1)
            if current_lst is not None:
                anomaly_c = round(current_lst - clim_mean, 1)

        return {
            "lst_current_c":     round(current_lst, 1) if current_lst else None,
            "lst_clim_mean_c":   clim_mean,
            "lst_anomaly_c":     anomaly_c,
            "lst_clim_years":    len(historical_lsts),
        }

    except Exception as e:
        log.warning(f"  LST anomaly fetch failed ({e})")
        return {"lst_current_c": None, "lst_anomaly_c": None}


# ─────────────────────────────────────────────
#  SENTINEL-1 SAR FALLBACK
# ─────────────────────────────────────────────

def fetch_weekly_sar(bounds: list, start_date: str, end_date: str,
                     tree_mask: ee.Image = None) -> dict:
    """
    Fetch Sentinel-1 C-band SAR backscatter as a cloud-independent
    moisture proxy.

    SAR penetrates clouds entirely, so this provides continuity
    during wet-season optical imagery gaps. VH polarisation is
    most sensitive to vegetation moisture; VV to soil moisture.

    The VH/VV ratio is a useful vegetation moisture indicator
    that's less affected by incidence angle than raw backscatter.

    Dataset: COPERNICUS/S1_GRD (Ground Range Detected)
    Bands: VV, VH (dB)
    """
    if not ENABLE_SAR:
        return {"sar_vh": None, "sar_vv": None, "sar_vh_vv_ratio": None,
                "sar_image_count": 0}

    geometry = ee.Geometry.Rectangle(bounds)

    try:
        collection = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(geometry)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .select(["VV", "VH"])
        )

        count = collection.size().getInfo()
        if count == 0:
            return {"sar_vh": None, "sar_vv": None, "sar_vh_vv_ratio": None,
                    "sar_image_count": 0}

        composite = collection.mean()

        if tree_mask is not None:
            composite = composite.updateMask(tree_mask)

        stats = composite.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=250,
            maxPixels=1e9,
            bestEffort=True,
        ).getInfo()

        vv = stats.get("VV")
        vh = stats.get("VH")

        # VH/VV ratio in dB space: VH_dB - VV_dB = 10*log10(VH_lin/VV_lin)
        vh_vv_ratio = None
        if vv is not None and vh is not None:
            vh_vv_ratio = round(vh - vv, 3)

        return {
            "sar_vh":           round(vh, 3) if vh is not None else None,
            "sar_vv":           round(vv, 3) if vv is not None else None,
            "sar_vh_vv_ratio":  vh_vv_ratio,
            "sar_image_count":  count,
        }

    except Exception as e:
        log.warning(f"  Sentinel-1 SAR fetch failed ({e}) — continuing without radar data")
        return {"sar_vh": None, "sar_vv": None, "sar_vh_vv_ratio": None,
                "sar_image_count": 0}


# ─────────────────────────────────────────────
#  CHIRPS LONGER-TERM RAINFALL CONTEXT
# ─────────────────────────────────────────────

def fetch_chirps_context(bounds: list, end_date_str: str) -> dict:
    """
    Fetch 30-day, 60-day, and 90-day cumulative CHIRPS rainfall
    ending at end_date. Provides backward-looking precipitation
    context that is far more robust than sparse station data.
    """
    if not ENABLE_CHIRPS:
        return {}

    geometry = ee.Geometry.Rectangle(bounds)
    end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
    result = {}

    for days, key in [(30, "chirps_30d_mm"), (60, "chirps_60d_mm"), (90, "chirps_90d_mm")]:
        start_dt = end_dt - timedelta(days=days)
        try:
            collection = (
                ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
                .filterBounds(geometry)
                .filterDate(start_dt.strftime("%Y-%m-%d"), end_date_str)
                .select("precipitation")
            )
            total = collection.sum().reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=5000,
                maxPixels=1e9,
                bestEffort=True,
            ).getInfo()
            val = total.get("precipitation")
            result[key] = round(val, 1) if val else None
        except Exception:
            result[key] = None

    return result


# ─────────────────────────────────────────────
#  STRESS FLAGS
# ─────────────────────────────────────────────

def compute_stress_flags(evi: float, ndvi: float, ndmi: float,
                         lst: float, soil_moisture: float = None,
                         evi_prev: float = None,
                         ndmi_prev: float = None,
                         soil_moisture_prev: float = None,
                         stress_frac_ndmi: float = None,
                         stress_frac_evi: float = None,
                         week_key: str = None) -> list:
    """
    Apply agronomic thresholds to flag crop stress conditions.
    Thresholds calibrated for tropical West African cocoa agroforestry.

    v2 changes:
      - EVI is primary vegetation index (NDVI secondary)
      - NDMI replaces old NDWI with correct thresholds
      - SMAP soil moisture flags as leading indicator
      - Spatial stress fractions flag localised drought
      - Week key used for dry-season suppression

    NDMI thresholds (Gao index, NIR-SWIR):
      Dense healthy tropical:  +0.20 to +0.50
      Normal range:            +0.10 to +0.30
      Mild stress:              0.00 to +0.10
      Moderate stress:         -0.10 to  0.00
      Severe stress:           < -0.10
    """
    flags = []

    # Determine if we're in the West African dry season (Nov–Mar)
    dry_season = False
    if week_key:
        try:
            cal_month = int(week_key[5:7])
            dry_season = cal_month in (11, 12, 1, 2, 3)
        except (ValueError, IndexError):
            pass

    # ── EVI (primary vegetation health) ──────────────────
    if evi is not None:
        if evi < 0.20:
            flags.append({
                "type":     "CRITICAL",
                "signal":   "Very Low EVI",
                "value":    evi,
                "message":  "Severe vegetation stress or significant canopy loss"
            })
        elif evi < 0.30:
            flags.append({
                "type":     "WARNING",
                "signal":   "Low EVI",
                "value":    evi,
                "message":  "Below-normal vegetation health — monitor closely"
            })
        elif evi > 0.50:
            flags.append({
                "type":     "POSITIVE",
                "signal":   "Strong EVI",
                "value":    evi,
                "message":  "Strong vegetation health — good canopy conditions"
            })

        # Week-on-week EVI deterioration (suppressed in dry season)
        if evi_prev and evi < evi_prev * 0.90 and not dry_season:
            flags.append({
                "type":     "WARNING",
                "signal":   "EVI Declining",
                "value":    round((evi - evi_prev) / evi_prev * 100, 1),
                "message":  f"EVI dropped >10% week-on-week ({evi_prev:.3f} → {evi:.3f})"
            })

    # ── NDMI (canopy/leaf moisture — correct index) ─────
    if ndmi is not None:
        if ndmi < -0.10:
            flags.append({
                "type":     "CRITICAL",
                "signal":   "Severe Moisture Stress (NDMI)",
                "value":    ndmi,
                "message":  "Severe leaf moisture depletion — significant yield risk"
            })
        elif ndmi < 0.0:
            flags.append({
                "type":     "WARNING",
                "signal":   "Moderate Moisture Stress (NDMI)",
                "value":    ndmi,
                "message":  (
                    f"Below-normal leaf moisture "
                    f"({'dry season — monitor' if dry_season else 'stress likely'})"
                )
            })
        elif ndmi > 0.30:
            flags.append({
                "type":     "POSITIVE",
                "signal":   "Good Moisture (NDMI)",
                "value":    ndmi,
                "message":  "Leaf moisture levels healthy — favourable growing conditions"
            })

        # Week-on-week NDMI deterioration
        if ndmi_prev is not None and ndmi < ndmi_prev - 0.05 and not dry_season:
            flags.append({
                "type":     "WARNING",
                "signal":   "NDMI Declining",
                "value":    round(ndmi - ndmi_prev, 4),
                "message":  f"NDMI dropped >0.05 week-on-week ({ndmi_prev:.3f} → {ndmi:.3f})"
            })

    # ── SMAP Soil Moisture (leading indicator) ──────────
    if soil_moisture is not None:
        # West African cocoa belt soil moisture norms (cm³/cm³):
        #   Healthy:      0.25 – 0.45
        #   Mild stress:  0.15 – 0.25
        #   Moderate:     0.10 – 0.15
        #   Severe:       < 0.10
        if soil_moisture < 0.10:
            flags.append({
                "type":     "CRITICAL",
                "signal":   "Very Low Soil Moisture (SMAP)",
                "value":    soil_moisture,
                "message":  "Critically dry soil — vegetation stress will follow in 2-4 weeks"
            })
        elif soil_moisture < 0.15:
            flags.append({
                "type":     "WARNING",
                "signal":   "Low Soil Moisture (SMAP)",
                "value":    soil_moisture,
                "message":  "Soil moisture declining — leading indicator of coming vegetation stress"
            })
        elif soil_moisture > 0.35:
            flags.append({
                "type":     "POSITIVE",
                "signal":   "Good Soil Moisture (SMAP)",
                "value":    soil_moisture,
                "message":  "Soil moisture adequate — supports healthy root zone"
            })

        # Soil-to-canopy divergence: soil drying but NDMI still holding
        if (soil_moisture_prev is not None and
                soil_moisture < soil_moisture_prev * 0.85 and
                ndmi is not None and ndmi_prev is not None and
                ndmi >= ndmi_prev * 0.97):
            flags.append({
                "type":     "WARNING",
                "signal":   "Soil-Canopy Divergence",
                "value":    round(soil_moisture - soil_moisture_prev, 4),
                "message":  (
                    f"Soil moisture falling (SMAP: {soil_moisture_prev:.3f} → {soil_moisture:.3f}) "
                    f"but canopy NDMI holding ({ndmi_prev:.3f} → {ndmi:.3f}). "
                    f"EARLY WARNING — vegetation stress likely in 2-4 weeks."
                )
            })

    # ── LST (heat stress) ──────────────────────────────
    if lst is not None:
        if lst > 38:
            flags.append({
                "type":     "CRITICAL",
                "signal":   "Extreme Heat Stress (LST)",
                "value":    lst,
                "message":  f"Land surface temperature {lst:.1f}°C — above cocoa heat tolerance"
            })
        elif lst > 34:
            flags.append({
                "type":     "WARNING",
                "signal":   "Elevated Temperature (LST)",
                "value":    lst,
                "message":  f"Temperature {lst:.1f}°C — approaching upper stress threshold"
            })

    # ── Spatial stress fractions ────────────────────────
    if stress_frac_ndmi is not None and stress_frac_ndmi > 25:
        severity = "CRITICAL" if stress_frac_ndmi > 50 else "WARNING"
        flags.append({
            "type":     severity,
            "signal":   "Localised Moisture Stress",
            "value":    stress_frac_ndmi,
            "message":  (
                f"{stress_frac_ndmi:.0f}% of tree-cover pixels below NDMI stress threshold. "
                f"Regional mean may mask localised drought pockets."
            )
        })

    if stress_frac_evi is not None and stress_frac_evi > 30:
        severity = "CRITICAL" if stress_frac_evi > 50 else "WARNING"
        flags.append({
            "type":     severity,
            "signal":   "Localised Vegetation Stress",
            "value":    stress_frac_evi,
            "message":  (
                f"{stress_frac_evi:.0f}% of tree-cover pixels below EVI stress threshold. "
                f"Indicates spatially widespread canopy deterioration."
            )
        })

    return flags


# ─────────────────────────────────────────────
#  SEASONAL ANOMALIES
# ─────────────────────────────────────────────

def compute_seasonal_anomalies(weekly_data: dict) -> dict:
    """
    Enrich each week's data with seasonal anomaly fields by comparing
    each reading to the historical average for the same calendar week
    across all available years.

    v2 changes: Computes anomalies for NDMI (replacing NDWI), EVI
    (replacing NDVI as primary), and soil moisture.
    """
    # Group values by calendar week (WW) across all years
    by_cal_week = {}
    for week_key, data in weekly_data.items():
        try:
            dt = datetime.strptime(week_key, "%Y-%m-%d")
            cal_week = dt.strftime("%W")
        except ValueError:
            cal_week = week_key[5:7]

        if cal_week not in by_cal_week:
            by_cal_week[cal_week] = {"ndmi": [], "evi": [], "soil_moisture": []}
        if data.get("ndmi") is not None:
            by_cal_week[cal_week]["ndmi"].append(data["ndmi"])
        if data.get("evi") is not None:
            by_cal_week[cal_week]["evi"].append(data["evi"])
        if data.get("soil_moisture") is not None:
            by_cal_week[cal_week]["soil_moisture"].append(data["soil_moisture"])

    # Enrich each week
    enriched = {}
    for week_key, data in weekly_data.items():
        try:
            dt = datetime.strptime(week_key, "%Y-%m-%d")
            cal_week = dt.strftime("%W")
        except ValueError:
            cal_week = week_key[5:7]

        hist  = by_cal_week.get(cal_week, {"ndmi": [], "evi": [], "soil_moisture": []})
        entry = dict(data)

        for key in ["ndmi", "evi", "soil_moisture"]:
            vals    = hist.get(key, [])
            current = data.get(key)
            n       = len(vals)

            if current is None or n == 0:
                entry[f"{key}_seasonal_mean"] = None
                entry[f"{key}_anomaly"]       = None
                entry[f"{key}_anomaly_pct"]   = None
                entry[f"{key}_zscore"]        = None
                continue

            s_mean  = round(smean(vals), 4)
            anomaly = round(current - s_mean, 4)
            anomaly_pct = round(anomaly / abs(s_mean) * 100, 1) if s_mean != 0 else None

            zscore = None
            if n >= 3:
                sd = sstdev(vals)
                zscore = round(anomaly / sd, 2) if sd > 0 else 0.0

            entry[f"{key}_seasonal_mean"] = s_mean
            entry[f"{key}_anomaly"]       = anomaly
            entry[f"{key}_anomaly_pct"]   = anomaly_pct
            entry[f"{key}_zscore"]        = zscore

        # Seasonal context label
        ndmi_anom = entry.get("ndmi_anomaly")
        n_periods = len(hist.get("ndmi", []))
        if ndmi_anom is None or n_periods < 2:
            label = f"Seasonal baseline building ({n_periods} period(s) of data)"
        elif ndmi_anom > 0.05:
            label = f"Above seasonal average (NDMI {ndmi_anom:+.3f} vs norm, n={n_periods})"
        elif ndmi_anom < -0.05:
            label = f"Below seasonal average (NDMI {ndmi_anom:+.3f} vs norm, n={n_periods})"
        else:
            label = f"Near seasonal average (NDMI {ndmi_anom:+.3f} vs norm, n={n_periods})"

        entry["seasonal_n"]     = n_periods
        entry["seasonal_label"] = label

        # Relative (seasonal) stress flags
        rel_flags = []
        ndmi_z = entry.get("ndmi_zscore")
        evi_z  = entry.get("evi_zscore")
        sm_z   = entry.get("soil_moisture_zscore")

        if ndmi_z is not None:
            if ndmi_z <= -2.0:
                rel_flags.append({
                    "type": "CRITICAL", "signal": "Anomalous Dry (NDMI)",
                    "value": ndmi_z,
                    "message": f"NDMI is {ndmi_z:.1f} std devs below seasonal norm — unusually dry"
                })
            elif ndmi_z <= -1.0:
                rel_flags.append({
                    "type": "WARNING", "signal": "Drier Than Seasonal Norm (NDMI)",
                    "value": ndmi_z,
                    "message": f"NDMI {ndmi_z:.1f} std devs below seasonal avg"
                })
            elif ndmi_z >= 1.0:
                rel_flags.append({
                    "type": "POSITIVE", "signal": "Wetter Than Seasonal Norm (NDMI)",
                    "value": ndmi_z,
                    "message": f"NDMI {ndmi_z:.1f} std devs above seasonal avg"
                })

        if evi_z is not None:
            if evi_z <= -1.5:
                rel_flags.append({
                    "type": "WARNING", "signal": "Below-Seasonal Vegetation (EVI)",
                    "value": evi_z,
                    "message": f"EVI {evi_z:.1f} std devs below seasonal norm"
                })
            elif evi_z >= 1.5:
                rel_flags.append({
                    "type": "POSITIVE", "signal": "Above-Seasonal Vegetation (EVI)",
                    "value": evi_z,
                    "message": f"EVI {evi_z:.1f} std devs above seasonal norm"
                })

        if sm_z is not None and sm_z <= -1.5:
            rel_flags.append({
                "type": "WARNING", "signal": "Anomalous Dry Soil (SMAP)",
                "value": sm_z,
                "message": f"Soil moisture {sm_z:.1f} std devs below seasonal norm — leading indicator"
            })

        entry["seasonal_flags"] = rel_flags
        enriched[week_key] = entry

    return enriched


# ─────────────────────────────────────────────
#  OVERALL CROP SIGNAL
# ─────────────────────────────────────────────

def overall_crop_signal(all_region_data: dict) -> dict:
    """
    Compute a weighted overall crop health signal across all regions.

    v2: EVI primary, NDMI replaces NDWI, soil moisture integrated,
    stress fractions feed into penalty, score and named signal decoupled.
    """
    weighted_evi   = 0.0
    weighted_ndmi  = 0.0
    weighted_sm    = 0.0
    total_weight   = 0.0
    sm_weight      = 0.0
    critical_count = 0
    warning_count  = 0
    spatial_penalty = 0.0

    for region_name, data in all_region_data.items():
        weight    = COCOA_REGIONS.get(region_name, {}).get("weight", 0)
        weekly    = data.get("weekly", data.get("monthly", {}))
        week_keys = sorted(weekly.keys())
        if not week_keys:
            continue

        # Find the most recent week that has at least SOME data.
        # If the latest week is all-None (e.g., just started, no
        # satellite has processed it yet), fall back to prior weeks.
        latest = None
        for wk in reversed(week_keys):
            candidate = weekly[wk]
            if (candidate.get("evi") is not None
                    or candidate.get("soil_moisture") is not None
                    or candidate.get("ndmi") is not None):
                latest = candidate
                break
        if latest is None:
            # Every week is empty for this region — skip it
            continue
        evi    = latest.get("evi")
        ndmi   = latest.get("ndmi")
        sm     = latest.get("soil_moisture")
        flags  = latest.get("stress_flags", [])

        if evi:
            weighted_evi += evi * weight
            total_weight += weight
        if ndmi:
            weighted_ndmi += ndmi * weight
        if sm:
            weighted_sm += sm * weight
            sm_weight   += weight

        critical_count += sum(1 for f in flags if f["type"] == "CRITICAL")
        warning_count  += sum(1 for f in flags if f["type"] == "WARNING")

        frac_ndmi = latest.get("stress_fraction_ndmi", 0) or 0
        if frac_ndmi > 25:
            spatial_penalty += (frac_ndmi - 25) * 0.1 * weight

    if total_weight == 0:
        return {"score": None, "signal": "Insufficient data", "bias": "NEUTRAL",
                "named_signal": "INSUFFICIENT_DATA"}

    avg_evi  = weighted_evi / total_weight
    avg_ndmi = weighted_ndmi / total_weight
    avg_sm   = weighted_sm / sm_weight if sm_weight > 0 else None

    evi_score  = min(100, max(0, (avg_evi - 0.15) / (0.55 - 0.15) * 100))
    ndmi_score = min(100, max(0, (avg_ndmi + 0.15) / (0.40 + 0.15) * 100))

    sm_modifier = 0
    if avg_sm is not None:
        sm_score = min(100, max(0, (avg_sm - 0.08) / (0.40 - 0.08) * 100))
        sm_modifier = (sm_score - 50) * 0.15

    base_score = (evi_score * 0.40 + ndmi_score * 0.40) + sm_modifier
    raw_penalty    = (critical_count * 8) + (warning_count * 3) + spatial_penalty
    stress_penalty = min(raw_penalty, base_score * 0.35)
    score = max(0, round(base_score - stress_penalty))

    # Named signal (decoupled from score)
    if critical_count >= 3 or score < 20:
        named_signal = "SEVERE_STRESS"
        bias   = "STRONGLY_BULLISH"
        signal = "Severe crop stress across multiple indicators — high supply risk"
    elif critical_count >= 1 or score < 35:
        named_signal = "MODERATE_STRESS"
        bias   = "BULLISH"
        signal = "Meaningful crop stress — supply risk emerging"
    elif warning_count >= 3 or score < 50:
        named_signal = "MILD_STRESS"
        bias   = "WEAKLY_BULLISH"
        signal = "Mild crop stress — some supply concern, monitor closely"
    elif score > 80:
        named_signal = "EXCELLENT_CONDITIONS"
        bias   = "BEARISH"
        signal = "Excellent crop health — strong supply outlook"
    elif score > 65:
        named_signal = "GOOD_CONDITIONS"
        bias   = "WEAKLY_BEARISH"
        signal = "Good crop health — supply broadly supportive"
    else:
        named_signal = "NORMAL_CONDITIONS"
        bias   = "NEUTRAL"
        signal = "Average crop conditions — no clear supply signal"

    return {
        "score":            score,
        "named_signal":     named_signal,
        "avg_evi":          round(avg_evi, 4),
        "avg_ndmi":         round(avg_ndmi, 4),
        "avg_soil_moisture": round(avg_sm, 4) if avg_sm else None,
        "signal":           signal,
        "bias":             bias,
        "critical_flags":   critical_count,
        "warning_flags":    warning_count,
    }


# ─────────────────────────────────────────────
#  WEEKLY DATA COLLECTION LOOP
# ─────────────────────────────────────────────

def generate_week_boundaries(lookback_months: int = LOOKBACK_MONTHS) -> list:
    """
    Generate (week_start, week_end, week_key) tuples covering
    the lookback period in 7-day increments aligned to Mondays.

    Stops before creating a current-week entry that covers fewer
    than 3 days — no satellite source will have processed data
    for a week that just started, so including it guarantees an
    all-None row that overwrites the actual latest readings.
    """
    today  = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start  = today - relativedelta(months=lookback_months)
    start  = start - timedelta(days=start.weekday())

    weeks = []
    current = start
    while current < today:
        week_end = min(current + timedelta(days=7), today)
        # Skip the current partial week if it covers fewer than 3 days —
        # satellites need processing time, so a 1-2 day window will always
        # return all Nones and mask the real latest data.
        days_in_week = (week_end - current).days
        if days_in_week < 3:
            break
        weeks.append((
            current.strftime("%Y-%m-%d"),
            week_end.strftime("%Y-%m-%d"),
            current.strftime("%Y-%m-%d"),
        ))
        current += timedelta(days=7)

    return weeks


def collect_region_data(region_name: str, bounds: list,
                        lookback_months: int = LOOKBACK_MONTHS) -> dict:
    """
    Collect weekly vegetation, moisture, soil, and rainfall data
    for a single region over the lookback period.
    """
    log.info(f"  Processing region: {region_name}")

    geometry = ee.Geometry.Rectangle(bounds)
    tree_mask = get_tree_cover_mask(geometry)

    weeks   = generate_week_boundaries(lookback_months)
    weekly  = {}

    prev_evi  = None
    prev_ndmi = None
    prev_sm   = None

    for start_str, end_str, week_key in weeks:
        log.info(f"    → {week_key}...")

        s2_data     = fetch_weekly_indices(region_name, bounds, start_str, end_str, tree_mask)
        lst_data    = fetch_weekly_lst(bounds, start_str, end_str)
        smap_data   = fetch_weekly_smap(bounds, start_str, end_str)
        chirps_data = fetch_weekly_chirps(bounds, start_str, end_str)
        sar_data    = fetch_weekly_sar(bounds, start_str, end_str, tree_mask)
        lai_data    = fetch_weekly_lai(bounds, start_str, end_str)
        rain_anom   = fetch_rainfall_anomaly(bounds, start_str, end_str)
        lst_anom    = fetch_lst_anomaly(bounds, start_str, end_str)

        optical_gap = s2_data.get("image_count", 0) == 0
        if optical_gap and sar_data.get("sar_image_count", 0) > 0:
            s2_data["optical_gap"] = True
            s2_data["sar_fallback"] = True
            log.info(f"      ⚡ No optical imagery — using SAR fallback")

        flags = compute_stress_flags(
            evi=s2_data.get("evi"),
            ndvi=s2_data.get("ndvi"),
            ndmi=s2_data.get("ndmi"),
            lst=lst_data.get("lst_celsius"),
            soil_moisture=smap_data.get("soil_moisture"),
            evi_prev=prev_evi,
            ndmi_prev=prev_ndmi,
            soil_moisture_prev=prev_sm,
            stress_frac_ndmi=s2_data.get("stress_fraction_ndmi"),
            stress_frac_evi=s2_data.get("stress_fraction_evi"),
            week_key=week_key,
        )

        weekly[week_key] = {
            **s2_data,
            "lst_celsius":          lst_data.get("lst_celsius"),
            "soil_moisture":            smap_data.get("soil_moisture"),
            "soil_moisture_rootzone":   smap_data.get("soil_moisture_rootzone"),
            "smap_image_count":        smap_data.get("smap_image_count", 0),
            "smap_source":             smap_data.get("smap_source", ""),
            "chirps_rainfall_mm":   chirps_data.get("chirps_rainfall_mm"),
            "chirps_daily_mean_mm": chirps_data.get("chirps_daily_mean_mm"),
            # LAI
            "lai": lai_data.get("lai"),
            "lai_image_count": lai_data.get("lai_image_count", 0),
            # Rainfall anomaly vs climatology
            "rainfall_current_mm":   rain_anom.get("rainfall_current_mm"),
            "rainfall_clim_mean_mm": rain_anom.get("rainfall_clim_mean_mm"),
            "rainfall_anomaly_pct":  rain_anom.get("rainfall_anomaly_pct"),
            # LST anomaly vs climatology
            "lst_anomaly_c":     lst_anom.get("lst_anomaly_c"),
            "lst_clim_mean_c":   lst_anom.get("lst_clim_mean_c"),
            **sar_data,
            "stress_flags":         flags,
            "date_range":           f"{start_str} to {end_str}",
        }

        prev_evi  = s2_data.get("evi")
        prev_ndmi = s2_data.get("ndmi")
        prev_sm   = smap_data.get("soil_moisture")

    weekly = compute_seasonal_anomalies(weekly)

    # Fetch CHIRPS 30/60/90d context
    latest_week = sorted(weekly.keys())[-1] if weekly else None
    chirps_context = {}
    if latest_week:
        chirps_context = fetch_chirps_context(bounds, latest_week)

    return {
        "region_name":    region_name,
        "country":        COCOA_REGIONS[region_name]["country"],
        "description":    COCOA_REGIONS[region_name]["description"],
        "weight":         COCOA_REGIONS[region_name]["weight"],
        "weekly":         weekly,
        "chirps_context": chirps_context,
    }


# ─────────────────────────────────────────────
#  CHART GENERATION
# ─────────────────────────────────────────────

def generate_charts(all_region_data: dict, output_path: str = OUTPUT_CHART):
    """Generate a 3-panel chart: EVI, NDMI, and Soil Moisture trends."""
    log.info("Generating trend charts...")

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.patch.set_facecolor("#1a1a2e")

    colors = {
        "CoteIvoire_West":    "#e94560",
        "CoteIvoire_Central": "#f5a623",
        "Ghana_Ashanti":      "#4ecdc4",
        "Ghana_Western":      "#a8e6cf",
    }

    for ax in axes:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444")
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color("white")

    def get_series(data, field):
        weekly = data.get("weekly", data.get("monthly", {}))
        sorted_keys = sorted(weekly.keys())
        dates, vals = [], []
        for k in sorted_keys:
            v = weekly[k].get(field)
            if v is not None:
                try:
                    dates.append(datetime.strptime(k, "%Y-%m-%d"))
                except ValueError:
                    dates.append(datetime.strptime(k, "%Y-%m"))
                vals.append(v)
        return dates, vals

    # Panel 1: EVI
    ax1 = axes[0]
    for rn, data in all_region_data.items():
        d, v = get_series(data, "evi")
        if d:
            ax1.plot(d, v, marker=".", markersize=3,
                     color=colors.get(rn, "white"),
                     label=rn.replace("_", " "), linewidth=1.5, alpha=0.8)
    ax1.axhspan(0.0, 0.20, alpha=0.15, color="red", label="Critical")
    ax1.axhspan(0.20, 0.30, alpha=0.10, color="orange", label="Warning")
    ax1.axhspan(0.45, 0.80, alpha=0.08, color="green", label="Healthy")
    ax1.set_title("EVI — Vegetation Health (Weekly)", color="white", fontsize=11, pad=8)
    ax1.set_ylabel("EVI", color="white")
    ax1.set_ylim(0, 0.7)
    ax1.legend(loc="upper left", fontsize=7, facecolor="#16213e", labelcolor="white", framealpha=0.7)
    ax1.grid(True, alpha=0.2, color="white")

    # Panel 2: NDMI
    ax2 = axes[1]
    for rn, data in all_region_data.items():
        d, v = get_series(data, "ndmi")
        if d:
            ax2.plot(d, v, marker=".", markersize=3,
                     color=colors.get(rn, "white"),
                     label=rn.replace("_", " "), linewidth=1.5, alpha=0.8)
    ax2.axhline(0.0, color="orange", linestyle="--", alpha=0.6, linewidth=1, label="Moderate stress")
    ax2.axhline(-0.10, color="red", linestyle="--", alpha=0.6, linewidth=1, label="Severe stress")
    ax2.axhspan(-0.5, -0.10, alpha=0.12, color="red")
    ax2.set_title("NDMI — Leaf/Canopy Moisture (Weekly)", color="white", fontsize=11, pad=8)
    ax2.set_ylabel("NDMI", color="white")
    ax2.set_ylim(-0.3, 0.5)
    ax2.legend(loc="upper left", fontsize=7, facecolor="#16213e", labelcolor="white", framealpha=0.7)
    ax2.grid(True, alpha=0.2, color="white")

    # Panel 3: Soil Moisture
    ax3 = axes[2]
    for rn, data in all_region_data.items():
        d, v = get_series(data, "soil_moisture")
        if d:
            ax3.plot(d, v, marker=".", markersize=3,
                     color=colors.get(rn, "white"),
                     label=rn.replace("_", " "), linewidth=1.5, alpha=0.8)
    ax3.axhline(0.15, color="orange", linestyle="--", alpha=0.6, linewidth=1, label="Stress threshold")
    ax3.axhline(0.10, color="red", linestyle="--", alpha=0.6, linewidth=1, label="Critical threshold")
    ax3.set_title("Soil Moisture — SMAP (Leading Indicator)", color="white", fontsize=11, pad=8)
    ax3.set_ylabel("cm³/cm³", color="white")
    ax3.set_ylim(0, 0.50)
    ax3.legend(loc="upper left", fontsize=7, facecolor="#16213e", labelcolor="white", framealpha=0.7)
    ax3.grid(True, alpha=0.2, color="white")

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.suptitle(
        f"🍫 Cocoa Crop Health Monitor v2 — West Africa\n"
        f"Generated: {datetime.now().strftime('%Y-%m-%d')}",
        color="white", fontsize=13, y=1.01
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    log.info(f"  ✅ Chart saved to: {output_path}")


# ─────────────────────────────────────────────
#  CACHE AND DIFF
# ─────────────────────────────────────────────

def compute_diff(old_snapshot: dict, new_snapshot: dict) -> dict:
    """
    Compare the latest readings between two snapshots and produce
    a structured diff showing what changed since the last run.
    """
    diff = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "old_generated_at": old_snapshot.get("generated_at"),
        "regions": {},
    }

    old_regions = old_snapshot.get("regions", {})
    new_regions = new_snapshot.get("regions", {})

    for region_name in new_regions:
        old_data = old_regions.get(region_name, {})
        new_data = new_regions.get(region_name, {})

        old_weekly = old_data.get("weekly", old_data.get("monthly", {}))
        new_weekly = new_data.get("weekly", new_data.get("monthly", {}))

        # Find latest week with actual data in each snapshot
        def find_latest_with_data(weekly_dict):
            for wk in sorted(weekly_dict.keys(), reverse=True):
                c = weekly_dict[wk]
                if (c.get("evi") is not None
                        or c.get("soil_moisture") is not None
                        or c.get("ndmi") is not None):
                    return wk
            return sorted(weekly_dict.keys())[-1] if weekly_dict else None

        old_latest_key = find_latest_with_data(old_weekly) if old_weekly else None
        new_latest_key = find_latest_with_data(new_weekly) if new_weekly else None
        if not old_latest_key or not new_latest_key:
            continue

        old_latest = old_weekly[old_latest_key]
        new_latest = new_weekly[new_latest_key]

        region_diff = {"old_week": old_latest_key, "new_week": new_latest_key}

        for field in ["evi", "ndvi", "ndmi", "lst_celsius", "soil_moisture",
                      "chirps_rainfall_mm", "stress_fraction_ndmi", "stress_fraction_evi"]:
            old_val = old_latest.get(field)
            new_val = new_latest.get(field)
            if old_val is not None and new_val is not None:
                change = round(new_val - old_val, 4)
                pct = round(change / abs(old_val) * 100, 1) if old_val != 0 else None
                region_diff[field] = {"old": old_val, "new": new_val,
                                      "change": change, "change_pct": pct}

        diff["regions"][region_name] = region_diff

    old_overall = old_snapshot.get("overall_signal", {})
    new_overall = new_snapshot.get("overall_signal", {})
    old_score = old_overall.get("score")
    new_score = new_overall.get("score")
    if old_score is not None and new_score is not None:
        diff["overall_score_change"] = new_score - old_score
        diff["old_score"] = old_score
        diff["new_score"] = new_score
        diff["old_named_signal"] = old_overall.get("named_signal", old_overall.get("bias"))
        diff["new_named_signal"] = new_overall.get("named_signal", new_overall.get("bias"))

    return diff


def save_diff(diff: dict, filepath: str = DIFF_JSON):
    """Save the diff cache to disk."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(diff, f, indent=2, default=str)
    log.info(f"  ✅ Diff saved to: {filepath}")


# ─────────────────────────────────────────────
#  CONSOLE SUMMARY
# ─────────────────────────────────────────────

def print_summary(snapshot: dict):
    overall = snapshot.get("overall_signal", {})
    print("\n" + "=" * 62)
    print("  🍫 COCOA CROP HEALTH MONITOR v2 — Summary")
    print("=" * 62)
    print(f"  Generated   : {snapshot.get('generated_at', '')[:16]}")
    print(f"  Period      : {snapshot.get('period', '')}")
    print()
    print("  ── Overall Signal ───────────────────────────────────")
    print(f"  Health Score:    {overall.get('score')} / 100")
    print(f"  Named Signal:    {overall.get('named_signal', 'N/A')}")
    print(f"  Avg EVI:         {overall.get('avg_evi')}")
    print(f"  Avg NDMI:        {overall.get('avg_ndmi')}")
    print(f"  Avg Soil Moist.: {overall.get('avg_soil_moisture', 'N/A')}")
    print(f"  Bias:            {overall.get('bias')}")
    print(f"  Signal:          {overall.get('signal')}")
    print(f"  Flags:           🔴 {overall.get('critical_flags')} critical  "
          f"⚠️  {overall.get('warning_flags')} warnings")
    print()

    for region_name, data in snapshot.get("regions", {}).items():
        weekly = data.get("weekly", data.get("monthly", {}))
        if not weekly:
            continue

        # Find the most recent week with actual data (not all-None)
        latest = None
        latest_key = None
        for wk in sorted(weekly.keys(), reverse=True):
            candidate = weekly[wk]
            if (candidate.get("evi") is not None
                    or candidate.get("soil_moisture") is not None
                    or candidate.get("ndmi") is not None):
                latest = candidate
                latest_key = wk
                break
        if latest is None:
            # Fall back to the most recent key even if all None
            latest_key = sorted(weekly.keys())[-1]
            latest = weekly[latest_key]
        print(f"\n  {region_name} ({data.get('country')})  [week: {latest_key}]")
        print(f"    EVI:  {latest.get('evi')}   NDVI: {latest.get('ndvi')}   "
              f"NDMI: {latest.get('ndmi')}   LST: {latest.get('lst_celsius')}°C   "
              f"LAI: {latest.get('lai')}")
        print(f"    Soil moisture: {latest.get('soil_moisture', 'N/A')} cm³/cm³   "
              f"CHIRPS rain: {latest.get('chirps_rainfall_mm', 'N/A')} mm")

        rain_anom = latest.get("rainfall_anomaly_pct")
        if rain_anom is not None:
            emoji = "🟢" if abs(rain_anom) < 20 else "🟡" if abs(rain_anom) < 40 else "🔴"
            print(f"    Rainfall vs normal: {rain_anom:+.0f}% {emoji} "
                  f"(clim avg: {latest.get('rainfall_clim_mean_mm', '?')} mm)")

        lst_anom = latest.get("lst_anomaly_c")
        if lst_anom is not None:
            emoji = "🟢" if abs(lst_anom) < 1.0 else "🟡" if abs(lst_anom) < 2.0 else "🔴"
            print(f"    LST vs normal: {lst_anom:+.1f}°C {emoji}")

        print(f"    Stress pixels: NDMI {latest.get('stress_fraction_ndmi', 'N/A')}%   "
              f"EVI {latest.get('stress_fraction_evi', 'N/A')}%")

        if latest.get("optical_gap"):
            print(f"    ⚡ OPTICAL GAP — using SAR fallback this week")

        flags = latest.get("stress_flags", []) + latest.get("seasonal_flags", [])
        flag_str = " ".join(["🔴" if f["type"] == "CRITICAL" else
                             "⚠️" if f["type"] == "WARNING" else "🟢"
                             for f in flags]) or "✅"
        print(f"    Flags: {flag_str}")
        for f in flags:
            print(f"      → [{f['type']}] {f['signal']}: {f['message']}")

    print("\n" + "=" * 62 + "\n")


# ─────────────────────────────────────────────
#  INCREMENTAL UPDATE
# ─────────────────────────────────────────────

def collect_new_weeks_for_region(region_name: str, bounds: list,
                                 existing_weekly: dict) -> dict:
    """Fetch only weeks that are missing or need refreshing."""
    today = datetime.now(timezone.utc)
    geometry = ee.Geometry.Rectangle(bounds)
    tree_mask = get_tree_cover_mask(geometry)

    all_weeks = generate_week_boundaries()
    current_week_key = all_weeks[-1][2] if all_weeks else None
    prev_week_key    = all_weeks[-2][2] if len(all_weeks) > 1 else None

    weeks_to_fetch = []
    for start_str, end_str, week_key in all_weeks:
        existing = existing_weekly.get(week_key)
        if existing is None:
            weeks_to_fetch.append((start_str, end_str, week_key, "missing"))
        elif week_key == current_week_key:
            weeks_to_fetch.append((start_str, end_str, week_key, "refresh_current"))
        elif week_key == prev_week_key and (existing.get("image_count", 1) == 0):
            weeks_to_fetch.append((start_str, end_str, week_key, "was_empty"))

    if not weeks_to_fetch:
        log.info(f"  {region_name}: all weeks up to date")
        return {}

    log.info(f"  {region_name}: fetching {len(weeks_to_fetch)} week(s)")

    new_data  = {}
    prev_evi  = None
    prev_ndmi = None
    prev_sm   = None

    sorted_existing = sorted(existing_weekly.keys())
    if sorted_existing:
        first_fetch = weeks_to_fetch[0][2]
        prior = [k for k in sorted_existing if k < first_fetch]
        if prior:
            prev_data = existing_weekly[prior[-1]]
            prev_evi  = prev_data.get("evi")
            prev_ndmi = prev_data.get("ndmi")
            prev_sm   = prev_data.get("soil_moisture")

    for start_str, end_str, week_key, reason in weeks_to_fetch:
        log.info(f"    → {week_key} ({reason})...")

        s2_data   = fetch_weekly_indices(region_name, bounds, start_str, end_str, tree_mask)
        lst_data  = fetch_weekly_lst(bounds, start_str, end_str)
        smap_data = fetch_weekly_smap(bounds, start_str, end_str)
        chirps_data = fetch_weekly_chirps(bounds, start_str, end_str)
        sar_data  = fetch_weekly_sar(bounds, start_str, end_str, tree_mask)
        lai_data  = fetch_weekly_lai(bounds, start_str, end_str)
        rain_anom = fetch_rainfall_anomaly(bounds, start_str, end_str)
        lst_anom  = fetch_lst_anomaly(bounds, start_str, end_str)

        optical_gap = s2_data.get("image_count", 0) == 0
        if optical_gap and sar_data.get("sar_image_count", 0) > 0:
            s2_data["optical_gap"] = True
            s2_data["sar_fallback"] = True

        flags = compute_stress_flags(
            evi=s2_data.get("evi"), ndvi=s2_data.get("ndvi"),
            ndmi=s2_data.get("ndmi"), lst=lst_data.get("lst_celsius"),
            soil_moisture=smap_data.get("soil_moisture"),
            evi_prev=prev_evi, ndmi_prev=prev_ndmi, soil_moisture_prev=prev_sm,
            stress_frac_ndmi=s2_data.get("stress_fraction_ndmi"),
            stress_frac_evi=s2_data.get("stress_fraction_evi"),
            week_key=week_key,
        )

        new_data[week_key] = {
            **s2_data,
            "lst_celsius": lst_data.get("lst_celsius"),
            "soil_moisture": smap_data.get("soil_moisture"),
            "soil_moisture_rootzone": smap_data.get("soil_moisture_rootzone"),
            "smap_image_count": smap_data.get("smap_image_count", 0),
            "smap_source": smap_data.get("smap_source", ""),
            "chirps_rainfall_mm": chirps_data.get("chirps_rainfall_mm"),
            "chirps_daily_mean_mm": chirps_data.get("chirps_daily_mean_mm"),
            # LAI
            "lai": lai_data.get("lai"),
            "lai_image_count": lai_data.get("lai_image_count", 0),
            # Rainfall anomaly vs climatology
            "rainfall_current_mm":   rain_anom.get("rainfall_current_mm"),
            "rainfall_clim_mean_mm": rain_anom.get("rainfall_clim_mean_mm"),
            "rainfall_anomaly_pct":  rain_anom.get("rainfall_anomaly_pct"),
            # LST anomaly vs climatology
            "lst_anomaly_c":     lst_anom.get("lst_anomaly_c"),
            "lst_clim_mean_c":   lst_anom.get("lst_clim_mean_c"),
            **sar_data,
            "stress_flags": flags,
            "date_range": f"{start_str} to {end_str}",
        }

        prev_evi  = s2_data.get("evi")
        prev_ndmi = s2_data.get("ndmi")
        prev_sm   = smap_data.get("soil_moisture")

    return new_data


def run_incremental_update(generate_chart: bool = True) -> dict:
    """Incremental update — only fetches new or stale weeks."""
    log.info("=" * 62)
    log.info("  COCOA CROP HEALTH MONITOR v2 — Incremental Update")
    log.info(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 62)

    existing_snapshot = None
    try:
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            existing_snapshot = json.load(f)
        existing_regions = existing_snapshot.get("regions", {})
        log.info(f"Loaded existing data from {OUTPUT_JSON}")
    except FileNotFoundError:
        log.info("No existing data found — running full pull")
        return run_crop_monitor(generate_chart=generate_chart)

    initialise_gee()
    today = datetime.now(timezone.utc)

    all_region_data = {}
    total_new_weeks = 0

    for region_name, config in COCOA_REGIONS.items():
        existing_region = existing_regions.get(region_name, {})
        existing_weekly = existing_region.get("weekly", existing_region.get("monthly", {}))

        try:
            new_weeks = collect_new_weeks_for_region(region_name, config["bounds"], existing_weekly)
            merged_weekly = {**existing_weekly, **new_weeks}
            cutoff = (today - relativedelta(months=LOOKBACK_MONTHS)).strftime("%Y-%m-%d")
            merged_weekly = {k: v for k, v in merged_weekly.items() if k >= cutoff}
            merged_weekly = compute_seasonal_anomalies(merged_weekly)

            chirps_context = {}
            if merged_weekly:
                latest_key = sorted(merged_weekly.keys())[-1]
                chirps_context = fetch_chirps_context(config["bounds"], latest_key)

            all_region_data[region_name] = {
                "region_name": region_name,
                "country": COCOA_REGIONS[region_name]["country"],
                "description": COCOA_REGIONS[region_name]["description"],
                "weight": COCOA_REGIONS[region_name]["weight"],
                "weekly": merged_weekly,
                "chirps_context": chirps_context,
            }
            total_new_weeks += len(new_weeks)

        except Exception as e:
            log.error(f"Failed to update {region_name}: {e}")
            all_region_data[region_name] = existing_region

    log.info(f"Incremental update complete: {total_new_weeks} week(s) refreshed")

    overall = overall_crop_signal(all_region_data)
    start_date = (today - relativedelta(months=LOOKBACK_MONTHS)).strftime("%Y-%m-%d")

    snapshot = {
        "generated_at": today.isoformat(),
        "period": f"{start_date} to {today.strftime('%Y-%m-%d')}",
        "lookback_months": LOOKBACK_MONTHS,
        "version": "2.0",
        "granularity": "weekly",
        "last_full_run": existing_snapshot.get("last_full_run",
                         existing_snapshot.get("generated_at")),
        "overall_signal": overall,
        "regions": all_region_data,
    }

    diff = compute_diff(existing_snapshot, snapshot)
    save_diff(diff)
    snapshot["last_diff"] = diff

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, default=str)
    log.info(f"  ✅ Updated data saved to: {OUTPUT_JSON}")

    if generate_chart:
        try:
            generate_charts(all_region_data)
        except Exception as e:
            log.warning(f"Chart generation failed (non-critical): {e}")

    print_summary(snapshot)
    return snapshot


# ─────────────────────────────────────────────
#  FULL RUN
# ─────────────────────────────────────────────

def run_crop_monitor(generate_chart: bool = True) -> dict:
    """Full pipeline: initialise GEE → collect → signal → save → chart → diff."""
    log.info("=" * 62)
    log.info("  COCOA CROP HEALTH MONITOR v2")
    log.info(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 62)

    prev_snapshot = None
    try:
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            prev_snapshot = json.load(f)
    except FileNotFoundError:
        pass

    initialise_gee()
    today = datetime.now(timezone.utc)
    start_date = (today - relativedelta(months=LOOKBACK_MONTHS)).strftime("%Y-%m-%d")
    end_date = today.strftime("%Y-%m-%d")

    all_region_data = {}
    log.info(f"Collecting {LOOKBACK_MONTHS} months of weekly satellite data "
             f"for {len(COCOA_REGIONS)} regions...")

    for region_name, config in COCOA_REGIONS.items():
        try:
            all_region_data[region_name] = collect_region_data(
                region_name, config["bounds"], LOOKBACK_MONTHS)
        except Exception as e:
            log.error(f"Failed to collect data for {region_name}: {e}")
            all_region_data[region_name] = {
                "region_name": region_name, "error": str(e), "weekly": {}}

    overall = overall_crop_signal(all_region_data)

    snapshot = {
        "generated_at": today.isoformat(),
        "last_full_run": today.isoformat(),
        "period": f"{start_date} to {end_date}",
        "lookback_months": LOOKBACK_MONTHS,
        "version": "2.0",
        "granularity": "weekly",
        "overall_signal": overall,
        "regions": all_region_data,
    }

    if prev_snapshot:
        diff = compute_diff(prev_snapshot, snapshot)
        save_diff(diff)
        snapshot["last_diff"] = diff

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, default=str)
    log.info(f"  ✅ Data saved to: {OUTPUT_JSON}")

    if generate_chart:
        try:
            generate_charts(all_region_data)
        except Exception as e:
            log.warning(f"Chart generation failed (non-critical): {e}")

    print_summary(snapshot)
    return snapshot


# ─────────────────────────────────────────────
#  INTEGRATION HELPER
# ─────────────────────────────────────────────

def load_crop_health_for_agent(filepath: str = OUTPUT_JSON) -> dict:
    """
    Load and summarise crop health data for injection into
    the trading agent's prompt. v2: NDMI, EVI, SMAP, CHIRPS, SAR, diff.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {"error": "Crop health data not available — run cocoa_crop_monitor.py"}

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

        # Find the most recent week with actual data
        latest_key = None
        latest = None
        for wk in sorted(weekly.keys(), reverse=True):
            candidate = weekly[wk]
            if (candidate.get("evi") is not None
                    or candidate.get("soil_moisture") is not None
                    or candidate.get("ndmi") is not None):
                latest_key = wk
                latest = candidate
                break
        if latest is None:
            latest_key = sorted(weekly.keys())[-1]
            latest = weekly[latest_key]
        flags     = [f["signal"] for f in latest.get("stress_flags", [])]
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
        "stale": age_days > 14 if age_days else None,
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


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cocoa Crop Health Monitor v2")
    parser.add_argument("--no-chart", action="store_true", help="Skip chart generation")
    parser.add_argument("--incremental", action="store_true",
                        help="Only fetch new/missing weeks (much faster)")
    parser.add_argument("--lookback", type=int, default=LOOKBACK_MONTHS,
                        help=f"Months of history (default: {LOOKBACK_MONTHS})")
    parser.add_argument("--no-smap", action="store_true", help="Disable SMAP soil moisture")
    parser.add_argument("--no-chirps", action="store_true", help="Disable CHIRPS rainfall")
    parser.add_argument("--no-sar", action="store_true", help="Disable Sentinel-1 SAR")
    args = parser.parse_args()

    LOOKBACK_MONTHS = args.lookback
    if args.no_smap:
        ENABLE_SMAP = False
    if args.no_chirps:
        ENABLE_CHIRPS = False
    if args.no_sar:
        ENABLE_SAR = False

    if args.incremental:
        run_incremental_update(generate_chart=not args.no_chart)
    else:
        run_crop_monitor(generate_chart=not args.no_chart)
