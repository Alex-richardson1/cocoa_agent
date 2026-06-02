"""
=============================================================
  COCOA TRADING ASSISTANT — Combined Stress Signal  v2.0
=============================================================
  Synthesises multi-sensor satellite data with weather forecasts
  to produce a structured supply stress signal for the agent.

  v2.0 CHANGES:
  ─────────────────────────────────────────────────────────
  1.  NDMI replaces NDWI — correct moisture index (Gao)
  2.  ET₀ (evapotranspiration) integrated — computes moisture
      deficit ratio (rainfall / ET₀) as a drought indicator
  3.  CHIRPS historical rainfall replaces sparse station data
      for backward-looking precipitation assessment
  4.  SMAP soil moisture as leading indicator input
  5.  Scoring DECOUPLED from signal — named interaction signal
      leads the prompt, score calibrates severity within it
  6.  Prompt restructured: signal → context → score
      (was: score → signal → context)

  The core NDMI × rainfall interaction matrix is preserved
  but enhanced with new data dimensions.
=============================================================
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  THRESHOLDS  (v2 — recalibrated for NDMI)
# ─────────────────────────────────────────────

# NDMI thresholds (Gao index — NIR-SWIR / NIR+SWIR)
# Dense healthy tropical canopy: +0.20 to +0.50
# Normal range:                  +0.10 to +0.30
NDMI_SEVERE_STRESS   = -0.10    # Severe depletion of leaf water
NDMI_MODERATE_STRESS =  0.00    # Moderate stress — yield risk emerging
NDMI_MILD_STRESS     =  0.10    # Mild stress — early warning
NDMI_HEALTHY         =  0.20    # Adequate / healthy moisture

# 7-day forecast rainfall thresholds (mm)
RAIN_VERY_DRY   =   5
RAIN_DRY        =  20
RAIN_ADEQUATE   =  50
RAIN_GOOD       =  80
RAIN_EXCESSIVE  = 120

# SMAP soil moisture thresholds (cm³/cm³)
SOIL_CRITICAL   = 0.10
SOIL_STRESS     = 0.15
SOIL_ADEQUATE   = 0.25
SOIL_GOOD       = 0.35

# ET₀ moisture deficit thresholds
# Deficit ratio = (actual rainfall) / (ET₀ demand)
# < 0.3  = severe deficit (rain covers <30% of demand)
# < 0.6  = moderate deficit
# < 1.0  = mild deficit
# >= 1.0 = surplus
DEFICIT_SEVERE   = 0.30
DEFICIT_MODERATE = 0.60
DEFICIT_MILD     = 1.00

# Cocoa belt region weights
REGION_WEIGHTS = {
    "CoteIvoire_West":    0.40,
    "CoteIvoire_Central": 0.25,
    "Ghana_Ashanti":      0.20,
    "Ghana_Western":      0.15,
}

# Map weather location names to crop region names
WEATHER_TO_REGION = {
    "Abidjan_CoteIvoire": ["CoteIvoire_West", "CoteIvoire_Central"],
    "Yamoussoukro_CI":    ["CoteIvoire_Central", "CoteIvoire_West"],
    "Accra_Ghana":        ["Ghana_Ashanti", "Ghana_Western"],
    "Kumasi_Ghana":       ["Ghana_Ashanti", "Ghana_Western"],
}


# ─────────────────────────────────────────────
#  CROP CALENDAR  (unchanged from v1)
# ─────────────────────────────────────────────

CROP_CALENDAR = {
    1:  {"multiplier": 1.20, "stage": "Main Crop Pod Filling",
         "crop": "Main Crop", "sensitivity": "HIGH",
         "notes": "Peak pod filling. Moisture stress reduces bean size/weight."},
    2:  {"multiplier": 1.15, "stage": "Main Crop Late Pod Fill / Early Harvest",
         "crop": "Main Crop", "sensitivity": "HIGH",
         "notes": "Pod filling largely complete; early harvest beginning."},
    3:  {"multiplier": 0.90, "stage": "Main Crop Harvest Peak",
         "crop": "Main Crop", "sensitivity": "MEDIUM",
         "notes": "Peak main crop harvest. Stress affects quality more than volume."},
    4:  {"multiplier": 1.10, "stage": "Mid Crop Flowering",
         "crop": "Mid Crop", "sensitivity": "HIGH",
         "notes": "Mid crop flowering onset. Moisture stress reduces pod set."},
    5:  {"multiplier": 1.15, "stage": "Mid Crop Pod Development",
         "crop": "Mid Crop", "sensitivity": "HIGH",
         "notes": "Critical moisture window — stress causes pod abortion."},
    6:  {"multiplier": 1.05, "stage": "Mid Crop Pod Filling",
         "crop": "Mid Crop", "sensitivity": "MEDIUM-HIGH",
         "notes": "Mid crop pod filling. Moisture needed for bean development."},
    7:  {"multiplier": 0.85, "stage": "Mid Crop Harvest / Vegetative Rest",
         "crop": "Mid Crop", "sensitivity": "LOW-MEDIUM",
         "notes": "Harvesting. Trees entering recovery phase."},
    8:  {"multiplier": 0.80, "stage": "Vegetative Recovery",
         "crop": "Inter-Crop", "sensitivity": "LOW",
         "notes": "Recovery period. Chronic stress weakens main crop flowering."},
    9:  {"multiplier": 1.30, "stage": "Main Crop Flowering",
         "crop": "Main Crop", "sensitivity": "CRITICAL",
         "notes": "Most supply-sensitive month. Bad September moves markets for 6 months."},
    10: {"multiplier": 1.35, "stage": "Main Crop Flowering / Early Pod Set",
         "crop": "Main Crop", "sensitivity": "CRITICAL",
         "notes": "Peak flowering + critical pod set. Highest sensitivity month."},
    11: {"multiplier": 1.25, "stage": "Main Crop Early Pod Development",
         "crop": "Main Crop", "sensitivity": "VERY HIGH",
         "notes": "Young pods highly vulnerable. Second most critical month."},
    12: {"multiplier": 1.20, "stage": "Main Crop Pod Development",
         "crop": "Main Crop", "sensitivity": "HIGH",
         "notes": "Continued pod development. Moisture stress causes pod loss."},
}


def get_crop_calendar_context(month: int = None) -> dict:
    if month is None:
        month = datetime.now(timezone.utc).month
    return {
        "month":       month,
        "month_name":  datetime(2000, month, 1).strftime("%B"),
        **CROP_CALENDAR.get(month, {
            "multiplier": 1.0, "stage": "Unknown", "crop": "Unknown",
            "sensitivity": "UNKNOWN", "notes": "No crop calendar data.",
        })
    }


# ─────────────────────────────────────────────
#  CLASSIFIERS
# ─────────────────────────────────────────────

def classify_ndmi(ndmi: float) -> dict:
    """Classify an NDMI value into a stress level with label and score."""
    if ndmi is None:
        return {"level": "unknown", "label": "No data", "score": 50}
    if ndmi <= NDMI_SEVERE_STRESS:
        return {"level": "severe",   "label": "Severe moisture stress",   "score": 10}
    elif ndmi <= NDMI_MODERATE_STRESS:
        return {"level": "moderate", "label": "Moderate moisture stress", "score": 30}
    elif ndmi <= NDMI_MILD_STRESS:
        return {"level": "mild",     "label": "Mild moisture stress",     "score": 50}
    elif ndmi <= NDMI_HEALTHY:
        return {"level": "adequate", "label": "Adequate moisture",        "score": 70}
    else:
        return {"level": "healthy",  "label": "Well hydrated",            "score": 90}


def classify_rainfall(rain_7d: float) -> dict:
    """Classify 7-day forecast rainfall into a supply impact level."""
    if rain_7d is None:
        return {"level": "unknown", "label": "No forecast data", "score": 50,
                "trend": "unknown"}
    if rain_7d <= RAIN_VERY_DRY:
        return {"level": "very_dry",   "label": "Essentially no rain",
                "score": 5,  "trend": "worsening"}
    elif rain_7d <= RAIN_DRY:
        return {"level": "dry",        "label": "Below-normal rainfall",
                "score": 25, "trend": "worsening"}
    elif rain_7d <= RAIN_ADEQUATE:
        return {"level": "adequate",   "label": "Moderate rainfall",
                "score": 55, "trend": "stable"}
    elif rain_7d <= RAIN_GOOD:
        return {"level": "good",       "label": "Good rainfall",
                "score": 75, "trend": "improving"}
    elif rain_7d <= RAIN_EXCESSIVE:
        return {"level": "excellent",  "label": "Excellent rainfall",
                "score": 90, "trend": "improving"}
    else:
        return {"level": "excessive",  "label": "Excessive rainfall — disease risk",
                "score": 40, "trend": "risk"}


def classify_soil_moisture(sm: float) -> dict:
    """Classify SMAP soil moisture into stress level."""
    if sm is None:
        return {"level": "unknown", "label": "No SMAP data", "score": 50}
    if sm < SOIL_CRITICAL:
        return {"level": "critical",  "label": "Critically dry soil",    "score": 5}
    elif sm < SOIL_STRESS:
        return {"level": "stressed",  "label": "Soil moisture depleted", "score": 25}
    elif sm < SOIL_ADEQUATE:
        return {"level": "mild",      "label": "Below-normal soil moisture", "score": 50}
    elif sm < SOIL_GOOD:
        return {"level": "adequate",  "label": "Adequate soil moisture", "score": 70}
    else:
        return {"level": "good",      "label": "Good soil moisture",     "score": 90}


def classify_moisture_deficit(rain_7d: float, et0_7d: float) -> dict:
    """
    Classify the moisture deficit ratio: actual rain / ET₀ demand.

    ET₀ is the atmospheric demand for moisture. If rain is much less
    than ET₀, the soil is being depleted faster than recharged — even
    if rainfall looks "adequate" in isolation, the deficit may be severe.
    """
    if rain_7d is None or et0_7d is None or et0_7d <= 0:
        return {"ratio": None, "level": "unknown", "label": "No ET₀ data"}

    ratio = rain_7d / et0_7d

    if ratio < DEFICIT_SEVERE:
        return {"ratio": round(ratio, 2), "level": "severe",
                "label": f"Severe moisture deficit (rain covers {ratio*100:.0f}% of ET₀ demand)"}
    elif ratio < DEFICIT_MODERATE:
        return {"ratio": round(ratio, 2), "level": "moderate",
                "label": f"Moderate moisture deficit (rain covers {ratio*100:.0f}% of ET₀ demand)"}
    elif ratio < DEFICIT_MILD:
        return {"ratio": round(ratio, 2), "level": "mild",
                "label": f"Mild moisture deficit (rain covers {ratio*100:.0f}% of ET₀ demand)"}
    else:
        return {"ratio": round(ratio, 2), "level": "surplus",
                "label": f"Moisture surplus (rain exceeds ET₀ demand by {(ratio-1)*100:.0f}%)"}


# ─────────────────────────────────────────────
#  INTERACTION SIGNAL  (unchanged logic)
# ─────────────────────────────────────────────

def interaction_signal(ndmi_level: str, rain_level: str) -> dict:
    """
    Compute the combined trading signal from NDMI × rainfall.
    Core 4×5 matrix logic — same as v1 but uses NDMI levels.
    """
    # Stressed NDMI interactions
    if ndmi_level in ("severe", "moderate"):
        if rain_level in ("very_dry", "dry"):
            return {"signal": "STRESS_DEEPENING", "bias": "STRONGLY_BULLISH",
                    "label": "Chronic drought worsening", "confidence": "HIGH",
                    "summary": "Crops are moisture-stressed and forecast is dry. Stress accumulating.",
                    "urgency": "HIGH"}
        elif rain_level == "adequate":
            return {"signal": "STRESS_PLATEAUING", "bias": "BULLISH",
                    "label": "Partial relief — stress not yet recovering", "confidence": "MEDIUM",
                    "summary": "Stressed but moderate rain forecast. Insufficient to fully recover.",
                    "urgency": "MEDIUM"}
        elif rain_level in ("good", "excellent"):
            return {"signal": "STRESS_RECOVERING", "bias": "WEAKLY_BULLISH",
                    "label": "Stress recovery underway", "confidence": "MEDIUM",
                    "summary": "Stress visible but good rain coming. NDMI will lag 2-4 weeks.",
                    "urgency": "LOW"}
        elif rain_level == "excessive":
            return {"signal": "STRESS_TO_DISEASE_RISK", "bias": "NEUTRAL",
                    "label": "Drought replaced by disease risk", "confidence": "MEDIUM",
                    "summary": "Drought easing but excessive rain raises black pod risk.",
                    "urgency": "MEDIUM"}

    # Mild stress
    elif ndmi_level == "mild":
        if rain_level in ("very_dry", "dry"):
            return {"signal": "STRESS_EMERGING", "bias": "BULLISH",
                    "label": "Early stress — conditions deteriorating", "confidence": "MEDIUM",
                    "summary": "Mild stress visible, forecast dry. Watch next 2 weeks.",
                    "urgency": "MEDIUM"}
        elif rain_level in ("adequate", "good", "excellent"):
            return {"signal": "MILD_STRESS_STABILISING", "bias": "NEUTRAL",
                    "label": "Mild stress with recovery forecast", "confidence": "MEDIUM",
                    "summary": "Mild stress but rain should prevent deterioration.",
                    "urgency": "LOW"}
        elif rain_level == "excessive":
            return {"signal": "DISEASE_RISK_LOW", "bias": "NEUTRAL",
                    "label": "Mild stress, excessive rain — mixed", "confidence": "LOW",
                    "summary": "Mild stress + excessive rain. Drought eases but disease risk.",
                    "urgency": "LOW"}

    # Healthy NDMI
    elif ndmi_level in ("adequate", "healthy"):
        if rain_level in ("very_dry", "dry"):
            return {"signal": "EARLY_WARNING", "bias": "NEUTRAL_TO_BULLISH",
                    "label": "Healthy crops facing dry spell", "confidence": "LOW",
                    "summary": "Adequate moisture but dry forecast. Not yet a concern.",
                    "urgency": "LOW"}
        elif rain_level in ("adequate", "good"):
            return {"signal": "OPTIMAL_CONDITIONS", "bias": "BEARISH",
                    "label": "Good crop conditions — supply supportive", "confidence": "HIGH",
                    "summary": "Well-hydrated crops with adequate rain. No weather premium.",
                    "urgency": "LOW"}
        elif rain_level == "excellent":
            return {"signal": "OPTIMAL_CONDITIONS", "bias": "BEARISH",
                    "label": "Excellent crop conditions", "confidence": "HIGH",
                    "summary": "Strong health confirmed by satellite + excellent rain forecast.",
                    "urgency": "LOW"}
        elif rain_level == "excessive":
            return {"signal": "DISEASE_RISK_MODERATE", "bias": "NEUTRAL_TO_BULLISH",
                    "label": "Healthy crops, excessive rain — disease risk", "confidence": "MEDIUM",
                    "summary": "Healthy but excessive rain raises black pod disease risk.",
                    "urgency": "MEDIUM"}

    return {"signal": "INSUFFICIENT_DATA", "bias": "NEUTRAL",
            "label": "Cannot determine signal", "confidence": "LOW",
            "summary": "One or both data sources unavailable.", "urgency": "LOW"}


# ─────────────────────────────────────────────
#  WEIGHTED REGIONAL AGGREGATION
# ─────────────────────────────────────────────

def aggregate_regional_weather(weather_data: dict) -> dict:
    """
    Aggregate 7-day moisture data across weather stations into
    production-weighted regional averages. Includes ET₀.
    """
    region_rain      = {r: [] for r in REGION_WEIGHTS}
    region_effective = {r: [] for r in REGION_WEIGHTS}
    region_non_rain  = {r: [] for r in REGION_WEIGHTS}
    region_et0       = {r: [] for r in REGION_WEIGHTS}

    locations = weather_data.get("locations", {})
    for station_name, station_data in locations.items():
        if "error" in station_data:
            continue

        rain      = station_data.get("rain_7d_mm")
        effective = station_data.get("effective_moisture_7d_mm")
        non_rain  = station_data.get("non_rain_precip_7d_mm", 0)
        et0       = station_data.get("et0_7d_mm")

        if effective is None:
            effective = rain

        for region in WEATHER_TO_REGION.get(station_name, []):
            if region in region_rain:
                if rain is not None:
                    region_rain[region].append(rain)
                if effective is not None:
                    region_effective[region].append(effective)
                if non_rain is not None:
                    region_non_rain[region].append(non_rain)
                if et0 is not None:
                    region_et0[region].append(et0)

    def region_avg(region_dict):
        return {r: sum(v) / len(v) for r, v in region_dict.items() if v}

    rain_avgs      = region_avg(region_rain)
    effective_avgs = region_avg(region_effective)
    non_rain_avgs  = region_avg(region_non_rain)
    et0_avgs       = region_avg(region_et0)

    if not effective_avgs:
        return {
            "weighted_avg_rain_7d":      None,
            "weighted_avg_effective_7d": None,
            "weighted_avg_non_rain_7d":  None,
            "weighted_avg_et0_7d":       None,
            "by_region":                 {},
        }

    def weighted_average(avgs_dict):
        total_w, total_v = 0.0, 0.0
        for region, val in avgs_dict.items():
            w = REGION_WEIGHTS.get(region, 0)
            total_v += val * w
            total_w += w
        return round(total_v / total_w, 1) if total_w > 0 else None

    return {
        "weighted_avg_rain_7d":      weighted_average(rain_avgs),
        "weighted_avg_effective_7d": weighted_average(effective_avgs),
        "weighted_avg_non_rain_7d":  weighted_average(non_rain_avgs),
        "weighted_avg_et0_7d":       weighted_average(et0_avgs),
        "by_region": {
            r: {
                "rain_mm":      round(rain_avgs.get(r, 0), 1),
                "non_rain_mm":  round(non_rain_avgs.get(r, 0), 1),
                "effective_mm": round(effective_avgs.get(r, 0), 1),
                "et0_mm":       round(et0_avgs.get(r, 0), 1) if r in et0_avgs else None,
            }
            for r in REGION_WEIGHTS if r in effective_avgs
        },
    }


def aggregate_regional_ndmi(crop_data: dict) -> dict:
    """Extract production-weighted NDMI from crop health data."""
    regions = crop_data.get("regions", [])
    if not regions:
        return {"weighted_avg_ndmi": crop_data.get("avg_ndmi"),
                "weighted_avg_soil_moisture": crop_data.get("avg_soil_moisture"),
                "by_region": {}}

    weighted_ndmi = 0.0
    weighted_sm   = 0.0
    total_weight  = 0.0
    sm_weight     = 0.0
    by_region     = {}

    for r in regions:
        region_name = r.get("region")
        ndmi        = r.get("ndmi")
        sm          = r.get("soil_moisture")
        weight      = REGION_WEIGHTS.get(region_name, 0)
        if ndmi is not None and weight > 0:
            weighted_ndmi += ndmi * weight
            total_weight  += weight
            by_region[region_name] = {"ndmi": ndmi, "soil_moisture": sm}
        if sm is not None and weight > 0:
            weighted_sm += sm * weight
            sm_weight   += weight

    weighted_avg_ndmi = round(weighted_ndmi / total_weight, 4) if total_weight > 0 else None
    weighted_avg_sm   = round(weighted_sm / sm_weight, 4) if sm_weight > 0 else None

    return {
        "weighted_avg_ndmi":           weighted_avg_ndmi,
        "weighted_avg_soil_moisture":  weighted_avg_sm,
        "by_region":                   by_region,
    }


# ─────────────────────────────────────────────
#  TRAJECTORY ANALYSIS
# ─────────────────────────────────────────────

def compute_ndmi_trajectory(crop_data: dict) -> dict:
    """
    Look at NDMI trend over the last 6 weeks to determine
    whether conditions are improving or deteriorating.
    v2: uses weekly data for finer granularity.
    """
    regions = crop_data.get("regions", [])
    monthly_ndmi = {}

    for r in regions:
        region_name = r.get("region", "")
        monthly = r.get("monthly", r.get("weekly", {}))
        if monthly:
            weight = REGION_WEIGHTS.get(region_name, 0)
            for key, data in monthly.items():
                ndmi = data.get("ndmi")
                if ndmi is not None:
                    if key not in monthly_ndmi:
                        monthly_ndmi[key] = []
                    monthly_ndmi[key].append((ndmi, weight))

    if len(monthly_ndmi) < 2:
        return {"trend": "unknown", "change": None, "description": "Insufficient history"}

    sorted_keys = sorted(monthly_ndmi.keys())
    recent = sorted_keys[-6:]   # Last 6 periods

    def weighted_avg(readings):
        total_w = sum(w for _, w in readings)
        return sum(v * w for v, w in readings) / total_w if total_w > 0 else None

    earliest = weighted_avg(monthly_ndmi[recent[0]])
    latest   = weighted_avg(monthly_ndmi[recent[-1]])

    if earliest is None or latest is None:
        return {"trend": "unknown", "change": None, "description": "Insufficient data"}

    change     = round(latest - earliest, 4)
    change_pct = round(change / abs(earliest) * 100, 1) if earliest != 0 else 0

    if change < -0.06:
        trend = "rapidly_deteriorating"
        desc  = f"NDMI dropped {abs(change_pct)}% — accelerating moisture stress"
    elif change < -0.02:
        trend = "deteriorating"
        desc  = f"NDMI declining gradually ({abs(change_pct)}%)"
    elif change > 0.06:
        trend = "rapidly_improving"
        desc  = f"NDMI improved {change_pct}% — meaningful recovery"
    elif change > 0.02:
        trend = "improving"
        desc  = f"NDMI improving modestly ({change_pct}%)"
    else:
        trend = "stable"
        desc  = "NDMI broadly stable"

    return {
        "trend":          trend,
        "change":         change,
        "change_pct":     change_pct,
        "from_ndmi":      round(earliest, 4),
        "to_ndmi":        round(latest, 4),
        "periods_analysed": recent,
        "description":    desc,
    }


# ─────────────────────────────────────────────
#  MAIN FUNCTION
# ─────────────────────────────────────────────

def compute_combined_stress_signal(
    weather_data: dict,
    crop_data: dict,
) -> dict:
    """
    Synthesise NDMI + SMAP + 7-day rainfall forecast + ET₀ + CHIRPS
    into a single structured supply stress signal.

    v2: Named signal LEADS (qualitative state), score CALIBRATES
    severity within that state. ET₀ moisture deficit integrated.
    """
    log.info("Computing combined crop stress signal (v2)...")

    # ── Extract weighted averages ──────────────────────────
    rain_agg = aggregate_regional_weather(weather_data)
    ndmi_agg = aggregate_regional_ndmi(crop_data)

    weighted_rain      = rain_agg.get("weighted_avg_rain_7d")
    weighted_effective = rain_agg.get("weighted_avg_effective_7d", weighted_rain)
    weighted_non_rain  = rain_agg.get("weighted_avg_non_rain_7d", 0)
    weighted_et0       = rain_agg.get("weighted_avg_et0_7d")
    weighted_ndmi      = ndmi_agg.get("weighted_avg_ndmi")
    weighted_sm        = ndmi_agg.get("weighted_avg_soil_moisture")

    moisture_for_classification = weighted_effective if weighted_effective is not None else weighted_rain

    # ── Classify each dimension ────────────────────────────
    ndmi_class    = classify_ndmi(weighted_ndmi)
    rain_class    = classify_rainfall(moisture_for_classification)
    sm_class      = classify_soil_moisture(weighted_sm)
    deficit_class = classify_moisture_deficit(weighted_effective, weighted_et0)

    # ── Compute interaction signal ─────────────────────────
    signal = interaction_signal(ndmi_class["level"], rain_class["level"])

    # ── NDMI trajectory ────────────────────────────────────
    trajectory = compute_ndmi_trajectory(crop_data)

    # ── Trajectory modifier ────────────────────────────────
    trajectory_modifier = ""
    if trajectory["trend"] in ("rapidly_deteriorating", "deteriorating"):
        if "BULLISH" in signal["bias"]:
            trajectory_modifier = (
                "⚠️ AMPLIFIED: NDMI trajectory is deteriorating — persistent trend, "
                "not a sudden event. Heightens conviction on bullish supply risk."
            )
        else:
            trajectory_modifier = (
                "Note: NDMI trend is declining despite adequate current conditions "
                "— underlying moisture trajectory bears watching."
            )
    elif trajectory["trend"] in ("rapidly_improving", "improving"):
        if "BULLISH" in signal["bias"]:
            trajectory_modifier = (
                "Note: Despite current stress, NDMI trend is improving. "
                "Bullish thesis requires re-validation at next satellite pass."
            )

    # ── Soil moisture leading indicator ─────────────────────
    soil_warning = ""
    if sm_class["level"] in ("critical", "stressed") and ndmi_class["level"] in ("adequate", "healthy"):
        soil_warning = (
            "⚠️ LEADING INDICATOR: Soil moisture (SMAP) is declining while canopy NDMI "
            "still appears healthy. Vegetation stress is likely 2-4 weeks away. "
            "Consider this a PRE-STRESS signal — do not wait for NDMI to confirm."
        )
    elif sm_class["level"] in ("critical", "stressed") and ndmi_class["level"] in ("severe", "moderate"):
        soil_warning = (
            "🔴 CONVERGENT: Both soil moisture and canopy moisture are depleted. "
            "Stress is deep and confirmed across multiple layers."
        )

    # ── ET₀ moisture deficit note ───────────────────────────
    et0_note = ""
    if deficit_class.get("ratio") is not None:
        if deficit_class["level"] == "severe":
            et0_note = (
                f"🔴 SEVERE MOISTURE DEFICIT: Rain covers only "
                f"{deficit_class['ratio']*100:.0f}% of atmospheric demand (ET₀). "
                f"Soil depletion is accelerating regardless of apparent rainfall."
            )
        elif deficit_class["level"] == "moderate":
            et0_note = (
                f"🟠 MODERATE DEFICIT: Rain covers {deficit_class['ratio']*100:.0f}% "
                f"of ET₀ demand. Soil buffer being drawn down."
            )
        elif deficit_class["level"] == "surplus":
            et0_note = (
                f"🟢 Moisture surplus: Rain exceeds ET₀ demand — "
                f"soil recharge active."
            )

    # ── CHIRPS backward-looking context ────────────────────
    chirps_note = ""
    # Extract CHIRPS from the first region that has it
    for r in crop_data.get("regions", []):
        c30 = r.get("chirps_30d_mm")
        c60 = r.get("chirps_60d_mm")
        c90 = r.get("chirps_90d_mm")
        if c30 is not None:
            chirps_note = (
                f"CHIRPS historical rainfall (region-avg): "
                f"30d={c30}mm  60d={c60}mm  90d={c90}mm"
            )
            break

    # ── Crop calendar context ──────────────────────────────
    calendar        = get_crop_calendar_context()
    cal_multiplier  = calendar["multiplier"]
    cal_stage       = calendar["stage"]
    cal_sensitivity = calendar["sensitivity"]
    cal_notes       = calendar["notes"]
    cal_crop        = calendar["crop"]

    # ── Stress score (0–100, higher = more stressed) ───────
    moisture_stress = 100 - ndmi_class["score"]
    rain_deficit    = 100 - rain_class["score"]

    # Base: NDMI 45%, rain forecast 30%, soil moisture 25%
    sm_stress = 100 - sm_class["score"] if sm_class["level"] != "unknown" else 50
    base_stress_score = (moisture_stress * 0.45 +
                         rain_deficit * 0.30 +
                         sm_stress * 0.25)

    # ET₀ deficit amplifier: severe deficit adds up to 10 points
    if deficit_class.get("level") == "severe":
        base_stress_score = min(100, base_stress_score + 10)
    elif deficit_class.get("level") == "moderate":
        base_stress_score = min(100, base_stress_score + 5)

    # Crop calendar multiplier
    calendar_adjusted = base_stress_score * cal_multiplier
    combined_stress_score = min(100, round(calendar_adjusted))

    # Trajectory amplifier
    if trajectory["trend"] == "rapidly_deteriorating":
        combined_stress_score = min(100, round(combined_stress_score * 1.15))
    elif trajectory["trend"] == "deteriorating":
        combined_stress_score = min(100, round(combined_stress_score * 1.08))
    elif trajectory["trend"] in ("improving", "rapidly_improving"):
        combined_stress_score = max(0, round(combined_stress_score * 0.90))

    # Calendar impact note
    if cal_multiplier >= 1.25:
        calendar_impact_note = (
            f"🔴 CRITICAL WINDOW ({cal_stage}): Stress multiplier {cal_multiplier}x — "
            f"moisture stress is maximally damaging right now. {cal_notes}"
        )
    elif cal_multiplier >= 1.10:
        calendar_impact_note = (
            f"🟠 HIGH SENSITIVITY ({cal_stage}): Stress multiplier {cal_multiplier}x. {cal_notes}"
        )
    elif cal_multiplier >= 0.95:
        calendar_impact_note = (
            f"🟡 NORMAL SENSITIVITY ({cal_stage}): Stress multiplier {cal_multiplier}x. {cal_notes}"
        )
    else:
        calendar_impact_note = (
            f"🟢 LOW SENSITIVITY ({cal_stage}): Stress multiplier {cal_multiplier}x. {cal_notes}"
        )

    # ── Crop health cross-check ────────────────────────────
    crop_score = crop_data.get("overall_score")
    crop_bias  = crop_data.get("overall_bias", "NEUTRAL")
    crop_named = crop_data.get("named_signal", "UNKNOWN")
    data_age   = crop_data.get("data_age_days", 0)
    stale      = data_age > 14 if data_age else False

    # ── Per-region breakdown ───────────────────────────────
    region_details = []
    for region_name, weight in REGION_WEIGHTS.items():
        r_data    = ndmi_agg["by_region"].get(region_name, {})
        r_ndmi    = r_data.get("ndmi") if isinstance(r_data, dict) else r_data
        r_sm      = r_data.get("soil_moisture") if isinstance(r_data, dict) else None
        region_wx = rain_agg["by_region"].get(region_name, {})
        rain_val  = region_wx.get("rain_mm") if isinstance(region_wx, dict) else region_wx
        eff_val   = region_wx.get("effective_mm", rain_val) if isinstance(region_wx, dict) else rain_val
        et0_val   = region_wx.get("et0_mm") if isinstance(region_wx, dict) else None
        non_rain  = region_wx.get("non_rain_mm", 0) if isinstance(region_wx, dict) else 0

        if r_ndmi is None and eff_val is None:
            continue

        rc_ndmi    = classify_ndmi(r_ndmi)
        rc_rain    = classify_rainfall(eff_val)
        rc_sm      = classify_soil_moisture(r_sm)
        rc_deficit = classify_moisture_deficit(eff_val, et0_val)
        rc_sig     = interaction_signal(rc_ndmi["level"], rc_rain["level"])

        region_details.append({
            "region":          region_name,
            "weight_pct":      int(weight * 100),
            "ndmi":            r_ndmi,
            "ndmi_label":      rc_ndmi["label"],
            "soil_moisture":   r_sm,
            "soil_label":      rc_sm["label"],
            "rain_7d_mm":      rain_val,
            "non_rain_mm":     round(non_rain, 1) if non_rain else 0,
            "effective_mm":    round(eff_val, 1) if eff_val else None,
            "et0_mm":          round(et0_val, 1) if et0_val else None,
            "deficit_ratio":   rc_deficit.get("ratio"),
            "rain_label":      rc_rain["label"],
            "signal":          rc_sig["signal"],
            "bias":            rc_sig["bias"],
        })

    # ── Assemble output ────────────────────────────────────
    result = {
        "generated_at":          datetime.now(timezone.utc).isoformat(),
        "data_age_days":         data_age,
        "stale_warning":         stale,

        # Inputs
        "weighted_ndmi":              weighted_ndmi,
        "ndmi_classification":        ndmi_class,
        "weighted_soil_moisture":     weighted_sm,
        "soil_classification":        sm_class,
        "weighted_rain_7d_mm":        weighted_rain,
        "weighted_non_rain_7d_mm":    weighted_non_rain,
        "weighted_effective_7d_mm":   weighted_effective,
        "weighted_et0_7d_mm":         weighted_et0,
        "rain_classification":        rain_class,
        "moisture_deficit":           deficit_class,
        "moisture_classified_on":     "effective_moisture",

        # Trajectory
        "ndmi_trajectory":       trajectory,

        # Combined signal (v2: SIGNAL LEADS, score calibrates)
        "signal":                signal["signal"],
        "bias":                  signal["bias"],
        "label":                 signal["label"],
        "confidence":            signal["confidence"],
        "summary":               signal["summary"],
        "urgency":               signal["urgency"],

        # Score (severity calibration within the signal)
        "combined_stress_score": combined_stress_score,
        "base_stress_score":     round(base_stress_score),

        # Contextual notes
        "trajectory_modifier":   trajectory_modifier,
        "soil_warning":          soil_warning,
        "et0_note":              et0_note,
        "chirps_note":           chirps_note,

        # Regional breakdown
        "region_details":        region_details,

        # Cross-check
        "satellite_crop_bias":   crop_bias,
        "satellite_crop_named":  crop_named,
        "satellite_score":       crop_score,
        "signals_agree":         (
            ("BULLISH" in signal["bias"] and "BULLISH" in crop_bias) or
            ("BEARISH" in signal["bias"] and "BEARISH" in crop_bias) or
            (signal["bias"] == "NEUTRAL" and crop_bias == "NEUTRAL")
        ),

        # Crop calendar
        "crop_calendar": {
            "month_name":       calendar["month_name"],
            "stage":            cal_stage,
            "crop":             cal_crop,
            "sensitivity":      cal_sensitivity,
            "multiplier":       cal_multiplier,
            "base_score":       round(base_stress_score),
            "calendar_adjusted_score": min(100, round(calendar_adjusted)),
            "impact_note":      calendar_impact_note,
        },
    }

    log.info(f"  Signal              : {signal['signal']}")
    log.info(f"  Bias                : {signal['bias']}")
    log.info(f"  Stress score        : {combined_stress_score}/100")
    log.info(f"  Soil moisture       : {weighted_sm} → {sm_class['label']}")
    log.info(f"  ET₀ deficit         : {deficit_class.get('ratio', 'N/A')}")
    log.info(f"  Crop stage          : {cal_stage} ({cal_sensitivity}, {cal_multiplier}x)")

    return result


# ─────────────────────────────────────────────
#  PROMPT FORMATTER  (v2 — restructured)
# ─────────────────────────────────────────────

def format_for_prompt(signal: dict) -> str:
    """
    Format the combined stress signal for the Claude agent prompt.

    v2 RESTRUCTURE: Signal LEADS, then context, then score.
    Old v1 format led with scores — the agent would anchor on
    the number rather than understanding the qualitative state.

    New hierarchy:
      1. Named signal + bias (what's happening)
      2. Leading indicators (what's coming)
      3. Context (why it's happening)
      4. Score (how severe)
    """
    if not signal:
        return "\n## COMBINED CROP STRESS SIGNAL\n  Data unavailable\n"

    stale_note = (
        f"\n  ⚠️  SATELLITE DATA IS {signal.get('data_age_days')} DAYS OLD — "
        "treat with reduced confidence\n"
        if signal.get("stale_warning") else ""
    )

    agree_str = (
        "✅ Satellite crop score and combined signal AGREE"
        if signal.get("signals_agree")
        else "⚠️  Satellite crop score and combined signal DIVERGE — examine carefully"
    )

    region_lines = "\n".join([
        f"  {r['region']} ({r['weight_pct']}%): "
        f"NDMI {r['ndmi']} ({r['ndmi_label']}) | "
        f"Soil {r.get('soil_moisture', 'N/A')} ({r.get('soil_label', 'N/A')}) | "
        f"Rain {r.get('rain_7d_mm', 0)}mm eff.{r.get('effective_mm', 0)}mm "
        f"(deficit ratio: {r.get('deficit_ratio', 'N/A')}) "
        f"→ {r['bias']}"
        for r in signal.get("region_details", [])
    ])

    traj = signal.get("ndmi_trajectory", {})
    traj_line = (
        f"{traj.get('trend', 'unknown').replace('_', ' ').title()}: "
        f"{traj.get('description', 'N/A')}"
    )

    modifier    = signal.get("trajectory_modifier", "")
    soil_warn   = signal.get("soil_warning", "")
    et0_note    = signal.get("et0_note", "")
    chirps_note = signal.get("chirps_note", "")
    cal         = signal.get("crop_calendar", {})
    deficit     = signal.get("moisture_deficit", {})

    cal_str = (
        f"{cal.get('month_name')} — {cal.get('stage')} "
        f"({cal.get('crop')}, sensitivity: {cal.get('sensitivity')}, "
        f"multiplier: {cal.get('multiplier')}x)"
    ) if cal else "No calendar data"

    return f"""
## COMBINED CROP STRESS SIGNAL v2 (NDMI × Rain × Soil × ET₀ × Calendar)
{stale_note}
┌─ SIGNAL (what's happening) ──────────────────────────────
│ Signal:     {signal.get('signal')}
│ Bias:       {signal.get('bias')}
│ Confidence: {signal.get('confidence')}
│ Urgency:    {signal.get('urgency')}
│ Label:      {signal.get('label')}
└──────────────────────────────────────────────────────────

Summary: {signal.get('summary')}

┌─ LEADING INDICATORS (what's coming) ────────────────────
│ Soil Moisture (SMAP): {signal.get('weighted_soil_moisture')} → {signal.get('soil_classification', {}).get('label', 'N/A')}
│ ET₀ Deficit Ratio:    {deficit.get('ratio', 'N/A')} → {deficit.get('label', 'N/A')}
│ NDMI Trajectory:      {traj_line}
└──────────────────────────────────────────────────────────
{('  ' + soil_warn) if soil_warn else ''}
{('  ' + et0_note) if et0_note else ''}
{('  ' + modifier) if modifier else ''}

┌─ CURRENT STATE ──────────────────────────────────────────
│ Weighted NDMI:    {signal.get('weighted_ndmi')} → {signal.get('ndmi_classification', {}).get('label')}
│ 7d Moisture:      Rain: {signal.get('weighted_rain_7d_mm')}mm  Dew: {signal.get('weighted_non_rain_7d_mm')}mm
│                   → Effective: {signal.get('weighted_effective_7d_mm')}mm → {signal.get('rain_classification', {}).get('label')}
│ ET₀ (7d):         {signal.get('weighted_et0_7d_mm')}mm
│ {chirps_note}
└──────────────────────────────────────────────────────────

┌─ SEVERITY SCORE ─────────────────────────────────────────
│ Base stress score:        {signal.get('base_stress_score')} / 100
│ Calendar-adjusted score:  {cal.get('calendar_adjusted_score', 'N/A')} / 100
│ Final score (w/ trend):   {signal.get('combined_stress_score')} / 100  (higher = more supply risk)
└──────────────────────────────────────────────────────────

Crop Calendar: {cal_str}
  {cal.get('impact_note', '')}

By Region:
{region_lines}

Cross-check: {agree_str}
  Satellite: score={signal.get('satellite_score')}, signal={signal.get('satellite_crop_named')}, bias={signal.get('satellite_crop_bias')}
"""


# ─────────────────────────────────────────────
#  STANDALONE RUNNER
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    SNAPSHOT_FILE    = "cocoa_daily_snapshot.json"
    CROP_HEALTH_FILE = "cocoa_crop_health.json"

    print("=" * 62)
    print("  COMBINED CROP STRESS SIGNAL v2 — Diagnostic")
    print("=" * 62)

    try:
        with open(SNAPSHOT_FILE, "r", encoding="utf-8") as f:
            snapshot = json.load(f)
        weather_data = snapshot.get("weather", {})
        print(f"Loaded snapshot: {SNAPSHOT_FILE}")
    except FileNotFoundError:
        print(f"ERROR: {SNAPSHOT_FILE} not found.")
        sys.exit(1)

    try:
        from cocoa_crop_monitor import load_crop_health_for_agent
        crop_data = load_crop_health_for_agent(CROP_HEALTH_FILE)
        print(f"Loaded crop health: {CROP_HEALTH_FILE}")
    except Exception as e:
        print(f"WARNING: Could not load crop health data: {e}")
        crop_data = {}

    result = compute_combined_stress_signal(weather_data, crop_data)
    print(format_for_prompt(result))
    print("\n--- Raw signal JSON ---")
    print(json.dumps(result, indent=2, default=str))
