"""
=============================================================
  COCOA TRADING ASSISTANT — AI Agent Module
=============================================================
  Loads the daily snapshot from cocoa_data_gatherer.py,
  builds a structured prompt, calls the Claude API, and
  delivers a formatted trading recommendation report.

  Delivery options (configure in .env):
    - Telegram Bot (recommended — instant, free)
    - Email via Gmail SMTP

  SETUP:
    pip install anthropic python-dotenv

  .env file:
    ANTHROPIC_API_KEY=your_key_here

    # Telegram (recommended):
    TELEGRAM_BOT_TOKEN=your_bot_token
    TELEGRAM_CHAT_ID=your_chat_id

    # Email (alternative):
    EMAIL_FROM=you@gmail.com
    EMAIL_TO=you@gmail.com
    EMAIL_APP_PASSWORD=your_gmail_app_password

  HOW TO GET TELEGRAM CREDENTIALS:
    1. Message @BotFather on Telegram → /newbot → copy the token
    2. Message @userinfobot on Telegram → copy your Chat ID
=============================================================
"""

import os
import json
import logging
import smtplib
import argparse
from datetime import datetime, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

import anthropic

# Optional — crop health monitor integration
try:
    from cocoa_crop_monitor import load_crop_health_for_agent
    CROP_MONITOR_AVAILABLE = True
except ImportError:
    CROP_MONITOR_AVAILABLE = False

# Optional — combined stress signal
try:
    from cocoa_stress_signal import format_for_prompt as format_stress_signal
    STRESS_SIGNAL_AVAILABLE = True
except ImportError:
    STRESS_SIGNAL_AVAILABLE = False

# Optional — feedback & learning loop
try:
    from cocoa_feedback import (
        record_prediction,
        evaluate_pending,
        build_feedback_prompt,
        extract_recommendation,
        get_ledger_stats,
    )
    FEEDBACK_AVAILABLE = True
except ImportError:
    FEEDBACK_AVAILABLE = False

# Optional — opportunity scoring
try:
    from cocoa_opportunity_scorer import (
        score_opportunity,
        log_opportunity,
        format_watchlist_alert,
        format_opportunity_alert,
    )
    SCORER_AVAILABLE = True
except ImportError:
    SCORER_AVAILABLE = False

# Optional — weekly review & continuous learning
try:
    from cocoa_weekly_review import (
        record_shadow_prediction, score_shadow_predictions,
        check_big_misses, generate_weekly_report,
        should_generate_weekly_report, build_learning_prompt,
    )
    WEEKLY_REVIEW_AVAILABLE = True
except ImportError:
    WEEKLY_REVIEW_AVAILABLE = False

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

ANTHROPIC_API_KEY    = os.getenv("ANTHROPIC_API_KEY", "")
SNAPSHOT_FILE        = os.getenv("SNAPSHOT_FILE", "cocoa_daily_snapshot.json")
REPORT_OUTPUT_FILE   = "cocoa_daily_report.md"

# Delivery config
TELEGRAM_BOT_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID     = os.getenv("TELEGRAM_CHAT_ID", "")

# Fallback: read Telegram credentials from memory store if not in env
if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    for creds_path in [
        "/mnt/memory/cocoa-surveillance-memory/state/telegram_creds.json",
        "telegram_creds.json",
    ]:
        try:
            import json as _json
            with open(creds_path, "r") as f:
                _creds = _json.load(f)
            TELEGRAM_BOT_TOKEN = TELEGRAM_BOT_TOKEN or _creds.get("bot_token", "")
            TELEGRAM_CHAT_ID = TELEGRAM_CHAT_ID or _creds.get("chat_id", "")
            break
        except (FileNotFoundError, ValueError):
            continue
EMAIL_FROM           = os.getenv("EMAIL_FROM", "")
EMAIL_TO             = os.getenv("EMAIL_TO", "")
EMAIL_APP_PASSWORD   = os.getenv("EMAIL_APP_PASSWORD", "")

# Claude model to use
CLAUDE_MODEL         = "claude-opus-4-6"   # Best for nuanced financial reasoning

# Instrument details (for report header and prompt context)
INSTRUMENT_NAME      = "ICE New York Cocoa (CC=F)"
INSTRUMENT_EXCHANGE  = "ICE Futures US"
INSTRUMENT_CURRENCY  = "USD"
INSTRUMENT_UNIT      = "USD/tonne"


# ─────────────────────────────────────────────
#  SYSTEM PROMPT  (the agent's "personality")
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a specialist cocoa commodity trading analyst with deep expertise in:
- ICE cocoa futures market (CC=F, USD/tonne)
- West African agricultural cycles and crop dynamics
- Technical analysis for commodity futures
- Macro factors affecting soft commodities

Your job is to assess the cocoa market and identify whether current conditions
represent an actionable trading opportunity. You are a SURVEILLANCE agent —
most days you will find nothing exceptional, and that is correct behaviour.
Your output will be scored across five dimensions (valuation gap, positioning,
catalyst proximity, satellite signals, track record) to determine whether
to alert the trader or remain silent.

The trader does NOT need daily trading advice. They need to be alerted ONLY
when a high-conviction setup emerges — defined as a meaningful valuation gap
amplified by extreme positioning, with an identifiable catalyst and
confirming satellite data. This may happen only a few times per quarter.

Produce your analysis honestly: if the market is fairly valued with no edge,
say so clearly. A "Fairly Valued" assessment with no actionable setup is
the CORRECT output most days. Do not manufacture opportunities.

## Your analytical framework (apply in this order of priority):

1. SUPPLY FUNDAMENTALS (highest weight)
   - The COMBINED CROP STRESS SIGNAL section is your single most important input —
     it synthesises satellite NDWI moisture data with the 7-day rainfall forecast
     into one scored, classified signal. Use it as your primary supply view.
   - Stress score >70 = significant supply risk, weight bullish heavily
   - Stress score 40–70 = mixed signal, look for corroboration from other factors
   - Stress score <40 = benign supply conditions, bearish weight
   - Where the satellite bias and combined signal AGREE, confidence is higher
   - Where they DIVERGE, flag the conflict explicitly and explain which you weight more
   - Côte d'Ivoire + Ghana = ~65% of global supply — regional breakdowns matter
   - Current harvest season (Main Crop Oct–Mar vs Mid Crop Apr–Sep) affects sensitivity
   - ICCO supply/demand balance signals

   NEW SATELLITE SIGNALS — use these for early warning:
   - LAI (Leaf Area Index): measures canopy structure, not just colour. Healthy
     cocoa LAI is 3.0–5.0. A drop below 2.5, or a declining trend, signals
     structural canopy problems (poor pod set, defoliation, disease) that standard
     NDVI/EVI will only show weeks later. LAI is the BEST satellite proxy for
     detecting cherelle formation problems early.
   - RAINFALL ANOMALY (%): compares this week's rainfall to the 10-year average
     for the same calendar week. -30% or worse during flowering/pod development
     is a meaningful supply stress signal. Don't just report "58mm" — report
     "58mm, which is 25% below normal for this week."
   - LST ANOMALY (°C): compares current Land Surface Temperature to the 10-year
     average. +2°C or more during mid-crop flowering is a significant heat stress
     signal that directly reduces cherelle survival. This is an EARLY WARNING
     indicator — heat stress during flowering reduces pod set before any visible
     canopy impact.

2. DEMAND FUNDAMENTALS (high weight)
   - GRINDING DATA is the direct demand proxy — ECA/NCA/CAA quarterly figures
   - Trend and YoY direction matter more than absolute levels
   - ECA is most market-moving — largest volume, released first
   - Rising grindings YoY = demand growth = bullish; falling = demand destruction = bearish
   - If a release is imminent (grinding_release_alert), treat as a scheduled catalyst
   - GRINDING RELEASE IMPACT MODEL: if available, use the scenario analysis
     (bear/base/bull) to quantify the expected price reaction

3. WAREHOUSE STOCKS (high weight — daily physical signal)
   - ICE-certified European warehouses, in 60kg bags (~16,000 bags = 1,000t)
   - RISING stocks = bearish (supply building); FALLING = bullish (demand absorbing)
   - Compare stock trend vs price trend for confirmation or divergence signals
   - Context: ~500k bags normal; >800k historically high (bearish); <200k very tight (bullish)

4. TECHNICAL PICTURE (medium-high weight — for timing, not valuation)
   - EMA stack alignment (20/50/200) defines the primary trend
   - MACD, RSI, Bollinger Bands for momentum and volatility context
   - Support/resistance levels inform ENTRY TIMING, not fair value
   - OBV trend for volume confirmation
   - NOTE: technicals tell you WHEN to act, not WHERE value is

5. SPECULATIVE POSITIONING — CFTC COT (medium-high weight)
   - Managed Money net position and its percentile rank vs 3yr history
   - EXTREMELY SHORT (<15th percentile) = HIGH short squeeze risk on any
     bullish catalyst — price can rally violently as shorts cover
   - EXTREMELY LONG (>85th percentile) = HIGH liquidation risk on any
     bearish catalyst — crowded longs unwind fast
   - Week-on-week change in net position shows the direction of flow
   - Commercial (Producer/Merchant) net position diverging from price
     is a contrarian signal — commercials are usually right at extremes
   - This data lags by ~3 days (released Friday, data as of Tuesday)
   - Use this to TEMPER or AMPLIFY your valuation bias:
     * If you believe overvalued BUT specs are extremely short → squeeze
       risk means your short entry timing must be precise
     * If you believe undervalued AND specs are extremely short → strong
       confirmation, higher conviction

5. SENTIMENT & NEWS (medium weight)
   - Aggregate news sentiment direction
   - ICCO report releases, government policy changes
   - Confectionery demand signals

6. MACRO CONTEXT (medium weight)
   - USD strength (inverse relationship — stronger USD = lower cocoa price)
   - Risk-on/risk-off signals from equities

7. SEASONAL CONTEXT (background weight)
   - Crop cycle position and typical seasonal patterns

## PREDICTION ACCURACY FEEDBACK

If a PREDICTION ACCURACY FEEDBACK section is provided in the data, you MUST use it
to calibrate your analysis. This section contains your track record — how often your
past predictions were correct, which factors led to accurate calls vs misses, and
whether your confidence levels have been well-calibrated.

Rules for using feedback:
- If your direction accuracy is below 50%, you are doing WORSE than a coin flip.
  Actively reconsider your analytical framework. Ask yourself what you keep getting wrong.
- If a factor shows <40% accuracy when cited, DOWNWEIGHT it heavily today.
  Do not rely on a signal that has repeatedly led you astray.
- If a factor shows >65% accuracy, it is a reliable signal — lean into it.
- If HIGH confidence calls are less accurate than LOW confidence calls, you are
  systematically overconfident. Cap your stated confidence at MEDIUM until calibration improves.
- If there is a systematic bullish or bearish bias, actively correct in the opposite direction.
- Reference specific recent misses when they are relevant to today's setup.
  E.g., "Last week's MACD-based call was wrong because [X] — this time the setup differs because [Y]"
  or "Similar conditions led to a miss on [date] — I am less confident this time."
- Do NOT ignore this section. It is the most important self-correction mechanism you have.

## Output format:

Always structure your response EXACTLY as follows — use these exact section headers:

### 🌍 MARKET OVERVIEW
[2-3 sentences on where cocoa stands today — price, trend, and the single most important factor]

### 📊 TECHNICAL ANALYSIS
[Concise TA picture — trend, momentum, volatility, key levels. Flag significant signals/divergences]

### 🌦️ SUPPLY & WEATHER
[West Africa weather assessment and near-term supply implications. Note crop season.]

### 📰 NEWS & SENTIMENT
[News flow and market sentiment summary. Call out market-moving headlines.]

### 🌐 MACRO FACTORS
[USD, equities — how they affect cocoa right now]

### ✅ VALUATION BIAS

Your core output. Where SHOULD this instrument be trading vs where IS it trading?

**Assessment:** [Overvalued / Undervalued / Fairly Valued — pick one]
**Magnitude:** [Slightly (1–5%) / Moderately (5–15%) / Significantly (>15%)]
**Estimated Fair Range:** [USD/t range, e.g. "7,500–8,200 USD/t"]
**Primary Driver of Gap:** [One sentence explaining WHY the gap exists — e.g. "Market is pricing
  in demand recovery that grinding data does not support" or "Supply stress from mid-crop
  weather risk is not yet reflected in price"]
**Confidence:** [HIGH / MEDIUM / LOW]
**Rationale:** [3-4 sentences synthesising the fundamental case for your fair value estimate.
  Weigh supply, demand, stocks, and macro. Cite specific data points.]

### ⏱️ TIMING SIGNAL

Should the trader act on the valuation gap NOW, or wait?

**Timing:** [Act Now / Wait for Catalyst / Wait for Level]
**Catalyst:** [If "Wait for Catalyst" — name the specific event and date, e.g.
  "ECA Q1 grinding data April 16 — if prints -7% YoY or worse, confirms demand destruction
  thesis and opens the gap further"]
**Entry:** [If "Wait for Level" — specify the price level and condition, e.g.
  "Enter short on a rally to 2,450+ / EMA-20 rejection" or "Enter long on a dip to
  2,320–2,350 near Bollinger lower band"]
**Key Invalidation:** [The single thing that would FLIP this view entirely, e.g.
  "A close above 2,550 (EMA-50) would signal the downtrend is breaking and the
  bearish valuation thesis is wrong"]

### ⚠️ KEY RISKS TO THIS VIEW
[2-3 things that would invalidate or flip this recommendation]

### 👀 WATCH TODAY
[1-3 specific things to monitor — data releases, price levels, news to track]

---
*Disclaimer: This is algorithmically generated analysis for informational purposes only. Not financial advice. Always apply your own judgement and risk management.*

## Style guidelines:

**Tone — calibrated, not dramatic**
- Write like a senior analyst briefing a professional trader, not a journalist
- Avoid hyperbole — let the data carry the weight
- Quantify where possible: price levels, percentages, dates

**Valuation honesty — do not fake precision**
- Your fair range should be wide enough to be honest about uncertainty
  (typically 5-10% band) but narrow enough to be useful
- If you genuinely cannot estimate fair value (conflicting signals, missing data),
  say "Fairly Valued — no clear mispricing signal" with LOW confidence
- Do not anchor to round numbers unless there is a structural reason

**Model outputs — caveat limitations**
- Crop stress signal caps at 100 — it means "at or beyond upper range", not a precise severity
- TA indicators are computed directly on CC=F USD prices — levels are accurate
- CC=F is a continuous futures contract. Rollover gaps are ratio-adjusted in the
  history so indicators are smooth, but if a rollover_note is present in the price
  data, mention it as a caveat. The daily change uses open-to-close (not close-to-close)
  specifically to avoid rollover distortion.
- EMA-200 may be absent until 200 days of data accumulate

**Alternative interpretations — always acknowledge the other side**
- For every directional thesis, spend 1-2 sentences on the credible counter-argument
- The market is not wrong: if price diverges from your fair value, explain why
  the market might know something you don't

**Confidence calibration — be internally consistent**
- HIGH: signals clearly aligned, you would size a full position
- MEDIUM: meaningful signals but genuine uncertainty on timing or magnitude
- LOW: directional lean but material risk of being wrong — trade small or wait
- If signals conflict, confidence is at most MEDIUM
- If the PREDICTION ACCURACY FEEDBACK shows systematic overconfidence, cap at MEDIUM

**GBP equivalents — include for key figures**
- The data includes a GBP/USD rate. For the main price levels in your output
  (current price, fair range, entry levels, invalidation level), include the
  approximate GBP equivalent in brackets after the USD figure.
  E.g., "Fair Range: 7,500–8,200 USD/t (≈ 5,650–6,180 GBP/t)"
- All analysis and calculations remain in USD — GBP is display-only.
- If the GBP/USD rate is unavailable, omit the GBP figures silently.

**Keep the total report under 600 words — concise and actionable.**
"""


# ─────────────────────────────────────────────
#  PROMPT BUILDER
# ─────────────────────────────────────────────

def build_user_prompt(snapshot: dict) -> str:
    """
    Converts the raw JSON snapshot into a well-structured,
    token-efficient prompt for Claude.
    """

    ta      = snapshot.get("technicals", {})
    px      = ta.get("price", {})
    trend   = ta.get("trend", {})
    mom     = ta.get("momentum", {})
    vol     = ta.get("volatility", {})
    levels  = ta.get("levels", {})
    volume  = ta.get("volume", {})
    related = snapshot.get("related_markets", {})
    news    = snapshot.get("news", {})
    weather = snapshot.get("weather", {})
    wx_sum  = weather.get("summary", {})
    seas    = snapshot.get("seasonal", {})

    # ── Price section ─────────────────────────────────────────
    hist_days = snapshot.get("history_days", 0)
    gbpusd    = snapshot.get("gbpusd_rate")
    price_gbp = snapshot.get("price_gbp")
    gbp_note  = f" (≈ {price_gbp} GBP/t at GBPUSD {gbpusd})" if price_gbp else ""

    price_block = f"""
## PRICE DATA (ICE New York Cocoa CC=F — Yahoo Finance, USD/tonne)
- Last Close:     {px.get('current')} USD/tonne{gbp_note}
- Today's Open:   {px.get('open')} USD/tonne
- Daily Change:   {px.get('change_1d_pct')}% (open to close)
- Prev Close:     {px.get('prev_close')} USD/tonne
- Source:         {snapshot.get("ta_source", "CC=F (Yahoo Finance)")}
- History:        {hist_days} trading days of data
- 52w High/Low:   {px.get('week52_high')} / {px.get('week52_low')}
- 1-Week Change:  {px.get('change_1w_pct')}%
- 1-Month Change: {px.get('change_1m_pct')}%
{"- ⚠️ " + px.get('rollover_note') if px.get('rollover_note') else ""}
- GBP/USD Rate:   {gbpusd or 'unavailable'}
"""

    # ── Technicals section ────────────────────────────────────
    ta_block = f"""
## TECHNICAL INDICATORS
Trend:
- Overall Trend:  {trend.get('label')}
- EMA 20:         {trend.get('ema_20')}  (price vs EMA20: {trend.get('price_vs_ema20')})
- EMA 50:         {trend.get('ema_50')}
- EMA 200:        {trend.get('ema_200')}  (price vs EMA200: {trend.get('price_vs_ema200')})

Momentum:
- RSI (14):       {mom.get('rsi_14')} → {mom.get('rsi_signal')}
- MACD Line:      {mom.get('macd')}
- MACD Signal:    {mom.get('macd_signal')}
- MACD Histogram: {mom.get('macd_hist')}
- MACD Crossover: {mom.get('macd_cross')}
- Stochastic K/D: {mom.get('stoch_k')} / {mom.get('stoch_d')}

Volatility:
- ATR (14):       {vol.get('atr_14')}
- BB Upper:       {vol.get('bb_upper')}
- BB Mid:         {vol.get('bb_mid')}
- BB Lower:       {vol.get('bb_lower')}
- BB Width:       {vol.get('bb_width_pct')}%
- BB Position:    {vol.get('bb_position')}

Key Levels:
- 20-Day Resistance: {levels.get('resistance_20d')}
- 20-Day Support:    {levels.get('support_20d')}
- 50-Day Resistance: {levels.get('resistance_50d')}
- 50-Day Support:    {levels.get('support_50d')}

Volume:
- OBV Trend:      {volume.get('obv_trend')}
"""

    # ── Related markets section ───────────────────────────────
    def fmt_mkt(name):
        m = related.get(name, {})
        price = m.get("price")
        if price is None:
            return "unavailable"
        chg = m.get("change_pct")
        chg_str = f"  {'+' if chg >= 0 else ''}{chg}%" if chg is not None else ""
        return f"{price}{chg_str}"

    related_block = f"""
## RELATED MARKETS (stooq.com)
  [MACRO]
- USD Index:    {fmt_mkt('USD_Index')}
- GBP/USD:      {fmt_mkt('GBPUSD')}
- S&P 500:      {fmt_mkt('SP500')}

  [SOFT COMMODITIES]
- Sugar #11:    {fmt_mkt('Sugar')}
- Coffee C:     {fmt_mkt('Coffee')}
"""

    # ── Weather section ───────────────────────────────────────
    wx_locations = weather.get("locations", {})
    wx_lines = []
    for loc, data in wx_locations.items():
        if "error" not in data:
            flags = "; ".join(data.get("crop_stress_flags", []))
            wx_lines.append(
                f"- {loc}: {data.get('rain_7d_mm')}mm rain (7d), "
                f"avg max {data.get('avg_max_temp_c')}°C | {flags}"
            )

    weather_block = f"""
## WEST AFRICA WEATHER (7-day forecast)
Overall Condition: {wx_sum.get('overall_condition')}
Average 7-day Rainfall: {wx_sum.get('avg_7d_rainfall_mm')} mm

By Location:
{chr(10).join(wx_lines) if wx_lines else '  No data available'}

Stress Alerts:
{chr(10).join(['- ' + a for a in wx_sum.get('significant_alerts', [])]) or '  None'}
"""

    # ── News section (uses news intelligence when available) ──
    news_intel = snapshot.get("news_intelligence", {})

    if news_intel and "error" not in news_intel:
        # Rich news intelligence available — use the Claude-summarised brief
        key_articles = news_intel.get("key_articles", [])
        article_lines = []
        for a in key_articles[:8]:
            article_lines.append(
                f"  - [{a.get('source', '?')}] {a.get('title', '')} "
                f"({a.get('age', '')})\n"
                f"    Relevance: {a.get('relevance', '')}"
            )

        demand_fc = news_intel.get("demand_forecast", {})
        demand_section = ""
        if demand_fc and "error" not in demand_fc and demand_fc.get("full_year_outlook"):
            demand_section = f"""
Demand Forecast (from analyst sources):
  - Outlook:       {demand_fc.get('full_year_outlook')}
  - Price Targets: {demand_fc.get('price_targets', 'None cited')}
  - Upside Risk:   {demand_fc.get('key_risks_upside', 'N/A')}
  - Downside Risk: {demand_fc.get('key_risks_downside', 'N/A')}
"""

        news_block = f"""
## NEWS & SENTIMENT (AI-summarised intelligence brief)
Directional Signal: {news_intel.get('directional_signal')} ({news_intel.get('signal_confidence')} confidence)
Rationale: {news_intel.get('signal_rationale', '')}

Recent Summary (last 7 days):
  {news_intel.get('recent_summary', 'No summary available')}

Background Context:
  {news_intel.get('background_context', 'No background available')}

Key Themes: {', '.join(news_intel.get('key_themes', []))}

Key Articles:
{chr(10).join(article_lines) if article_lines else '  No key articles identified'}
{demand_section}
Watch Items:
{chr(10).join(['  - ' + w for w in news_intel.get('watch_items', [])]) or '  None'}

Data Quality: {news_intel.get('data_quality', 'N/A')}
Article Counts: {news_intel.get('article_counts', {})}
"""
    else:
        # Fallback: use basic sentiment + article list (old format)
        sentiment = news.get("sentiment", {})
        articles  = news.get("articles", [])[:10]

        news_lines = []
        for a in articles:
            sentiment_tag = {"Bullish": "🟢", "Bearish": "🔴", "Neutral": "⚪"}.get(
                a.get("sentiment", "Neutral"), "⚪"
            )
            news_lines.append(
                f"  {sentiment_tag} [{a.get('source', '')}] {a.get('title', '')} "
                f"({a.get('published', '')[:10]})"
            )

        news_block = f"""
## NEWS & SENTIMENT
Overall Sentiment: {sentiment.get('overall')} 
(🟢 Bullish: {sentiment.get('bullish')} | 🔴 Bearish: {sentiment.get('bearish')} | ⚪ Neutral: {sentiment.get('neutral')} | Total: {sentiment.get('total')})

Recent Headlines:
{chr(10).join(news_lines) if news_lines else '  No relevant articles found'}
"""

    # ── Seasonal section ──────────────────────────────────────
    seasonal_block = f"""
## SEASONAL CONTEXT
- Current Season:  {seas.get('current_season')} ({seas.get('month')})
- Season Notes:    {seas.get('season_notes')}
- Seasonal Alert:  {seas.get('seasonal_alert') or 'None'}
"""

    # ── ICE Warehouse stocks ─────────────────────────────────
    warehouse = snapshot.get("ice_warehouse", {})
    if warehouse and "error" not in warehouse:
        wh_hist = warehouse.get("history_7d", {})
        wh_hist_lines = "\n".join(
            f"    {d}: {v:,} bags"
            for d, v in sorted(wh_hist.items(), reverse=True)
        ) if wh_hist else "    No history available"

        warehouse_block = f"""
## ICE WAREHOUSE STOCKS (certified European warehouses)
- As Of:         {warehouse.get('as_of')}
- Current:       {warehouse.get('current_bags'):,} bags  ({warehouse.get('current_tonnes'):,} tonnes)
- Previous:      {warehouse.get('prev_bags'):,} bags
- Change:        {warehouse.get('change_bags'):+,} bags  ({warehouse.get('change_pct')}%)
- 5-Day Trend:   {warehouse.get('trend_5d') or 'N/A'}
- 7-Day Trend:   {warehouse.get('trend_7d') or 'N/A'}
- 20-Day Trend:  {warehouse.get('trend_20d') or 'N/A'}
- Signal:        {warehouse.get('signal')}

Recent History (7 days):
{wh_hist_lines}
"""
    else:
        warehouse_block = "\n## ICE WAREHOUSE STOCKS\n  Data unavailable\n"

    # ── Grinding data (quarterly) ────────────────────────────
    grinding = snapshot.get("grinding_data", {})
    if grinding and "error" not in grinding:
        eca = grinding.get("ECA", {})
        nca = grinding.get("NCA", {})
        caa = grinding.get("CAA", {})
        icco = grinding.get("icco_balance", {})
        release_alert = snapshot.get("grinding_release_alert", "")

        # Safe formatting — handle None volumes
        eca_vol = f"{eca['volume_tonnes']:,}t" if eca.get('volume_tonnes') else "N/A"
        eca_fy  = f"{eca['full_year_2025']:,}t" if eca.get('full_year_2025') else "N/A"

        grinding_block = f"""
## GRINDING DATA (quarterly demand proxy)
Latest Quarter: {grinding.get('latest_quarter')} (released {grinding.get('release_date')})
{('⚠️  ' + release_alert) if release_alert else ''}

ECA (Europe — largest, most market-moving):
  - Volume:      {eca_vol}  ({eca.get('quarter')})
  - YoY Change:  {eca.get('yoy_change_pct')}%
  - Full Year:   {eca_fy}  (YoY: {eca.get('full_year_yoy')}%)
  - Note:        {eca.get('note', '')}

NCA (North America):
  - YoY Change:  {nca.get('yoy_change_pct') or 'awaiting release'}
  - Full Year:   YoY {nca.get('full_year_yoy')}%
  - Note:        {nca.get('note', '')}

CAA (Asia):
  - YoY Change:  {caa.get('yoy_change_pct')}%  ({caa.get('quarter')})
  - Trailing 12m: {caa.get('trailing_12m_yoy')}%
  - Note:        {caa.get('note', '')}

ICCO Supply/Demand Balance ({icco.get('season', 'N/A')}):
  - Balance:     {icco.get('surplus_deficit')}
  - Production:  {icco.get('global_production')}
  - Grindings:   {icco.get('global_grindings')}
  - Note:        {icco.get('note', '')}

Trend:  {grinding.get('trend_summary', '')}
"""
    else:
        grinding_block = "\n## GRINDING DATA\n  Data unavailable\n"

    # ── Grinding release impact model ────────────────────────
    impact = snapshot.get("grinding_impact", {})
    if impact and "error" not in impact:
        forecasts = impact.get("forecasts", [])
        bucket_stats = impact.get("bucket_stats", {})

        # Format bucket stats summary
        bucket_lines = []
        for bucket_name in ["large_miss", "moderate_miss", "in_line",
                            "moderate_beat", "large_beat"]:
            bs = bucket_stats.get(bucket_name, {})
            if bs.get("count", 0) == 0:
                continue
            t5_avg = bs.get("t5_avg")
            t5_range = bs.get("t5_range")
            t5_str = (
                f"T+5 avg={t5_avg:+.1f}%, "
                f"range=[{t5_range[0]:+.1f}%, {t5_range[1]:+.1f}%]"
                if t5_avg is not None and t5_range
                else "insufficient price data"
            )
            bucket_lines.append(
                f"  [{bucket_name.replace('_',' ').upper()}] "
                f"n={bs['count']}: {t5_str}"
            )

        # Format forecast scenarios
        forecast_lines = []
        for fc in forecasts:
            days = fc.get("days_away", "?")
            forecast_lines.append(
                f"\n  {fc['release']} — {fc['date']} ({days} days away)"
                f"\n  Consensus: {fc.get('consensus_yoy', 0):+.1f}% YoY"
            )
            for skey in ["bear", "base", "bull"]:
                s = fc.get("scenarios", {}).get(skey, {})
                t5 = s.get("impact_t5", {})
                avg = t5.get("avg")
                rng = t5.get("range")
                gbp_avg = s.get("gbp_impact_t5_avg")
                n = t5.get("count", 0)
                if avg is not None:
                    rng_str = (f"[{rng[0]:+.1f}%, {rng[1]:+.1f}%]"
                               if rng else "N/A")
                    gbp_str = (f"~£{gbp_avg:+.0f}/t"
                               if gbp_avg is not None else "")
                    forecast_lines.append(
                        f"    {skey.upper():5s}: {s.get('label', '')} "
                        f"→ T+5 avg {avg:+.1f}% {rng_str} {gbp_str} (n={n})"
                    )
                else:
                    forecast_lines.append(
                        f"    {skey.upper():5s}: {s.get('label', '')} "
                        f"→ insufficient data"
                    )

        grinding_impact_block = f"""
## GRINDING RELEASE IMPACT MODEL (historical price reaction analysis)
Data: {impact.get('data_note', '')}

Historical Bucket Statistics (% price change from T-1, by surprise magnitude):
{chr(10).join(bucket_lines) if bucket_lines else '  Insufficient price data to compute bucket stats'}

Upcoming Release Forecasts:
{chr(10).join(forecast_lines) if forecast_lines else '  No upcoming releases in forecast window'}
"""
    else:
        grinding_impact_block = ""

    # ── Crop health (satellite data — v2) ───────────────────
    crop_block = ""
    if CROP_MONITOR_AVAILABLE:
        try:
            crop = load_crop_health_for_agent()
            if "error" not in crop:
                age = crop.get("data_age_days", 0)
                if age and age > 21:
                    stale_note = f" ⚠️ STALE ({age} days old — re-run crop monitor)"
                elif age and age > 7:
                    stale_note = f" ℹ️ {age} days old"
                else:
                    stale_note = f" ✅ {age} days old" if age else ""

                region_lines = []
                for r in crop.get("regions", []):
                    src = r.get("source", "?")
                    src_note = f" [SAR fallback]" if src == "sentinel1_sar" else ""
                    line = (
                        f"  - {r['region']} ({r['country']}) [{r.get('period', '?')}]{src_note}:\n"
                        f"    EVI={r.get('evi')}  NDMI={r.get('ndmi')}  "
                        f"LST={r.get('lst_c')}°C  "
                        f"SM={r.get('soil_moisture')} "
                        f"{'(rootzone='+str(r.get('soil_moisture_rootzone'))+')' if r.get('soil_moisture_rootzone') else ''}  "
                        f"CHIRPS={r.get('chirps_rain_mm')}mm"
                    )
                    # Stressed pixel fraction
                    frac = r.get("ndmi_stressed_fraction")
                    if frac is not None:
                        line += f"\n    Stressed pixels: {frac*100:.0f}% below NDMI threshold"
                    # Seasonal anomaly
                    if r.get("ndmi_seasonal_mean") is not None:
                        n = r.get("seasonal_n", 0)
                        z_ndmi = r.get("ndmi_zscore")
                        z_evi  = r.get("evi_zscore")
                        z_str = f"NDMI z={z_ndmi:+.1f}" if z_ndmi is not None else ""
                        if z_evi is not None:
                            z_str += f", EVI z={z_evi:+.1f}"
                        line += f"\n    Seasonal: {z_str} ({n}yr baseline)"
                    # Flags
                    all_flags = (r.get("flags", []) or []) + (r.get("seasonal_flags", []) or [])
                    if all_flags:
                        line += f"\n    Flags: {', '.join(all_flags)}"
                    region_lines.append(line)

                region_lines = "\n".join(region_lines)

                # Run diff
                diff = crop.get("run_diff", {})
                diff_line = ""
                if diff.get("available"):
                    diff_line = f"\nChanges Since Last Run:\n  {diff.get('summary', 'None')}"

                crop_block = f"""
## SATELLITE CROP HEALTH v2 (S2 + SMAP + CHIRPS + SAR) [{stale_note}]
Overall Health Score: {crop.get('overall_score')} / 100
Overall Bias:         {crop.get('overall_bias')}
Signal:               {crop.get('overall_signal')}
Avg EVI:              {crop.get('avg_evi')}  (canopy activity; dense tropical canopy primary indicator)
Avg NDMI (Gao):       {crop.get('avg_ndmi')}  (leaf water content; stress < 0.10, healthy > 0.25)
Avg Soil Moisture:    {crop.get('avg_soil_moisture')}  (SMAP; leading indicator, dries 2-4wks before canopy)
Critical Flags:       {crop.get('critical_flags')}
Warning Flags:        {crop.get('warning_flags')}
{diff_line}

By Region (latest period):
{region_lines}
"""
        except Exception as e:
            crop_block = f"\n## SATELLITE CROP HEALTH\n  Data unavailable: {e}\n"

    # ── Combined stress signal ────────────────────────────────
    stress_block = ""
    if STRESS_SIGNAL_AVAILABLE:
        combined = snapshot.get("combined_stress", {})
        if combined and "error" not in combined:
            stress_block = format_stress_signal(combined)
        else:
            stress_block = "\n## COMBINED CROP STRESS SIGNAL\n  Not available — run cocoa_data_gatherer.py with crop health data present\n"

    # ── COT positioning data ────────────────────────────────
    cot = snapshot.get("cot", {})
    cot_block = ""
    if cot and "error" not in cot:
        def _fmt(val, fmt=","):
            """Format a number safely, returning 'N/A' for None."""
            if val is None:
                return "N/A"
            try:
                return f"{val:{fmt}}"
            except (ValueError, TypeError):
                return str(val)

        cot_block = f"""
## CFTC COMMITMENTS OF TRADERS (speculative positioning)
  Report Date:          {cot.get('report_date', 'N/A')}
  Contract:             {cot.get('contract', 'N/A')}
  Managed Money Long:   {_fmt(cot.get('managed_money_long'))} contracts
  Managed Money Short:  {_fmt(cot.get('managed_money_short'))} contracts
  Managed Money Net:    {_fmt(cot.get('managed_money_net'), '+,')} contracts
  Net Position Pctile:  {cot.get('net_position_percentile', 'N/A — building history')}% (vs 3yr range)
  Total Open Interest:  {_fmt(cot.get('total_open_interest'))} contracts
  MM Long % of OI:      {cot.get('mm_long_pct_oi', 'N/A')}%
  MM Short % of OI:     {cot.get('mm_short_pct_oi', 'N/A')}%
  Positioning Signal:   {cot.get('positioning_signal', 'N/A — building history')}
  Bias:                 {cot.get('positioning_bias', 'N/A')}
  Note:                 {cot.get('positioning_note', '')}

  Producer/Merchant Net: {_fmt(cot.get('producer_merchant_net'), '+,')} contracts
"""
    else:
        cot_block = "\n## CFTC COT POSITIONING\n  Data unavailable\n"

    # ── Continuous learning prompt ──────────────────────────
    learning_block = ""
    if WEEKLY_REVIEW_AVAILABLE:
        try:
            learning_block = build_learning_prompt()
            if learning_block:
                log.info("  ✅ Learning block injected into prompt")
        except Exception as e:
            log.warning(f"Learning prompt generation failed (non-critical): {e}")

    # ── Prediction accuracy feedback ─────────────────────────
    feedback_block = ""
    if FEEDBACK_AVAILABLE:
        try:
            feedback_text = build_feedback_prompt()
            if feedback_text:
                feedback_block = f"\n{feedback_text}\n"
                log.info("  ✅ Feedback block injected into prompt")
            else:
                feedback_block = (
                    "\n## PREDICTION ACCURACY FEEDBACK\n"
                    "No scored predictions yet — feedback loop will activate after "
                    "predictions have been recorded and evaluated.\n"
                )
        except Exception as e:
            log.warning(f"Feedback generation failed (non-critical): {e}")

    # ── Continuous learning block (shadow accuracy + post-mortems) ──
    learning_block = ""
    if WEEKLY_REVIEW_AVAILABLE:
        try:
            learning_text = build_learning_prompt()
            if learning_text:
                learning_block = f"\n{learning_text}\n"
        except Exception as e:
            log.warning(f"Learning prompt generation failed (non-critical): {e}")

    # ── Assemble full prompt ──────────────────────────────────
    generated_at = snapshot.get("generated_at", datetime.now(timezone.utc).isoformat())

    prompt = f"""Please analyse the following cocoa market data and produce your daily valuation assessment.

Data snapshot generated at: {generated_at}

{price_block}
{ta_block}
{related_block}
{weather_block}
{news_block}
{seasonal_block}
{warehouse_block}
{grinding_block}
{grinding_impact_block}
{crop_block}
{stress_block}
{cot_block}
{learning_block}
{feedback_block}
{learning_block}
Apply your analytical framework: assess fair value from fundamentals (supply, demand, stocks, macro),
then use technicals to determine timing. Produce your VALUATION BIAS and TIMING SIGNAL.
"""
    return prompt


# ─────────────────────────────────────────────
#  CLAUDE API CALL
# ─────────────────────────────────────────────

def call_claude(user_prompt: str) -> str:
    """Send the prompt to Claude and return the report text."""
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not set in environment / .env file")

    log.info(f"Calling Claude API ({CLAUDE_MODEL})...")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=3000,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_prompt}
        ]
    )

    report = message.content[0].text
    log.info(f"  → Report generated ({len(report)} chars, "
             f"~{message.usage.input_tokens} in / {message.usage.output_tokens} out tokens)")
    return report


# ─────────────────────────────────────────────
#  REPORT FORMATTING
# ─────────────────────────────────────────────

def format_report(report: str, snapshot: dict) -> str:
    """Wrap the Claude output in a clean report header."""
    generated_at = snapshot.get("generated_at", "")[:16].replace("T", " ")
    price        = snapshot.get("technicals", {}).get("price", {}).get("current", "N/A")
    price_gbp    = snapshot.get("price_gbp")
    gbp_str      = f" (≈ {price_gbp} GBP/t)" if price_gbp else ""
    change       = snapshot.get("technicals", {}).get("price", {}).get("change_1d_pct", "N/A")
    season       = snapshot.get("seasonal", {}).get("current_season", "N/A")

    # Feedback stats line
    feedback_line = ""
    if FEEDBACK_AVAILABLE:
        try:
            stats = get_ledger_stats()
            if stats["scored"] > 0:
                feedback_line = f"  |  **Feedback:** {stats['scored']} scored predictions"
        except Exception:
            pass

    header = f"""# 🍫 COCOA TRADING ASSISTANT — Daily Valuation Report
**Instrument:** ICE New York Cocoa CC=F (USD/tonne)
**Generated:** {generated_at} UTC  |  **Price:** {price} USD/t{gbp_str} ({change}%)  |  **Season:** {season}{feedback_line}

---

"""
    return header + report


# ─────────────────────────────────────────────
#  DELIVERY: TELEGRAM
# ─────────────────────────────────────────────

def send_telegram(text: str) -> bool:
    """
    Send report via Telegram bot.
    Telegram has a 4096 char limit per message, so we split if needed.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.info("Telegram credentials not set — skipping Telegram delivery")
        return False

    try:
        import urllib.request
        import urllib.parse

        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

        # Split into chunks if needed (Telegram 4096 char limit)
        chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
        log.info(f"Sending to Telegram ({len(chunks)} message(s))...")

        for i, chunk in enumerate(chunks):
            payload = json.dumps({
                "chat_id":    TELEGRAM_CHAT_ID,
                "text":       chunk,
                "parse_mode": "Markdown",
            }).encode("utf-8")

            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read())
                if not result.get("ok"):
                    log.error(f"Telegram error: {result}")
                    return False

        log.info("  ✅ Sent via Telegram")
        return True

    except Exception as e:
        log.error(f"Telegram delivery failed: {e}")
        return False


# ─────────────────────────────────────────────
#  DELIVERY: EMAIL
# ─────────────────────────────────────────────

def send_email(report_md: str, snapshot: dict) -> bool:
    """Send report via Gmail SMTP."""
    if not all([EMAIL_FROM, EMAIL_TO, EMAIL_APP_PASSWORD]):
        log.info("Email credentials not set — skipping email delivery")
        return False

    try:
        price    = snapshot.get("technicals", {}).get("price", {}).get("current", "N/A")
        change   = snapshot.get("technicals", {}).get("price", {}).get("change_1d_pct", "N/A")
        ta       = snapshot.get("technicals", {})
        trend    = ta.get("trend", {}).get("label", "N/A")
        date_str = datetime.now().strftime("%d %b %Y")

        subject = f"🍫 Cocoa Daily Report — {date_str} | {price} USD/t ({change}%) | {trend}"

        # Convert markdown to simple HTML
        html_body = report_md.replace("\n", "<br>").replace("###", "<h3>").replace("**", "<b>")

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = EMAIL_FROM
        msg["To"]      = EMAIL_TO

        msg.attach(MIMEText(report_md,  "plain"))
        msg.attach(MIMEText(f"<html><body>{html_body}</body></html>", "html"))

        log.info(f"Sending email to {EMAIL_TO}...")
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_FROM, EMAIL_APP_PASSWORD)
            server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())

        log.info("  ✅ Sent via email")
        return True

    except Exception as e:
        log.error(f"Email delivery failed: {e}")
        return False


# ─────────────────────────────────────────────
#  SAVE REPORT TO FILE
# ─────────────────────────────────────────────

def save_report(report: str, filepath: str = REPORT_OUTPUT_FILE):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)
    log.info(f"  ✅ Report saved to: {filepath}")


# ─────────────────────────────────────────────
#  DAILY RECOMMENDATION PERSISTENCE
#  Extracts key fields from the Claude report
#  so the intra-day agent can anchor against them.
# ─────────────────────────────────────────────

DAILY_REC_FILE = "cocoa_daily_rec.json"

def extract_and_save_daily_recommendation(report: str, snapshot: dict):
    """
    Parse the structured report to extract the valuation bias and timing
    signal fields, save as JSON for the intra-day agent, and record the
    prediction in the feedback ledger for future scoring.
    """
    import re

    def extract_field(pattern, text, default="N/A"):
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else default

    # ── Extract from Valuation Bias section ─────────────────
    vb_section = ""
    m = re.search(
        r"(?:###?\s*)?(?:✅\s*)?VALUATION BIAS(.*?)(?=###?\s*(?:⏱|TIMING)|$)",
        report, re.DOTALL | re.IGNORECASE
    )
    if m:
        vb_section = m.group(1)

    # ── Extract from Timing Signal section ──────────────────
    ts_section = ""
    m = re.search(
        r"(?:###?\s*)?(?:⏱\s*)?TIMING SIGNAL(.*?)(?=###?\s*(?:⚠|KEY RISKS|WATCH)|$)",
        report, re.DOTALL | re.IGNORECASE
    )
    if m:
        ts_section = m.group(1)

    rec = {
        "generated_at":    snapshot.get("generated_at", ""),
        "price_at_report": snapshot.get("technicals", {}).get("price", {}).get("current"),
        "valuation_bias": {
            "assessment":     extract_field(r"\*\*Assessment:\*\*\s*(.+?)(?:\n|$)", vb_section),
            "magnitude":      extract_field(r"\*\*Magnitude:\*\*\s*(.+?)(?:\n|$)", vb_section),
            "fair_range":     extract_field(r"\*\*(?:Estimated )?Fair Range:\*\*\s*(.+?)(?:\n|$)", vb_section),
            "primary_driver": extract_field(r"\*\*Primary Driver.*?:\*\*\s*(.+?)(?:\n|$)", vb_section),
            "confidence":     extract_field(r"\*\*Confidence:\*\*\s*(.+?)(?:\n|$)", vb_section),
            "rationale":      extract_field(r"\*\*Rationale:\*\*\s*(.+?)(?:\n\n|\*\*|\Z)", vb_section),
        },
        "timing_signal": {
            "timing":        extract_field(r"\*\*Timing:\*\*\s*(.+?)(?:\n|$)", ts_section),
            "catalyst":      extract_field(r"\*\*Catalyst:\*\*\s*(.+?)(?:\n|$)", ts_section),
            "entry":         extract_field(r"\*\*Entry:\*\*\s*(.+?)(?:\n|$)", ts_section),
            "invalidation":  extract_field(r"\*\*(?:Key )?Invalidation:\*\*\s*(.+?)(?:\n|$)", ts_section),
        },
    }

    try:
        with open(DAILY_REC_FILE, "w") as f:
            json.dump(rec, f, indent=2, default=str)
        log.info(f"  ✅ Daily recommendation saved to: {DAILY_REC_FILE}")
    except Exception as e:
        log.warning(f"  Failed to save daily rec file: {e}")

    # ── Record prediction in the feedback ledger ────────────
    if FEEDBACK_AVAILABLE:
        try:
            parsed = extract_recommendation(report)
            record_prediction(report, snapshot, parsed)
        except Exception as e:
            log.warning(f"  Failed to record prediction in feedback ledger: {e}")

    return rec


# ─────────────────────────────────────────────
#  MAIN ORCHESTRATOR
# ─────────────────────────────────────────────

def run_agent(snapshot_file: str = SNAPSHOT_FILE, dry_run: bool = False):
    """
    Main entry point.
    Loads snapshot → builds prompt → calls Claude → delivers report.

    Args:
        snapshot_file: Path to the JSON snapshot from data_gatherer.py
        dry_run:       If True, print prompt only — don't call Claude API
    """
    log.info("=" * 55)
    log.info("  COCOA AGENT — Generating Daily Valuation Assessment")
    log.info(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 55)

    # ── Evaluate pending predictions from previous runs ───
    if FEEDBACK_AVAILABLE:
        try:
            stats = get_ledger_stats()
            log.info(f"Feedback ledger: {stats['total']} predictions "
                     f"({stats['pending']} pending, {stats['scored']} scored)")
        except Exception as e:
            log.warning(f"Feedback stats failed: {e}")

    # ── Load snapshot ──────────────────────────────────────
    log.info(f"Loading snapshot from: {snapshot_file}")
    try:
        with open(snapshot_file, "r") as f:
            snapshot = json.load(f)
    except FileNotFoundError:
        log.error(f"Snapshot file not found: {snapshot_file}")
        log.error("Run cocoa_data_gatherer.py first to generate the snapshot.")
        return
    except json.JSONDecodeError as e:
        log.error(f"Failed to parse snapshot JSON: {e}")
        return

    snapshot_age = None
    generated_at_str = snapshot.get("generated_at")
    if generated_at_str:
        try:
            generated_at = datetime.fromisoformat(generated_at_str)
            snapshot_age = (datetime.now(timezone.utc) - generated_at).total_seconds() / 3600
            log.info(f"Snapshot age: {snapshot_age:.1f} hours")
            if snapshot_age > 12:
                log.warning("⚠️  Snapshot is over 12 hours old — consider re-running the data gatherer")
        except Exception:
            pass

    # ── Build prompt ───────────────────────────────────────
    # First, evaluate any pending predictions using the current price
    if FEEDBACK_AVAILABLE:
        try:
            # Get current price from snapshot for evaluation
            eval_price = None
            px = snapshot.get("technicals", {}).get("price", {})
            if px.get("current"):
                eval_price = float(px["current"])
            if eval_price:
                n_scored = evaluate_pending(current_price=eval_price)
                if n_scored > 0:
                    log.info(f"  📊 Scored {n_scored} past prediction(s) at {eval_price} USD/t")
        except Exception as e:
            log.warning(f"Prediction evaluation failed (non-critical): {e}")

    # ── Shadow prediction scoring & big-miss detection ────
    if WEEKLY_REVIEW_AVAILABLE:
        try:
            eval_price = float(
                snapshot.get("technicals", {}).get("price", {}).get("current", 0)
            ) if snapshot.get("technicals", {}).get("price", {}).get("current") else None

            if eval_price:
                n_shadow = score_shadow_predictions(eval_price)
                if n_shadow > 0:
                    log.info(f"  📊 Scored {n_shadow} shadow prediction horizon(s)")
                misses = check_big_misses(eval_price)
                if misses:
                    log.warning(f"  ⚠️  {len(misses)} big miss(es) detected this run")
        except Exception as e:
            log.warning(f"Shadow scoring / big-miss detection failed (non-critical): {e}")

    log.info("Building prompt...")
    user_prompt = build_user_prompt(snapshot)

    if dry_run:
        print("\n" + "=" * 55)
        print("  DRY RUN — PROMPT PREVIEW (no API call made)")
        print("=" * 55)
        print(user_prompt)
        return

    # ── Call Claude ────────────────────────────────────────
    raw_report = call_claude(user_prompt)

    # ── Format ─────────────────────────────────────────────
    formatted_report = format_report(raw_report, snapshot)

    # ── Save to file (always — even SILENT runs) ──────────
    save_report(formatted_report)

    # ── Extract & persist recommendation ──────────────────
    rec = extract_and_save_daily_recommendation(raw_report, snapshot)

    # ── Opportunity Scoring ───────────────────────────────
    # This is the core decision: should we alert the user or stay silent?
    alert_level = "OPPORTUNITY"   # Default: always send (fallback if scorer unavailable)
    opp_result  = None
    parsed_rec  = {}              # Initialised here so it's always in scope below

    if SCORER_AVAILABLE and FEEDBACK_AVAILABLE:
        try:
            # Parse the recommendation for scoring
            parsed_rec = extract_recommendation(raw_report)

            # Load feedback stats for track record scoring
            import json as _json
            feedback_stats = None
            try:
                with open("cocoa_feedback_summary.json", "r") as f:
                    feedback_stats = _json.load(f)
            except (FileNotFoundError, _json.JSONDecodeError):
                pass

            opp_result = score_opportunity(snapshot, parsed_rec, feedback_stats)
            alert_level = opp_result["alert_level"]

            # Log the opportunity score (always)
            log_opportunity(opp_result, snapshot)

        except Exception as e:
            log.warning(f"Opportunity scoring failed (non-critical): {e}")
            alert_level = "OPPORTUNITY"  # Fail open — send the report

    elif SCORER_AVAILABLE:
        try:
            parsed_rec = {}
            # Try basic extraction without full feedback
            try:
                from cocoa_feedback import extract_recommendation as _extract
                parsed_rec = _extract(raw_report)
            except ImportError:
                pass
            opp_result = score_opportunity(snapshot, parsed_rec, None)
            alert_level = opp_result["alert_level"]
            log_opportunity(opp_result, snapshot)
        except Exception as e:
            log.warning(f"Opportunity scoring failed: {e}")

    # ── Shadow prediction recording (every run, regardless of alert level) ──
    if WEEKLY_REVIEW_AVAILABLE:
        try:
            record_shadow_prediction(snapshot, parsed_rec, opp_result)
        except Exception as e:
            log.warning(f"Shadow prediction recording failed (non-critical): {e}")

        # ── Weekly report (Sundays only, once per day) ────────────────────
        try:
            if should_generate_weekly_report():
                current_price = snapshot.get("technicals", {}).get("price", {}).get("current")
                if current_price:
                    weekly_report = generate_weekly_report(
                        float(current_price), snapshot, opp_result
                    )
                    log.info("  📋 Weekly report generated — sending via Telegram")
                    send_telegram(weekly_report)
        except Exception as e:
            log.warning(f"Weekly report generation failed (non-critical): {e}")

    # Override alert level if --force-alert was set
    if os.environ.get("COCOA_FORCE_ALERT") == "1":
        log.info("  --force-alert: overriding alert level to OPPORTUNITY")
        alert_level = "OPPORTUNITY"

    # ── Record shadow prediction (always, regardless of alert level) ──
    if WEEKLY_REVIEW_AVAILABLE and FEEDBACK_AVAILABLE:
        try:
            parsed = extract_recommendation(raw_report)
            record_shadow_prediction(snapshot, parsed, opp_result)
        except Exception as e:
            log.warning(f"Shadow prediction recording failed (non-critical): {e}")

    # ── Weekly report (on designated day) ─────────────────
    weekly_report = None
    if WEEKLY_REVIEW_AVAILABLE and should_generate_weekly_report():
        try:
            eval_price = snapshot.get("technicals", {}).get("price", {}).get("current")
            if eval_price:
                weekly_report = generate_weekly_report(
                    float(eval_price), snapshot, opp_result
                )
                log.info("  📋 Weekly report generated")
        except Exception as e:
            log.warning(f"Weekly report generation failed (non-critical): {e}")

    # ── Print to console (always — for logging/debugging) ─
    print("\n" + formatted_report)
    if opp_result:
        print(f"\n🎯 Opportunity Score: {opp_result['total_score']}/100 → {alert_level}")
        for comp_name, comp_data in opp_result["components"].items():
            print(f"   {comp_name:<18} {comp_data['score']:>5.1f}  {comp_data.get('rationale', '')[:70]}")

    # ── Deliver based on alert level ──────────────────────
    delivered = False

    if alert_level == "SILENT":
        log.info("  📊 SILENT — no actionable opportunity. Logged internally.")

    elif alert_level == "MONITOR":
        log.info(f"  👁️ MONITOR — {opp_result['summary'] if opp_result else 'monitoring'}")
        # Save a brief monitoring log entry but don't alert
        _save_monitor_entry(opp_result, snapshot)

    elif alert_level == "WATCHLIST":
        log.info(f"  ⚠️ WATCHLIST — sending alert")
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID and opp_result:
            watchlist_msg = format_watchlist_alert(opp_result, snapshot)
            delivered = send_telegram(watchlist_msg)

    elif alert_level == "OPPORTUNITY":
        log.info(f"  🔔 OPPORTUNITY — sending full report")
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            if opp_result:
                opp_msg = format_opportunity_alert(opp_result, formatted_report, snapshot)
                delivered = send_telegram(opp_msg)
            else:
                delivered = send_telegram(formatted_report)

        if EMAIL_FROM and EMAIL_TO and EMAIL_APP_PASSWORD:
            delivered = send_email(formatted_report, snapshot) or delivered

    if alert_level in ("WATCHLIST", "OPPORTUNITY") and not delivered:
        log.info("No delivery method configured — report saved to file only.")

    # ── Send weekly report if generated ───────────────────
    if weekly_report and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        # Truncate for Telegram if needed (4096 char limit)
        if len(weekly_report) > 4000:
            weekly_report = weekly_report[:3950] + "\n\n... (truncated)"
        send_telegram(weekly_report)
        log.info("  📋 Weekly report sent to Telegram")

    log.info(f"\n✅ Agent run complete. Alert level: {alert_level}")
    return formatted_report


def _save_monitor_entry(opp_result: dict, snapshot: dict):
    """Save a brief monitoring log entry (not sent to user)."""
    import json as _json
    MONITOR_LOG = "cocoa_monitor_log.json"
    entry = {
        "date": datetime.now(timezone.utc).isoformat()[:19],
        "score": opp_result["total_score"] if opp_result else None,
        "direction": opp_result["direction"] if opp_result else None,
        "summary": opp_result["summary"] if opp_result else None,
        "price": snapshot.get("technicals", {}).get("price", {}).get("current"),
    }
    try:
        with open(MONITOR_LOG, "r") as f:
            log_data = _json.load(f)
    except (FileNotFoundError, _json.JSONDecodeError):
        log_data = []
    log_data.append(entry)
    log_data = log_data[-90:]  # Keep last 90 days
    try:
        with open(MONITOR_LOG, "w") as f:
            _json.dump(log_data, f, indent=2)
    except Exception:
        pass


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cocoa Trading Assistant — AI Agent")
    parser.add_argument(
        "--snapshot", "-s",
        default=SNAPSHOT_FILE,
        help=f"Path to snapshot JSON file (default: {SNAPSHOT_FILE})"
    )
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="Preview the prompt without calling the Claude API"
    )
    parser.add_argument(
        "--force-alert",
        action="store_true",
        help="Force OPPORTUNITY alert level regardless of score (for testing)"
    )
    args = parser.parse_args()

    if args.force_alert:
        os.environ["COCOA_FORCE_ALERT"] = "1"

    run_agent(snapshot_file=args.snapshot, dry_run=args.dry_run)
