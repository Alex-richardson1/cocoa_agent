"""
=============================================================
  COCOA NEWS INTELLIGENCE AGENT
=============================================================
  Standalone script that:
    1. Fetches recent (7-day) and background (90-day) articles
       via Google News RSS across targeted cocoa topics
    2. Fetches from specialist commodity/market sources
    3. Passes all gathered content to a Claude summarisation
       agent that produces a structured intelligence brief
    4. Writes the brief to cocoa_news_intelligence.json
       (also merges into cocoa_daily_snapshot.json if present)

  Run standalone:
    python cocoa_news_agent.py

  Or imported by cocoa_data_gatherer.py:
    from cocoa_news_agent import fetch_news_intelligence

  Requirements:
    pip install requests feedparser anthropic python-dotenv
=============================================================
"""

import os
import json
import logging
import re
import requests
import feedparser
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import anthropic

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

ANTHROPIC_API_KEY     = os.getenv("ANTHROPIC_API_KEY", "")
NEWS_OUTPUT_FILE      = "cocoa_news_intelligence.json"
SNAPSHOT_FILE         = "cocoa_daily_snapshot.json"

# ─────────────────────────────────────────────
#  SEARCH QUERIES
# ─────────────────────────────────────────────

# (label, query string, window_days)
# Recent = last 7 days for price-sensitive signals
# Background = last 90 days for structural context
GOOGLE_NEWS_QUERIES = [
    # ── Recent market signals (7-day, keep queries short and broad) ────────
    ("prices_recent",      "cocoa prices",                              7),
    ("futures_recent",     "cocoa futures market",                      7),
    ("supply_recent",      "cocoa supply",                              7),
    ("harvest_recent",     "cocoa harvest Africa",                      7),
    ("demand_recent",      "chocolate demand",                          7),
    ("inventory_recent",   "cocoa inventory warehouse",                 7),
    ("weather_recent",     "cocoa weather",                             7),
    ("disease_recent",     "cocoa crop disease",                        7),
    ("regulation_recent",  "cocoa EUDR",                                7),
    ("wire_recent",  "cocoa site:cnbcafrica.com OR site:businessday.co.za",  7),

    # ── Background / structural context (90-day, slightly more specific) ──
    ("outlook_background", "cocoa price forecast 2026",                90),
    ("supply_background",  "cocoa crop forecast Africa",               90),
    ("grindings_bg",       "cocoa grindings",                          90),
    ("demand_background",  "chocolate sales consumption",              90),
    ("macro_background",   "cocoa surplus deficit",                    90),
    ("policy_background",  "cocoa deforestation regulation",           90),

]

SPECIALIST_FEEDS = [
    # ── Confirmed working ────────────────────────────────────────────
    ("SpreadCharts",        "https://spreadcharts.com/feed"),
    ("Food Dive",           "https://fooddive.com/feeds/news"),
    ("FoodNavigator-USA",   "https://www.foodnavigator-usa.com/arc/outboundfeeds/rss"),

    # ── Very likely working (same Arc XP platform, active sites) ─────
    ("FoodNavigator EU",    "https://www.foodnavigator.com/arc/outboundfeeds/rss"),
    ("Confectionery News",  "https://www.confectionerynews.com/arc/outboundfeeds/rss"),

    # ── WordPress standard feeds (active sites, infrequent publish) ──
    ("ICCO",                "https://www.icco.org/feed/"),
    ("GhanaWeb Business",   "https://www.ghanaweb.com/GhanaHomePage/business/rss.xml"),

    ("CNBC Africa",         "https://www.cnbcafrica.com/feed/"), # Reuters Abidjan wire republisher
    ("BusinessDay SA",      "https://www.businessday.co.za/feed/")
]

# Source authority weights for ranking
SOURCE_WEIGHTS = {
    "reuters":          10,
    "bloomberg":        10,
    "financial times":   9,
    "ft.com":            9,
    "wall street":       9,
    "icco":              9,
    "barchart":          8,
    "agrimoney":         8,
    "confectionery":     7,
    "foodnavigator":     6,
    "businessgreen":     5,
    "nasdaq":            5,
    "marketwatch":       5,
    "cnbc africa":       8,    # Republishes Reuters Abidjan wire (Ange Aboa) same-day
    "cnbcafrica":        8,
    "businessday":       7,    # SA — carries Reuters commodity wires
}

COCOA_KEYWORDS = [
    "cocoa", "cacao", "chocolate", "cocobod", "barry callebaut",
    "mondelez", "ivory coast", "côte d'ivoire", "ghana", "west africa",
    "grinding", "grindings", "icco", "soft commodity", "softs",
    "harmattan", "main crop", "mid crop", "lcc", "certified stock",
    "swollen shoot", "black pod", "eudr", "olam", "cargill cocoa",
    "cameroon", "nigeria cocoa", "ecuador cocoa",
]


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def is_cocoa_relevant(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in COCOA_KEYWORDS)


def get_source_weight(name: str) -> int:
    n = name.lower()
    for key, w in SOURCE_WEIGHTS.items():
        if key in n:
            return w
    return 3


def parse_date(entry) -> datetime | None:
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        try:
            return datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        except Exception:
            pass
    return None


def age_label(dt: datetime | None) -> str:
    if not dt:
        return "unknown date"
    days = (datetime.now(timezone.utc) - dt).days
    if days == 0:
        return "today"
    elif days == 1:
        return "yesterday"
    elif days < 7:
        return f"{days} days ago"
    elif days < 30:
        return f"{days // 7} week(s) ago"
    else:
        return f"{days // 30} month(s) ago"


ANALYST_FORECAST_URLS = [
    ("Capital.com",   "https://capital.com/en-gb/analysis/cocoa-price-forecast"),
    ("TradingEcon",   "https://tradingeconomics.com/commodity/cocoa"),
]

# Max characters to extract per analyst page (keeps prompt size manageable)
FORECAST_EXTRACT_CHARS = 3000


# ─────────────────────────────────────────────
#  ANALYST FORECAST SCRAPER
# ─────────────────────────────────────────────

def fetch_analyst_forecasts() -> list:
    """
    Fetch and extract text from known analyst forecast pages.
    Returns a list of dicts with source, url, and extracted text.
    These are passed directly to the summary agent as additional context.
    """
    from bs4 import BeautifulSoup
    results = []
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0.0.0 Safari/537.36"
        )
    }

    for source, url in ANALYST_FORECAST_URLS:
        try:
            r = requests.get(url, headers=headers, timeout=20)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")

            # Remove nav, footer, script, style noise
            for tag in soup(["script", "style", "nav", "footer", "header",
                              "aside", "form", "iframe"]):
                tag.decompose()

            text = soup.get_text("\n", strip=True)
            # Collapse excessive whitespace
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = re.sub(r" {2,}", " ", text)

            # Try to find the most relevant section (contains "cocoa" and "forecast"/"price")
            lines = text.split("\n")
            relevant_start = 0
            for i, line in enumerate(lines):
                if "cocoa" in line.lower() and any(
                    kw in line.lower() for kw in ["forecast", "price", "outlook", "predict"]
                ):
                    relevant_start = max(0, i - 2)
                    break

            extract = "\n".join(lines[relevant_start:])[:FORECAST_EXTRACT_CHARS]

            results.append({
                "source": source,
                "url":    url,
                "text":   extract,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            })
            log.info(f"  ✅ Analyst forecast fetched: {source} ({len(extract)} chars)")

        except Exception as e:
            log.warning(f"  Analyst forecast failed ({source}): {e}")
            results.append({"source": source, "url": url, "error": str(e)})

    return results




def fetch_google_news(query: str, days_back: int, max_results: int = 10) -> list:
    """Fetch articles from Google News RSS for a given query and time window."""
    encoded = requests.utils.quote(query)
    url     = (
        f"https://news.google.com/rss/search?q={encoded}"
        f"&hl=en-GB&gl=GB&ceid=GB:en"
    )
    cutoff  = datetime.now(timezone.utc) - timedelta(days=days_back)
    articles = []

    try:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            title   = entry.get("title", "") or ""
            summary = entry.get("summary", "") or ""
            link    = entry.get("link", "")
            pub_dt  = parse_date(entry)

            if pub_dt and pub_dt < cutoff:
                continue
            if not is_cocoa_relevant(title + " " + summary):
                continue

            # Google News encodes publisher in title as "Headline - Publisher"
            publisher = "Google News"
            if " - " in title:
                publisher = title.rsplit(" - ", 1)[-1].strip()
                title     = title.rsplit(" - ", 1)[0].strip()

            articles.append({
                "title":     title,
                "summary":   re.sub(r"<[^>]+>", "", summary)[:500].strip(),
                "link":      link,
                "published": pub_dt.isoformat() if pub_dt else None,
                "age":       age_label(pub_dt),
                "source":    publisher,
                "weight":    get_source_weight(publisher),
            })

            if len(articles) >= max_results:
                break

    except Exception as e:
        log.warning(f"  Google News query '{query}' failed: {e}")

    return articles


# ─────────────────────────────────────────────
#  2. SPECIALIST FEEDS
# ─────────────────────────────────────────────

def fetch_specialist_feeds(days_back: int = 30) -> list:
    """Fetch from known specialist commodity news sources."""
    cutoff   = datetime.now(timezone.utc) - timedelta(days=days_back)
    articles = []

    for source_name, url in SPECIALIST_FEEDS:
        count = 0
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                title   = entry.get("title", "") or ""
                summary = entry.get("summary", "") or ""
                link    = entry.get("link", "")
                pub_dt  = parse_date(entry)

                if pub_dt and pub_dt < cutoff:
                    continue
                if not is_cocoa_relevant(title + " " + summary):
                    continue

                articles.append({
                    "title":     title,
                    "summary":   re.sub(r"<[^>]+>", "", summary)[:500].strip(),
                    "link":      link,
                    "published": pub_dt.isoformat() if pub_dt else None,
                    "age":       age_label(pub_dt),
                    "source":    source_name,
                    "weight":    get_source_weight(source_name),
                })
                count += 1
                if count >= 8:
                    break

            log.info(f"  {source_name}: {count} articles")
        except Exception as e:
            log.warning(f"  Specialist feed failed ({source_name}): {e}")

    return articles


# ─────────────────────────────────────────────
#  3. GATHER & DEDUPLICATE
# ─────────────────────────────────────────────

def gather_all_articles() -> dict:
    """
    Run all queries and feeds, deduplicate, and split into
    recent (≤7 days) and background (>7 days) buckets.
    """
    log.info("=" * 55)
    log.info("  COCOA NEWS INTELLIGENCE — Gathering Articles")
    log.info("=" * 55)

    all_articles = []
    seen_titles  = set()

    # Google News queries
    for label, query, days in GOOGLE_NEWS_QUERIES:
        log.info(f"  Querying: '{query}' ({days}d window)")
        fetched = fetch_google_news(query, days_back=days, max_results=10)
        new = 0
        for a in fetched:
            key = a["title"].lower().strip()
            if key not in seen_titles:
                seen_titles.add(key)
                a["query_label"] = label
                all_articles.append(a)
                new += 1
        log.info(f"    → {new} new articles")

    # Specialist feeds (30-day window)
    log.info("  Fetching specialist feeds...")
    for a in fetch_specialist_feeds(days_back=30):
        key = a["title"].lower().strip()
        if key not in seen_titles:
            seen_titles.add(key)
            a["query_label"] = "specialist"
            all_articles.append(a)

    # Sort by weight then recency
    all_articles.sort(key=lambda a: (
        -a.get("weight", 0),
        -(datetime.fromisoformat(a["published"]).timestamp()
          if a.get("published") else 0)
    ))

    # Split into recent vs background
    cutoff_recent = datetime.now(timezone.utc) - timedelta(days=7)
    recent     = []
    background = []
    for a in all_articles:
        pub = datetime.fromisoformat(a["published"]) if a.get("published") else None
        if pub and pub >= cutoff_recent:
            recent.append(a)
        else:
            background.append(a)

    log.info(f"\n  Total: {len(all_articles)} articles "
             f"({len(recent)} recent ≤7d, {len(background)} background)")

    # Analyst forecast pages
    log.info("  Fetching analyst forecast pages...")
    analyst_forecasts = fetch_analyst_forecasts()
    good = sum(1 for f in analyst_forecasts if "error" not in f)
    log.info(f"  → {good}/{len(analyst_forecasts)} analyst pages fetched")

    return {
        "recent":            recent[:30],
        "background":        background[:20],
        "all":               all_articles,
        "analyst_forecasts": analyst_forecasts,
    }


# ─────────────────────────────────────────────
#  4. SUMMARISATION AGENT
# ─────────────────────────────────────────────

SUMMARY_SYSTEM_PROMPT = """You are a specialist cocoa commodity market intelligence analyst.

Your job is to synthesise a set of news articles and market reports into a 
structured intelligence brief for a professional cocoa trader.

You will be given two buckets of articles:
  - RECENT (last 7 days): price-sensitive, actionable signals
  - BACKGROUND (last 7-90 days): structural context and trends

Your output must be a valid JSON object with exactly this structure:
{
  "generated_at": "<ISO timestamp>",
  "recent_summary": "<3-5 sentences on what has happened in the last 7 days — price moves, key events, data releases>",
  "background_context": "<3-5 sentences on the structural picture — supply/demand balance, grinding trends, seasonal factors, producer country dynamics>",
  "key_themes": ["<theme 1>", "<theme 2>", "<theme 3>"],
  "directional_signal": "Bullish" | "Bearish" | "Neutral" | "Mixed",
  "signal_confidence": "High" | "Medium" | "Low",
  "signal_rationale": "<1-2 sentences on why you rated the signal this way>",
  "demand_forecast": {
    "full_year_outlook": "<one sentence on full-year demand direction for 2026>",
    "price_targets": "<any analyst price targets mentioned — e.g. £2,400 by Q3 2026>",
    "key_risks_upside": "<main upside risk to prices>",
    "key_risks_downside": "<main downside risk to prices>",
    "source_summary": "<which sources contributed to this forecast section>"
  },
  "key_articles": [
    {
      "title": "<headline>",
      "source": "<publisher>",
      "age": "<e.g. 2 days ago>",
      "relevance": "<one sentence on why this article matters>",
      "url": "<link>"
    }
  ],
  "watch_items": ["<specific thing to monitor 1>", "<specific thing to monitor 2>"],
  "data_quality": "<brief note on coverage gaps or low article count>"
}

key_articles: include the 5-8 most market-relevant articles only. Prioritise Reuters, Bloomberg, FT, ICCO, Barchart over general press.

Style rules:
- Be factual and precise — cite specific numbers from articles where available
- Do not invent or extrapolate beyond what the articles say
- If article coverage is thin, say so honestly in data_quality
- Directional signal should reflect the BALANCE of evidence, not just the most recent headline
- Output raw JSON only — no markdown, no preamble, no code fences
"""


def format_articles_for_prompt(articles: list, label: str, max_articles: int) -> str:
    if not articles:
        return f"## {label}\nNo articles found.\n"

    lines = [f"## {label} ({len(articles[:max_articles])} articles)\n"]
    for i, a in enumerate(articles[:max_articles], 1):
        lines.append(f"{i}. [{a.get('source','?')} | {a.get('age','?')}] {a['title']}")
        if a.get("summary"):
            lines.append(f"   {a['summary'][:300]}")
        if a.get("link"):
            lines.append(f"   URL: {a['link']}")
        lines.append("")
    return "\n".join(lines)


def _call_claude(client, system: str, user: str, max_tokens: int, label: str) -> dict:
    """
    Make a single Claude API call, parse JSON response, return dict.
    Raises on failure so callers can decide how to handle.
    """
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    raw = response.content[0].text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        log.error(f"  {label} returned invalid JSON: {e}")
        log.debug(f"  Raw ({len(raw)} chars): {raw[:300]}")
        raise


def run_summary_agent(article_buckets: dict) -> dict:
    """
    Two-call approach to avoid max_tokens truncation:
      Call 1 — news summary (recent, background, signal, articles, watch items)
      Call 2 — demand forecast (analyst pages only)
    Results are merged into one brief dict.
    """
    if not ANTHROPIC_API_KEY:
        log.warning("ANTHROPIC_API_KEY not set — skipping summarisation agent")
        return {"error": "ANTHROPIC_API_KEY not configured"}

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    today  = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    recent_text     = format_articles_for_prompt(
        article_buckets["recent"],     "RECENT NEWS (last 7 days)", max_articles=20
    )
    background_text = format_articles_for_prompt(
        article_buckets["background"], "BACKGROUND CONTEXT (7–90 days)", max_articles=12
    )

    # ── Call 1: news summary ─────────────────────────────────────────────────
    log.info("  News agent call 1/2: article summary...")

    call1_system = """You are a cocoa commodity market analyst. Analyse the provided articles
and return ONLY a valid JSON object with these exact fields:
{
  "recent_summary": "<3-5 sentences on last 7 days — price moves, key events, data releases>",
  "background_context": "<3-5 sentences on structural picture — supply/demand, grinding trends, seasonal factors>",
  "key_themes": ["<theme 1>", "<theme 2>", "<theme 3>"],
  "directional_signal": "Bullish" or "Bearish" or "Neutral" or "Mixed",
  "signal_confidence": "High" or "Medium" or "Low",
  "signal_rationale": "<1-2 sentences>",
  "key_articles": [
    {"title": "<headline>", "source": "<publisher>", "age": "<e.g. 2 days ago>", "relevance": "<one sentence>", "url": "<link>"}
  ],
  "watch_items": ["<item 1>", "<item 2>", "<item 3>"],
  "data_quality": "<one sentence on coverage quality>"
}
key_articles: 5-8 most relevant only. Prioritise Reuters, Bloomberg, FT, ICCO over general press.
Be factual. No markdown. No preamble. Raw JSON only."""

    call1_prompt = f"""{recent_text}

{background_text}

Today: {today}
Return only the JSON object."""

    try:
        brief = _call_claude(client, call1_system, call1_prompt,
                             max_tokens=3000, label="Call 1")
        log.info(f"  ✅ Call 1 done: signal={brief.get('directional_signal')}")
    except Exception as e:
        return {"error": f"News summary call failed: {e}"}

    # ── Call 2: demand forecast from analyst pages ────────────────────────────
    log.info("  News agent call 2/2: demand forecast...")

    forecast_text = ""
    for af in article_buckets.get("analyst_forecasts", []):
        if "error" not in af and af.get("text"):
            forecast_text += (
                f"SOURCE: {af['source']} ({af['url']})\n"
                f"{af['text'][:2000]}\n\n---\n"
            )

    call2_system = """You are a cocoa commodity analyst. Based on the analyst forecast content provided,
return ONLY a valid JSON object with this exact structure:
{
  "full_year_outlook": "<one sentence on full-year 2026 demand/price direction>",
  "price_targets": "<any specific price targets mentioned, or 'None cited'>",
  "key_risks_upside": "<main factor that could push prices higher>",
  "key_risks_downside": "<main factor that could push prices lower>",
  "source_summary": "<which sources contributed>"
}
If no analyst pages were provided, populate fields with null.
No markdown. No preamble. Raw JSON only."""

    call2_prompt = (
        forecast_text if forecast_text
        else "No analyst forecast pages were retrieved. Return all fields as null."
    )

    try:
        demand_forecast = _call_claude(client, call2_system, call2_prompt,
                                       max_tokens=800, label="Call 2")
        brief["demand_forecast"] = demand_forecast
        log.info("  ✅ Call 2 done: demand forecast extracted")
    except Exception as e:
        log.warning(f"  Call 2 failed (non-critical): {e}")
        brief["demand_forecast"] = {"error": str(e)}

    # ── Metadata ─────────────────────────────────────────────────────────────
    brief["generated_at"] = datetime.now(timezone.utc).isoformat()
    brief["article_counts"] = {
        "recent":     len(article_buckets["recent"]),
        "background": len(article_buckets["background"]),
        "total":      len(article_buckets["all"]),
    }

    log.info(f"  ✅ Intelligence brief complete: "
             f"signal={brief.get('directional_signal')} "
             f"({brief.get('signal_confidence')} confidence)")
    return brief


# ─────────────────────────────────────────────
#  5. MAIN ENTRY POINT
# ─────────────────────────────────────────────

def fetch_news_intelligence() -> dict:
    """
    Full pipeline: gather → summarise → return brief.
    Called by cocoa_data_gatherer.py or run standalone.
    """
    article_buckets = gather_all_articles()
    brief           = run_summary_agent(article_buckets)

    # Save standalone output
    with open(NEWS_OUTPUT_FILE, "w") as f:
        json.dump(brief, f, indent=2, default=str)
    log.info(f"  News intelligence saved to {NEWS_OUTPUT_FILE}")

    # Merge into snapshot if it exists
    if os.path.exists(SNAPSHOT_FILE):
        try:
            with open(SNAPSHOT_FILE) as f:
                snapshot = json.load(f)
            snapshot["news_intelligence"] = brief
            # Keep raw articles for the agent prompt too
            snapshot["news"] = {
                "recent_articles":     article_buckets["recent"][:20],
                "background_articles": article_buckets["background"][:10],
                "sentiment": {
                    "overall":    brief.get("directional_signal", "Unknown"),
                    "confidence": brief.get("signal_confidence",  "Unknown"),
                },
            }
            with open(SNAPSHOT_FILE, "w") as f:
                json.dump(snapshot, f, indent=2, default=str)
            log.info(f"  Merged into {SNAPSHOT_FILE}")
        except Exception as e:
            log.warning(f"  Could not merge into snapshot: {e}")

    return brief


if __name__ == "__main__":
    brief = fetch_news_intelligence()
    print("\n" + "=" * 55)
    print("  COCOA NEWS INTELLIGENCE BRIEF")
    print("=" * 55)
    print(f"  Signal:     {brief.get('directional_signal')} ({brief.get('signal_confidence')} confidence)")
    print(f"  Rationale:  {brief.get('signal_rationale','')}")
    print(f"\n  Recent Summary:")
    print(f"  {brief.get('recent_summary','')}")
    print(f"\n  Background:")
    print(f"  {brief.get('background_context','')}")
    print(f"\n  Key Themes: {', '.join(brief.get('key_themes', []))}")
    print(f"\n  Watch Items:")
    for w in brief.get("watch_items", []):
        print(f"    • {w}")
    print(f"\n  Articles: {brief.get('article_counts', {})}")
