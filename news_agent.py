import json
import os
import textwrap
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from html import escape
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Dict, List
from urllib.parse import parse_qs, urlencode, urlparse

import requests
from dotenv import load_dotenv
from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parent
STYLES_PATH = BASE_DIR / "static" / "styles.css"

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "").strip()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192").strip()
HOST = os.getenv("HOST", "0.0.0.0").strip()
PORT = int(os.getenv("PORT", "8000"))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "900"))

NEWS_ENDPOINT = "https://newsapi.org/v2/everything"

CATEGORY_QUERIES = OrderedDict(
    [
        (
            "Geopolitics",
            {
                "query": '("geopolitics" OR diplomacy OR sanctions OR "border dispute" OR conflict OR "foreign policy")',
                "audience": "Foreign policy shifts, conflict zones, diplomacy, and strategic competition.",
            },
        ),
        (
            "New Tech",
            {
                "query": '("artificial intelligence" OR semiconductor OR robotics OR cybersecurity OR startup OR "digital policy")',
                "audience": "AI, chips, cyber, platforms, startups, and strategic technology.",
            },
        ),
        (
            "Education",
            {
                "query": '("education policy" OR university OR schools OR edtech OR skilling OR curriculum)',
                "audience": "Schooling, higher education, skills, edtech, and learning policy.",
            },
        ),
        (
            "Economics",
            {
                "query": '(inflation OR trade OR "central bank" OR manufacturing OR GDP OR tariffs OR jobs)',
                "audience": "Growth, trade, prices, jobs, investment, and monetary policy.",
            },
        ),
    ]
)

FALLBACK_STORIES = [
    {
        "category": "Geopolitics",
        "title": "Strategic tensions reshape trade and security ties",
        "source": "Fallback Briefing",
        "published_at": "Live feed unavailable",
        "description": "Governments are recalibrating alliances, supply chains, and defense priorities amid persistent regional tensions.",
        "url": "#",
        "summary": "Countries are balancing security concerns with trade needs as rival blocs harden.",
        "cause": "Long-running competition over territory, technology access, and supply-chain control is pushing governments into new alignments.",
        "impact_india": "India gains room to deepen ties with multiple partners, but it also faces pressure to secure energy, defence readiness, and resilient imports.",
        "impact_world": "The US, China, the EU, Russia, and Indo-Pacific powers all face harder choices on sanctions, shipping routes, and strategic partnerships.",
        "impact_citizens": "Citizens can feel the effects through higher prices, stricter visa rules, slower shipping, and a stronger policy focus on national security.",
        "watchlist": "Watch for changes in sanctions, shipping disruptions, and defense agreements.",
    },
    {
        "category": "New Tech",
        "title": "AI adoption accelerates while regulation catches up",
        "source": "Fallback Briefing",
        "published_at": "Live feed unavailable",
        "description": "Companies and governments are pushing AI into work, public services, and industry while debating rules and safety.",
        "url": "#",
        "summary": "AI is moving from experimentation to deployment, making policy and workforce adaptation more urgent.",
        "cause": "Cheaper models, stronger infrastructure, and competitive pressure are driving rapid deployment across sectors.",
        "impact_india": "India can benefit through software exports, public digital infrastructure, and skilling, but it must invest in compute, data governance, and talent pipelines.",
        "impact_world": "Major economies are racing to capture productivity gains while managing cyber risk, labor shifts, and platform concentration.",
        "impact_citizens": "Citizens may see better services and new jobs, but also misinformation, automation anxiety, and privacy concerns.",
        "watchlist": "Watch for chip supply constraints, AI rules, and large public-private investments.",
    },
    {
        "category": "Education",
        "title": "Skills and employability dominate education reform",
        "source": "Fallback Briefing",
        "published_at": "Live feed unavailable",
        "description": "Education systems are under pressure to connect learning outcomes with employability and digital literacy.",
        "url": "#",
        "summary": "Policy focus is shifting toward practical skills, digital access, and better transitions from classrooms to careers.",
        "cause": "Technology shifts and labor-market disruption are exposing gaps between degrees, skills, and employer demand.",
        "impact_india": "India stands to gain from aligning higher education and vocational systems with growth sectors, especially AI, manufacturing, and services.",
        "impact_world": "Large economies are rethinking student debt, teacher shortages, and how quickly schools can adapt to digital learning needs.",
        "impact_citizens": "Families are affected by costs, quality gaps, and the pressure to keep pace with changing skill requirements.",
        "watchlist": "Watch for public spending changes, edtech adoption, and curriculum reforms.",
    },
    {
        "category": "Economics",
        "title": "High interest rates and trade frictions test global growth",
        "source": "Fallback Briefing",
        "published_at": "Live feed unavailable",
        "description": "Economies are balancing inflation control with growth, jobs, industrial policy, and cross-border trade competition.",
        "url": "#",
        "summary": "Governments are trying to protect jobs and lower inflation without stalling growth.",
        "cause": "Persistent price pressures, geopolitical risk, and industrial subsidies are distorting the normal growth cycle.",
        "impact_india": "India could benefit from supply-chain diversification and manufacturing investment, but imported inflation and export softness remain risks.",
        "impact_world": "The US, China, Europe, Japan, and emerging markets face different mixes of debt, demand, and trade exposure.",
        "impact_citizens": "Citizens experience the economy through borrowing costs, fuel and food prices, wage growth, and job security.",
        "watchlist": "Watch for central-bank signals, oil prices, and trade-policy moves.",
    },
]

CACHE = {"expires_at": None, "payload": None}


def groq_client() -> OpenAI | None:
    if not GROQ_API_KEY:
        return None
    return OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")


def iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC")


def clean_text(value: str, fallback: str = "No summary available.") -> str:
    value = (value or "").strip()
    if not value:
        return fallback
    return value


def fetch_news_for_category(category: str, query: str, page_size: int = 6) -> List[Dict]:
    if not NEWS_API_KEY:
        return []

    from_date = (datetime.now(timezone.utc) - timedelta(days=4)).date().isoformat()
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "from": from_date,
        "apiKey": NEWS_API_KEY,
    }
    response = requests.get(NEWS_ENDPOINT, params=params, timeout=20)
    response.raise_for_status()
    data = response.json()

    articles = []
    for article in data.get("articles", []):
        title = clean_text(article.get("title"), "")
        description = clean_text(article.get("description"), "")
        url = clean_text(article.get("url"), "#")
        if not title or url == "#":
            continue
        articles.append(
            {
                "category": category,
                "title": title,
                "description": description,
                "url": url,
                "source": clean_text(article.get("source", {}).get("name"), "Unknown source"),
                "published_at": clean_text(article.get("publishedAt"), "Unknown time"),
            }
        )
    return articles


def fetch_all_news() -> List[Dict]:
    stories: List[Dict] = []
    seen_urls = set()

    for category, meta in CATEGORY_QUERIES.items():
        try:
            category_articles = fetch_news_for_category(category, meta["query"])
        except Exception:
            category_articles = []

        for article in category_articles:
            if article["url"] in seen_urls:
                continue
            seen_urls.add(article["url"])
            stories.append(article)

    return stories


def strip_fences(payload: str) -> str:
    payload = payload.strip()
    if payload.startswith("```"):
        lines = payload.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        payload = "\n".join(lines).strip()
    return payload


def analyze_news(stories: List[Dict]) -> List[Dict]:
    client = groq_client()
    if not client or not stories:
        return []

    compact_stories = []
    for index, story in enumerate(stories, start=1):
        compact_stories.append(
            {
                "id": index,
                "category": story["category"],
                "title": story["title"],
                "description": story["description"],
                "source": story["source"],
                "published_at": story["published_at"],
            }
        )

    prompt = textwrap.dedent(
        f"""
        You are a current-affairs editor building a world news briefing website for Indian readers.
        Analyze the events below and return valid JSON only.

        JSON schema:
        {{
          "stories": [
            {{
              "id": 1,
              "summary": "1-2 sentences",
              "cause": "What likely caused or triggered this event",
              "impact_india": "Impact on India",
              "impact_world": "Impact on other major countries like the US, China, EU, Russia, Japan, Gulf states, or others when relevant",
              "impact_citizens": "Impact on ordinary citizens across affected regions",
              "watchlist": "What to watch next"
            }}
          ]
        }}

        Rules:
        - Stay grounded in the article title and description.
        - If the cause is uncertain, say "likely driven by" and infer cautiously.
        - Keep each field concise, specific, and readable on a news card.
        - Do not add markdown fences or commentary.

        Stories:
        {json.dumps(compact_stories, ensure_ascii=True)}
        """
    ).strip()

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.3,
        messages=[
            {
                "role": "system",
                "content": "Return strict JSON. No markdown. No prose outside the JSON object.",
            },
            {"role": "user", "content": prompt},
        ],
    )

    content = response.choices[0].message.content or ""
    parsed = json.loads(strip_fences(content))
    return parsed.get("stories", [])


def combine_story_data(raw_stories: List[Dict], analyses: List[Dict]) -> List[Dict]:
    analysis_map = {item.get("id"): item for item in analyses if isinstance(item, dict)}
    enriched = []

    for index, story in enumerate(raw_stories, start=1):
        analysis = analysis_map.get(index, {})
        enriched.append(
            {
                **story,
                "summary": clean_text(
                    analysis.get("summary"),
                    story["description"] or "A fast-moving event is reshaping this sector.",
                ),
                "cause": clean_text(
                    analysis.get("cause"),
                    "This event appears to be driven by policy shifts, market pressure, or strategic competition.",
                ),
                "impact_india": clean_text(
                    analysis.get("impact_india"),
                    "India may feel the effects through trade, policy, energy costs, talent demand, or strategic positioning.",
                ),
                "impact_world": clean_text(
                    analysis.get("impact_world"),
                    "Major economies could see knock-on effects in trade, security, regulation, investment, or supply chains.",
                ),
                "impact_citizens": clean_text(
                    analysis.get("impact_citizens"),
                    "Citizens may notice the impact through prices, jobs, public services, mobility, or access to technology.",
                ),
                "watchlist": clean_text(
                    analysis.get("watchlist"),
                    "Watch for policy updates, market reactions, and official statements.",
                ),
            }
        )

    return enriched


def build_highlights(groups: Dict[str, List[Dict]]) -> List[Dict]:
    highlights = []
    for category, items in groups.items():
        if not items:
            continue
        lead = items[0]
        highlights.append(
            {
                "category": category,
                "title": lead["title"],
                "summary": lead["summary"],
                "anchor": f"#{category.lower().replace(' ', '-')}",
            }
        )
    return highlights[:4]


def build_global_snapshot(groups: Dict[str, List[Dict]]) -> List[Dict]:
    snapshot = []
    for category, items in groups.items():
        snapshot.append(
            {
                "category": category,
                "count": len(items),
                "audience": CATEGORY_QUERIES.get(category, {}).get("audience", ""),
            }
        )
    return snapshot


def build_payload(force_refresh: bool = False) -> Dict:
    now = datetime.now(timezone.utc)
    expires_at = CACHE["expires_at"]
    if not force_refresh and CACHE["payload"] and expires_at and now < expires_at:
        return CACHE["payload"]

    raw_stories = fetch_all_news()
    source_mode = "live"
    error_message = ""

    if not raw_stories:
        raw_stories = FALLBACK_STORIES
        analyses = []
        source_mode = "fallback"
        error_message = "Live news could not be loaded, so the page is showing a smart fallback briefing layout."
    else:
        try:
            analyses = analyze_news(raw_stories)
        except Exception:
            analyses = []
            error_message = "Headlines loaded, but AI analysis is using simplified fallback text right now."

    stories = combine_story_data(raw_stories, analyses)

    grouped = OrderedDict((category, []) for category in CATEGORY_QUERIES.keys())
    for story in stories:
        grouped.setdefault(story["category"], []).append(story)

    payload = {
        "updated_at": iso_now(),
        "source_mode": source_mode,
        "error_message": error_message,
        "groups": grouped,
        "highlights": build_highlights(grouped),
        "snapshot": build_global_snapshot(grouped),
        "story_count": len(stories),
    }

    CACHE["payload"] = payload
    CACHE["expires_at"] = now + timedelta(seconds=CACHE_TTL_SECONDS)
    return payload


def render_story_card(story: Dict) -> str:
    url = escape(story["url"], quote=True)
    is_external = url != "#"
    cta = (
        f'<a class="story-link" href="{url}" target="_blank" rel="noreferrer">Open source</a>'
        if is_external
        else '<span class="story-link disabled">Source unavailable</span>'
    )

    return f"""
    <article class="story-card">
      <div class="story-meta">
        <span>{escape(story["source"])}</span>
        <span>{escape(story["published_at"])}</span>
      </div>
      <h3>{escape(story["title"])}</h3>
      <p class="story-summary">{escape(story["summary"])}</p>
      <div class="insight-grid">
        <section>
          <h4>Cause Behind Event</h4>
          <p>{escape(story["cause"])}</p>
        </section>
        <section>
          <h4>Impact on India</h4>
          <p>{escape(story["impact_india"])}</p>
        </section>
        <section>
          <h4>Impact on Major Countries</h4>
          <p>{escape(story["impact_world"])}</p>
        </section>
        <section>
          <h4>Impact on Citizens</h4>
          <p>{escape(story["impact_citizens"])}</p>
        </section>
      </div>
      <div class="watch-row">
        <strong>What to watch:</strong>
        <span>{escape(story["watchlist"])}</span>
      </div>
      <div class="story-footer">{cta}</div>
    </article>
    """


def render_feature_panel(payload: Dict) -> str:
    highlights = []
    for item in payload["highlights"]:
        highlights.append(
            f"""
            <a class="highlight-card" href="{escape(item['anchor'], quote=True)}">
              <span class="highlight-tag">{escape(item['category'])}</span>
              <h3>{escape(item['title'])}</h3>
              <p>{escape(item['summary'])}</p>
            </a>
            """
        )

    snapshot = []
    for item in payload["snapshot"]:
        snapshot.append(
            f"""
            <div class="snapshot-card">
              <span class="snapshot-count">{item['count']}</span>
              <div>
                <strong>{escape(item['category'])}</strong>
                <p>{escape(item['audience'])}</p>
              </div>
            </div>
            """
        )

    return f"""
    <section class="feature-layout">
      <div class="feature-panel">
        <div class="feature-heading">
          <p class="section-kicker">Top radar</p>
          <h2>What looks most important right now</h2>
          <p>Jump straight into the strongest live themes across diplomacy, markets, technology, and social change.</p>
        </div>
        <div class="highlight-grid">
          {''.join(highlights) or '<div class="empty-state">Highlights will appear here once stories load.</div>'}
        </div>
      </div>
      <aside class="snapshot-panel">
        <div class="feature-heading">
          <p class="section-kicker">Coverage map</p>
          <h2>Today’s editorial spread</h2>
        </div>
        <div class="snapshot-list">
          {''.join(snapshot)}
        </div>
      </aside>
    </section>
    """


def render_explainer_panel() -> str:
    return """
    <section class="explainer-strip">
      <div class="explainer-card">
        <p class="section-kicker">How to read this site</p>
        <h2>From headline to consequence in one scan</h2>
        <div class="explainer-grid">
          <div>
            <strong>Cause</strong>
            <p>Why the event happened or what likely triggered it.</p>
          </div>
          <div>
            <strong>India impact</strong>
            <p>Trade, policy, security, jobs, prices, and strategic consequences for India.</p>
          </div>
          <div>
            <strong>World impact</strong>
            <p>How major countries and power centers may respond or be affected.</p>
          </div>
          <div>
            <strong>Citizen impact</strong>
            <p>What families, workers, students, consumers, and travelers may feel directly.</p>
          </div>
        </div>
      </div>
    </section>
    """


def render_home(payload: Dict) -> str:
    sections = []
    for category, items in payload["groups"].items():
        cards = "".join(render_story_card(item) for item in items)
        section_body = cards or '<div class="empty-state">No fresh stories were available for this section.</div>'
        audience = CATEGORY_QUERIES.get(category, {}).get("audience", "")
        sections.append(
            f"""
            <section class="category-section" id="{escape(category.lower().replace(' ', '-'))}">
              <div class="section-heading">
                <div>
                  <p class="eyebrow">{escape(category)}</p>
                  <h2>{escape(category)} Briefing</h2>
                </div>
                <p>{escape(audience)}</p>
              </div>
              <div class="card-grid">
                {section_body}
              </div>
            </section>
            """
        )

    status_badge = "Live analysis" if payload["source_mode"] == "live" else "Fallback mode"
    notice = (
        f'<div class="notice">{escape(payload["error_message"])}</div>'
        if payload["error_message"]
        else ""
    )

    refresh_query = urlencode({"refresh": "1"})
    category_links = "".join(
        f'<a href="#{escape(category.lower().replace(" ", "-"), quote=True)}">{escape(category)}</a>'
        for category in payload["groups"].keys()
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>World Current Affairs Pulse</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet" />
  <link rel="stylesheet" href="/static/styles.css" />
</head>
<body>
  <div class="page-shell">
    <header class="hero">
      <nav class="topbar">
        <span class="brand">World Current Affairs Pulse</span>
        <div class="topbar-actions">
          <a class="mini-link" href="#briefing">Briefing</a>
          <a class="mini-link" href="#method">Method</a>
          <a class="refresh-link" href="/?{refresh_query}">Refresh briefing</a>
        </div>
      </nav>
      <div class="hero-copy">
        <p class="eyebrow">Global affairs, explained for impact</p>
        <h1>Track geopolitics, technology, education, and economics with a clear India lens.</h1>
        <p class="hero-text">Each story highlights what happened, what likely caused it, how India is affected, what it means for other major countries, and how everyday citizens may feel the impact.</p>
      </div>
      <div class="quick-nav">
        {category_links}
      </div>
      <div class="hero-stats">
        <div>
          <span class="stat-label">Coverage</span>
          <strong>{payload["story_count"]} current stories</strong>
        </div>
        <div>
          <span class="stat-label">Updated</span>
          <strong>{escape(payload["updated_at"])}</strong>
        </div>
        <div>
          <span class="stat-label">Mode</span>
          <strong>{status_badge}</strong>
        </div>
      </div>
    </header>

    {notice}

    <main class="content-stack" id="briefing">
      {render_feature_panel(payload)}
      {render_explainer_panel()}
      {''.join(sections)}
    </main>

    <footer class="site-footer" id="method">
      <div>
        <p class="section-kicker">Editorial method</p>
        <h2>Built for fast, high-context reading</h2>
      </div>
      <p>The site fetches recent headlines by category, then generates structured analysis focused on causes and consequences. If live feeds are unavailable, it falls back to a built-in briefing so the experience stays usable.</p>
    </footer>
  </div>
</body>
</html>
"""


class NewsRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/static/styles.css":
            self.serve_styles()
            return

        if parsed.path != "/":
            self.send_error(404, "Page not found")
            return

        query = parse_qs(parsed.query)
        force_refresh = query.get("refresh") == ["1"]
        payload = build_payload(force_refresh=force_refresh)
        body = render_home(payload).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_styles(self) -> None:
        if not STYLES_PATH.exists():
            self.send_error(404, "Stylesheet missing")
            return

        body = STYLES_PATH.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/css; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        return


def run_server() -> None:
    server = HTTPServer((HOST, PORT), NewsRequestHandler)
    print(f"Serving World Current Affairs Pulse at http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
