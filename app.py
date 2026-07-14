# -*- coding: utf-8 -*-
"""
Query Opportunity Mapper — Google Suggest Miner + Query Fan-out Generator
==========================================================================
Two opportunity-mapping engines from a list of seed terms:

  1. GOOGLE SUGGEST MINER
     Large-scale expansion via Google's undocumented Suggest endpoint
     (a-z alphabet soup, question/commercial/preposition modifiers,
     and optional level-2 recursion).

  2. QUERY FAN-OUT GENERATOR (LLM)
     Simulates the query decomposition behavior of Google AI Mode and
     ChatGPT using the Anthropic API, classifying each fan-out by type
     and search intent.

Run locally:  streamlit run app.py
"""

import json
import random
import re
import time
from datetime import datetime
from io import BytesIO

import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Query Opportunity Mapper",
    page_icon="🔎",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; }
      div[data-testid="stMetricValue"] { font-size: 1.6rem; }
      .stTabs [data-baseweb="tab"] { font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Constants — expansion modifiers (pt-BR and en)
# ---------------------------------------------------------------------------
ALPHABET = list("abcdefghijklmnopqrstuvwxyz")

MODIFIERS = {
    "pt-BR": {
        "Questions": [
            "como", "o que", "qual", "quais", "quando", "onde",
            "por que", "para que", "quanto custa", "vale a pena",
        ],
        "Commercial": [
            "preço", "melhor", "barato", "promoção", "comprar",
            "vs", "ou", "custo benefício", "review", "é bom",
        ],
        "Prepositions": ["para", "com", "sem", "de", "em", "por"],
    },
    "en": {
        "Questions": [
            "how", "what", "which", "when", "where", "why",
            "how much", "is it worth",
        ],
        "Commercial": [
            "price", "best", "cheap", "buy", "vs", "or",
            "review", "alternatives", "deals",
        ],
        "Prepositions": ["for", "with", "without", "near", "to"],
    },
}

SUGGEST_URL = "https://suggestqueries.google.com/complete/search"

FANOUT_TYPES_DESC = """
- reformulation: rewrites of the original query with different syntax/vocabulary
- related: related queries that broaden the topic
- implicit: implicit sub-questions the user didn't type but wants answered
- comparative: comparisons with alternatives, competitors, or substitutes
- entity_expansion: expansion by entities (brands, models, categories, attributes)
- follow_up: follow-up questions typical of multi-turn conversation (ChatGPT style)
"""

# ---------------------------------------------------------------------------
# Google Suggest functions
# ---------------------------------------------------------------------------
def fetch_suggestions(query: str, hl: str, gl: str, session: requests.Session) -> list[str]:
    """Query the Google Suggest endpoint and return the list of suggestions."""
    params = {
        "client": "firefox",   # returns clean JSON
        "hl": hl,
        "gl": gl,
        "q": query,
    }
    try:
        r = session.get(SUGGEST_URL, params=params, timeout=6)
        if r.status_code == 200:
            data = json.loads(r.content.decode("utf-8", errors="ignore"))
            if isinstance(data, list) and len(data) >= 2:
                return [s for s in data[1] if isinstance(s, str)]
    except (requests.RequestException, json.JSONDecodeError, UnicodeDecodeError):
        pass
    return []


def build_probe_queries(seed: str, lang: str, use_alphabet: bool,
                        groups: list[str]) -> list[tuple[str, str]]:
    """Build probe query variations for a seed.
    Returns tuples of (probe, expansion_pattern)."""
    probes = [(seed, "base")]

    if use_alphabet:
        for letter in ALPHABET:
            probes.append((f"{seed} {letter}", f"alphabet ({letter})"))

    mods = MODIFIERS.get(lang, MODIFIERS["en"])
    for group in groups:
        for mod in mods.get(group, []):
            probes.append((f"{mod} {seed}", f"{group.lower()}: {mod} [prefix]"))
            probes.append((f"{seed} {mod}", f"{group.lower()}: {mod} [suffix]"))

    return probes


def mine_google_suggest(seeds: list[str], hl: str, gl: str, lang: str,
                        use_alphabet: bool, groups: list[str],
                        depth2: bool, depth2_limit: int,
                        delay_range: tuple[float, float],
                        progress_cb=None) -> pd.DataFrame:
    """Run the full Google Suggest mining for all seeds."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    })

    rows = []
    seen: set[str] = set()

    # build the full probe queue for the progress bar
    all_probes: list[tuple[str, str, str, int]] = []  # (seed, probe, pattern, level)
    for seed in seeds:
        for probe, pattern in build_probe_queries(seed, lang, use_alphabet, groups):
            all_probes.append((seed, probe, pattern, 1))

    level2_queue: list[tuple[str, str]] = []  # (original_seed, level-1 suggestion)
    total_est = len(all_probes)
    done = 0

    for seed, probe, pattern, level in all_probes:
        suggestions = fetch_suggestions(probe, hl, gl, session)
        for s in suggestions:
            key = s.lower().strip()
            if key and key not in seen and key != seed.lower():
                seen.add(key)
                rows.append({
                    "query": s,
                    "seed": seed,
                    "source": "google_suggest",
                    "expansion_pattern": pattern,
                    "level": level,
                })
                if depth2:
                    level2_queue.append((seed, s))
        done += 1
        if progress_cb:
            progress_cb(done / max(total_est, 1),
                        f"Level 1 — {done}/{total_est} requests · {len(seen)} unique queries")
        time.sleep(random.uniform(*delay_range))

    # ---- Level 2: suggestions become new seeds (capped) ----
    if depth2 and level2_queue:
        random.shuffle(level2_queue)
        queue = level2_queue[:depth2_limit]
        for i, (seed, sub_seed) in enumerate(queue, start=1):
            suggestions = fetch_suggestions(sub_seed, hl, gl, session)
            for s in suggestions:
                key = s.lower().strip()
                if key and key not in seen:
                    seen.add(key)
                    rows.append({
                        "query": s,
                        "seed": seed,
                        "source": "google_suggest",
                        "expansion_pattern": f"recursion_l2 (via: {sub_seed})",
                        "level": 2,
                    })
            if progress_cb:
                progress_cb(i / len(queue),
                            f"Level 2 — {i}/{len(queue)} requests · {len(seen)} unique queries")
            time.sleep(random.uniform(*delay_range))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Observed fan-outs — free capture from the real AI Mode interface
# (DevTools console script + saved HTML / HAR parsing; no external APIs)
# ---------------------------------------------------------------------------

# matches google search links embedded in HTML/HAR/JSON payloads
GOOGLE_SEARCH_LINK_RE = re.compile(
    r"(?:https?:)?(?:\\/\\/|//)(?:www\.)?google\.[a-z.]{2,6}(?:\\/|/)search\?[^\"'\s<>\\]+"
)


def build_ai_mode_url(seed: str, hl: str, gl: str) -> str:
    """Build the Google AI Mode URL (udm=50) for a seed."""
    from urllib.parse import quote_plus
    return (f"https://www.google.com/search?udm=50&q={quote_plus(seed)}"
            f"&hl={hl}&gl={gl}")


CONSOLE_SCRIPT = r"""(() => {
  const qs = new Set();
  const grab = (u) => {
    try {
      const p = new URL(u, location.origin);
      if (!p.pathname.includes('/search')) return;
      const q = p.searchParams.get('q');
      if (q) qs.add(q.trim());
    } catch (e) {}
  };
  // 1) anchors rendered inside the AI Mode answer
  document.querySelectorAll('a[href*="/search"]').forEach(a => grab(a.href));
  // 2) search links embedded in the raw page payload (streamed response data)
  const re = /(?:\\\/|\/)search\?[^"'\\\s<>]*?q=([^"'&\\\s<>]+)/g;
  const html = document.documentElement.innerHTML;
  let m;
  while ((m = re.exec(html))) {
    try { qs.add(decodeURIComponent(m[1].replace(/\+/g, ' ')).trim()); } catch (e) {}
  }
  const seed = (new URL(location.href)).searchParams.get('q') || '';
  const out = [...qs].filter(q =>
    q && q.length <= 120 &&
    q.toLowerCase() !== seed.toLowerCase() &&
    !q.startsWith('http') && !q.includes('site:'));
  copy(JSON.stringify({ seed: seed, queries: out }));
  console.log(out.length + ' queries copied to clipboard - paste them back into the app.');
})();"""


def parse_pasted_extraction(raw: str) -> list[dict]:
    """Parse JSON pasted back from the console script.
    Accepts one or more {seed, queries[]} objects, or plain one-per-line text."""
    rows = []
    raw = raw.strip()
    if not raw:
        return rows
    # try to find JSON objects first (user may paste several, one per seed)
    decoder = json.JSONDecoder()
    idx, parsed_any = 0, False
    while idx < len(raw):
        try:
            obj, end = decoder.raw_decode(raw, idx)
        except json.JSONDecodeError:
            idx += 1
            continue
        parsed_any = True
        idx = end
        if isinstance(obj, dict):
            seed = str(obj.get("seed", "")).strip() or "console_extraction"
            for q in obj.get("queries", []):
                if isinstance(q, str) and q.strip():
                    rows.append({"query": q.strip(), "seed": seed,
                                 "source": "ai_mode_observed"})
    if not parsed_any:  # fallback: plain lines
        for line in raw.splitlines():
            line = line.strip().strip('",')
            if line:
                rows.append({"query": line, "seed": "console_extraction",
                             "source": "ai_mode_observed"})
    return rows


def _clean_embedded_url(raw: str) -> str:
    """Normalize escaped URLs found inside HTML/JSON payloads."""
    return (raw.replace("\\/", "/")
               .replace("\\u0026", "&")
               .replace("&amp;", "&")
               .replace("\\u003d", "="))


def _q_params_from_url(url: str) -> list[str]:
    from urllib.parse import urlparse, parse_qs, unquote_plus
    try:
        qs = parse_qs(urlparse(url).query)
        return [unquote_plus(q).strip() for q in qs.get("q", []) if q.strip()]
    except ValueError:
        return []


def _walk_for_search_queries(obj, found: set):
    """Recursively scan a JSON structure for embedded google search links."""
    if isinstance(obj, dict):
        for v in obj.values():
            _walk_for_search_queries(v, found)
    elif isinstance(obj, list):
        for v in obj:
            _walk_for_search_queries(v, found)
    elif isinstance(obj, str) and "google." in obj and "/search" in obj:
        for m in GOOGLE_SEARCH_LINK_RE.finditer(obj):
            for q in _q_params_from_url(_clean_embedded_url(m.group(0))):
                found.add(q)


def parse_capture_file(raw_bytes: bytes, filename: str) -> set[str]:
    """Extract google search queries from a user-captured file
    (saved AI Mode/SERP HTML, DevTools HAR export, or raw text)."""
    text = raw_bytes.decode("utf-8", errors="ignore")
    found: set = set()
    for m in GOOGLE_SEARCH_LINK_RE.finditer(text):
        for q in _q_params_from_url(_clean_embedded_url(m.group(0))):
            found.add(q)
    # HAR files: also walk the JSON structure (URLs may be split across fields)
    if filename.lower().endswith(".har"):
        try:
            _walk_for_search_queries(json.loads(text), found)
        except json.JSONDecodeError:
            pass
    # drop obvious noise (navigation params, empty, very long payload strings)
    return {q for q in found if 0 < len(q) <= 120 and not q.startswith(("http", "site:"))}


# ---------------------------------------------------------------------------
# Query fan-out via LLM (Anthropic API or Google AI Studio / Gemini API)
# ---------------------------------------------------------------------------
PROVIDERS = {
    "Anthropic (Claude)": {
        "id": "anthropic",
        "fallback_models": ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
        "key_help": "Get your key at console.anthropic.com",
    },
    "Google AI Studio (Gemini)": {
        "id": "google",
        "fallback_models": ["gemini-flash-latest", "gemini-3.5-flash", "gemini-3.1-flash-lite"],
        "key_help": "Get your key at aistudio.google.com/apikey",
    },
}

# model name fragments that are not text-generation chat models
_NON_CHAT_FRAGMENTS = ("embedding", "tts", "image", "veo", "imagen",
                       "live", "audio", "robotics", "aqa")


@st.cache_data(ttl=3600, show_spinner=False)
def list_available_models(provider_id: str, api_key: str) -> list[str]:
    """Fetch the models actually available to this API key.
    Raises on failure — caller decides the fallback."""
    if provider_id == "google":
        r = requests.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            headers={"x-goog-api-key": api_key},
            params={"pageSize": 1000},
            timeout=15,
        )
        r.raise_for_status()
        models = []
        for m in r.json().get("models", []):
            if "generateContent" not in m.get("supportedGenerationMethods", []):
                continue
            name = m.get("name", "").split("/")[-1]
            if name and not any(frag in name for frag in _NON_CHAT_FRAGMENTS):
                models.append(name)
        # flash models first (cheaper / lower demand), then the rest
        models.sort(key=lambda n: ("flash" not in n, n))
        return models

    # anthropic
    r = requests.get(
        "https://api.anthropic.com/v1/models",
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
        timeout=15,
    )
    r.raise_for_status()
    return [m["id"] for m in r.json().get("data", [])]

FANOUT_SYSTEM = """You are a query fan-out engine that faithfully replicates the query
decomposition behavior of two systems:

1. GOOGLE AI MODE / AI OVERVIEWS — given a query, the system generates dozens of
   synthetic queries in parallel (reformulations, related, implicit, comparative,
   entity_expansion) to retrieve passages and compose the answer.

2. CHATGPT / CONVERSATIONAL ASSISTANTS — decomposition into sub-questions and
   follow-ups typical of multi-turn conversations (follow_up).

Respond ONLY with valid JSON — no markdown, no backticks, no extra text."""


def build_fanout_prompt(seed: str, lang: str, n_per_type: int,
                        business_context: str) -> str:
    lang_name = "Brazilian Portuguese" if lang == "pt-BR" else "English"
    ctx = f"\nBusiness context (use it to make fan-outs specific): {business_context}" if business_context else ""
    return f"""Original query: "{seed}"{ctx}

Generate fan-outs in {lang_name} for this query, covering the types below:
{FANOUT_TYPES_DESC}

Generate up to {n_per_type} queries per type. Each query must be realistic —
something the system would actually generate or a real user would type/ask.
For "entity_expansion", include plausible domain entities (brands, models,
attributes, adjacent categories).

Response format (pure JSON):
{{
  "fanouts": [
    {{"query": "...", "type": "reformulation", "engine": "google_ai_mode", "intent": "informational|commercial|transactional|navigational"}},
    ...
  ]
}}

Rules:
- "engine" = "google_ai_mode" for reformulation/related/implicit/comparative/entity_expansion
- "engine" = "chatgpt" for follow_up
- No duplicates, no overly generic queries."""


def call_anthropic(prompt: str, api_key: str, model: str) -> str:
    """Call the Anthropic Messages API and return the raw text response."""
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 4000,
        "system": FANOUT_SYSTEM,
        "messages": [{"role": "user", "content": prompt}],
    }
    r = requests.post("https://api.anthropic.com/v1/messages",
                      headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    return "".join(
        block.get("text", "")
        for block in r.json().get("content", [])
        if block.get("type") == "text"
    )


def call_gemini(prompt: str, api_key: str, model: str) -> str:
    """Call the Gemini API (Google AI Studio key) and return the raw text response."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {
        "x-goog-api-key": api_key,
        "content-type": "application/json",
    }
    payload = {
        "system_instruction": {"parts": [{"text": FANOUT_SYSTEM}]},
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": 8192,
            "responseMimeType": "application/json",
        },
    }
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    candidates = r.json().get("candidates", [])
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts if "text" in p)


RETRYABLE_STATUS = {429, 500, 502, 503, 529}  # rate limit / overloaded / transient
MAX_RETRIES = 4


def call_with_retry(call_fn, prompt: str, api_key: str, model: str,
                    status_cb=None) -> str:
    """Call an LLM provider with exponential backoff on transient errors."""
    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            return call_fn(prompt, api_key, model)
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status not in RETRYABLE_STATUS or attempt == MAX_RETRIES:
                raise
            last_err = e
            # honor Retry-After when present, otherwise exponential backoff + jitter
            retry_after = e.response.headers.get("retry-after") if e.response is not None else None
            try:
                wait = float(retry_after)
            except (TypeError, ValueError):
                wait = (2 ** attempt) * 2 + random.uniform(0, 1.5)  # 2s, 4s, 8s, 16s...
            if status_cb:
                status_cb(f"⏳ Provider returned {status} (high demand). "
                          f"Retry {attempt + 1}/{MAX_RETRIES} in {wait:.0f}s...")
            time.sleep(wait)
        except requests.ConnectionError as e:
            if attempt == MAX_RETRIES:
                raise
            last_err = e
            time.sleep((2 ** attempt) * 2)
    raise last_err  # pragma: no cover


def generate_fanouts(seeds: list[str], provider_id: str, api_key: str, lang: str,
                     n_per_type: int, business_context: str,
                     model: str, progress_cb=None) -> pd.DataFrame:
    """Generate fan-outs for each seed using the selected LLM provider."""
    rows = []
    call_fn = call_anthropic if provider_id == "anthropic" else call_gemini
    status_box = st.empty()

    for i, seed in enumerate(seeds, start=1):
        try:
            text = call_with_retry(
                call_fn,
                build_fanout_prompt(seed, lang, n_per_type, business_context),
                api_key, model,
                status_cb=lambda msg: status_box.caption(msg),
            )
            status_box.empty()
            # defensive cleanup of code fences
            text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
            data = json.loads(text)
            for item in data.get("fanouts", []):
                q = str(item.get("query", "")).strip()
                if q:
                    rows.append({
                        "query": q,
                        "seed": seed,
                        "source": item.get("engine", "google_ai_mode"),
                        "fanout_type": item.get("type", ""),
                        "intent": item.get("intent", ""),
                    })
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else "?"
            if status in RETRYABLE_STATUS:
                st.error(f"Seed '{seed}': provider still overloaded after {MAX_RETRIES} retries "
                         f"({status}). Try again in a few minutes or pick a different model "
                         f"in the sidebar (flash models usually have lower demand).")
            else:
                st.error(f"API error for seed '{seed}': {status} — {e.response.text[:300]}")
        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            st.warning(f"Failed to process seed '{seed}': {e}")

        if progress_cb:
            progress_cb(i / len(seeds), f"Fan-outs — {i}/{len(seeds)} seeds · {len(rows)} queries generated")

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["query"], keep="first")
    return df


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------
def to_xlsx(dfs: dict[str, pd.DataFrame]) -> bytes:
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for sheet, df in dfs.items():
            df.to_excel(writer, sheet_name=sheet[:31], index=False)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# UI — Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")

lang = st.sidebar.selectbox("Expansion language", ["pt-BR", "en"], index=0)
hl = st.sidebar.text_input("hl (Google interface language)", value="pt-BR" if lang == "pt-BR" else "en")
gl = st.sidebar.text_input("gl (country)", value="br" if lang == "pt-BR" else "us")

st.sidebar.divider()
st.sidebar.subheader("Google Suggest")
use_alphabet = st.sidebar.checkbox("Alphabet expansion (a–z)", value=True)
groups = st.sidebar.multiselect(
    "Modifier groups",
    ["Questions", "Commercial", "Prepositions"],
    default=["Questions", "Commercial"],
)
depth2 = st.sidebar.checkbox("Level-2 recursion (suggestions become seeds)", value=False)
depth2_limit = st.sidebar.slider("Level-2 request cap", 10, 300, 60, step=10,
                                 disabled=not depth2)
delay = st.sidebar.slider("Delay between requests (seconds)", 0.1, 2.0, (0.2, 0.6))

st.sidebar.divider()
st.sidebar.subheader("Query Fan-out (LLM)")
provider_name = st.sidebar.selectbox("LLM provider", list(PROVIDERS.keys()), index=0)
provider = PROVIDERS[provider_name]
api_key = st.sidebar.text_input(f"{provider_name} API Key", type="password",
                                help=f"Only required for the fan-out tab. {provider['key_help']}.")

model_options = provider["fallback_models"]
models_are_live = False
if api_key:
    try:
        live_models = list_available_models(provider["id"], api_key)
        if live_models:
            model_options = live_models
            models_are_live = True
    except requests.HTTPError as e:
        code = e.response.status_code if e.response is not None else "?"
        if code in (401, 403):
            st.sidebar.error("API key rejected by the provider. Check the key and try again.")
        else:
            st.sidebar.warning(f"Couldn't fetch the model list ({code}). Using defaults.")
    except requests.RequestException:
        st.sidebar.warning("Couldn't reach the provider to list models. Using defaults.")

model = st.sidebar.selectbox(
    "Model", model_options, index=0,
    help="Fetched live from your account — only models your key can use."
         if models_are_live else
         "Default list — enter a valid API key to load the models available to your account.",
)
n_per_type = st.sidebar.slider("Fan-outs per type", 3, 15, 8)
business_context = st.sidebar.text_area(
    "Business context (optional)",
    placeholder="E.g.: home appliances e-commerce in Brazil, focused on white goods...",
    height=80,
)

# ---------------------------------------------------------------------------
# UI — Main body
# ---------------------------------------------------------------------------
st.title("🔎 Query Opportunity Mapper")
st.caption("Google Suggest Miner + Query Fan-out Generator (Google AI Mode & ChatGPT) — SEO/GEO opportunity mapping from seed terms.")

seeds_raw = st.text_area(
    "Seeds (one term/topic per line)",
    placeholder="frost free refrigerator\nair fryer\nfront load washer",
    height=140,
)
seeds = [s.strip() for s in seeds_raw.splitlines() if s.strip()]
seeds = list(dict.fromkeys(seeds))  # dedup preserving order

if seeds:
    st.caption(f"**{len(seeds)}** seed(s) loaded.")

tab_suggest, tab_fanout, tab_observed, tab_results = st.tabs(
    ["🅶 Google Suggest Miner", "🤖 Query Fan-out Generator",
     "👁 Observed Fan-outs", "📊 Results & Export"]
)

# ---- Tab 1: Google Suggest ----
with tab_suggest:
    st.markdown(
        "Mines Google's autocomplete endpoint with alphabet expansion, "
        "modifiers, and optional recursion. Every suggestion is tracked by seed, "
        "expansion pattern, and level."
    )
    n_probes = len(build_probe_queries("x", lang, use_alphabet, groups)) * max(len(seeds), 1)
    st.caption(f"Estimate: ~{n_probes} endpoint requests at level 1.")

    if st.button("▶️ Mine Google Suggest", type="primary", disabled=not seeds):
        bar = st.progress(0.0, text="Starting...")
        cb = lambda p, msg: bar.progress(min(p, 1.0), text=msg)
        with st.spinner("Mining suggestions..."):
            df_sug = mine_google_suggest(
                seeds, hl, gl, lang, use_alphabet, groups,
                depth2, depth2_limit, delay, progress_cb=cb,
            )
        bar.empty()
        st.session_state["df_suggest"] = df_sug
        if df_sug.empty:
            st.warning("No suggestions returned. Check connectivity/rate limits and try increasing the delay.")
        else:
            st.success(f"{len(df_sug)} unique queries extracted from Google Suggest.")

    if "df_suggest" in st.session_state and not st.session_state["df_suggest"].empty:
        df = st.session_state["df_suggest"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Unique queries", len(df))
        c2.metric("Seeds covered", df["seed"].nunique())
        c3.metric("Expansion patterns", df["expansion_pattern"].nunique())
        st.dataframe(df, use_container_width=True, height=420)

# ---- Tab 2: Fan-outs ----
with tab_fanout:
    st.markdown(
        "Generates synthetic fan-outs replicating **Google AI Mode** decomposition "
        "(reformulations, related, implicit, comparative, entity expansion) and "
        "**ChatGPT conversational follow-ups**, classified by type and search intent."
    )
    if st.button("▶️ Generate Fan-outs", type="primary", disabled=not (seeds and api_key)):
        bar = st.progress(0.0, text="Starting...")
        cb = lambda p, msg: bar.progress(min(p, 1.0), text=msg)
        with st.spinner("Generating fan-outs via LLM..."):
            df_fan = generate_fanouts(seeds, provider["id"], api_key, lang,
                                      n_per_type, business_context, model,
                                      progress_cb=cb)
        bar.empty()
        st.session_state["df_fanout"] = df_fan
        if not df_fan.empty:
            st.success(f"{len(df_fan)} fan-outs generated.")

    if not api_key:
        st.info("Select your LLM provider (Anthropic or Google AI Studio) and enter the "
                "corresponding API key in the sidebar to enable this engine.")

    if "df_fanout" in st.session_state and not st.session_state["df_fanout"].empty:
        df = st.session_state["df_fanout"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Unique fan-outs", len(df))
        c2.metric("Google AI Mode", int((df["source"] == "google_ai_mode").sum()))
        c3.metric("ChatGPT (follow-ups)", int((df["source"] == "chatgpt").sum()))
        f1, f2 = st.columns(2)
        type_sel = f1.multiselect("Filter by type", sorted(df["fanout_type"].unique()))
        int_sel = f2.multiselect("Filter by intent", sorted(df["intent"].unique()))
        view = df
        if type_sel:
            view = view[view["fanout_type"].isin(type_sel)]
        if int_sel:
            view = view[view["intent"].isin(int_sel)]
        st.dataframe(view, use_container_width=True, height=420)

# ---- Tab 3: Observed fan-outs (free capture from the real interface) ----
with tab_observed:
    st.markdown(
        "Captures **real fan-out queries exposed by the Google AI Mode interface** — "
        "no LLM simulation, no external APIs, no paid tools. After the AI Mode answer "
        "finishes loading, the executed sub-queries are embedded in the page as search "
        "links and streamed payload data. Two free routes to extract them from your "
        "own browsing session:"
    )

    st.markdown("##### Route 1 — DevTools console script (recommended)")
    st.markdown(
        "**Step 1.** Open the AI Mode link for each seed below and wait for the "
        "answer to fully load (expand sections if shown):"
    )
    if seeds:
        for seed in seeds:
            st.markdown(f"- [{seed}]({build_ai_mode_url(seed, hl, gl)})")
    else:
        st.caption("Add seeds above to generate the AI Mode links.")

    st.markdown(
        "**Step 2.** Press `F12` → *Console* tab, paste the script below and hit "
        "Enter. It scans the rendered answer and the streamed payload for embedded "
        "search queries and copies them to your clipboard as JSON. "
        "(If the console blocks pasting, type `allow pasting` first — that's a "
        "Chrome safety prompt.)"
    )
    st.code(CONSOLE_SCRIPT, language="javascript")

    st.markdown("**Step 3.** Paste the JSON here (repeat per seed — paste one after the other):")
    pasted = st.text_area(
        "Extraction output",
        placeholder='{"seed": "air fryer", "queries": ["air fryer vs oven", ...]}',
        height=120,
        label_visibility="collapsed",
    )
    if st.button("➕ Parse & Add", type="primary", disabled=not pasted.strip()):
        new_rows = parse_pasted_extraction(pasted)
        if new_rows:
            df_new = pd.DataFrame(new_rows)
            prev = st.session_state.get("df_observed", pd.DataFrame())
            merged = (pd.concat([prev, df_new], ignore_index=True)
                        .drop_duplicates(subset=["query"], keep="first"))
            st.session_state["df_observed"] = merged
            st.success(f"{len(df_new)} queries parsed and added.")
        else:
            st.warning("Couldn't parse any queries from the pasted text.")

    st.markdown("##### Route 2 — Saved page / HAR upload")
    st.caption(
        "Alternative: on the loaded AI Mode page, save it (`Ctrl+S` → 'Webpage, "
        "HTML Only') or export a HAR (`F12` → *Network* → ⬇ export). Upload the "
        "files and the app extracts every embedded search sub-query — same data, "
        "no console needed."
    )
    uploads = st.file_uploader(
        "Upload captures (.html, .har, .txt)",
        type=["html", "har", "txt", "mhtml"],
        accept_multiple_files=True,
    )
    if uploads:
        manual_rows = []
        for up in uploads:
            queries = parse_capture_file(up.getvalue(), up.name)
            for q in queries:
                manual_rows.append({
                    "query": q,
                    "seed": f"capture: {up.name}",
                    "source": "ai_mode_observed",
                })
            st.caption(f"`{up.name}` → {len(queries)} queries extracted")
        if manual_rows:
            df_man = pd.DataFrame(manual_rows).drop_duplicates(subset=["query"])
            prev = st.session_state.get("df_observed", pd.DataFrame())
            merged = (pd.concat([prev, df_man], ignore_index=True)
                        .drop_duplicates(subset=["query"], keep="first"))
            st.session_state["df_observed"] = merged
            st.success(f"{len(df_man)} queries added from uploaded captures.")

    if "df_observed" in st.session_state and not st.session_state["df_observed"].empty:
        df = st.session_state["df_observed"]
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Observed queries", len(df))
        c2.metric("Seeds/captures", df["seed"].nunique())
        c3.metric("Source", "AI Mode (real)")
        st.dataframe(df, use_container_width=True, height=380)
        if st.button("🗑 Clear observed data"):
            st.session_state["df_observed"] = pd.DataFrame()
            st.rerun()

# ---- Tab 4: Consolidated results & export ----
with tab_results:
    df_s = st.session_state.get("df_suggest", pd.DataFrame())
    df_f = st.session_state.get("df_fanout", pd.DataFrame())
    df_o = st.session_state.get("df_observed", pd.DataFrame())

    if df_s.empty and df_f.empty and df_o.empty:
        st.info("Run at least one of the engines to consolidate results here.")
    else:
        frames = []
        if not df_s.empty:
            frames.append(df_s.assign(fanout_type="", intent=""))
        if not df_f.empty:
            frames.append(df_f.assign(expansion_pattern="", level=""))
        if not df_o.empty:
            frames.append(df_o.assign(expansion_pattern="", level="",
                                      fanout_type="", intent=""))
        df_all = pd.concat(frames, ignore_index=True)
        df_all = df_all.drop_duplicates(subset=["query"], keep="first")
        cols = ["query", "seed", "source", "expansion_pattern", "level", "fanout_type", "intent"]
        df_all = df_all[[c for c in cols if c in df_all.columns]]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Consolidated total", len(df_all))
        c2.metric("Google Suggest", int((df_all["source"] == "google_suggest").sum()))
        c3.metric("Fan-outs (LLM)", int(df_all["source"].isin(["google_ai_mode", "chatgpt"]).sum()))
        c4.metric("Observed (AI Mode)", int((df_all["source"] == "ai_mode_observed").sum()))

        search = st.text_input("🔍 Search the consolidated list")
        view = df_all[df_all["query"].str.contains(search, case=False, na=False)] if search else df_all
        st.dataframe(view, use_container_width=True, height=420)

        ts = datetime.now().strftime("%Y%m%d_%H%M")
        d1, d2 = st.columns(2)
        d1.download_button(
            "⬇️ Download CSV",
            data=df_all.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"query_opportunities_{ts}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        sheets = {"consolidated": df_all}
        if not df_s.empty:
            sheets["google_suggest"] = df_s
        if not df_f.empty:
            sheets["llm_fanouts"] = df_f
        if not df_o.empty:
            sheets["observed_ai_mode"] = df_o
        d2.download_button(
            "⬇️ Download XLSX (multi-sheet)",
            data=to_xlsx(sheets),
            file_name=f"query_opportunities_{ts}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
