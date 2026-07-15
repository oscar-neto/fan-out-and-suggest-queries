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
    "es": {
        "Questions": [
            "cómo", "qué", "cuál", "cuáles", "cuándo", "dónde",
            "por qué", "para qué", "cuánto cuesta", "vale la pena",
        ],
        "Commercial": [
            "precio", "mejor", "barato", "oferta", "comprar",
            "vs", "o", "calidad precio", "opiniones", "es bueno",
        ],
        "Prepositions": ["para", "con", "sin", "de", "en", "por"],
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


def _is_excluded(query_lower: str, exclude_terms: list[str]) -> bool:
    """True when the query contains any excluded term (case-insensitive)."""
    return any(term in query_lower for term in exclude_terms)


def mine_google_suggest(seeds: list[str], hl: str, gl: str, lang: str,
                        use_alphabet: bool, groups: list[str],
                        depth2: bool, depth2_limit: int,
                        delay_range: tuple[float, float],
                        exclude_terms: list[str] | None = None,
                        progress_cb=None) -> pd.DataFrame:
    """Run the full Google Suggest mining for all seeds."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    })

    exclude_terms = [t.lower() for t in (exclude_terms or [])]
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
            if key and key not in seen and key != seed.lower() \
                    and not _is_excluded(key, exclude_terms):
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
                if key and key not in seen \
                        and not _is_excluded(key, exclude_terms):
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


def fetch_openai_fanouts(seed: str, api_key: str, model: str,
                         lang: str = "pt-BR", country: str = "br") -> list[str]:
    """Fully automated: run the seed through the OpenAI Responses API with the
    web_search tool and return the actual search queries the model executed
    (web_search_call -> action.query / action.queries). Query language is
    forced via prompt; search localization via the tool's user_location."""
    lang_name = _lang_name(lang)
    web_tool = {"type": "web_search"}
    if country and len(country.strip()) == 2:
        web_tool["user_location"] = {"type": "approximate",
                                     "country": country.strip().upper()}
    r = requests.post(
        "https://api.openai.com/v1/responses",
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        json={
            "model": model,
            "input": (
                "You are researching to build a comprehensive answer about this "
                f"topic: {seed}\n\n"
                "Use the web search tool to run AT LEAST 6 DISTINCT searches, each "
                "with a DIFFERENT query: cover subtopics, comparisons vs alternatives, "
                "buying criteria, prices/reviews when relevant, and related questions "
                "people ask. Never repeat the topic verbatim as your only search. "
                f"Write ALL search queries in {lang_name}, regardless of the "
                "language of this instruction. After searching, reply with a single "
                "short sentence."
            ),
            "tools": [web_tool],
            "tool_choice": "auto",
        },
        timeout=180,
    )
    r.raise_for_status()
    queries = []
    for item in r.json().get("output", []):
        if item.get("type") == "web_search_call":
            action = item.get("action") or {}
            q = action.get("query")
            if isinstance(q, str) and q.strip():
                queries.append(q.strip())
            for q in action.get("queries") or []:
                if isinstance(q, str) and q.strip():
                    queries.append(q.strip())
    return list(dict.fromkeys(queries))  # dedupe preserving order


def fetch_gemini_grounding_fanouts(seed: str, api_key: str, model: str,
                                   lang: str = "pt-BR") -> list[str]:
    """Fully automated: run the seed through the Gemini API with Google Search
    grounding and return the actual queries Google executed
    (groundingMetadata.webSearchQueries). Query language forced via prompt."""
    lang_name = _lang_name(lang)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    r = requests.post(
        url,
        headers={"x-goog-api-key": api_key, "content-type": "application/json"},
        json={
            "contents": [{"parts": [{"text": (
                "You are researching to build a comprehensive answer about this "
                f"topic: {seed}\n\n"
                "Use Google Search to run MULTIPLE DISTINCT searches (at least 6), "
                "each with a DIFFERENT query: cover subtopics, comparisons vs "
                "alternatives, buying criteria, prices/reviews when relevant, and "
                f"related questions people ask. Write ALL search queries in "
                f"{lang_name}, regardless of the language of this instruction. "
                "Then reply with a single short sentence."
            )}]}],
            "tools": [{"google_search": {}}],
        },
        timeout=180,
    )
    r.raise_for_status()
    queries = []
    for cand in r.json().get("candidates", []):
        gm = cand.get("groundingMetadata") or {}
        for q in gm.get("webSearchQueries") or []:
            if isinstance(q, str) and q.strip():
                queries.append(q.strip())
    return list(dict.fromkeys(queries))  # dedupe preserving order


CHATGPT_CONSOLE_SCRIPT = r"""(async () => {
  const id = (location.pathname.split('/c/')[1] || '').split('/')[0];
  if (!id) { console.log('Open a ChatGPT conversation first (URL must contain /c/...).'); return; }
  const s = await fetch('/api/auth/session').then(r => r.json());
  if (!s || !s.accessToken) { console.log('Could not read session token - are you logged in?'); return; }
  const conv = await fetch('/backend-api/conversation/' + id, {
    headers: { authorization: 'Bearer ' + s.accessToken }
  }).then(r => r.json());

  const qs = new Set();
  const addQ = (q) => { q = String(q || '').trim(); if (q && q.length <= 120 && !q.startsWith('http')) qs.add(q); };

  // 1) structured fields anywhere in the conversation tree
  const walk = (o) => {
    if (!o || typeof o !== 'object') return;
    if (Array.isArray(o)) { o.forEach(walk); return; }
    for (const key of ['search_query', 'search_queries']) {
      const v = o[key];
      if (Array.isArray(v)) v.forEach(e => { if (typeof e === 'string') addQ(e); else if (e && e.q) addQ(e.q); });
      else if (typeof v === 'string') addQ(v);
    }
    Object.values(o).forEach(walk);
  };
  walk(conv);

  // 2) the model's web.run tool calls are serialized as JSON-strings inside
  //    message parts: {"search_query":[{"q":"..."}]} - plus legacy search("...").
  //    Scan the raw serialization (escaped quotes) AND an unescaped variant.
  const raw = JSON.stringify(conv);
  const unesc = (t) => t.split(String.fromCharCode(92) + String.fromCharCode(34)).join(String.fromCharCode(34));
  for (const text of [raw, unesc(raw)]) {
    let m;
    const blockRe = new RegExp('"search_quer(?:y|ies)"' + String.raw`\s*:\s*\[([\s\S]{0,4000}?)\]`, 'g');
    while ((m = blockRe.exec(text))) {
      const inner = m[1];
      if (inner.indexOf('"q"') !== -1) {
        let qm; const qRe = new RegExp(String.raw`"q"\s*:\s*"((?:[^"\\]|\\.)*?)"`, 'g');
        while ((qm = qRe.exec(inner))) addQ(unesc(qm[1]));
      } else {
        let sm; const sRe = new RegExp(String.raw`"((?:[^"\\]|\\.)+?)"`, 'g');
        while ((sm = sRe.exec(inner))) addQ(unesc(sm[1]));
      }
    }
    const callRe = new RegExp(String.raw`search\("((?:[^"\\]|\\.)+?)"\)`, 'g');
    while ((m = callRe.exec(text))) addQ(unesc(m[1]));
  }

  // exclude the user's own typed prompts - we only want the model's fan-outs
  const userTexts = new Set();
  const users = Object.values(conv.mapping || {})
    .map(n => n && n.message)
    .filter(msg => msg && msg.author && msg.author.role === 'user')
    .sort((a, b) => (a.create_time || 0) - (b.create_time || 0));
  users.forEach(msg => {
    const p = msg.content && msg.content.parts && msg.content.parts[0];
    if (typeof p === 'string') userTexts.add(p.trim().toLowerCase());
  });
  let seed = 'chatgpt';
  if (users.length) {
    const p = users[0].content && users[0].content.parts && users[0].content.parts[0];
    if (typeof p === 'string' && p.trim()) seed = p.trim().slice(0, 80);
  }
  const out = [...qs].filter(q => !userTexts.has(q.toLowerCase()));

  const payload = JSON.stringify({ seed: seed, platform: 'chatgpt', queries: out });
  window.__fanout = payload;  // kept for manual copy fallback
  try {
    await navigator.clipboard.writeText(payload);
    console.log(out.length + ' fan-out queries copied to clipboard - paste them back into the app.');
  } catch (e) {
    console.log(payload);
    console.log(out.length + ' fan-out queries extracted. Clipboard was blocked - copy the JSON above, or run:  copy(window.__fanout)');
  }
  if (!out.length) console.log('No fan-out queries found. Make sure the answer actually used web search (a sources/citations panel should be visible) - simple prompts may be answered from model knowledge without searching.');
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
            platform = str(obj.get("platform", "")).strip().lower()
            source = "chatgpt_observed" if platform == "chatgpt" else "ai_mode_observed"
            for q in obj.get("queries", []):
                if isinstance(q, str) and q.strip():
                    rows.append({"query": q.strip(), "seed": seed,
                                 "source": source})
    if not parsed_any:  # fallback: plain lines
        for line in raw.splitlines():
            line = line.strip().strip('",')
            if line:
                rows.append({"query": line, "seed": "console_extraction",
                             "source": "ai_mode_observed"})
    return rows


# ---------------------------------------------------------------------------
# Query fan-out via LLM (Anthropic API or Google AI Studio / Gemini API)
# ---------------------------------------------------------------------------
# languages available for fan-out generation/extraction (independent from the
# Google Suggest expansion language, which needs dedicated modifier sets)
FANOUT_LANGUAGES = {
    "pt-BR": "Brazilian Portuguese",
    "en": "English",
    "es": "Spanish",
    "es-MX": "Mexican Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
}


def _lang_name(lang: str) -> str:
    return FANOUT_LANGUAGES.get(lang, "English")


PROVIDERS = {
    "Anthropic Claude — simulated fan-outs": {
        "id": "anthropic", "mode": "simulated",
        "fallback_models": ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
        "key_help": "Get your key at console.anthropic.com",
    },
    "Google AI Studio (Gemini) — simulated fan-outs": {
        "id": "google", "mode": "simulated",
        "fallback_models": ["gemini-flash-latest", "gemini-3.5-flash", "gemini-3.1-flash-lite"],
        "key_help": "Get your key at aistudio.google.com/apikey",
    },
    "OpenAI (ChatGPT) — real extracted fan-outs": {
        "id": "openai", "mode": "observed",
        "fallback_models": ["gpt-5-mini", "gpt-5", "gpt-4.1-mini"],
        "key_help": "Get your key at platform.openai.com/api-keys — the web_search "
                    "tool returns the searches the model actually executed",
    },
    "Google Search grounding (Gemini) — real extracted fan-outs": {
        "id": "google_grounding", "mode": "observed",
        "fallback_models": ["gemini-flash-latest", "gemini-3.5-flash", "gemini-3.1-flash-lite"],
        "key_help": "Uses your Google AI Studio key — grounding returns the real "
                    "webSearchQueries Google executed (free tier works)",
    },
}

# model name fragments that are not text-generation chat models
_NON_CHAT_FRAGMENTS = ("embedding", "tts", "image", "veo", "imagen",
                       "live", "audio", "robotics", "aqa")


@st.cache_data(ttl=3600, show_spinner=False)
def list_available_models(provider_id: str, api_key: str) -> list[str]:
    """Fetch the models actually available to this API key.
    Raises on failure — caller decides the fallback."""
    if provider_id == "openai":
        r = requests.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        r.raise_for_status()
        models = [m["id"] for m in r.json().get("data", [])
                  if m.get("id", "").startswith("gpt-")
                  and not any(frag in m["id"] for frag in _NON_CHAT_FRAGMENTS)]
        models.sort(key=lambda n: ("mini" not in n, n))
        return models

    if provider_id in ("google", "google_grounding"):
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
    lang_name = _lang_name(lang)
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


def generate_fanouts(seeds: list[str], provider: dict, api_key: str, lang: str,
                     n_per_type: int, business_context: str,
                     model: str, country: str = "br",
                     progress_cb=None) -> pd.DataFrame:
    """Generate fan-outs for each seed with the selected provider.
    Simulated providers (Claude/Gemini) prompt the model to replicate fan-out
    behavior; observed providers (OpenAI web_search / Gemini grounding) return
    the real search queries the system executed."""
    rows = []
    provider_id, mode = provider["id"], provider.get("mode", "simulated")
    call_fn = call_anthropic if provider_id == "anthropic" else call_gemini
    status_box = st.empty()

    for i, seed in enumerate(seeds, start=1):
        try:
            if mode == "observed":
                if provider_id == "openai":
                    queries = fetch_openai_fanouts(seed, api_key, model,
                                                   lang=lang, country=country)
                    src_label = "chatgpt_observed"
                else:
                    queries = fetch_gemini_grounding_fanouts(seed, api_key, model,
                                                             lang=lang)
                    src_label = "google_grounding_observed"
                for q in queries:
                    rows.append({"query": q, "seed": seed, "source": src_label,
                                 "fanout_type": "observed_search", "intent": ""})
                if progress_cb:
                    progress_cb(i / len(seeds),
                                f"Fan-outs — {i}/{len(seeds)} seeds · {len(rows)} queries")
                continue

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

lang = st.sidebar.selectbox("Expansion language", ["pt-BR", "en", "es"], index=0)
_HL_DEFAULTS = {"pt-BR": "pt-BR", "en": "en", "es": "es"}
_GL_DEFAULTS = {"pt-BR": "br", "en": "us", "es": "mx"}
hl = st.sidebar.text_input("hl (Google interface language)",
                           value=_HL_DEFAULTS.get(lang, "en"))
gl = st.sidebar.text_input("gl (country)", value=_GL_DEFAULTS.get(lang, "us"),
                           help="For Spanish, set the market: mx (Mexico), "
                                "ar (Argentina), cl (Chile), es (Spain)...")

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
exclude_raw = st.sidebar.text_area(
    "Exclude terms (one per line or comma-separated)",
    placeholder="electrolux\nbrastemp, samsung",
    height=80,
    help="Any Suggest result containing one of these words is dropped from the "
         "final list — e.g. exclude 'electrolux' to remove 'geladeira electrolux'.",
)
exclude_terms = [t.strip().lower() for t in re.split(r"[,\n]", exclude_raw) if t.strip()]

st.sidebar.divider()
st.sidebar.subheader("Query Fan-out (LLM)")
_fanout_lang_keys = list(FANOUT_LANGUAGES.keys())
fanout_lang = st.sidebar.selectbox(
    "Fan-out language",
    _fanout_lang_keys,
    index=_fanout_lang_keys.index(lang) if lang in _fanout_lang_keys else 0,
    format_func=lambda k: f"{FANOUT_LANGUAGES[k]} ({k})",
    help="Language of the generated/extracted fan-out queries. Independent from "
         "the Suggest expansion language above.",
)
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
            # recommended models first (they drive the best fan-out behavior),
            # then the rest of what the key can access
            preferred = [m for m in provider["fallback_models"] if m in live_models]
            model_options = preferred + [m for m in live_models if m not in preferred]
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
if provider.get("mode") == "simulated":
    n_per_type = st.sidebar.slider("Fan-outs per type", 3, 15, 8)
    business_context = st.sidebar.text_area(
        "Business context (optional)",
        placeholder="E.g.: home appliances e-commerce in Brazil, focused on white goods...",
        height=80,
    )
else:
    n_per_type, business_context = 0, ""
    st.sidebar.caption("Real extraction mode: the provider returns the searches "
                       "it actually executed — no simulation settings needed.")

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
                depth2, depth2_limit, delay,
                exclude_terms=exclude_terms, progress_cb=cb,
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
        "Two modes, selected via the **LLM provider** in the sidebar. "
        "**Simulated** (Claude / Gemini): replicates Google AI Mode decomposition "
        "(reformulations, related, implicit, comparative, entity expansion) and "
        "ChatGPT follow-ups, classified by type and intent. "
        "**Real extracted** (OpenAI / Gemini grounding): runs each seed through the "
        "provider's live web search and returns the queries the system *actually* "
        "executed (`web_search_call.action` / `webSearchQueries`) — zero simulation."
    )
    if st.button("▶️ Generate Fan-outs", type="primary", disabled=not (seeds and api_key)):
        bar = st.progress(0.0, text="Starting...")
        cb = lambda p, msg: bar.progress(min(p, 1.0), text=msg)
        with st.spinner("Generating fan-outs via LLM..."):
            df_fan = generate_fanouts(seeds, provider, api_key, fanout_lang,
                                      n_per_type, business_context, model,
                                      country=gl, progress_cb=cb)
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
        c2.metric("Simulated", int(df["source"].isin(["google_ai_mode", "chatgpt"]).sum()))
        c3.metric("Real extracted", int(df["source"].isin(
            ["chatgpt_observed", "google_grounding_observed"]).sum()))
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
        "Manual capture of the **real fan-out queries exposed by the consumer "
        "interfaces** (Google AI Mode and chatgpt.com) — useful to validate what the "
        "actual products execute, complementing the automated extraction available "
        "in the Fan-out Generator. Follow the steps, run the script in your own "
        "browser, and paste the JSON back here."
    )

    subtab_gam, subtab_gpt = st.tabs(["Google AI Mode", "ChatGPT"])

    with subtab_gam:
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

    with subtab_gpt:
        st.markdown(
            "**Step 1.** On [chatgpt.com](https://chatgpt.com), ask each seed as a "
            "prompt (enable *Search the web* if it doesn't trigger automatically) "
            "and wait for the full answer. The executed sub-searches live in the "
            "conversation JSON, not in the visible page."
        )
        if seeds:
            st.caption("Suggested prompts (one conversation per seed): " +
                       " · ".join(f"`{s}`" for s in seeds))
        st.markdown(
            "**Step 2.** With the conversation open (URL containing `/c/...`), press "
            "`F12` → *Console*, paste the script below and hit Enter. It fetches the "
            "conversation JSON from your own logged-in session, walks it for every "
            "`search_queries` entry and `search(\"...\")` tool call, and copies the "
            "result to your clipboard as JSON."
        )
        st.code(CHATGPT_CONSOLE_SCRIPT, language="javascript")

    st.markdown("**Step 3.** Paste the JSON here (repeat per seed/conversation — paste one after the other):")
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

    if "df_observed" in st.session_state and not st.session_state["df_observed"].empty:
        df = st.session_state["df_observed"]
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Observed queries", len(df))
        c2.metric("Google (AI Mode/grounding)", int(df["source"].isin(
            ["ai_mode_observed", "google_grounding_observed"]).sum()))
        c3.metric("ChatGPT", int((df["source"] == "chatgpt_observed").sum()))
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
        c4.metric("Observed (real)", int(df_all["source"].isin(
            ["ai_mode_observed", "chatgpt_observed", "google_grounding_observed"]).sum()))

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
