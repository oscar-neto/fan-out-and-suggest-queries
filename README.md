# 🔎 Query Opportunity Mapper

SEO/GEO opportunity mapping tool with two engines, built with Streamlit:

1. **Google Suggest Miner** — large-scale keyword expansion via Google's autocomplete endpoint, using alphabet soup (a–z), question/commercial/preposition modifiers (pt-BR and en), and optional level-2 recursion.
2. **Query Fan-out Generator (LLM)** — simulates the query decomposition behavior of **Google AI Mode** (reformulations, related, implicit, comparative, entity expansion) and **ChatGPT** (conversational follow-ups) using the Anthropic API, classifying each fan-out by type and search intent.

Results are consolidated, deduplicated, and exportable as CSV or multi-sheet XLSX.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository (`app.py` + `requirements.txt` at the root).
2. Go to [share.streamlit.io](https://share.streamlit.io), click **New app**, and select the repo, branch, and `app.py` as the main file.
3. Deploy. No secrets are required — the Anthropic API key is entered by the user in the sidebar at runtime and is never stored.

## Usage

1. Paste your seed terms/topics in the main text area (one per line).
2. **Google Suggest Miner tab** — configure modifier groups, alphabet expansion, and recursion in the sidebar, then click *Mine Google Suggest*. If results come back empty, increase the request delay (rate limiting).
3. **Query Fan-out Generator tab** — enter your Anthropic API key in the sidebar, optionally add business context (this makes entity expansions domain-specific), then click *Generate Fan-outs*.
4. **Results & Export tab** — search, review, and download the consolidated dataset (CSV / XLSX).

## Output schema

| Column | Description |
|---|---|
| `query` | The discovered/generated query |
| `seed` | Originating seed term |
| `source` | `google_suggest`, `google_ai_mode`, or `chatgpt` |
| `expansion_pattern` | Suggest probe pattern (base, alphabet, modifier, recursion) |
| `level` | Suggest recursion depth (1 or 2) |
| `fanout_type` | reformulation, related, implicit, comparative, entity_expansion, follow_up |
| `intent` | informational, commercial, transactional, navigational |

## Notes

- The Google Suggest endpoint is undocumented and rate-limited; randomized delays between requests are built in.
- LLM fan-outs are synthetic simulations of engine behavior, intended for content-gap and coverage analysis — not an official Google/OpenAI data source.
