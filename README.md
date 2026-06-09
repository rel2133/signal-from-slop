# Signal from the Slop

`Signal from the Slop` is a local MVP for surfacing research leads from Reddit stock discussions. It loads Reddit-style posts and comments, extracts likely tickers, classifies each item with Ollama, stores the results in SQLite, and displays them in a filterable Streamlit UI.

This is a research workflow, not a financial advice tool.

## MVP Scope

The current prototype covers the dashboard expansion MVP:

- Manage subreddit and thread URL sources in SQLite
- Run source-filtered fake-data analysis over multi-week time windows
- Extract tickers and company names from a local CSV catalog
- Send each item to Ollama over `http://localhost:11434/api/chat`
- Save raw items, classifications, item-level ticker mentions, ticker summaries, and time buckets to SQLite
- Explore results in dashboard, trends, and export pages

The Reddit ingestion layer supports both the bundled fake dataset and no-key live Reddit collection through public RSS feeds.

## Project Layout

```text
.
├── app.py
├── requirements.txt
├── schema.sql
├── .env.example
├── data
│   ├── default_sources.json
│   ├── fake_reddit_data.json
│   └── tickers.csv
└── signal_from_the_slop
    ├── __init__.py
    ├── analytics.py
    ├── database.py
    ├── ollama_classifier.py
    ├── reddit_client.py
    ├── scoring.py
    └── ticker_extractor.py
```

## Requirements

- Python 3.10+
- Ollama installed locally
- A pulled Ollama model, for example `llama3.1:8b`

## Setup

### Quick Copy/Paste

Terminal 1:

```bash
cd "/Users/rowanellis/Documents/Signal from the Slop"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp -n .env.example .env
streamlit run app.py
```

Terminal 2:

```bash
ollama serve
```

Optional, if you want the default model used in `.env.example`:

```bash
ollama pull llama3.1:8b
```

If `llama3.1:8b` is not installed locally, the app will let you choose any installed Ollama model from the sidebar.

1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy environment settings:

```bash
cp .env.example .env
```

4. Start Ollama and ensure the model exists:

```bash
ollama serve
ollama pull llama3.1:8b
```

5. Run the app:

```bash
streamlit run app.py
```

## Troubleshooting

- Do not run `npm install`. This MVP is Python-only and does not include a `package.json`.
- If `python3 -m venv .venv` fails and `.venv/bin/activate` is missing, remove the partial `.venv` and try again after freeing disk space.
- If `pip install -r requirements.txt` fails with `No space left on device`, clear caches or remove files until you have at least a few GB free before retrying.

## How It Works

1. The app loads and filters multi-week fake Reddit items from [data/fake_reddit_data.json](/Users/rowanellis/Documents/Signal from the Slop/data/fake_reddit_data.json).
2. [signal_from_the_slop/reddit_client.py](/Users/rowanellis/Documents/Signal from the Slop/signal_from_the_slop/reddit_client.py) normalizes subreddit or thread URL sources and matches them to fake items.
3. [signal_from_the_slop/ticker_extractor.py](/Users/rowanellis/Documents/Signal from the Slop/signal_from_the_slop/ticker_extractor.py) matches `$TICKER`, uppercase ticker symbols, and company names from [data/tickers.csv](/Users/rowanellis/Documents/Signal from the Slop/data/tickers.csv).
4. [signal_from_the_slop/ollama_classifier.py](/Users/rowanellis/Documents/Signal from the Slop/signal_from_the_slop/ollama_classifier.py) sends each item to Ollama using `requests.post(...)` against `/api/chat` and requests strict JSON output.
5. If Ollama is unavailable or returns invalid JSON, the app falls back to deterministic heuristics so the dashboard remains testable offline.
6. [signal_from_the_slop/analytics.py](/Users/rowanellis/Documents/Signal from the Slop/signal_from_the_slop/analytics.py) builds long-format mention rows, ticker summaries, and time-bucketed acceleration metrics.
7. [signal_from_the_slop/database.py](/Users/rowanellis/Documents/Signal from the Slop/signal_from_the_slop/database.py) stores sources, runs, raw items, classifications, long-format mentions, summaries, and trend buckets in SQLite.
8. [app.py](/Users/rowanellis/Documents/Signal from the Slop/app.py) exposes the Streamlit pages: `Sources`, `Run Analysis`, `Results Dashboard`, `Ticker Trends`, `Export Data`, and `Settings`.

## Environment Variables

```env
OLLAMA_URL=http://localhost:11434/api/chat
OLLAMA_MODEL=llama3.1:8b
SQLITE_PATH=signal_from_the_slop.db
REDDIT_USER_AGENT=signal-from-the-slop/0.1
```

## Live Reddit Smoke Test

Run this to test the no-key Reddit scraper:

```bash
cd "/Users/rowanellis/Documents/Signal from the Slop"
source .venv/bin/activate
python scripts/check_live_reddit.py --subreddit stocks --days 1 --max-posts 3 --max-comments 2
```

That command exercises the same `collect_live_items(...)` path the Streamlit app uses and prints the fetched item counts plus the newest post permalink.

## Notes

- The classifier prompt forces strict JSON output and uses `temperature: 0`.
- The app does not invent tickers. Unknown references remain empty unless the extractor finds a catalog match.
- Live Reddit scraping uses public Reddit RSS feeds. It does not need Reddit API keys, but it can be rate-limited and may only expose recent public posts/comments.
- Local source edits are stored in SQLite. A fresh Streamlit Community Cloud deployment starts from [data/default_sources.json](/Users/rowanellis/Documents/Signal from the Slop/data/default_sources.json), not from your local `signal_from_the_slop.db`.
