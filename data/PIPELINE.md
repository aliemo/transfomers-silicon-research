# Auto papers pipeline — CI/CD + free AI analysis + website

## Single source of truth

```text
data/papers.yaml   ← only editable catalog
        │
        ├── scripts/generator.py     → README.md, data/papers.csv, plot
        └── website/build.py         → website/dist/*
```

Pipeline / admin write **only** `data/papers.yaml`, then regenerate derived artifacts.

Canonical paper fields: `title`, `year`, `type`, `doi`, `url`, `pdf`, `ignore`, `silicon`,
`platform`, `model`, `publisher`, `pubname`, `authors`, `category`,
`review_pass`, `review_pass1..3`, `review_notes`.

Removed (unused): `method`, `pubkey`, `reserved`.

## Weekly GitHub Actions (ready to use)

Workflow: `.github/workflows/papers-pipeline.yml`

| Trigger | When |
|---|---|
| **Schedule** | Every **Monday 06:00 UTC** |
| **Manual** | Actions → *Papers Auto-Ingest* → Run workflow |

What it does:
1. Search last **21 days** (OpenAlex + arXiv)
2. Analyze (Groq/Gemini if secrets set, else heuristic)
3. Append new related papers to `papers.yaml` (**Pass 1**, `ignore: check`)
4. Rebuild README / CSV / plot / `website/dist`
5. Open a **Pull Request** (`auto/papers-ingest`) — merge when ready

Your Pass 1/2/3 admin decisions are **kept** (CI only appends).

### One-time GitHub setup
1. Push this repo to GitHub
2. **Settings → Actions → General** → allow Actions / allow creating PRs
3. Optional secrets: `GROQ_API_KEY`, `GEMINI_API_KEY`
4. After first push, open **Actions** and run *Papers Auto-Ingest* once (manual test)

## Config layout

```text
configs/
  queries.yaml       # Scholar / OpenAlex / arXiv query expressions
  categories.yaml    # journal | conference | arxiv | thesis | other
  pipeline.yaml      # fetch / analyze / catalog / CI settings
  website.yaml       # site title + filter UI options
  review.yaml        # Pass 1 / 2 / 3 workflow
```

## Local commands

```bash
pip install -r requirements.txt

# Weekly-style run (same as CI)
python scripts/pipeline/run_pipeline.py --lookback-days 21

# Dry run (no YAML write)
python scripts/pipeline/run_pipeline.py --dry-run --lookback-days 21

# Deep historical backfill
python scripts/pipeline/fetch_papers.py --from-year 2017 --to-year 2026 --max-per-query 200
```

## Website (GitHub Pages)

Live site: **https://aliemo.github.io/transfomers-silicon-research/**

CI workflow: `.github/workflows/pages.yml` — rebuilds from `data/papers.yaml` on every relevant push to `main`.

```bash
python website/build.py
# open website/dist/index.html
```

One-time GitHub setup:
1. Merge/push this workflow, wait for **Deploy GitHub Pages** to create the `gh-pages` branch
2. **Settings → Pages → Build and deployment**
   - Source: **Deploy from a branch**
   - Branch: **gh-pages** / **/ (root)**
3. Open https://aliemo.github.io/transfomers-silicon-research/

Status chips: **Silicon · Non-silicon · Pass 1 · Pass 2 · Pass 3**

## Admin (Pass 1 / 2 / 3)

```bash
python scripts/admin_server.py
# http://127.0.0.1:8787/admin/
```

Commit + push `data/papers.yaml` after admin review so CI PRs merge cleanly on top of your decisions.
