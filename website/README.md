# Website

Filterable static catalog generated from **`data/papers.yaml`** (single source of truth).

## Build public site

```bash
python website/build.py
# open website/dist/index.html
```

Status chips: **Silicon** · **Non-silicon** · **Pass 1** · **Pass 2** · **Pass 3**  
(Pass values come from `review_pass` in papers.yaml / admin.)

## Admin panel

```bash
python scripts/admin_server.py
# http://127.0.0.1:8787/admin/
```

Capabilities:
- list / search / filter by Pass 1–3
- change `review_pass` and pass statuses
- Accept / Reject / Skip pass
- add / edit / save papers → `data/papers.yaml`
- ignore (soft) or delete (hard)
- analyze one paper (heuristic / Groq / Gemini)
- rebuild public site (+ README/CSV when pipeline regenerates)

## Config

- `configs/website.yaml` — title, filters, build paths, admin port
- `configs/categories.yaml` — journal / conference / arxiv / thesis rules
- `configs/review.yaml` — Pass 1 / 2 / 3 workflow
- `configs/queries.yaml` — ingest queries
