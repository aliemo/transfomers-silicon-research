# Website

Filterable static catalog generated from **`data/papers.yaml`** (single source of truth).

## Live site (GitHub Pages)

https://aliemo.github.io/transfomers-silicon-research/

Deployed by `.github/workflows/pages.yml` whenever `papers.yaml` / website / configs change on `main`.

Enable once: after the first successful deploy creates `gh-pages`, set  
**Settings → Pages → Source: Deploy from a branch → `gh-pages` / root**.

## Build locally

```bash
python website/build.py
# open website/dist/index.html
```

Status chips: **Silicon** · **Non-silicon** · **Pass 1** · **Pass 2** · **Pass 3**

## Admin panel

```bash
python scripts/admin_server.py
# http://127.0.0.1:8787/admin/
```

## Config

- `configs/website.yaml` — title, filters, build paths, admin port, Pages URL
- `configs/categories.yaml` — category rules
- `configs/review.yaml` — Pass 1 / 2 / 3 workflow
