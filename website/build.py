#!/usr/bin/env python
"""Build a filterable static website from papers.yaml + configs/."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "pipeline"))

from common import category_meta, classify_venue, load_categories, load_yaml  # noqa: E402

MISSING = {"", "none", "null", "false", "__no_data__", "no_data", "n/a", "na", "?", "platform", "unknown"}

SHORT_VENUE = [
    ("ISSCC", r"\bISSCC\b"),
    ("JSSC", r"\bJSSC\b|Journal of Solid[- ]State Circuits"),
    ("TVLSI", r"\bTVLSI\b|Very Large Scale Integration"),
    ("TCAD", r"\bTCAD\b|Computer-Aided Design"),
    ("ESSCIRC", r"\bESSCIRC\b"),
    ("MICRO", r"\bMICRO\b"),
    ("ISCA", r"\bISCA\b"),
    ("HPCA", r"\bHPCA\b"),
    ("DAC", r"\bDAC\b|Design Automation Conference"),
    ("ICCAD", r"\bICCAD\b"),
    ("FCCM", r"\bFCCM\b"),
    ("FPGA", r"\bFPGA\b.*Symposium|International Symposium on Field-Programmable Gate Arrays"),
    ("DATE", r"\bDATE\b|Design,? Automation.? (and|&) Test"),
    ("ISCAS", r"\bISCAS\b"),
    ("ASP-DAC", r"\bASP-?DAC\b"),
    ("ISVLSI", r"\bISVLSI\b"),
    ("ASAP", r"\bASAP\b"),
    ("MLSys", r"\bMLSys\b"),
    ("NeurIPS", r"\bNeurIPS\b|\bNIPS\b"),
    ("AAAI", r"\bAAAI\b"),
    ("ACL", r"\bACL\b"),
    ("USENIX", r"\bUSENIX\b"),
    ("arXiv", r"\barXiv\b|\bcs\.(AR|LG|CL|CV)\b"),
]


def clean_text(v) -> str:
    if v is None or v is False:
        return ""
    s = str(v).strip()
    if s.lower() in MISSING:
        return ""
    return s


def clean_link(v) -> str:
    s = clean_text(v)
    if not s:
        return ""
    if s.startswith("http"):
        return s
    if s.startswith("10."):
        return f"https://doi.org/{s}"
    return ""


def doi_text(doi_url: str) -> str:
    if not doi_url:
        return ""
    s = doi_url
    for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/"):
        if s.startswith(prefix):
            return s[len(prefix) :]
    if "arxiv.org/abs/" in s:
        return "arXiv:" + s.rsplit("/", 1)[-1]
    return s


def models_list(v) -> list[str]:
    if isinstance(v, list):
        vals = [clean_text(x) for x in v]
    elif clean_text(v):
        vals = [clean_text(v)]
    else:
        vals = []
    # drop hardware tokens accidentally stored as model
    drop = {"fpga", "asic", "gpu", "pim", "cim", "design", "dsa", "soc", "eda"}
    out = []
    for x in vals:
        if x.lower() in drop:
            continue
        if x and x not in out:
            out.append(x)
    return out


def short_venue(pubname: str, publisher: str, category: str, doi: str) -> str:
    blob = f"{pubname} {publisher} {doi}"
    for short, pat in SHORT_VENUE:
        if re.search(pat, blob, flags=re.I):
            return short
    if category == "arxiv" or "arxiv" in blob.lower():
        return "arXiv"
    if pubname:
        # take parenthetical acronym if present
        m = re.search(r"\(([A-Z][A-Z0-9\-]{1,12})\)", pubname)
        if m:
            return m.group(1)
        words = re.findall(r"[A-Za-z0-9]+", pubname)
        if words:
            return " ".join(words[:3])[:28]
    return publisher or category.title() or "Venue"


def status_info(paper: dict, review_pass: str) -> dict:
    """Public status: Pass 1/2/3 or Silicon / Non-silicon (replaces Under review)."""
    rp = str(review_pass or "").strip().lower()
    if rp == "pass1":
        return {"id": "pass1", "label": "Pass 1", "color": "#c9892d", "kind": "review"}
    if rp == "pass2":
        return {"id": "pass2", "label": "Pass 2", "color": "#d4a15b", "kind": "review"}
    if rp == "pass3":
        return {"id": "pass3", "label": "Pass 3", "color": "#e0a45c", "kind": "review"}

    # Done / included → silicon truth
    s = str(paper.get("silicon")).strip().lower()
    if s in {"false", "0", "no"}:
        return {"id": "no", "label": "Non-silicon", "color": "#8a8f98", "kind": "silicon"}
    if s in {"true", "1", "yes"}:
        return {"id": "silicon", "label": "Silicon", "color": "#1f8f5f", "kind": "silicon"}
    # Legacy check without review_pass → treat as Pass 1
    return {"id": "pass1", "label": "Pass 1", "color": "#c9892d", "kind": "review"}


def paper_keywords(title: str, models: list[str]) -> list[str]:
    """Lightweight keywords derived from title + model tags."""
    keys = []
    for m in models:
        if m not in keys:
            keys.append(m)
    patterns = [
        ("Accelerator", r"accelerator|acceleration"),
        ("FPGA", r"\bfpga\b"),
        ("ASIC", r"\basic\b|nm\b|tapeout|chip"),
        ("Attention", r"attention|softmax|mhsa"),
        ("Quantization", r"quantiz"),
        ("Sparsity", r"spars"),
        ("PIM", r"\bpim\b|in-memory|cim\b"),
        ("Edge", r"\bedge\b"),
        ("LLM", r"\bllm\b|large language"),
        ("ViT", r"vision transformer|\bvit\b"),
    ]
    t = title.lower()
    for label, pat in patterns:
        if re.search(pat, t) and label not in keys:
            keys.append(label)
    return keys[:8]


def meta_keywords(platform: str, short: str, paper_kw: list[str]) -> list[str]:
    out = []
    seen = {k.lower() for k in paper_kw}
    drop_cats = {"journal", "conference", "arxiv", "thesis / report", "thesis", "report", "other"}
    for x in (platform, short):
        x = clean_text(x)
        if not x or x.lower() in MISSING:
            continue
        if x.lower() in seen or x.lower() in drop_cats:
            continue
        out.append(x)
        seen.add(x.lower())
    return out


def load_site_papers(cfg: dict):
    papers_path = ROOT / cfg.get("build", {}).get("papers_yaml", "data/papers.yaml")
    cat_path = ROOT / cfg.get("build", {}).get("categories_file", "configs/categories.yaml")
    include_ignored = bool(cfg.get("build", {}).get("include_ignored", False))
    max_papers = int(cfg.get("build", {}).get("max_papers") or 0)

    raw = yaml.safe_load(papers_path.read_text(encoding="utf-8")) or {}
    categories_cfg = load_categories(cat_path)
    cat_meta = category_meta(categories_cfg)

    # Lazy import review inference
    from review import attach_review  # noqa: WPS440

    papers = []
    for pid, p in raw.items():
        if not include_ignored and str(p.get("ignore")).lower() in {"true", "ignore"}:
            continue

        reviewed = attach_review(p)
        category = reviewed.get("category") or classify_venue(reviewed, categories_cfg)
        cat_label = cat_meta.get(category, {}).get("label", category.title())
        cat_color = cat_meta.get(category, {}).get("color", "#9aa7b8")

        doi = clean_link(reviewed.get("doi"))
        url = clean_link(reviewed.get("url")) or doi
        pdf = clean_link(reviewed.get("pdf"))
        download = pdf or doi or url

        publisher = clean_text(reviewed.get("publisher")) or ("arXiv" if category == "arxiv" else "Publisher TBD")
        if publisher.lower() in {"openalex", "arxive"}:
            publisher = "arXiv" if "arxiv" in publisher.lower() or category == "arxiv" else publisher
        year = reviewed.get("year") or ""
        pubname = clean_text(reviewed.get("pubname"))
        if not pubname:
            pubname = {
                "journal": f"{publisher} Journal",
                "conference": f"{publisher} Conference",
                "arxiv": "arXiv preprint",
                "thesis": "Thesis / Dissertation",
            }.get(category, publisher)

        platform = clean_text(reviewed.get("platform"))
        models = models_list(reviewed.get("model"))
        short = short_venue(pubname, publisher, category, doi)
        status = status_info(reviewed, reviewed.get("review_pass"))
        paper_kw = paper_keywords(clean_text(reviewed.get("title")), models)
        meta_kw = meta_keywords(platform, short, paper_kw)
        authors = reviewed.get("authors") or []
        if isinstance(authors, str):
            authors = [a.strip() for a in authors.split(",") if a.strip()]
        authors = [clean_text(a) for a in authors if clean_text(a)]

        papers.append(
            {
                "id": int(pid) if str(pid).isdigit() else pid,
                "title": clean_text(reviewed.get("title")) or "Untitled",
                "authors": authors,
                "authors_text": ", ".join(authors),
                "venue_full": pubname,
                "venue_short": short,
                "year": year,
                "publisher": publisher,
                "publisher_year": f"{publisher}|{year}" if year else publisher,
                "category": category,
                "category_label": cat_label,
                "category_color": cat_color,
                "doi": doi,
                "doi_text": doi_text(doi),
                "url": url,
                "pdf": pdf,
                "download": download,
                "platform": platform,
                "model": models,
                "keywords_paper": paper_kw,
                "keywords_meta": meta_kw,
                "status": status["id"],
                "status_label": status["label"],
                "status_color": status["color"],
                "status_kind": status["kind"],
                "review_pass": reviewed.get("review_pass"),
                "silicon_raw": reviewed.get("silicon"),
            }
        )

    papers.sort(key=lambda x: (-int(x["year"] or 0), x["title"].lower()))
    if max_papers > 0:
        papers = papers[:max_papers]
    return papers, cat_meta


def write_assets(out: Path, site: dict, papers: list[dict], cat_meta: dict) -> None:
    years = sorted({str(p["year"]) for p in papers if p["year"]}, reverse=True)
    platforms = sorted({p["platform"] for p in papers if p["platform"]})
    models = sorted({m for p in papers for m in p["model"]})
    publishers = sorted({p["publisher"] for p in papers if p["publisher"]})
    shorts = sorted({p["venue_short"] for p in papers if p["venue_short"]})
    categories = list(cat_meta.values())

    status_filters = [
        {"id": "silicon", "label": "Silicon", "color": "#1f8f5f"},
        {"id": "no", "label": "Non-silicon", "color": "#8a8f98"},
        {"id": "pass1", "label": "Pass 1", "color": "#c9892d"},
        {"id": "pass2", "label": "Pass 2", "color": "#d4a15b"},
        {"id": "pass3", "label": "Pass 3", "color": "#e0a45c"},
    ]

    payload = {
        "site": site,
        "categories": categories,
        "filters": {
            "years": years,
            "platforms": platforms,
            "models": models,
            "publishers": publishers,
            "venues": shorts,
            "status": status_filters,
        },
        "papers": papers,
        "count": len(papers),
    }
    (out / "papers.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    (out / "styles.css").write_text(CSS, encoding="utf-8")
    (out / "app.js").write_text(JS, encoding="utf-8")

    cat_chips = "".join(
        f'<button type="button" class="chip" data-filter-chip="category:{c["id"]}" style="--chip:{c["color"]}">{c["label"]}</button>'
        for c in categories
    )
    status_chips = "".join(
        f'<button type="button" class="chip status-chip" data-filter-chip="status:{s["id"]}" style="--chip:{s["color"]}">{s["label"]}</button>'
        for s in status_filters
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{site.get("title", "Transformers on Silicon")}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,700;1,9..40,400&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet" />
  <link rel="stylesheet" href="./styles.css" />
</head>
<body>
  <div class="appbar">
    <div class="wrap appbar-inner">
      <div>
        <p class="eyebrow">Hardware · Transformers · Silicon</p>
        <h1>{site.get("title", "Transformers on Silicon")}</h1>
        <p class="lead">{site.get("tagline", "")}</p>
      </div>
      <div class="stats">
        <div class="stat"><strong id="visible-count">0</strong><span>shown</span></div>
        <div class="stat"><strong id="total-count">{len(papers)}</strong><span>total</span></div>
      </div>
    </div>
  </div>

  <section class="toolbar" id="filters">
    <div class="wrap">
      <div class="search-row m3-field">
        <input id="q" type="search" placeholder="Search title, authors, venue, DOI, keywords…" autocomplete="off" />
      </div>
      <div class="chips" id="category-chips">{cat_chips}</div>
      <div class="chips" id="status-chips">{status_chips}</div>
      <div class="filters">
        <label class="m3-field">Year
          <select id="year"><option value="">All years</option></select>
        </label>
        <label class="m3-field">Venue
          <select id="venue"><option value="">All venues</option></select>
        </label>
        <label class="m3-field">Publisher
          <select id="publisher"><option value="">All publishers</option></select>
        </label>
        <label class="m3-field">Platform
          <select id="platform"><option value="">All platforms</option></select>
        </label>
        <label class="m3-field">Model
          <select id="model"><option value="">All models</option></select>
        </label>
        <button id="reset" type="button" class="m3-btn tonal">Reset</button>
      </div>
    </div>
  </section>

  <main class="wrap">
    <div id="year-groups"></div>
    <p id="empty" class="empty hidden">No papers match these filters.</p>
  </main>

  <footer class="wrap foot">
    <span>Source of truth: <code>data/papers.yaml</code></span>
    <span class="foot-sep">·</span>
    <span>Generated catalog</span>
  </footer>
  <script src="./app.js"></script>
</body>
</html>
"""
    (out / "index.html").write_text(html, encoding="utf-8")


CSS = r"""
:root {
  --md-sys-color-surface: #f3f6f4;
  --md-sys-color-surface-container: #ffffff;
  --md-sys-color-surface-container-high: #e9efeb;
  --md-sys-color-on-surface: #15201b;
  --md-sys-color-on-surface-variant: #4f5d56;
  --md-sys-color-outline: #d0dbd4;
  --md-sys-color-primary: #0b6e56;
  --md-sys-color-on-primary: #ffffff;
  --md-sys-color-secondary-container: #d4ebe3;
  --md-sys-color-on-secondary-container: #12352c;
  --md-elevation-1: 0 1px 2px rgba(21,32,27,.07), 0 1px 3px 1px rgba(21,32,27,.05);
  --md-elevation-2: 0 4px 12px rgba(21,32,27,.1), 0 1px 3px rgba(21,32,27,.06);
  --md-radius: 14px;
  --md-font: "DM Sans", system-ui, sans-serif;
  --md-mono: "IBM Plex Mono", ui-monospace, monospace;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: var(--md-font);
  color: var(--md-sys-color-on-surface);
  background:
    radial-gradient(900px 420px at 8% -10%, rgba(11,110,86,.14), transparent 60%),
    radial-gradient(700px 360px at 92% 0%, rgba(30,77,123,.08), transparent 55%),
    var(--md-sys-color-surface);
  line-height: 1.45;
}
.wrap { width: min(1080px, calc(100% - 2rem)); margin-inline: auto; }

.appbar {
  background:
    linear-gradient(135deg, #0a5f4a 0%, #0b6e56 48%, #0d7a5f 100%);
  color: #f3faf7;
  padding: 1.7rem 0 1.4rem;
  box-shadow: var(--md-elevation-2);
}
.appbar-inner {
  display: flex; justify-content: space-between; gap: 1.5rem; align-items: end;
}
.eyebrow {
  margin: 0 0 .4rem; font-size: .72rem; letter-spacing: .14em;
  text-transform: uppercase; opacity: .82; font-weight: 500;
}
h1 {
  margin: 0; font-size: clamp(1.75rem, 3.8vw, 2.5rem); font-weight: 700; letter-spacing: -.03em;
}
.lead { margin: .55rem 0 0; max-width: 42rem; opacity: .9; font-size: .98rem; font-weight: 400; }
.stats { display: flex; gap: .7rem; }
.stat {
  min-width: 78px; text-align: center; background: rgba(255,255,255,.12);
  border: 1px solid rgba(255,255,255,.16); border-radius: 14px; padding: .55rem .7rem;
  backdrop-filter: blur(6px);
}
.stat strong { display: block; font-size: 1.25rem; }
.stat span { font-size: .72rem; opacity: .85; }

.toolbar {
  position: sticky; top: 0; z-index: 30;
  background: color-mix(in srgb, var(--md-sys-color-surface) 86%, white);
  backdrop-filter: blur(14px);
  border-bottom: 1px solid var(--md-sys-color-outline);
  padding: .85rem 0 .95rem;
}
.search-row input {
  width: 100%; border: 1px solid var(--md-sys-color-outline); background: var(--md-sys-color-surface-container);
  border-radius: 28px; padding: .95rem 1.15rem; font: inherit;
  box-shadow: var(--md-elevation-1);
  transition: border-color .15s ease, box-shadow .15s ease;
}
.search-row input:focus {
  outline: 2px solid color-mix(in srgb, var(--md-sys-color-primary) 45%, white);
  border-color: var(--md-sys-color-primary);
}
.chips { display: flex; flex-wrap: wrap; gap: .45rem; margin-top: .7rem; }
.chip {
  border: 1px solid color-mix(in srgb, var(--chip, var(--md-sys-color-primary)) 28%, white);
  background: color-mix(in srgb, var(--chip, var(--md-sys-color-primary)) 12%, white);
  color: var(--md-sys-color-on-surface); border-radius: 999px; padding: .38rem .8rem;
  font-size: .82rem; font-weight: 500; cursor: pointer;
  transition: transform .12s ease, background .12s ease, color .12s ease;
}
.chip:hover { transform: translateY(-1px); }
.chip.active {
  background: var(--chip, var(--md-sys-color-primary)); color: #fff; border-color: transparent;
  box-shadow: var(--md-elevation-1);
}
.filters {
  display: grid; grid-template-columns: repeat(6, minmax(0,1fr));
  gap: .65rem; margin-top: .8rem; align-items: end;
}
@media (max-width: 900px) {
  .appbar-inner { flex-direction: column; align-items: start; }
  .filters { grid-template-columns: repeat(2, minmax(0,1fr)); }
}
.m3-field { display: grid; gap: .28rem; font-size: .74rem; color: var(--md-sys-color-on-surface-variant); font-weight: 500; }
.m3-field select {
  border: 1px solid var(--md-sys-color-outline); background: var(--md-sys-color-surface-container);
  border-radius: 12px; padding: .62rem .75rem; font: inherit; color: var(--md-sys-color-on-surface);
}
.m3-btn {
  border: 0; border-radius: 999px; padding: .7rem 1rem; font: inherit; font-weight: 600; cursor: pointer;
}
.m3-btn.tonal {
  background: var(--md-sys-color-secondary-container); color: var(--md-sys-color-on-secondary-container);
}

.year-block { margin: 1.5rem 0 1.9rem; }
.year-block h2 {
  margin: 0 0 .9rem; font-size: 1.12rem; font-weight: 700;
  color: var(--md-sys-color-on-surface-variant);
  display: flex; align-items: center; gap: .55rem;
}
.year-block h2::after {
  content: ""; flex: 1; height: 1px; background: var(--md-sys-color-outline); opacity: .7;
}
.cards { display: grid; gap: .9rem; }
.card {
  background: var(--md-sys-color-surface-container);
  border: 1px solid var(--md-sys-color-outline);
  border-radius: var(--md-radius);
  padding: 1.05rem 1.15rem 1rem;
  box-shadow: var(--md-elevation-1);
  transition: transform .16s ease, box-shadow .16s ease, border-color .16s ease;
  animation: rise .35s ease both;
}
.card:hover {
  transform: translateY(-2px);
  box-shadow: var(--md-elevation-2);
  border-color: color-mix(in srgb, var(--md-sys-color-primary) 28%, var(--md-sys-color-outline));
}
@keyframes rise {
  from { opacity: 0; transform: translateY(6px); }
  to { opacity: 1; transform: translateY(0); }
}
.card h3 {
  margin: 0; font-size: 1.05rem; font-weight: 700; line-height: 1.3; letter-spacing: -.015em;
}
.card h3 a { color: inherit; text-decoration: none; }
.card h3 a:hover { color: var(--md-sys-color-primary); }
.authors {
  margin: .4rem 0 .2rem; color: var(--md-sys-color-on-surface-variant);
  font-size: .88rem; line-height: 1.35; font-weight: 500;
}
.venue-full {
  margin: .2rem 0 .55rem; color: var(--md-sys-color-on-surface-variant); font-size: .9rem;
}
.meta-line {
  display: flex; flex-wrap: wrap; gap: .35rem .45rem; align-items: center; margin-bottom: .55rem;
}
.mchip {
  display: inline-flex; align-items: center; border-radius: 8px;
  padding: .2rem .55rem; font-size: .73rem; font-weight: 600;
  background: var(--md-sys-color-surface-container-high); color: var(--md-sys-color-on-surface-variant);
}
.mchip.cat { background: var(--cat, var(--md-sys-color-primary)); color: #fff; }
.split-badge {
  display: inline-flex; align-items: stretch; overflow: hidden;
  border-radius: 8px; font-size: .73rem; font-weight: 700;
  line-height: 1; box-shadow: inset 0 0 0 1px rgba(20,40,70,.08);
}
.split-badge .sb-left,
.split-badge .sb-right {
  display: inline-flex; align-items: center; padding: .28rem .55rem;
}
.split-badge .sb-left { background: #1e4d7b; color: #fff; }
.split-badge .sb-right { background: #d7e8f8; color: #1e4d7b; }
.doi-row { margin: 0 0 .55rem; font-size: .88rem; }
.doi-row a { color: var(--md-sys-color-primary); text-decoration: none; font-weight: 600; }
.doi-row a:hover { text-decoration: underline; }
.doi-text {
  margin-left: .4rem; color: var(--md-sys-color-on-surface-variant);
  font-family: var(--md-mono); font-size: .8rem;
}
.kw {
  display: grid; gap: .4rem; margin-top: .15rem;
}
.kw-row { display: flex; flex-wrap: wrap; gap: .3rem; }
.kw strong {
  font-size: .68rem; letter-spacing: .05em; text-transform: uppercase;
  color: var(--md-sys-color-on-surface-variant); font-weight: 700;
}
.tag {
  border-radius: 999px; padding: .16rem .5rem; font-size: .74rem; font-weight: 500;
  background: #e8f1ec; color: #355348;
}
.tag.meta { background: #eef1f4; color: #46525c; }
.card-foot {
  display: flex; justify-content: space-between; gap: .8rem; align-items: center;
  margin-top: .9rem; padding-top: .75rem; border-top: 1px solid var(--md-sys-color-outline);
}
.status {
  display: inline-flex; align-items: center; gap: .45rem;
  font-size: .8rem; font-weight: 700;
  padding: .28rem .65rem; border-radius: 999px;
  background: color-mix(in srgb, var(--st) 14%, white);
  color: color-mix(in srgb, var(--st) 75%, #15201b);
  border: 1px solid color-mix(in srgb, var(--st) 28%, white);
}
.status i {
  width: .55rem; height: .55rem; border-radius: 50%; background: var(--st, #8a8f98);
}
.download {
  display: inline-flex; align-items: center; border-radius: 999px;
  background: var(--md-sys-color-primary); color: var(--md-sys-color-on-primary);
  text-decoration: none; font-weight: 600; padding: .45rem .9rem; font-size: .8rem;
  box-shadow: var(--md-elevation-1);
  transition: transform .12s ease, background .12s ease;
}
.download:hover { transform: translateY(-1px); }
.download.disabled { background: #c9c3b8; pointer-events: none; box-shadow: none; }
.empty { color: var(--md-sys-color-on-surface-variant); padding: 2rem 0; }
.hidden { display: none !important; }
.foot {
  color: var(--md-sys-color-on-surface-variant); font-size: .82rem;
  padding: 1.4rem 0 2.6rem; display: flex; flex-wrap: wrap; gap: .4rem; align-items: center;
}
.foot-sep { opacity: .5; }
"""

JS = r"""
async function main() {
  const res = await fetch('./papers.json');
  const data = await res.json();
  const papers = data.papers;
  const state = { q: '', category: '', year: '', platform: '', model: '', status: '', publisher: '', venue: '' };

  fillSelect('year', data.filters.years);
  fillSelect('platform', data.filters.platforms);
  fillSelect('model', data.filters.models);
  fillSelect('publisher', data.filters.publishers);
  fillSelect('venue', data.filters.venues);
  document.getElementById('total-count').textContent = String(data.count);

  const bind = (id, key) => {
    const el = document.getElementById(id);
    el.addEventListener('input', () => { state[key] = el.value.trim(); render(); });
    el.addEventListener('change', () => { state[key] = el.value.trim(); render(); });
  };
  bind('q', 'q');
  bind('year', 'year');
  bind('platform', 'platform');
  bind('model', 'model');
  bind('publisher', 'publisher');
  bind('venue', 'venue');

  document.querySelectorAll('[data-filter-chip]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const [key, val] = btn.dataset.filterChip.split(':');
      const group = key === 'status' ? 'status' : 'category';
      state[group] = state[group] === val ? '' : val;
      document.querySelectorAll(`[data-filter-chip^="${group}:"]`).forEach((b) => {
        b.classList.toggle('active', b.dataset.filterChip === `${group}:${state[group]}`);
      });
      render();
    });
  });

  document.getElementById('reset').addEventListener('click', () => {
    Object.keys(state).forEach((k) => state[k] = '');
    ['q','year','platform','model','publisher','venue'].forEach((id) => { document.getElementById(id).value = ''; });
    document.querySelectorAll('[data-filter-chip]').forEach((b) => b.classList.remove('active'));
    render();
  });

  function fillSelect(id, values) {
    const el = document.getElementById(id);
    (values || []).forEach((v) => {
      const opt = document.createElement('option');
      opt.value = v; opt.textContent = v; el.appendChild(opt);
    });
  }

  function matches(p) {
    if (state.category && p.category !== state.category) return false;
    if (state.year && String(p.year) !== state.year) return false;
    if (state.platform && p.platform !== state.platform) return false;
    if (state.model && !(p.model || []).includes(state.model)) return false;
    if (state.status && p.status !== state.status) return false;
    if (state.publisher && p.publisher !== state.publisher) return false;
    if (state.venue && p.venue_short !== state.venue) return false;
    if (state.q) {
      const hay = [
        p.title, p.authors_text, p.venue_full, p.venue_short, p.doi_text, p.doi, p.publisher,
        p.status_label, ...(p.keywords_paper || []), ...(p.keywords_meta || []), ...(p.model || [])
      ].join(' ').toLowerCase();
      if (!hay.includes(state.q.toLowerCase())) return false;
    }
    return true;
  }

  function escapeHtml(s) {
    return String(s ?? '')
      .replaceAll('&', '&amp;').replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;').replaceAll('"', '&quot;');
  }

  function tags(list, cls='') {
    return (list || []).map((t) => `<span class="tag ${cls}">${escapeHtml(t)}</span>`).join('');
  }

  function splitBadge(left, right, cls='') {
    const L = String(left || '').trim();
    const R = String(right || '').trim();
    if (!L && !R) return '';
    if (!R) return `<span class="mchip">${escapeHtml(L)}</span>`;
    if (!L) return `<span class="mchip">${escapeHtml(R)}</span>`;
    return `<span class="split-badge ${cls}"><span class="sb-left">${escapeHtml(L)}</span><span class="sb-right">${escapeHtml(R)}</span></span>`;
  }

  function formatAuthors(list) {
    const names = (list || []).map((n) => String(n || '').trim()).filter(Boolean);
    if (!names.length) return '';
    if (names.length <= 6) return names.join(', ');
    return `${names.slice(0, 5).join(', ')}, et al.`;
  }

  function cardHtml(p, i) {
    const href = p.url || p.doi || '#';
    const doiHtml = p.doi
      ? `<div class="doi-row"><a href="${escapeHtml(p.doi)}" target="_blank" rel="noopener">DOI</a><span class="doi-text">${escapeHtml(p.doi_text)}</span></div>`
      : '';
    const dl = p.download
      ? `<a class="download" href="${escapeHtml(p.download)}" target="_blank" rel="noopener">Download</a>`
      : `<span class="download disabled">No PDF</span>`;
    const authorsText = formatAuthors(p.authors);
    const authorsHtml = authorsText
      ? `<p class="authors">${escapeHtml(authorsText)}</p>`
      : '';
    const shortNeeded = p.venue_short && p.venue_short.toLowerCase() !== String(p.publisher || '').toLowerCase();
    const pubBadge = splitBadge(p.publisher, p.year);
    const delay = Math.min(i, 12) * 28;
    return `<article class="card" style="animation-delay:${delay}ms">
      <h3><a href="${escapeHtml(href)}" target="_blank" rel="noopener">${escapeHtml(p.title)}</a></h3>
      ${authorsHtml}
      <p class="venue-full">${escapeHtml(p.venue_full)}</p>
      <div class="meta-line">
        <span class="mchip cat" style="--cat:${escapeHtml(p.category_color)}">${escapeHtml(p.category_label)}</span>
        ${shortNeeded ? `<span class="mchip">${escapeHtml(p.venue_short)}</span>` : ''}
        ${pubBadge}
      </div>
      ${doiHtml}
      <div class="kw">
        <div><strong>Keywords</strong><div class="kw-row">${tags(p.keywords_paper) || '<span class="tag">—</span>'}</div></div>
        <div><strong>Meta</strong><div class="kw-row">${tags(p.keywords_meta, 'meta') || '<span class="tag">—</span>'}</div></div>
      </div>
      <div class="card-foot">
        <span class="status" style="--st:${escapeHtml(p.status_color)}"><i></i>${escapeHtml(p.status_label)}</span>
        ${dl}
      </div>
    </article>`;
  }

  function render() {
    const filtered = papers.filter(matches);
    document.getElementById('visible-count').textContent = String(filtered.length);
    document.getElementById('empty').classList.toggle('hidden', filtered.length > 0);
    const byYear = new Map();
    filtered.forEach((p) => {
      const y = String(p.year || 'Unknown');
      if (!byYear.has(y)) byYear.set(y, []);
      byYear.get(y).push(p);
    });
    document.getElementById('year-groups').innerHTML = [...byYear.entries()].map(([year, list]) => `
      <section class="year-block">
        <h2>${escapeHtml(year)} <span style="font-weight:500;font-size:.85rem;opacity:.7">(${list.length})</span></h2>
        <div class="cards">${list.map((p, i) => cardHtml(p, i)).join('')}</div>
      </section>
    `).join('');
  }

  render();
}

main().catch((err) => {
  document.getElementById('year-groups').innerHTML = `<p class="empty">Failed to load papers.json: ${err}</p>`;
});
"""


def build() -> None:
    cfg = load_yaml(ROOT / "configs" / "website.yaml") or {}
    out = ROOT / cfg.get("build", {}).get("out_dir", "website/dist")
    out.mkdir(parents=True, exist_ok=True)
    papers, cat_meta = load_site_papers(cfg)
    write_assets(out, cfg.get("site") or {}, papers, cat_meta)
    print(f"wrote {out}/index.html with {len(papers)} papers")


if __name__ == "__main__":
    build()
