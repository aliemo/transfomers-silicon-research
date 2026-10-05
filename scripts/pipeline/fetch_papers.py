"""Fetch candidate papers from OpenAlex and arXiv (Scholar-query equivalents).

Google Scholar has no public API; these sources use the same query intent
from configs/queries.yaml across the full date range.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import time
import urllib.parse
from typing import Any

from common import ROOT, http_get, http_json, load_config, norm_title, write_json


def _year_from_date(s: str | None) -> int | None:
    if not s:
        return None
    m = re.match(r"(\d{4})", s)
    return int(m.group(1)) if m else None


def _openalex_authors(work: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for a in work.get("authorships") or []:
        name = ((a.get("author") or {}).get("display_name") or "").strip()
        if name and name not in out:
            out.append(name)
    return out


def _arxiv_authors(block: str) -> list[str]:
    names = re.findall(r"<name>(.*?)</name>", block, flags=re.S)
    out: list[str] = []
    for n in names:
        name = re.sub(r"\s+", " ", n).strip()
        if name and name not in out:
            out.append(name)
    return out


def fetch_openalex(
    query: str,
    *,
    from_year: int,
    to_year: int,
    max_results: int,
    mailto: str,
    per_page: int = 50,
    from_date: str | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cursor = "*"
    pages = 0
    max_pages = max(60, (max_results // 50) + 5)
    start = from_date or f"{from_year}-01-01"
    while len(out) < max_results and cursor:
        params = {
            "search": query,
            "filter": f"from_publication_date:{start},to_publication_date:{to_year}-12-31",
            "per-page": str(min(per_page, 50)),
            "sort": "publication_date:desc",
            "mailto": mailto,
            "cursor": cursor,
        }
        url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
        data = http_json(url)
        results = data.get("results") or []
        if not results:
            break
        for w in results:
            title = (w.get("display_name") or "").strip()
            if not title:
                continue
            doi = w.get("doi")
            doi_url = doi if doi and str(doi).startswith("http") else (f"https://doi.org/{doi}" if doi else None)
            land = w.get("primary_location") or {}
            landing = land.get("landing_page_url") or doi_url or w.get("id")
            pdf = land.get("pdf_url") or False
            host = ((land.get("source") or {}).get("display_name")) or "__no_data__"
            abstract = ""
            inv = w.get("abstract_inverted_index") or {}
            if inv:
                positions: list[tuple[int, str]] = []
                for word, idxs in inv.items():
                    for i in idxs:
                        positions.append((i, word))
                abstract = " ".join(w for _, w in sorted(positions))
            out.append(
                {
                    "source": "openalex",
                    "title": title,
                    "year": w.get("publication_year") or _year_from_date(w.get("publication_date")),
                    "doi": doi_url or "__no_data__",
                    "url": landing or "__no_data__",
                    "pdf": pdf if pdf else False,
                    "publisher": "OpenAlex",
                    "pubname": host,
                    "authors": _openalex_authors(w),
                    "abstract": abstract[:2000],
                    "openalex_id": w.get("id"),
                    "query": query[:120],
                }
            )
            if len(out) >= max_results:
                break
        cursor = (data.get("meta") or {}).get("next_cursor")
        pages += 1
        if pages > max_pages:
            break
        time.sleep(0.12)
    return out


def fetch_arxiv(terms: str, *, from_year: int, to_year: int, max_results: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    start = 0
    page = min(100, max_results)
    while len(out) < max_results:
        q = urllib.parse.quote(terms)
        url = (
            "https://export.arxiv.org/api/query?"
            f"search_query=all:{q}&start={start}&max_results={page}"
            "&sortBy=submittedDate&sortOrder=descending"
        )
        xml = http_get(url, headers={"Accept": "application/atom+xml"}).decode("utf-8", errors="replace")
        entries = re.findall(r"<entry>(.*?)</entry>", xml, flags=re.S)
        if not entries:
            break
        stop = False
        for block in entries:
            title = re.search(r"<title>(.*?)</title>", block, flags=re.S)
            summary = re.search(r"<summary>(.*?)</summary>", block, flags=re.S)
            published = re.search(r"<published>(.*?)</published>", block)
            id_tag = re.search(r"<id>(.*?)</id>", block)
            cats = re.findall(r'term="([^"]+)"', block)
            if not title or not id_tag:
                continue
            abs_url = id_tag.group(1).strip()
            aid = abs_url.rsplit("/", 1)[-1]
            aid = re.sub(r"v\d+$", "", aid)
            pub_dt = None
            if published:
                try:
                    pub_dt = dt.datetime.fromisoformat(published.group(1).replace("Z", "+00:00"))
                except ValueError:
                    pub_dt = None
            year = pub_dt.year if pub_dt else None
            if year is not None and year < from_year:
                stop = True
                break
            if year is not None and year > to_year:
                continue
            out.append(
                {
                    "source": "arxiv",
                    "title": re.sub(r"\s+", " ", title.group(1)).strip(),
                    "year": year or to_year,
                    "doi": f"https://arxiv.org/abs/{aid}",
                    "url": f"https://arxiv.org/abs/{aid}",
                    "pdf": f"https://arxiv.org/pdf/{aid}",
                    "publisher": "Arxiv",
                    "pubname": cats[0] if cats else "cs.AR",
                    "authors": _arxiv_authors(block),
                    "abstract": re.sub(r"\s+", " ", (summary.group(1) if summary else "")).strip()[:2000],
                    "arxiv_id": aid,
                    "query": terms[:120],
                }
            )
            if len(out) >= max_results:
                break
        if stop or len(entries) < page:
            break
        start += page
        time.sleep(0.3)
    return out


def dedupe(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for c in candidates:
        key = norm_title(c["title"])
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def fetch_all(cfg: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    cfg = cfg or load_config()
    fetch_cfg = cfg["fetch"]
    to_year = int(fetch_cfg.get("to_year") or dt.date.today().year)
    lookback = int(fetch_cfg.get("lookback_days", 21))
    if fetch_cfg.get("from_year") is not None:
        from_year = int(fetch_cfg["from_year"])
        from_date = f"{from_year}-01-01"
    else:
        start = dt.date.today() - dt.timedelta(days=lookback)
        from_year = start.year
        from_date = start.isoformat()
    max_per = int(fetch_cfg.get("max_per_query", 100))
    mailto = fetch_cfg.get("mailto", "transformers-silicon-research@users.noreply.github.com")
    sources = set(fetch_cfg.get("sources", ["openalex", "arxiv"]))

    all_items: list[dict[str, Any]] = []
    for q in cfg.get("queries", []):
        name = q.get("name") or q.get("id")
        print(f"[fetch] query={name} years={from_year}-{to_year} from_date={from_date}")
        if "openalex" in sources and q.get("openalex"):
            try:
                got = fetch_openalex(
                    q["openalex"],
                    from_year=from_year,
                    to_year=to_year,
                    max_results=max_per,
                    mailto=mailto,
                    from_date=from_date,
                )
                print(f"  openalex={len(got)}")
                all_items.extend(got)
            except Exception as e:
                print(f"[warn] openalex '{name}' failed: {e}")
        arxiv_q = q.get("arxiv") or q.get("arxiv_terms")
        if "arxiv" in sources and arxiv_q:
            try:
                got = fetch_arxiv(arxiv_q, from_year=from_year, to_year=to_year, max_results=max_per)
                print(f"  arxiv={len(got)}")
                all_items.extend(got)
            except Exception as e:
                print(f"[warn] arxiv '{name}' failed: {e}")
    return dedupe(all_items)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch candidate transformer-silicon papers")
    parser.add_argument("-o", "--output", default="data/pipeline/candidates_raw.json")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--from-year", type=int, default=None)
    parser.add_argument("--to-year", type=int, default=None)
    parser.add_argument("--max-per-query", type=int, default=None)
    parser.add_argument("--lookback-days", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    if args.from_year is not None:
        cfg.setdefault("fetch", {})["from_year"] = args.from_year
    if args.to_year is not None:
        cfg.setdefault("fetch", {})["to_year"] = args.to_year
    if args.max_per_query is not None:
        cfg.setdefault("fetch", {})["max_per_query"] = args.max_per_query
    if args.lookback_days is not None:
        cfg.setdefault("fetch", {})["lookback_days"] = args.lookback_days
        # Weekly mode: clear pinned from_year so lookback applies
        if args.from_year is None:
            cfg.setdefault("fetch", {}).pop("from_year", None)

    items = fetch_all(cfg)
    out = ROOT / args.output
    write_json(
        out,
        {
            "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "from_year": cfg["fetch"].get("from_year"),
            "to_year": cfg["fetch"].get("to_year") or dt.date.today().year,
            "count": len(items),
            "items": items,
            "note": "Fetched via OpenAlex/arXiv using Scholar-equivalent queries (Google Scholar has no public API).",
        },
    )
    print(f"fetched={len(items)} -> {out}")


if __name__ == "__main__":
    main()
