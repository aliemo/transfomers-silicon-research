"""Enrich papers.yaml with author names via OpenAlex (DOI / arXiv / title)."""

from __future__ import annotations

import argparse
import re
import time
import urllib.parse
from typing import Any

from common import ROOT, format_paper_entry, http_json, load_papers_yaml, norm_title


def _doi_key(doi: str) -> str | None:
    s = str(doi or "").strip()
    if not s or s in {"__no_data__", "False", "None"}:
        return None
    s = re.sub(r"^https?://(dx\.)?doi\.org/", "", s, flags=re.I)
    if s.lower().startswith("https://arxiv.org/") or s.lower().startswith("http://arxiv.org/"):
        return s
    if re.match(r"10\.\d{4,9}/\S+", s):
        return s
    return None


def _authors_from_work(work: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for a in work.get("authorships") or []:
        name = ((a.get("author") or {}).get("display_name") or "").strip()
        if name and name not in out:
            out.append(name)
    return out


def _arxiv_id(paper: dict[str, Any]) -> str | None:
    for key in ("doi", "url", "pdf"):
        s = str(paper.get(key) or "")
        m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]+\.[0-9]+)(v\d+)?", s, flags=re.I)
        if m:
            return m.group(1)
    return None


def lookup_arxiv_authors(arxiv_id: str) -> list[str]:
    url = f"https://export.arxiv.org/api/query?id_list={urllib.parse.quote(arxiv_id)}"
    try:
        from common import http_get

        xml = http_get(url, headers={"Accept": "application/atom+xml"}).decode("utf-8", errors="replace")
    except Exception:
        return []
    block = re.search(r"<entry>(.*?)</entry>", xml, flags=re.S)
    if not block:
        return []
    names = re.findall(r"<name>(.*?)</name>", block.group(1), flags=re.S)
    out: list[str] = []
    for n in names:
        name = re.sub(r"\s+", " ", n).strip()
        if name and name not in out:
            out.append(name)
    return out


def lookup_authors(paper: dict[str, Any], mailto: str) -> list[str]:
    doi = _doi_key(paper.get("doi") or "")
    url = str(paper.get("url") or "")
    title = str(paper.get("title") or "").strip()

    candidates: list[str] = []
    if doi:
        if doi.lower().startswith("http"):
            candidates.append(f"https://api.openalex.org/works/{urllib.parse.quote(doi, safe='')}")
        else:
            candidates.append(f"https://api.openalex.org/works/doi:{urllib.parse.quote(doi, safe='')}")
    if "arxiv.org" in url.lower():
        candidates.append(f"https://api.openalex.org/works/{urllib.parse.quote(url, safe='')}")

    for endpoint in candidates:
        try:
            work = http_json(endpoint + ("&" if "?" in endpoint else "?") + f"mailto={urllib.parse.quote(mailto)}")
            authors = _authors_from_work(work)
            if authors:
                return authors
        except Exception:
            continue

    aid = _arxiv_id(paper)
    if aid:
        authors = lookup_arxiv_authors(aid)
        if authors:
            return authors

    if title:
        try:
            q = urllib.parse.urlencode(
                {
                    "search": title,
                    "per-page": "5",
                    "mailto": mailto,
                }
            )
            data = http_json(f"https://api.openalex.org/works?{q}")
            want = norm_title(title)
            for w in data.get("results") or []:
                if norm_title(w.get("display_name") or "") == want:
                    authors = _authors_from_work(w)
                    if authors:
                        return authors
            # fallback: first result if very close
            for w in data.get("results") or []:
                got = norm_title(w.get("display_name") or "")
                if want and (want in got or got in want):
                    authors = _authors_from_work(w)
                    if authors:
                        return authors
        except Exception:
            pass
    return []


def rewrite_catalog(papers: dict[int, dict[str, Any]], path) -> None:
    blocks = [format_paper_entry(idx, papers[idx]) for idx in sorted(papers)]
    path.write_text("".join(blocks).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich papers.yaml with authors from OpenAlex")
    parser.add_argument("--papers", default="data/papers.yaml")
    parser.add_argument("--mailto", default="transformers-silicon-research@users.noreply.github.com")
    parser.add_argument("--limit", type=int, default=0, help="Max papers to enrich (0=all)")
    parser.add_argument("--sleep", type=float, default=0.12)
    parser.add_argument("--force", action="store_true", help="Re-fetch even if authors exist")
    args = parser.parse_args()

    path = ROOT / args.papers
    papers = load_papers_yaml(path)
    updated = 0
    missing = 0
    checked = 0

    for idx in sorted(papers):
        p = papers[idx]
        existing = p.get("authors") or []
        if existing and not args.force:
            continue
        if args.limit and checked >= args.limit:
            break
        checked += 1
        authors = lookup_authors(p, args.mailto)
        time.sleep(args.sleep)
        if authors:
            p["authors"] = authors
            updated += 1
            print(f"[ok] {idx}: {len(authors)} authors — {authors[0]}{' et al.' if len(authors) > 1 else ''}", flush=True)
        else:
            p["authors"] = []
            missing += 1
            print(f"[miss] {idx}: {str(p.get('title', ''))[:70]}", flush=True)
        if checked % 25 == 0:
            rewrite_catalog(papers, path)
            print(f"[checkpoint] checked={checked} updated={updated}", flush=True)

    rewrite_catalog(papers, path)
    print(f"done checked={checked} updated={updated} missing={missing} -> {path}")


if __name__ == "__main__":
    main()
