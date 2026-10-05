#!/usr/bin/env python
"""Local admin API for papers.yaml CRUD + per-paper analysis.

Bind: 127.0.0.1 only
Run:  python scripts/admin_server.py
Open: http://127.0.0.1:8787/admin/
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "pipeline"))

from analyze_papers import analyze_one, heuristic_analyze  # noqa: E402
from common import (  # noqa: E402
    ROOT as PIPELINE_ROOT,
    category_meta,
    classify_venue,
    format_paper_entry,
    load_categories,
    load_config,
    load_papers_yaml,
    load_yaml,
    norm_title,
)
from review import (  # noqa: E402
    apply_review_decision,
    attach_review,
    default_review_fields,
    load_review_config,
    review_summary,
)

assert PIPELINE_ROOT == ROOT

PAPERS_PATH = ROOT / "data" / "papers.yaml"
ADMIN_DIR = ROOT / "website" / "admin"
DIST_DIR = ROOT / "website" / "dist"


def parse_boolish(v: Any) -> Any:
    if v is True or v is False:
        return v
    s = str(v).strip()
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    return s  # e.g. check


def normalize_paper(payload: dict[str, Any], existing: dict[str, Any] | None = None) -> dict[str, Any]:
    base = dict(existing or {})
    allowed = {
        "title",
        "year",
        "type",
        "doi",
        "url",
        "pdf",
        "ignore",
        "silicon",
        "platform",
        "model",
        "publisher",
        "pubname",
        "authors",
        "category",
        "review_pass",
        "review_pass1",
        "review_pass2",
        "review_pass3",
        "review_notes",
    }
    for k, v in payload.items():
        if k in allowed:
            base[k] = v

    # Drop legacy unused keys if present
    for dead in ("method", "pubkey", "reserved"):
        base.pop(dead, None)

    base.setdefault("type", "article")
    base.setdefault("doi", "__no_data__")
    base.setdefault("url", "__no_data__")
    base.setdefault("pdf", False)
    base.setdefault("ignore", "check")
    base.setdefault("silicon", "check")
    base.setdefault("platform", "__no_data__")
    base.setdefault("publisher", "__no_data__")
    base.setdefault("pubname", "__no_data__")
    base.setdefault("authors", [])

    # Ensure review fields exist for newly touched papers
    for k, v in default_review_fields().items():
        base.setdefault(k, v)

    if "year" in base:
        try:
            base["year"] = int(base["year"])
        except Exception:
            base["year"] = 0

    if isinstance(base.get("model"), str):
        parts = [x.strip() for x in re.split(r"[,/|]", base["model"]) if x.strip()]
        base["model"] = parts or ["Transformer"]
    elif not isinstance(base.get("model"), list):
        base["model"] = ["Transformer"]

    if isinstance(base.get("authors"), str):
        base["authors"] = [x.strip() for x in re.split(r"[,;]", base["authors"]) if x.strip()]
    elif not isinstance(base.get("authors"), list):
        base["authors"] = []

    base["ignore"] = parse_boolish(base.get("ignore", "check"))
    base["silicon"] = parse_boolish(base.get("silicon", "check"))

    pdf = base.get("pdf", False)
    if pdf in (False, "False", "false", None, "", "__no_data__"):
        base["pdf"] = False
    else:
        base["pdf"] = str(pdf)

    if not base.get("category"):
        base["category"] = classify_venue(base)
    return base


def save_papers(papers: dict[int, dict[str, Any]]) -> None:
    """Rewrite papers.yaml in canonical entry format."""
    blocks = []
    for idx in sorted(papers.keys()):
        # Normalize review fields before write
        papers[idx] = normalize_paper({}, papers[idx])
        blocks.append(format_paper_entry(idx, papers[idx]))
    PAPERS_PATH.write_text("".join(blocks).rstrip() + "\n", encoding="utf-8")


def paper_view(pid: int, p: dict[str, Any]) -> dict[str, Any]:
    cats = load_categories()
    reviewed = attach_review(p)
    category = reviewed.get("category") or classify_venue(reviewed, cats)
    meta = category_meta(cats).get(category, {})
    return {
        "id": pid,
        "title": reviewed.get("title"),
        "year": reviewed.get("year"),
        "type": reviewed.get("type"),
        "doi": reviewed.get("doi"),
        "url": reviewed.get("url"),
        "pdf": reviewed.get("pdf"),
        "ignore": reviewed.get("ignore"),
        "silicon": reviewed.get("silicon"),
        "platform": reviewed.get("platform"),
        "model": reviewed.get("model"),
        "publisher": reviewed.get("publisher"),
        "pubname": reviewed.get("pubname"),
        "authors": reviewed.get("authors") or [],
        "category": category,
        "category_label": meta.get("label", category),
        "review_pass": reviewed.get("review_pass"),
        "review_pass1": reviewed.get("review_pass1"),
        "review_pass2": reviewed.get("review_pass2"),
        "review_pass3": reviewed.get("review_pass3"),
        "review_notes": reviewed.get("review_notes") or "",
        "review": reviewed.get("_review"),
    }


def list_papers(query: dict[str, list[str]]) -> list[dict[str, Any]]:
    papers = load_papers_yaml(PAPERS_PATH)
    q = (query.get("q") or [""])[0].strip().lower()
    ignore = (query.get("ignore") or [""])[0].strip()
    silicon = (query.get("silicon") or [""])[0].strip()
    category = (query.get("category") or [""])[0].strip()
    year = (query.get("year") or [""])[0].strip()
    review_pass = (query.get("review_pass") or [""])[0].strip()
    review_status = (query.get("review_status") or [""])[0].strip()

    out = []
    for pid, p in papers.items():
        view = paper_view(pid, p)
        if ignore and str(view["ignore"]) != ignore:
            continue
        if silicon and str(view["silicon"]) != silicon:
            continue
        if category and view["category"] != category:
            continue
        if year and str(view["year"]) != year:
            continue
        if review_pass and str(view.get("review_pass")) != review_pass:
            continue
        if review_status:
            # status of the current pass (or explicit passN via review_pass filter)
            cur = view.get("review_pass") or "pass1"
            key = f"review_{cur}" if cur.startswith("pass") else None
            cur_status = view.get(key) if key else None
            if str(cur_status) != review_status:
                continue
        if q:
            hay = " ".join(
                [
                    str(view.get("title") or ""),
                    str(view.get("pubname") or ""),
                    str(view.get("doi") or ""),
                    str(view.get("publisher") or ""),
                    " ".join(view.get("model") or []),
                    str(view.get("review_notes") or ""),
                ]
            ).lower()
            if q not in hay:
                continue
        out.append(view)
    out.sort(key=lambda x: (-int(x.get("year") or 0), str(x.get("title") or "").lower()))
    return out


def rebuild_site() -> dict[str, Any]:
    cmd = [sys.executable, str(ROOT / "website" / "build.py")]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    return {
        "ok": proc.returncode == 0,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "code": proc.returncode,
    }


class AdminHandler(SimpleHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[admin] " + (fmt % args) + "\n")

    def _send(self, code: int, payload: Any, content_type: str = "application/json") -> None:
        raw = payload if isinstance(payload, (bytes, bytearray)) else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        data = self.rfile.read(length)
        if not data:
            return {}
        return json.loads(data.decode("utf-8"))

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,PUT,DELETE,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/health":
            return self._send(200, {"ok": True})

        if path == "/api/meta":
            cats = category_meta()
            return self._send(
                200,
                {
                    "categories": list(cats.values()),
                    "fields": {
                        "ignore": ["False", "check", "True"],
                        "silicon": ["True", "check", "False"],
                        "type": ["article", "inproceedings", "thesis", "preprint"],
                    },
                    "review": review_summary(),
                    "papers_path": str(PAPERS_PATH.relative_to(ROOT)),
                },
            )

        if path == "/api/papers":
            qs = parse_qs(parsed.query)
            items = list_papers(qs)
            return self._send(200, {"count": len(items), "items": items})

        m = re.fullmatch(r"/api/papers/(\d+)", path)
        if m:
            pid = int(m.group(1))
            papers = load_papers_yaml(PAPERS_PATH)
            if pid not in papers:
                return self._send(404, {"error": "not found"})
            return self._send(200, paper_view(pid, papers[pid]))

        # Static admin + optional public dist
        if path == "/" or path == "/admin" or path == "/admin/":
            return self._serve_file(ADMIN_DIR / "index.html", "text/html")
        if path.startswith("/admin/"):
            rel = path[len("/admin/") :]
            return self._serve_file(ADMIN_DIR / rel)
        if path.startswith("/site/"):
            rel = path[len("/site/") :]
            return self._serve_file(DIST_DIR / (rel or "index.html"))

        return self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            body = self._read_json()
        except Exception:
            return self._send(400, {"error": "invalid json"})

        if path == "/api/papers":
            papers = load_papers_yaml(PAPERS_PATH)
            title = str(body.get("title") or "").strip()
            if not title:
                return self._send(400, {"error": "title required"})
            index = {norm_title(p.get("title", "")): i for i, p in papers.items()}
            if norm_title(title) in index:
                return self._send(409, {"error": "duplicate title", "id": index[norm_title(title)]})
            pid = (max(papers.keys()) + 1) if papers else 1
            entry = normalize_paper(body)
            papers[pid] = entry
            save_papers(papers)
            return self._send(201, paper_view(pid, entry))

        m = re.fullmatch(r"/api/papers/(\d+)/analyze", path)
        if m:
            pid = int(m.group(1))
            papers = load_papers_yaml(PAPERS_PATH)
            if pid not in papers:
                return self._send(404, {"error": "not found"})
            paper = papers[pid]
            item = {
                "title": paper.get("title"),
                "abstract": body.get("abstract", ""),
                "publisher": paper.get("publisher"),
                "pubname": paper.get("pubname"),
                "year": paper.get("year"),
                "doi": paper.get("doi"),
                "url": paper.get("url"),
            }
            cfg = load_config()
            try:
                analysis = analyze_one(item, cfg)
            except Exception:
                analysis = heuristic_analyze(item)
            suggested = {
                "related": analysis.get("related"),
                "confidence": analysis.get("confidence"),
                "reason": analysis.get("reason"),
                "provider": analysis.get("provider"),
                "silicon": analysis.get("silicon"),
                "platform": analysis.get("platform"),
                "model": analysis.get("model"),
                "publisher": analysis.get("publisher") or paper.get("publisher"),
                "category": classify_venue({**paper, **{k: analysis.get(k) for k in ("publisher", "platform")}}),
            }
            apply = bool(body.get("apply"))
            if apply:
                paper = normalize_paper(
                    {
                        "silicon": suggested["silicon"],
                        "platform": suggested["platform"],
                        "model": suggested["model"],
                        "category": suggested["category"],
                        "ignore": "False" if suggested.get("related") else "check",
                    },
                    existing=paper,
                )
                papers[pid] = paper
                save_papers(papers)
                return self._send(200, {"analysis": suggested, "paper": paper_view(pid, paper), "applied": True})
            return self._send(200, {"analysis": suggested, "paper": paper_view(pid, paper), "applied": False})

        m = re.fullmatch(r"/api/papers/(\d+)/review", path)
        if m:
            pid = int(m.group(1))
            papers = load_papers_yaml(PAPERS_PATH)
            if pid not in papers:
                return self._send(404, {"error": "not found"})
            decision = body.get("decision")
            notes = body.get("notes")
            pass_id = body.get("pass")
            try:
                updated = apply_review_decision(papers[pid], decision=decision, notes=notes, pass_id=pass_id)
            except ValueError as e:
                return self._send(400, {"error": str(e)})
            updated = normalize_paper(updated, existing=updated)
            papers[pid] = updated
            save_papers(papers)
            return self._send(200, {"paper": paper_view(pid, updated), "decision": decision})

        if path == "/api/rebuild":
            result = rebuild_site()
            code = 200 if result["ok"] else 500
            return self._send(code, result)

        return self._send(404, {"error": "not found"})

    def do_PUT(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        m = re.fullmatch(r"/api/papers/(\d+)", parsed.path)
        if not m:
            return self._send(404, {"error": "not found"})
        pid = int(m.group(1))
        try:
            body = self._read_json()
        except Exception:
            return self._send(400, {"error": "invalid json"})
        papers = load_papers_yaml(PAPERS_PATH)
        if pid not in papers:
            return self._send(404, {"error": "not found"})
        entry = normalize_paper(body, existing=papers[pid])
        # reclassify if venue fields changed and category not forced
        if "category" not in body or not body.get("category"):
            entry["category"] = classify_venue(entry)
        papers[pid] = entry
        save_papers(papers)
        return self._send(200, paper_view(pid, entry))

    def do_DELETE(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        m = re.fullmatch(r"/api/papers/(\d+)", parsed.path)
        if not m:
            return self._send(404, {"error": "not found"})
        pid = int(m.group(1))
        qs = parse_qs(parsed.query)
        hard = (qs.get("hard") or ["0"])[0] in {"1", "true", "True"}
        papers = load_papers_yaml(PAPERS_PATH)
        if pid not in papers:
            return self._send(404, {"error": "not found"})
        if hard:
            del papers[pid]
            save_papers(papers)
            return self._send(200, {"deleted": pid, "hard": True})
        papers[pid]["ignore"] = True
        save_papers(papers)
        return self._send(200, {"deleted": pid, "hard": False, "paper": paper_view(pid, papers[pid])})

    def _serve_file(self, path: Path, content_type: str | None = None) -> None:
        path = path.resolve()
        allowed_roots = [ADMIN_DIR.resolve(), DIST_DIR.resolve()]
        if not any(str(path).startswith(str(root)) for root in allowed_roots):
            return self._send(403, {"error": "forbidden"})
        if path.is_dir():
            path = path / "index.html"
        if not path.exists() or not path.is_file():
            return self._send(404, {"error": "file not found", "path": str(path)})
        data = path.read_bytes()
        if content_type is None:
            if path.suffix == ".html":
                content_type = "text/html"
            elif path.suffix == ".js":
                content_type = "application/javascript"
            elif path.suffix == ".css":
                content_type = "text/css"
            elif path.suffix == ".json":
                content_type = "application/json"
            else:
                content_type = "application/octet-stream"
        self._send(200, data, content_type=content_type)

    def handle_one_request(self) -> None:
        try:
            super().handle_one_request()
        except Exception:
            traceback.print_exc()
            try:
                self._send(500, {"error": "internal server error"})
            except Exception:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Local papers admin panel")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()

    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("Refusing to bind non-local host. Use 127.0.0.1")

    ADMIN_DIR.mkdir(parents=True, exist_ok=True)
    httpd = ThreadingHTTPServer((args.host, args.port), AdminHandler)
    print(f"Admin panel: http://{args.host}:{args.port}/admin/")
    print(f"API health:  http://{args.host}:{args.port}/api/health")
    print(f"Catalog:     {PAPERS_PATH}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
