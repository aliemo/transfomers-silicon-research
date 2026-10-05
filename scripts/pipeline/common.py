"""Shared helpers for the papers auto-ingest pipeline."""

from __future__ import annotations

import json
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CTX = ssl.create_default_context()
try:
    import certifi

    CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    pass


def load_yaml(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load pipeline.yaml and attach queries/categories from configs/."""
    cfg_path = path or (ROOT / "configs" / "pipeline.yaml")
    # backward compatibility
    if not cfg_path.exists() and (ROOT / "data" / "pipeline_config.yaml").exists():
        cfg_path = ROOT / "data" / "pipeline_config.yaml"
    cfg = load_yaml(cfg_path) or {}

    paths = cfg.get("paths") or {}
    q_path = ROOT / paths.get("queries", "configs/queries.yaml")
    c_path = ROOT / paths.get("categories", "configs/categories.yaml")
    if q_path.exists():
        qcfg = load_yaml(q_path) or {}
        cfg["queries"] = [q for q in qcfg.get("queries", []) if q.get("enabled", True)]
        cfg["query_excludes"] = qcfg.get("excludes", {})
    if c_path.exists():
        cfg["categories"] = load_yaml(c_path) or {}
    return cfg


def load_categories(path: Path | None = None) -> dict[str, Any]:
    p = path or (ROOT / "configs" / "categories.yaml")
    return load_yaml(p) or {}


def classify_venue(paper: dict[str, Any], categories_cfg: dict[str, Any] | None = None) -> str:
    """Return category id: journal | conference | arxiv | thesis | other."""
    categories_cfg = categories_cfg or load_categories()
    publisher = str(paper.get("publisher") or "")
    overrides = categories_cfg.get("publisher_overrides") or {}
    for key, val in overrides.items():
        if publisher.lower() == str(key).lower() and val:
            return str(val)

    blob = " ".join(
        [
            publisher,
            str(paper.get("pubname") or ""),
            str(paper.get("doi") or ""),
            str(paper.get("url") or ""),
            str(paper.get("type") or ""),
        ]
    ).lower()

    for cat in categories_cfg.get("categories", []):
        cid = cat.get("id")
        if cid == "other":
            continue
        for term in cat.get("match_any") or []:
            if str(term).lower() in blob:
                return str(cid)
    return "other"


def category_meta(categories_cfg: dict[str, Any] | None = None) -> dict[str, dict[str, str]]:
    categories_cfg = categories_cfg or load_categories()
    out = {}
    for cat in categories_cfg.get("categories", []):
        out[cat["id"]] = {
            "id": cat["id"],
            "label": cat.get("label", cat["id"]),
            "description": cat.get("description", ""),
            "color": cat.get("color", "#9aa7b8"),
        }
    return out


def norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(title).lower()).strip()


def http_get(url: str, headers: dict[str, str] | None = None, timeout: int = 45) -> bytes:
    req_headers = {
        "User-Agent": "transformers-silicon-research/1.0 (open research catalog)",
        "Accept": "application/json",
    }
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, headers=req_headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=CTX) as resp:
            return resp.read()
    except urllib.error.URLError:
        insecure = ssl._create_unverified_context()
        with urllib.request.urlopen(req, timeout=timeout, context=insecure) as resp:
            return resp.read()


def http_json(url: str, headers: dict[str, str] | None = None) -> Any:
    return json.loads(http_get(url, headers=headers).decode("utf-8"))


def http_post_json(url: str, payload: dict[str, Any], headers: dict[str, str] | None = None) -> Any:
    data = json.dumps(payload).encode("utf-8")
    req_headers = {
        "User-Agent": "transformers-silicon-research/1.0",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, data=data, headers=req_headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60, context=CTX) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError:
        insecure = ssl._create_unverified_context()
        with urllib.request.urlopen(req, timeout=60, context=insecure) as resp:
            return json.loads(resp.read().decode("utf-8"))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_papers_yaml(path: Path) -> dict[int, dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return {int(k): v for k, v in raw.items()}


def existing_title_index(papers: dict[int, dict[str, Any]]) -> dict[str, int]:
    return {norm_title(p.get("title", "")): i for i, p in papers.items() if p.get("title")}


def yaml_bool(v: Any) -> str:
    if v is True:
        return "True"
    if v is False:
        return "False"
    return str(v)


def yaml_scalar(v: Any, *, default: str = "__no_data__") -> str:
    """Render a YAML scalar safely (quote risky values)."""
    if v is None:
        s = default
    else:
        s = str(v)
    if s == "" or s.lower() in {"true", "false", "null", "yes", "no"} or re.search(r'[:?#*&!|>%@`\'",\[\]\{\}]', s) or s.strip() != s:
        return '"' + s.replace('"', '\\"') + '"'
    return s


def format_authors(authors: Any) -> str:
    if not authors:
        return "[]"
    if isinstance(authors, str):
        authors = [a.strip() for a in authors.split(",") if a.strip()]
    cleaned = []
    for a in authors:
        name = str(a).strip().replace('"', "'")
        if name:
            cleaned.append(name)
    return "[" + ", ".join(f'"{n}"' for n in cleaned) + "]"


# Canonical catalog fields written to papers.yaml (single source of truth).
# Dropped: method, pubkey, reserved (unused / always empty or constant).
CANONICAL_FIELDS = (
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
)


def format_paper_entry(idx: int, value: dict[str, Any]) -> str:
    title = str(value["title"]).replace('"', '\\"')
    pdf = value.get("pdf", False)
    pdf_s = "False" if pdf is False else str(pdf)
    model = value.get("model", ["Transformer"])
    category = value.get("category") or "other"
    notes = str(value.get("review_notes") or "").replace('"', "'").replace("\n", " | ")
    lines = [
        f"{idx}:",
        f'  title: "{title}"',
        f"  year: {value.get('year', 0)}",
        f"  type: {yaml_scalar(value.get('type', 'article'))}",
        f"  doi: {yaml_scalar(value.get('doi', '__no_data__'))}",
        f"  url: {yaml_scalar(value.get('url', '__no_data__'))}",
        f"  pdf: {pdf_s}",
        f"  ignore: {yaml_bool(value.get('ignore', 'check'))}",
        f"  silicon: {yaml_bool(value.get('silicon', 'check'))}",
        f"  platform: {yaml_scalar(value.get('platform', '__no_data__'))}",
        f"  model: {model}",
        f"  publisher: {yaml_scalar(value.get('publisher', '__no_data__'))}",
        f'  pubname: {yaml_scalar(value.get("pubname", "__no_data__"))}',
        f"  authors: {format_authors(value.get('authors'))}",
        f"  category: {category}",
        f"  review_pass: {value.get('review_pass', 'pass1')}",
        f"  review_pass1: {value.get('review_pass1', 'pending')}",
        f"  review_pass2: {value.get('review_pass2', 'pending')}",
        f"  review_pass3: {value.get('review_pass3', 'pending')}",
        f'  review_notes: "{notes}"',
        "",
        "",
    ]
    return "\n".join(lines)


def shield_escape(text: str) -> str:
    """Escape text for shields.io badge labels."""
    s = str(text)
    s = s.replace("-", "--").replace("_", "__").replace(" ", "%20")
    s = re.sub(r"[^\w\-./%]+", "", s)
    return s or "n-a"
