"""Run the full papers auto-ingest pipeline."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

# Allow running as `python scripts/pipeline/run_pipeline.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_papers import analyze_items  # noqa: E402
from common import ROOT, load_config, write_json  # noqa: E402
from fetch_papers import fetch_all  # noqa: E402
from update_catalog import append_papers, regenerate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch → analyze → update catalog → regenerate README + website")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Do not modify papers.yaml / README")
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--candidates", default="data/pipeline/candidates_raw.json")
    parser.add_argument("--lookback-days", type=int, default=None, help="Recent-day window for weekly CI")
    parser.add_argument("--from-year", type=int, default=None)
    parser.add_argument("--to-year", type=int, default=None)
    parser.add_argument("--max-per-query", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    if args.lookback_days is not None:
        cfg.setdefault("fetch", {})["lookback_days"] = args.lookback_days
        if args.from_year is None:
            cfg["fetch"].pop("from_year", None)
    if args.from_year is not None:
        cfg.setdefault("fetch", {})["from_year"] = args.from_year
    if args.to_year is not None:
        cfg.setdefault("fetch", {})["to_year"] = args.to_year
    if args.max_per_query is not None:
        cfg.setdefault("fetch", {})["max_per_query"] = args.max_per_query

    out_dir = ROOT / "data" / "pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_fetch and (ROOT / args.candidates).exists():
        raw = json.loads((ROOT / args.candidates).read_text(encoding="utf-8"))
        items = raw["items"] if isinstance(raw, dict) else raw
    else:
        items = fetch_all(cfg)
        write_json(
            ROOT / args.candidates,
            {"fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(), "count": len(items), "items": items},
        )

    related = analyze_items(items, cfg)
    related_path = out_dir / "candidates_related.json"
    write_json(related_path, {"count_in": len(items), "count_related": len(related), "items": related})

    added = []
    if not args.dry_run:
        added = append_papers(ROOT / cfg["catalog"]["papers_yaml"], related)
        # Always regenerate derived artifacts so website/README stay in sync
        regenerate(cfg)

    report = {
        "ran_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        "fetched": len(items),
        "related": len(related),
        "added_count": len(added),
        "added_ids": [a["id"] for a in added],
        "lookback_days": cfg.get("fetch", {}).get("lookback_days"),
        "added": [
            {
                "id": a["id"],
                "title": a["title"],
                "doi": a.get("doi"),
                "platform": a.get("platform"),
                "silicon": a.get("silicon"),
                "analysis": a.get("_analysis"),
            }
            for a in added
        ],
        "ai_providers_configured": {
            "groq": bool(__import__("os").environ.get("GROQ_API_KEY")),
            "gemini": bool(__import__("os").environ.get("GEMINI_API_KEY") or __import__("os").environ.get("GOOGLE_API_KEY")),
        },
    }
    write_json(out_dir / "last_run_report.json", report)

    md = [
        f"# Pipeline run {report['ran_at']}",
        "",
        f"- fetched: **{report['fetched']}**",
        f"- related: **{report['related']}**",
        f"- added: **{report['added_count']}**",
        f"- lookback_days: `{report.get('lookback_days')}`",
        f"- dry_run: `{report['dry_run']}`",
        f"- groq: `{report['ai_providers_configured']['groq']}` gemini: `{report['ai_providers_configured']['gemini']}`",
        "",
    ]
    if added:
        md.append("## Added (Pass 1 / ignore: check — please finalize in admin)")
        md.append("")
        for a in added:
            md.append(f"- **{a['id']}** — {a['title']}")
    else:
        md.append("_No new related papers added._")
    (out_dir / "last_run_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps({k: report[k] for k in ("fetched", "related", "added_count", "added_ids", "dry_run", "lookback_days")}, indent=2))


if __name__ == "__main__":
    main()
