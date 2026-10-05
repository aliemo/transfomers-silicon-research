"""Merge analyzed papers into papers.yaml and regenerate README/CSV."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from common import (
    ROOT,
    existing_title_index,
    format_paper_entry,
    load_config,
    load_papers_yaml,
    norm_title,
    write_json,
)


def append_papers(papers_path: Path, new_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    papers = load_papers_yaml(papers_path)
    index = existing_title_index(papers)
    next_id = (max(papers.keys()) + 1) if papers else 1

    text = papers_path.read_text(encoding="utf-8")
    marker = "\n# 40x:"
    if marker in text:
        head, tail = text.split(marker, 1)
        tail = marker + tail
    else:
        head, tail = text.rstrip() + "\n\n", ""

    added: list[dict[str, Any]] = []
    blocks: list[str] = []
    for entry in new_entries:
        clean = {k: v for k, v in entry.items() if not k.startswith("_")}
        key = norm_title(clean["title"])
        if key in index:
            continue
        idx = next_id
        next_id += 1
        papers[idx] = clean
        index[key] = idx
        blocks.append(format_paper_entry(idx, clean))
        added.append({"id": idx, **clean, "_analysis": entry.get("_analysis")})

    if not blocks:
        return []

    new_text = head.rstrip() + "\n\n" + "".join(blocks)
    if tail:
        new_text = new_text.rstrip() + "\n\n" + tail.lstrip("\n")
    papers_path.write_text(new_text, encoding="utf-8")
    return added


def regenerate(cfg: dict[str, Any]) -> None:
    """Regenerate derived artifacts from papers.yaml (README, CSV, plot, website)."""
    cat = cfg["catalog"]
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "generator.py"),
        "-i",
        str(ROOT / cat["papers_yaml"]),
        "-o",
        str(ROOT / cat["readme"]),
        "-c",
        str(ROOT / cat["papers_csv"]),
        "-p",
        str(ROOT / cat["plot"]),
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    # Website is always generated from the same papers.yaml
    subprocess.run([sys.executable, str(ROOT / "website" / "build.py")], cwd=str(ROOT), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Update catalog from analyzed papers")
    parser.add_argument("-i", "--input", default="data/pipeline/candidates_related.json")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--no-generate", action="store_true")
    parser.add_argument("--report", default="data/pipeline/last_run_report.json")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    payload = json.loads((ROOT / args.input).read_text(encoding="utf-8"))
    items = payload.get("items", [])
    papers_path = ROOT / cfg["catalog"]["papers_yaml"]
    added = append_papers(papers_path, items)

    if added and not args.no_generate:
        regenerate(cfg)

    report = {
        "added_count": len(added),
        "added_ids": [a["id"] for a in added],
        "added": [{"id": a["id"], "title": a["title"], "doi": a.get("doi"), "analysis": a.get("_analysis")} for a in added],
    }
    write_json(ROOT / args.report, report)
    print(f"added={len(added)} ids={report['added_ids']}")


if __name__ == "__main__":
    main()
