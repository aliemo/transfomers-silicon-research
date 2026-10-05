#!/usr/bin/env python
"""Reformat papers.yaml using the canonical pipeline formatter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "pipeline"))

from common import format_paper_entry, load_papers_yaml  # noqa: E402
from review import attach_review  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize papers.yaml to canonical fields")
    parser.add_argument("-i", "--input", default="data/papers.yaml")
    parser.add_argument("-o", "--output", default="data/papers.yaml")
    args = parser.parse_args()

    path_in = ROOT / args.input
    path_out = ROOT / args.output
    papers = load_papers_yaml(path_in)
    blocks = []
    for idx in sorted(papers):
        p = attach_review(papers[idx])
        p.pop("_review", None)
        for dead in ("method", "pubkey", "reserved"):
            p.pop(dead, None)
        if not p.get("authors"):
            p["authors"] = []
        if not p.get("category"):
            from common import classify_venue

            p["category"] = classify_venue(p)
        blocks.append(format_paper_entry(idx, p))
    path_out.write_text("".join(blocks).rstrip() + "\n", encoding="utf-8")
    print(f"normalized {len(papers)} papers -> {path_out}")


if __name__ == "__main__":
    main()
