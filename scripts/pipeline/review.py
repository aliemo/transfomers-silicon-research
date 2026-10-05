"""Review-pass helpers (pass1 / pass2 / pass3)."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_review_config(path: Path | None = None) -> dict[str, Any]:
    p = path or (ROOT / "configs" / "review.yaml")
    with p.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def pass_map(cfg: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    cfg = cfg or load_review_config()
    return {p["id"]: p for p in cfg.get("passes", [])}


def default_review_fields(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_review_config()
    d = dict(cfg.get("defaults") or {})
    d.setdefault("review_pass", "pass1")
    d.setdefault("review_pass1", "pending")
    d.setdefault("review_pass2", "pending")
    d.setdefault("review_pass3", "pending")
    d.setdefault("review_notes", "")
    return d


def infer_review(paper: dict[str, Any], cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fill missing review_* fields from ignore flag / defaults."""
    cfg = cfg or load_review_config()
    out = default_review_fields(cfg)
    infer = cfg.get("infer") or {}
    ign = str(paper.get("ignore")).lower()
    if "review_pass" in paper or "review_pass1" in paper:
        # Prefer stored values
        for k in ("review_pass", "review_pass1", "review_pass2", "review_pass3", "review_notes"):
            if k in paper and paper[k] is not None:
                out[k] = paper[k]
        return out

    if ign == "check":
        out.update(infer.get("ignore_check") or {})
    elif ign == "true":
        out.update(infer.get("ignore_true") or {})
    elif ign == "false":
        out.update(infer.get("ignore_false") or {})
    return out


def attach_review(paper: dict[str, Any], cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_review_config()
    review = infer_review(paper, cfg)
    merged = dict(paper)
    merged.update(review)
    current = merged.get("review_pass") or "pass1"
    pmeta = pass_map(cfg).get(current) or {}
    merged["_review"] = {
        "current": current,
        "current_label": pmeta.get("label", current),
        "current_short": pmeta.get("short", current),
        "checklist": pmeta.get("checklist") or [],
        "description": pmeta.get("description") or "",
        "pass1": merged.get("review_pass1"),
        "pass2": merged.get("review_pass2"),
        "pass3": merged.get("review_pass3"),
        "notes": merged.get("review_notes") or "",
        "done": current == cfg.get("done_pass", "done"),
    }
    return merged


def apply_review_decision(
    paper: dict[str, Any],
    decision: str,
    notes: str | None = None,
    pass_id: str | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply accepted|rejected|skipped to the current (or specified) pass."""
    cfg = cfg or load_review_config()
    paper = attach_review(paper, cfg)
    decision = str(decision).lower().strip()
    if decision not in {"accepted", "rejected", "skipped"}:
        raise ValueError("decision must be accepted|rejected|skipped")

    current = pass_id or paper.get("review_pass") or "pass1"
    if current == cfg.get("done_pass", "done"):
        raise ValueError("review already done")

    pmeta = pass_map(cfg).get(current)
    if not pmeta:
        raise ValueError(f"unknown pass: {current}")

    updated = dict(paper)
    # drop helper
    updated.pop("_review", None)

    status_key = f"review_{current}"  # review_pass1 ...
    if status_key not in {"review_pass1", "review_pass2", "review_pass3"}:
        # allow only configured pass ids matching review_passN
        status_key = f"review_{current}"
    updated[status_key] = decision

    action = pmeta.get(f"on_{decision}") or {}
    for k, v in (action.get("set") or {}).items():
        updated[k] = v

    next_pass = action.get("next_pass") or cfg.get("done_pass", "done")
    updated["review_pass"] = next_pass

    if notes is not None:
        prev = str(updated.get("review_notes") or "").strip()
        stamp = f"[{current}/{decision}] {notes}".strip()
        updated["review_notes"] = (prev + "\n" + stamp).strip() if prev else stamp

    return updated


def review_summary(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_review_config()
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "statuses": cfg.get("statuses") or [],
        "passes": cfg.get("passes") or [],
        "done_pass": cfg.get("done_pass", "done"),
        "defaults": default_review_fields(cfg),
    }
