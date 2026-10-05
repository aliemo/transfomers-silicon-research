"""Analyze candidate papers with free AI providers + heuristic fallback."""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any

from common import ROOT, classify_venue, http_post_json, load_config, write_json
from review import default_review_fields

HARDWARE_TERMS = [
    "accelerator",
    "accelerators",
    "fpga",
    "asic",
    "vlsi",
    "cim",
    "pim",
    "systolic",
    "in-memory computing",
    "computing-in-memory",
    "near-memory",
    "hardware accelerator",
    "chip implementation",
    "tapeout",
    "tops/w",
    "risc-v",
    "overlay",
]

MODEL_TERMS = [
    "transformer",
    "transformers",
    "bert",
    "gpt",
    "llm",
    "large language model",
    "vision transformer",
    " vit ",
    "self-attention",
    "multi-head attention",
    "mixture-of-experts",
    "diffusion transformer",
]

EXCLUDE_TERMS = [
    "protein",
    "peptide",
    "genome",
    "sentiment",
    "machine translation",
    "clinical",
    "violence detection",
    "robotic navigation",
    "neutrino",
    "jailbreak",
    "fuzzing",
    "hdl code",
    "anomaly detection",
    "predictive maintenance",
    "timing closure",
    "agentic checkpoint",
]


def heuristic_analyze(item: dict[str, Any]) -> dict[str, Any]:
    title = str(item.get("title", "")).lower()
    abstract = str(item.get("abstract", "")).lower()
    text = f" {title} {abstract} "

    if any(t in text for t in EXCLUDE_TERMS):
        return {
            "related": False,
            "confidence": 0.9,
            "silicon": False,
            "platform": "__no_data__",
            "model": ["Transformer"],
            "publisher": item.get("publisher", "__no_data__"),
            "reason": "heuristic exclude-term",
            "provider": "heuristic",
        }

    hw_hits = [t for t in HARDWARE_TERMS if t in text]
    model_hits = [t for t in MODEL_TERMS if t in text]

    # Prefer title evidence for hardware+model; abstract alone is weak.
    title_hw = any(t in title for t in HARDWARE_TERMS)
    title_model = any(t in title for t in MODEL_TERMS)
    related = (title_hw and title_model) or (len(hw_hits) >= 1 and len(model_hits) >= 1 and title_hw)
    confidence = 0.0
    if related:
        confidence = 0.55
        if title_hw and title_model:
            confidence = 0.85
        confidence = min(1.0, confidence + 0.05 * (len(hw_hits) + len(model_hits)))

    platform = "__no_data__"
    for p in ("FPGA", "ASIC", "PIM", "GPU", "EDA"):
        keys = [p.lower()]
        if p == "PIM":
            keys += ["cim", "in-memory", "computing-in-memory", "near-memory"]
        if any(k in text for k in keys):
            platform = p
            break
    if platform == "__no_data__" and ("accelerator" in text or "hardware" in text):
        platform = "Design"

    models = []
    for tag, keys in [
        ("LLM", ["llm", "large language"]),
        ("BERT", ["bert"]),
        ("GPT", ["gpt"]),
        ("ViT", ["vit", "vision transformer"]),
        ("Transformer", ["transformer", "self-attention", "multi-head attention"]),
        ("MoE", ["moe", "mixture-of-experts"]),
        ("DiT", ["dit", "diffusion transformer"]),
    ]:
        if any(k in text for k in keys):
            models.append(tag)
    if not models:
        models = ["Transformer"]

    silicon: Any = "check"
    if any(x in text for x in ("asic", "chip implementation", "tapeout", "28nm", "jssc", "isscc", "tops/w")):
        silicon = True
    if not related:
        silicon = False

    return {
        "related": related,
        "confidence": round(confidence, 3),
        "silicon": silicon if related else False,
        "platform": platform if related else "__no_data__",
        "model": models,
        "publisher": item.get("publisher", "__no_data__"),
        "reason": f"heuristic hw={hw_hits[:3]} model={model_hits[:3]}",
        "provider": "heuristic",
    }


SYSTEM_PROMPT = """You classify research papers for a catalog about hardware/silicon implementations of Transformer/BERT/LLM/ViT models.
Return ONLY compact JSON with keys:
related (bool), confidence (0-1), silicon (true|false|"check"), platform (FPGA|ASIC|PIM|GPU|Design|EDA|framework|__no_data__),
model (array of short tags), publisher (short string), reason (one short sentence).
Mark related=true only if the paper is about accelerating, implementing, or co-designing Transformers/LLMs/ViT/BERT on hardware (ASIC/FPGA/CIM/PIM/chip/accelerator), or closely related silicon/EDA for those models.
Exclude pure NLP applications, biology/protein LMs, materials science, networking-only FPGA, and software-only surveys without hardware."""


def _extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None


def analyze_groq(item: dict[str, Any], model: str) -> dict[str, Any] | None:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return None
    payload = {
        "model": model,
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "title": item.get("title"),
                        "abstract": item.get("abstract", "")[:1500],
                        "venue": item.get("pubname"),
                        "year": item.get("year"),
                    }
                ),
            },
        ],
    }
    data = http_post_json(
        "https://api.groq.com/openai/v1/chat/completions",
        payload,
        headers={"Authorization": f"Bearer {key}"},
    )
    content = data["choices"][0]["message"]["content"]
    parsed = _extract_json(content)
    if not parsed:
        return None
    parsed["provider"] = "groq"
    return parsed


def analyze_gemini(item: dict[str, Any], model: str) -> dict[str, Any] | None:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        return None
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": SYSTEM_PROMPT
                        + "\n\nPaper:\n"
                        + json.dumps(
                            {
                                "title": item.get("title"),
                                "abstract": item.get("abstract", "")[:1500],
                                "venue": item.get("pubname"),
                                "year": item.get("year"),
                            }
                        )
                    }
                ]
            }
        ],
        "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"},
    }
    data = http_post_json(url, payload)
    content = data["candidates"][0]["content"]["parts"][0]["text"]
    parsed = _extract_json(content)
    if not parsed:
        return None
    parsed["provider"] = "gemini"
    return parsed


def analyze_one(item: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    a_cfg = cfg.get("analyze", {})
    providers = a_cfg.get("providers", ["groq", "gemini", "heuristic"])
    for name in providers:
        try:
            if name == "groq":
                got = analyze_groq(item, a_cfg.get("groq_model", "llama-3.1-8b-instant"))
            elif name == "gemini":
                got = analyze_gemini(item, a_cfg.get("gemini_model", "gemini-2.0-flash"))
            elif name == "heuristic":
                got = heuristic_analyze(item)
            else:
                continue
            if got:
                return got
        except Exception as e:
            print(f"[warn] provider {name} failed for '{item.get('title','')[:60]}': {e}")
    return heuristic_analyze(item)


def to_catalog_entry(item: dict[str, Any], analysis: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    cat = cfg.get("catalog", {})
    entry = {
        "title": item["title"],
        "year": item.get("year") or 0,
        "type": "article",
        "doi": item.get("doi") or "__no_data__",
        "url": item.get("url") or "__no_data__",
        "pdf": item.get("pdf") if item.get("pdf") else False,
        "ignore": cat.get("new_ignore", "check"),
        "silicon": analysis.get("silicon", cat.get("new_silicon_default", "check")),
        "platform": analysis.get("platform", "__no_data__"),
        "model": analysis.get("model") or ["Transformer"],
        "publisher": analysis.get("publisher") or item.get("publisher") or "__no_data__",
        "pubname": item.get("pubname") or "__no_data__",
        "authors": item.get("authors") or [],
        "_analysis": {
            "related": bool(analysis.get("related")),
            "confidence": analysis.get("confidence"),
            "reason": analysis.get("reason"),
            "provider": analysis.get("provider"),
            "source": item.get("source"),
        },
    }
    entry["category"] = classify_venue(entry, cfg.get("categories"))
    entry.update(default_review_fields())
    return entry


def analyze_items(items: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    threshold = float(cfg.get("analyze", {}).get("related_threshold", 0.55))
    accepted: list[dict[str, Any]] = []
    for item in items:
        analysis = analyze_one(item, cfg)
        entry = to_catalog_entry(item, analysis, cfg)
        related = bool(analysis.get("related"))
        conf = float(analysis.get("confidence") or 0)
        if related and conf >= threshold:
            accepted.append(entry)
        else:
            entry["_analysis"]["rejected"] = True
            # Keep rejected only in report, not catalog; skip append
    return accepted


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze candidate papers")
    parser.add_argument("-i", "--input", default="data/pipeline/candidates_raw.json")
    parser.add_argument("-o", "--output", default="data/pipeline/candidates_related.json")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    raw = json.loads((ROOT / args.input).read_text(encoding="utf-8"))
    items = raw["items"] if isinstance(raw, dict) else raw
    related = analyze_items(items, cfg)
    write_json(
        ROOT / args.output,
        {
            "count_in": len(items),
            "count_related": len(related),
            "items": related,
        },
    )
    print(f"related={len(related)}/{len(items)} -> {args.output}")


if __name__ == "__main__":
    main()
