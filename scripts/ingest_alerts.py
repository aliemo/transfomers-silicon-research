#!/usr/bin/env python
"""Ingest related Google Scholar alert papers into papers.yaml (ignore: check)."""

import json
import re
import ssl
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys_path_extra = ROOT / "scripts" / "pipeline"
import sys

sys.path.insert(0, str(sys_path_extra))
from common import format_paper_entry as _format_paper_entry  # noqa: E402
from review import default_review_fields  # noqa: E402

CTX = ssl._create_unverified_context()


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research-bot)"})
    with urllib.request.urlopen(req, timeout=30, context=CTX) as r:
        return r.read()


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(t).lower()).strip()


def crossref_by_title(title: str):
    q = urllib.parse.quote(title)
    url = f"https://api.crossref.org/works?query.bibliographic={q}&rows=5"
    try:
        data = json.loads(fetch(url).decode("utf-8"))
    except Exception as e:
        return None, str(e)
    items = data.get("message", {}).get("items", [])
    if not items:
        return None, "no items"
    nt = norm_title(title)
    best = None
    best_score = 0.0
    for it in items:
        titles = it.get("title") or []
        if not titles:
            continue
        ot = norm_title(titles[0])
        inter = len(set(nt.split()) & set(ot.split()))
        score = inter / max(1, len(set(nt.split())))
        if score > best_score:
            best_score = score
            best = it
    if best_score < 0.55:
        return None, f"low score {best_score:.2f}"
    return best, f"score {best_score:.2f}"


def arxiv_meta(aid: str) -> dict:
    data = fetch(f"https://export.arxiv.org/api/query?id_list={aid}").decode("utf-8")
    titles = re.findall(r"<title>(.*?)</title>", data, re.S)
    cats = re.findall(r'term="([^"]+)"', data)
    title = titles[1].strip().replace("\n", " ") if len(titles) > 1 else None
    primary = cats[0] if cats else "cs.AR"
    return {
        "title": title,
        "doi": f"https://arxiv.org/abs/{aid}",
        "url": f"https://arxiv.org/abs/{aid}",
        "pdf": f"https://arxiv.org/pdf/{aid}",
        "publisher": "Arxiv",
        "pubname": primary,
    }


RELATED = [
    dict(
        title="A Fully Reconfigurable and Programmable VLIW Architecture for Nonlinear Function Acceleration",
        year=2026,
        publisher="IEEE",
        pubname="IEEE Transactions on Very Large Scale Integration (VLSI) Systems",
        platform="ASIC",
        model=["Transformer", "ASIC"],
        silicon=True,
    ),
    dict(
        title="MorphAtt: A Neuromorphic Accelerator for Efficient Multi-Head Attention Processing in Spiking Vision Transformers",
        year=2026,
        arxiv="2609.33207",
        platform="ASIC",
        model=["ViT", "ASIC"],
        silicon=True,
    ),
    dict(
        title="From Acceleration to Accelerating Acceleration: Applications and Tools for Democratizing Hardware Design Using High-Level Synthesis",
        year=2026,
        publisher="Other",
        pubname="Georgia Tech Repository",
        platform="FPGA",
        model=["ViT", "FPGA"],
        silicon="check",
    ),
    dict(
        title="Defending Side-Channel Attacks in FPGA-based Convolutional Layers and Multi-Head Attention Layers with Channel-Level Parallelization",
        year=2026,
        publisher="ACM",
        pubname="ACM Transactions on Design Automation of Electronic Systems",
        platform="FPGA",
        model=["Transformer", "FPGA"],
        silicon=True,
    ),
    dict(
        title="HiS-CIM: A 72.65-TOPS/W Hetero-CIM-Based LLM Accelerator Exploiting Hierarchical Weight Compression and Multi-Granular Sparsity",
        year=2026,
        publisher="IEEE",
        pubname="IEEE Journal of Solid-State Circuits",
        platform="PIM",
        model=["LLM", "CIM"],
        silicon=True,
    ),
    dict(
        title="A Novel Projection-Wise Quantization Method and A Custom Accelerator Design for Efficient Sub-4-Bit Large Language Model Inference",
        year=2026,
        publisher="Other",
        pubname="eScholarship",
        platform="FPGA",
        model=["LLM", "FPGA"],
        silicon="check",
    ),
    dict(
        title="HeteroReason: Heterogeneous FPGA-GPU Acceleration for Disaggregated Speculative Reasoning",
        year=2026,
        publisher="ACM",
        pubname="MICRO 2026",
        platform="FPGA",
        model=["LLM", "FPGA"],
        silicon=True,
    ),
    dict(
        title="Reinforcement Learning-Guided High-Level Synthesis Optimization for Energy-Efficient Operator Fusion in Systolic Arrays: An Analytical Evaluation",
        year=2026,
        publisher="Other",
        pubname="Positive Sciences",
        platform="Design",
        model=["ViT", "Design"],
        silicon="check",
    ),
    dict(
        title="RVBSCop: A RISC-V-Based Coprocessor With 229.53 GOPS/W and 92.19 GOPS/mm2 for LLM Inference",
        year=2026,
        publisher="IEEE",
        pubname="IEEE Transactions on Very Large Scale Integration (VLSI) Systems",
        platform="ASIC",
        model=["LLM", "ASIC"],
        silicon=True,
    ),
    dict(
        title="A Comprehensive Analysis and Chip Implementation of Hybrid-Bonding-Based 3D-DRAM Process-Near-Memory Design for LLM Inference",
        year=2026,
        publisher="IEEE",
        pubname="IEEE Journal of Solid-State Circuits",
        platform="3D",
        model=["LLM", "PNM"],
        silicon=True,
    ),
    dict(
        title="GRADE-RTL: Evaluating LLM-Generated RTL Beyond Compilation",
        year=2026,
        arxiv="2609.25335",
        platform="EDA",
        model=["LLM", "EDA"],
        silicon="check",
    ),
    dict(
        title="LUT-DiT: A LUT-Based Diffusion Transformer Accelerator for Scalable LUT Inference",
        year=2026,
        publisher="IEEE",
        pubname="IEEE Transactions on Very Large Scale Integration (VLSI) Systems",
        platform="ASIC",
        model=["Transformer", "DiT"],
        silicon=True,
    ),
    dict(
        title="A Dual-Precision MX Datapath for Layer-Adaptive RLHF Quantization on FPGAs",
        year=2026,
        publisher="__no_data__",
        pubname="__no_data__",
        platform="FPGA",
        model=["LLM", "FPGA"],
        silicon=True,
    ),
    dict(
        title="Lockstep: Bit-Exact, Verifiable Transformer Inference at Parity Across Devices, for Dense and Mixture-of-Experts Models",
        year=2026,
        publisher="__no_data__",
        pubname="__no_data__",
        platform="framework",
        model=["Transformer", "MoE"],
        silicon="check",
    ),
    dict(
        title="Hermes: Adaptive Memory-Efficient Pipeline Inference for Large Models on Edge Devices",
        year=2026,
        publisher="ACM",
        pubname="ACM Transactions",
        platform="framework",
        model=["BERT", "GPT", "ViT"],
        silicon="check",
    ),
    dict(
        title="Can Agents Design Better Chips with a Higher Level Abstraction?",
        year=2026,
        arxiv="2609.21157",
        platform="EDA",
        model=["LLM", "EDA"],
        silicon="check",
    ),
    dict(
        title="ViT-CIM: A Computing-in-Memory based Vision Transformer Accelerator Featuring Multi-Head Self Attention and Token-Level Optimization",
        year=2026,
        publisher="Other",
        pubname="2026 5th International Conference",
        platform="PIM",
        model=["ViT", "CIM"],
        silicon=True,
    ),
    dict(
        title="HBQ: Hierarchical Scaling Block Quantization with Hardware-Efficiency-Aware Design for Accurate LLM Inference",
        year=2026,
        arxiv="2609.00450",
        platform="ASIC",
        model=["LLM", "ASIC"],
        silicon=True,
    ),
    dict(
        title="Memory-Efficient Acceleration for Emerging Applications via Hardware/Software Co-Design",
        year=2026,
        publisher="Other",
        pubname="Dissertation, UCF",
        platform="Design",
        model=["ViT", "ASIC"],
        silicon="check",
    ),
    dict(
        title="Cerium: A Multi-GPU Framework for Terabyte-Scale Encrypted Inference",
        year=2026,
        publisher="ACM",
        pubname="Proceedings of the ACM",
        platform="GPU",
        model=["LLM", "GPU"],
        silicon="check",
    ),
    dict(
        title="Silicon Photonics for Short-Reach Interconnects: Progress in High-Bandwidth and Energy-Efficient Optical I/O",
        year=2026,
        publisher="Wiley",
        pubname="Laser & Photonics Reviews",
        platform="Design",
        model=["LLM", "Photonics"],
        silicon="check",
    ),
]

EXCLUDED = [
    "THE EVOLUTION OF LARGE LANGUAGE MODEL ARCHITECTURES: FROM THE TRANSFORMER TO HYBRID MODELS (2017–2026)",
    "AVP-GPT2: Prompt-Conditioned Fine-Tuning of a GPT-2 Protein Language Model for Antiviral Peptide Identification",
    "SCALE: Simulation-Calibrated Amortized Learning for Energy Materials",
    "Evaluating the Role of Provider on Safety Alignment in Large Language Models",
    "Recurrent Looped Transformer",
    "Scalable Packet Tracking on FPGAs for Erasure-Coded RDMA over Lossy WANs",
    "18-2: Research on Lightweight Super-Resolution GAN Model on Low-Cost FPGA for Automotive Application",
    "The Athena Project–Energy-Sparing Agentic Microcomputer-Based Platform Advancing the Wheel of Reincarnation",
    "Performance Optimization and Scalability of LLM-Based Enterprise AI Under High-Load and Adversarial Conditions",
    "The Efficiency-Decentralization-Security Trilemma: A Co-Design Framework for Lightweight, Decentralized AI in Cyber-Physical Systems",
    "Transformer-Based Small Object Detection in UAV Aerial Imagery Using Edge Artificial Intelligence",
]


def format_entry(idx: int, value: dict) -> str:
    return _format_paper_entry(idx, value)


def main():
    yaml_path = ROOT / "data" / "papers.yaml"
    existing = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    existing_titles = {norm_title(p["title"]): k for k, p in existing.items()}
    next_id = max(existing.keys()) + 1

    candidates = []
    for p in RELATED:
        entry = {
            "title": p["title"],
            "year": p["year"],
            "type": "article",
            "doi": "__no_data__",
            "url": "__no_data__",
            "pdf": False,
            "ignore": "check",
            "silicon": p["silicon"],
            "platform": p["platform"],
            "model": p["model"],
            "publisher": p.get("publisher", "__no_data__"),
            "pubname": p.get("pubname", "__no_data__"),
            "authors": [],
            "category": "other",
        }
        entry.update(default_review_fields())
        if "arxiv" in p:
            meta = arxiv_meta(p["arxiv"])
            entry.update(
                {
                    "title": meta["title"] or p["title"],
                    "doi": meta["doi"],
                    "url": meta["url"],
                    "pdf": meta["pdf"],
                    "publisher": meta["publisher"],
                    "pubname": meta["pubname"],
                }
            )
            print("ARXIV", p["arxiv"], "OK")
        else:
            item, msg = crossref_by_title(p["title"])
            print("CROSSREF", p["title"][:70], "->", msg)
            if item:
                doi = item.get("DOI")
                if doi:
                    entry["doi"] = f"https://doi.org/{doi}"
                    entry["url"] = f"https://doi.org/{doi}"
                container = (item.get("container-title") or [None])[0]
                if container:
                    entry["pubname"] = container
                pub = item.get("publisher") or ""
                if "IEEE" in pub:
                    entry["publisher"] = "IEEE"
                elif "ACM" in pub:
                    entry["publisher"] = "ACM"
                elif "Elsevier" in pub:
                    entry["publisher"] = "Elsevier"
                elif pub:
                    entry["publisher"] = pub.split()[0]
                years = item.get("published-print") or item.get("published-online") or {}
                parts = years.get("date-parts", [[None]])[0]
                if parts and parts[0]:
                    entry["year"] = parts[0]
                if item.get("title"):
                    entry["title"] = item["title"][0]

        nt = norm_title(entry["title"])
        if nt in existing_titles:
            print("SKIP DUP", entry["title"][:80], "id=", existing_titles[nt])
            continue
        candidates.append(entry)

    # Append before the trailing template comment block if present
    text = yaml_path.read_text(encoding="utf-8")
    marker = "\n# 40x:"
    if marker in text:
        head, tail = text.split(marker, 1)
        tail = marker + tail
    else:
        head, tail = text.rstrip() + "\n\n", ""

    blocks = []
    start = next_id
    for i, entry in enumerate(candidates):
        idx = start + i
        existing[idx] = entry
        blocks.append(format_entry(idx, entry))

    new_text = head.rstrip() + "\n\n" + "".join(blocks)
    if tail:
        new_text = new_text.rstrip() + "\n\n" + tail.lstrip("\n")
    yaml_path.write_text(new_text, encoding="utf-8")

    review = {
        "source": "Unread Google Scholar Alert Papers.md.docx",
        "as_of": "2026-10-05",
        "added": [
            {
                "id": start + i,
                "title": c["title"],
                "doi": c["doi"],
                "ignore": c["ignore"],
                "silicon": c["silicon"],
                "platform": c["platform"],
                "model": c["model"],
            }
            for i, c in enumerate(candidates)
        ],
        "excluded": EXCLUDED,
        "notes": [
            "All new entries use ignore: check for manual finalization.",
            "silicon: check means hardware relevance is borderline or not a fabricated silicon chip.",
            "Missing DOI/url left as __no_data__ when Crossref/arXiv did not resolve.",
        ],
    }
    (ROOT / "data" / "alerts_review.json").write_text(json.dumps(review, indent=2), encoding="utf-8")

    md_lines = [
        "# Scholar Alerts Ingest Review (2026-10-05)",
        "",
        f"Added **{len(candidates)}** related papers to `data/papers.yaml` with `ignore: check`.",
        "",
        "## Added (please finalize)",
        "",
    ]
    for i, c in enumerate(candidates):
        md_lines.append(
            f"- **{start + i}** — {c['title']}  \n"
            f"  platform=`{c['platform']}` silicon=`{c['silicon']}` doi=`{c['doi']}`"
        )
    md_lines += ["", "## Excluded (not transformer-on-hardware focused)", ""]
    for t in EXCLUDED:
        md_lines.append(f"- {t}")
    (ROOT / "data" / "alerts_review.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"added={len(candidates)} next_id_start={start} last_id={start + len(candidates) - 1}")


if __name__ == "__main__":
    main()
