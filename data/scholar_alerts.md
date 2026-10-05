# Google Scholar Alerts — Transformers on Silicon

Goal: catch BERT / GPT / LLM / Transformer / ViT hardware papers (ASIC, FPGA, PIM, accelerators).

Canonical query definitions live in [`configs/queries.yaml`](../configs/queries.yaml).

Canonical query definitions live in [`configs/queries.yaml`](../configs/queries.yaml).

## Current alerts (as configured)

1. `(BERT OR GPT OR LLMs OR Transformers) AND (Accelerator OR Overlay OR FPGA OR PIM OR "Systolic Array" OR "Parallelism" OR ...`
   Mode: **Most relevant results**

2. `(BERT OR GPT OR LLMs OR Transformer OR LVM) AND (Accelerator OR Overlay OR FPGA OR PIM OR "Systolic Array" OR "Parallelism)`
   Mode: **All results**
   Issue: missing closing `"` after `Parallelism`.

## Recommended replacements (paste into Scholar → Create alert)

### Alert A — Core (Most relevant)

```text
(BERT OR GPT OR LLM OR LLMs OR Transformer OR Transformers OR ViT OR "vision transformer") AND (Accelerator OR Accelerators OR Overlay OR FPGA OR ASIC OR PIM OR CIM OR "Systolic Array" OR "in-memory computing" OR VLSI OR "chip" OR "hardware accelerator")
```

### Alert B — Broader recall (All results)

```text
(BERT OR GPT OR LLM OR LLMs OR Transformer OR Transformers OR LVM OR VLM OR ViT) AND (Accelerator OR Overlay OR FPGA OR ASIC OR PIM OR CIM OR "Systolic Array" OR Parallelism OR "near-memory" OR "computing-in-memory" OR "RISC-V" OR Softmax OR "self-attention")
```

### Alert C — Silicon / chip venues (Most relevant) — optional 3rd alert

```text
(Transformer OR BERT OR LLM OR ViT OR GPT) AND (ASIC OR "solid-state circuits" OR JSSC OR ISSCC OR TVLSI OR "TOPS/W" OR tapeout OR "28 nm" OR "28nm" OR "chip implementation")
```

### Alert D — LLM accelerators only (Most relevant) — optional

```text
("large language model" OR LLM OR LLMs OR GPT OR MoE OR "mixture-of-experts") AND (Accelerator OR FPGA OR ASIC OR PIM OR CIM OR "KV cache" OR quantization OR sparsity)
```

## Fixes vs your current queries

| Issue | Fix |
|---|---|
| Unclosed `"Parallelism` | Use `Parallelism` or `"Systolic Array"` only for multi-word phrases |
| Trailing `OR ...` | Remove incomplete `OR` |
| `Transformers` only | Add singular `Transformer` (many titles use singular) |
| `LLMs` only | Add `LLM` |
| `LVM` | Keep if you want vision/language models; also add `VLM` / `ViT` |
| Missing ASIC / CIM | Add — common in your catalog (HiS-CIM, MorphAtt, HBQ, …) |
| `Parallelism` alone | Noisy; keep in Alert B only, drop from Alert A |

## Suggested config

| Alert | Results mode | Frequency |
|---|---|---|
| **A** | Most relevant | as new results appear |
| **B** | All results | as new results appear |
| **C** (optional) | Most relevant | if you want stronger silicon/chip signal |
| **D** (optional) | Most relevant | if LLM silicon is the main focus now |

## Light noise reduction (append only if mail is too noisy)

```text
-protein -peptide -sentiment -translation -chemistry -genome
```

Example (Alert A + excludes):

```text
(BERT OR GPT OR LLM OR LLMs OR Transformer OR Transformers OR ViT OR "vision transformer") AND (Accelerator OR Accelerators OR Overlay OR FPGA OR ASIC OR PIM OR CIM OR "Systolic Array" OR "in-memory computing" OR VLSI OR "hardware accelerator") -protein -peptide -sentiment -chemistry
```

## Keyword banks (for future edits)

**Models:** BERT, GPT, LLM, LLMs, Transformer, Transformers, ViT, VLM, LVM, MoE, DiT
**Hardware:** Accelerator, Overlay, FPGA, ASIC, PIM, CIM, Systolic Array, VLSI, RISC-V, chip, TOPS/W
**Methods:** Parallelism, Softmax, self-attention, KV cache, quantization, sparsity, HLS, co-design
