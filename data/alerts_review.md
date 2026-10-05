# Scholar Alerts Ingest Review (2026-10-05)

Source: `Unread Google Scholar Alert Papers.md.docx`
Added **21** related papers to `data/papers.yaml` as IDs **445–465**, all with `ignore: check` for your final pass.

## How to finalize

For each new entry in `data/papers.yaml`:
- set `ignore: False` to keep, or `ignore: True` to drop from generated views
- set `silicon: True/False` if currently `check`
- fill any `__no_data__` DOI/url/publisher fields when you have them

## Added (related)

| ID | Title | Platform | silicon | DOI/URL |
|---|---|---|---|---|
| 445 | A Fully Reconfigurable and Programmable VLIW Architecture for Nonlinear Function Acceleration | ASIC | True | https://doi.org/10.1109/tvlsi.2026.3729817 |
| 446 | MorphAtt: Neuromorphic Accelerator for MHSA in Spiking Vision Transformers | ASIC | True | https://arxiv.org/abs/2609.33207 |
| 447 | From Acceleration to Accelerating Acceleration (HLS / FCCM) | FPGA | check | https://doi.org/10.1109/fccm68464.2026.00082 |
| 448 | Defending Side-Channel Attacks in FPGA CNN + Multi-Head Attention | FPGA | True | https://doi.org/10.1145/3847663 |
| 449 | HiS-CIM: Hetero-CIM LLM Accelerator (72.65 TOPS/W) | PIM | True | https://doi.org/10.1109/jssc.2026.3734350 |
| 450 | Projection-Wise Quantization + Accelerator for Sub-4-Bit LLM | FPGA | check | `__no_data__` |
| 451 | HeteroReason: FPGA-GPU Speculative LLM Reasoning | FPGA | True | `__no_data__` |
| 452 | RL-Guided HLS Operator Fusion for ViT Systolic Arrays | Design | check | `__no_data__` |
| 453 | RVBSCop: RISC-V Coprocessor for LLM Inference | ASIC | True | https://doi.org/10.1109/tvlsi.2026.3733137 |
| 454 | Hybrid-Bonding 3D-DRAM PNM for LLM Inference | 3D | True | https://doi.org/10.1109/jssc.2026.3733213 |
| 455 | GRADE-RTL: Evaluating LLM-Generated RTL | EDA | check | https://arxiv.org/abs/2609.25335 |
| 456 | LUT-DiT: LUT-Based Diffusion Transformer Accelerator | ASIC | True | https://doi.org/10.1109/tvlsi.2026.3732472 |
| 457 | Dual-Precision MX Datapath for RLHF Quantization on FPGAs | FPGA | True | `__no_data__` |
| 458 | Lockstep: Bit-Exact Verifiable Transformer/MoE Inference | framework | check | `__no_data__` |
| 459 | Hermes+: Adaptive Pipeline Inference on Edge Devices | framework | check | https://doi.org/10.1145/3845607 |
| 460 | Can Agents Design Better Chips with Higher-Level Abstraction? | EDA | check | https://arxiv.org/abs/2609.21157 |
| 461 | ViT-CIM: CIM Vision Transformer Accelerator | PIM | True | https://doi.org/10.1109/isset70600.2026.11682578 |
| 462 | HBQ: Hierarchical Scaling Block Quantization (ASIC LLM) | ASIC | True | https://arxiv.org/abs/2609.00450 |
| 463 | Memory-Efficient Acceleration via HW/SW Co-Design (UCF dissertation) | Design | check | `__no_data__` |
| 464 | Cerium: Multi-GPU Encrypted LLM Inference | GPU | check | https://doi.org/10.1145/3830418.3843864 |
| 465 | Silicon Photonics for Short-Reach Optical I/O | Design | check | https://doi.org/10.1002/lpor.71894 |

## Excluded (not transformer-on-hardware focused)

- THE EVOLUTION OF LARGE LANGUAGE MODEL ARCHITECTURES: FROM THE TRANSFORMER TO HYBRID MODELS (2017–2026)
- AVP-GPT2: Prompt-Conditioned Fine-Tuning of a GPT-2 Protein Language Model for Antiviral Peptide Identification
- SCALE: Simulation-Calibrated Amortized Learning for Energy Materials
- Evaluating the Role of Provider on Safety Alignment in Large Language Models
- Recurrent Looped Transformer
- Scalable Packet Tracking on FPGAs for Erasure-Coded RDMA over Lossy WANs
- 18-2: Research on Lightweight Super-Resolution GAN Model on Low-Cost FPGA for Automotive Application
- The Athena Project–Energy-Sparing Agentic Microcomputer-Based Platform Advancing the Wheel of Reincarnation
- Performance Optimization and Scalability of LLM-Based Enterprise AI Under High-Load and Adversarial Conditions
- The Efficiency-Decentralization-Security Trilemma: A Co-Design Framework for Lightweight, Decentralized AI in Cyber-Physical Systems
- Transformer-Based Small Object Detection in UAV Aerial Imagery Using Edge Artificial Intelligence

## Notes

- All new entries are staged with `ignore: check` so nothing is finalized until you edit them.
- `silicon: check` marks borderline / software-framework / EDA / dissertation cases.
- Helper script used: `scripts/ingest_alerts.py`
- Raw extract kept at `data/alerts_raw.txt`
