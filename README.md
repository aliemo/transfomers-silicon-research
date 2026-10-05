# Transformer Models Silicon Research

> **Curated research catalog of hardware implementations of Transformer, BERT, LLM, and ViT models**  
> ASIC / FPGA / CIM-PIM / accelerators / edge silicon

**Repository:** [github.com/aliemo/transfomers-silicon-research](https://github.com/aliemo/transfomers-silicon-research)

[![GitHub](https://img.shields.io/badge/GitHub-aliemo%2Ftransfomers--silicon--research-181717?logo=github&style=flat-square)](https://github.com/aliemo/transfomers-silicon-research)
[![Website](https://img.shields.io/badge/Website-GitHub%20Pages-0b6e56?style=flat-square)](https://aliemo.github.io/transfomers-silicon-research/)
[![Weekly CI](https://img.shields.io/badge/CI-weekly%20ingest-2088FF?logo=githubactions&style=flat-square)](https://github.com/aliemo/transfomers-silicon-research/actions/workflows/papers-pipeline.yml)
[![Pages](https://img.shields.io/badge/Pages-auto%20deploy-222?logo=github&style=flat-square)](https://github.com/aliemo/transfomers-silicon-research/actions/workflows/pages.yml)
[![License](https://img.shields.io/badge/License-see%20LICENSE-lightgrey?style=flat-square)](https://github.com/aliemo/transfomers-silicon-research/blob/main/LICENSE)

| | |
|---|---|
| **Live catalog** | [aliemo.github.io/transfomers-silicon-research](https://aliemo.github.io/transfomers-silicon-research/) |
| **Source of truth** | [`data/papers.yaml`](https://github.com/aliemo/transfomers-silicon-research/blob/main/data/papers.yaml) (~580 papers; public site hides ignored) |
| **Coverage** | Publication years **2018-2026** (and growing) |
| **Pipeline docs** | [`data/PIPELINE.md`](https://github.com/aliemo/transfomers-silicon-research/blob/main/data/PIPELINE.md) |

---

## Overview

This repository tracks **Transformers on silicon**: papers that map attention / BERT / LLM / ViT workloads onto real hardware - FPGAs, ASICs, processing-in-memory (CIM/PIM), and related accelerators.

**What you get:**

- A YAML catalog (`data/papers.yaml`) used as the **single source of truth**
- A generated [`README`](https://github.com/aliemo/transfomers-silicon-research/blob/main/README.md) paper list (below)
- A **filterable website** (search, category, Pass/silicon status, venue, year, ...)
- **Weekly GitHub Actions** ingest (OpenAlex + arXiv) that opens a review PR
- A **local admin** UI for Pass 1 / Pass 2 / Pass 3 curation

```text
data/papers.yaml
      |
      +-- scripts/generator.py   -> README.md + data/papers.csv + plot
      +-- website/build.py       -> website/dist -> GitHub Pages
```

---

## Website

**Browse:** https://aliemo.github.io/transfomers-silicon-research/

Filters include category (journal / conference / arXiv / ...), **Silicon / Non-silicon / Pass 1 / Pass 2 / Pass 3**, year, venue, publisher, platform, and model tags.

Rebuild locally:

```bash
python website/build.py
# open website/dist/index.html
```

GitHub Pages deploys automatically on relevant pushes to `main` (workflow: [pages.yml](https://github.com/aliemo/transfomers-silicon-research/blob/main/.github/workflows/pages.yml)).

---

## How to Contribute

1. Check whether the paper is already in [`data/papers.yaml`](https://github.com/aliemo/transfomers-silicon-research/blob/main/data/papers.yaml)
2. Add a new entry **or** fix metadata / review fields (`review_pass`, DOI, venue, platform, models, authors, ...)
3. Open a pull request against [this repository](https://github.com/aliemo/transfomers-silicon-research)

### Local admin (Pass 1 / 2 / 3)

```bash
pip install -r requirements.txt
python scripts/admin_server.py
# http://127.0.0.1:8787/admin/
```

| Pass | Purpose |
|---|---|
| **Pass 1** | Relevance - transformer/LLM/ViT on hardware? |
| **Pass 2** | Metadata - DOI, venue, platform, models, silicon |
| **Pass 3** | Final include (`ignore: False`) or drop |

Accept / Reject / Skip in admin updates `review_pass*` in `papers.yaml`. **Commit and push** after review so weekly CI PRs merge cleanly on top of your decisions.

### Weekly auto-ingest (CI/CD)

Every **Monday 06:00 UTC**, [Papers Auto-Ingest](https://github.com/aliemo/transfomers-silicon-research/actions/workflows/papers-pipeline.yml):

1. Searches recent OpenAlex + arXiv results (default lookback ~21 days)
2. Filters related hardware/transformer papers
3. Appends new entries with `ignore: check` and **Pass 1**
4. Regenerates README / CSV / website assets
5. Opens a PR (`auto/papers-ingest`) for you to merge

Optional secrets for stronger AI triage: `GROQ_API_KEY`, `GEMINI_API_KEY` (heuristic fallback always works).

Manual run: **Actions -> Papers Auto-Ingest -> Run workflow**.

Deep historical fetch (local):

```bash
python scripts/pipeline/run_pipeline.py --lookback-days 21
python scripts/pipeline/fetch_papers.py --from-year 2017 --to-year 2026 --max-per-query 200
```

---

## Repository layout

| Path | Role |
|---|---|
| `data/papers.yaml` | Catalog (edit this) |
| `configs/` | Queries, categories, pipeline, review, website |
| `scripts/pipeline/` | Fetch -> analyze -> update catalog |
| `scripts/admin_server.py` | Local CRUD + review UI |
| `website/` | Static site builder + admin UI assets |
| `.github/workflows/` | Weekly ingest + Pages deploy |

---

## Transformer and BERT Model

* BERT is a method of **pre-training language representations**, meaning that we **train a general-purpose *language understanding model*** on a large text corpus (like Wikipedia) and then use that model for downstream NLP tasks.

* BERT was created and **published in 2018 by Jacob Devlin and his colleagues from Google**. BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks.

<p align="center">
  <img src="./data/img/BERT-ARCH.png" width='480' />
</p>

* **BERT is a Transformer-based model.**
    * The architecture of BERT is similar to the original Transformer model, except that BERT has two separate Transformer models: one for the left-to-right direction (the "encoder") and one for the right-to-left direction (the "encoder").
    * The output of each model is the hidden state output by the final Transformer layer. The two models are pre-trained jointly on a large corpus of unlabeled text. The pre-training task is a simple and straightforward masked language modeling objective.
    * The pre-trained BERT model can then be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.

---
### Reference Papers

**1. Attention Is All You Need**

![](https://img.shields.io/badge/arXiv-2017-skyblue?colorstyle=plastic) [![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.1706.03762-sandybrown?style=flat-square?&style=plastic)](https://arxiv.org/abs/1706.03762) [![PDF-Download](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/1706.03762.pdf)

[![Code-Link](https://img.shields.io/badge/Code-PyTorch-red?style=plastic)](https://github.com/jadore801120/attention-is-all-you-need-pytorch) [![Code-Link](https://img.shields.io/badge/Code-TensorFlow-orange?style=plastic)](https://github.com/lsdefine/attention-is-all-you-need-keras)

<details>
<summary><img src="https://img.shields.io/badge/ABSTRACT-9575cd?&style=plastic"/></summary>
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 Englishto-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.
</details>



**2. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**

![](https://img.shields.io/badge/arXiv-2018-skyblue?colorstyle=plastic) [![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.1810.04805-sandybrown?style=flat-square?&style=plastic)](https://arxiv.org/abs/1810.04805) [![PDF-Download](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/1810.04805.pdf) [![Code-Link](https://img.shields.io/badge/Code-TensorFlow-orange?style=plastic)](https://github.com/google-research/bert) [![Code-Link](https://img.shields.io/badge/Code-PyTorch-red?style=plastic)](https://github.com/codertimo/BERT-pytorch)

<details>
<summary><img src="https://img.shields.io/badge/ABSTRACT-9575cd?&style=plastic"/></summary>
We introduce a new language representation model called BERT, which stands for
Bidirectional Encoder Representations from Transformers. Unlike recent language representation models (Peters et al., 2018a; Radford et al., 2018), BERT is designed to pretrain deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be finetuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial taskspecific architecture modifications.
<br>
BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute
improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).
</details>

---

# Hardware Research

Papers below are generated from [`data/papers.yaml`](https://github.com/aliemo/transfomers-silicon-research/blob/main/data/papers.yaml). Prefer the [filterable website](https://aliemo.github.io/transfomers-silicon-research/) for browsing.


### 2018
**Algorithm-Hardware Co-Design of Single Shot Detector for Fast Object Detection on FPGAs**

![](https://img.shields.io/badge/IEEE/ACM-2018-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3240765.3240775-sandybrown?style=flat-square)](https://doi.org/10.1145/3240765.3240775)


**SparseNN: An energy-efficient neural network accelerator exploiting input and output sparsity**

![](https://img.shields.io/badge/None-2018-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.23919/DATE.2018.8342010-sandybrown?style=flat-square)](https://doi.org/10.23919/DATE.2018.8342010)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://doi.org/10.23919/DATE.2018.8342010)

---
### 2019
**A Power Efficient Neural Network Implementation on Heterogeneous FPGA and GPU Devices**

![](https://img.shields.io/badge/IEEE-2019-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/IRI.2019.00040-sandybrown?style=flat-square)](https://doi.org/10.1109/IRI.2019.00040)


**A Simple and Effective Approach to Automatic Post-Editing with Transfer Learning**

![](https://img.shields.io/badge/arXiv-2019-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.1906.06253-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.1906.06253)


**An Evaluation of Transfer Learning for Classifying Sales Engagement Emails at Large Scale**

![](https://img.shields.io/badge/IEEE-2019-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/CCGRID.2019.00069-sandybrown?style=flat-square)](https://doi.org/10.1109/CCGRID.2019.00069)


**MAGNet: A Modular Accelerator Generator for Neural Networks**

![](https://img.shields.io/badge/IEEE-2019-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ICCAD45719.2019.8942127-sandybrown?style=flat-square)](https://doi.org/10.1109/ICCAD45719.2019.8942127)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://people.eecs.berkeley.edu/~ysshao/assets/papers/magnet2019-iccad.pdf)

**mRNA: Enabling Efficient Mapping Space Exploration for a Reconfiguration Neural Accelerator**

![](https://img.shields.io/badge/IEEE-2019-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ISPASS.2019.00040-sandybrown?style=flat-square)](https://doi.org/10.1109/ISPASS.2019.00040)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://bpb-us-w2.wpmucdn.com/sites.gatech.edu/dist/c/332/files/2019/02/mrna_ispass2019.pdf)

**Pre-trained bert-gru model for relation extraction**

![](https://img.shields.io/badge/ACM-2019-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3373509.3373533-sandybrown?style=flat-square)](https://doi.org/10.1145/3373509.3373533)


**Q8BERT: Quantized 8Bit BERT**

![](https://img.shields.io/badge/IEEE-2019-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/EMC2--NIPS53020.2019.00016-sandybrown?style=flat-square)](https://doi.org/10.1109/EMC2-NIPS53020.2019.00016)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/1910.06188.pdf)

**Structured pruning of a BERT-based question answering model**

![](https://img.shields.io/badge/Arxiv-2019-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/1910.06360-sandybrown?style=flat-square)](https://arxiv.org/abs/1910.06360)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/1910.06360)

**Structured pruning of large language models**

![](https://img.shields.io/badge/Arxiv-2019-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/1910.04732-sandybrown?style=flat-square)](https://arxiv.org/abs/1910.04732)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/1910.04732)

**Tinybert: Distilling bert for natural language understanding**

![](https://img.shields.io/badge/Arxiv-2019-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/1909.10351-sandybrown?style=flat-square)](https://arxiv.org/abs/1909.10351)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/1909.10351)

---
### 2020
**A Low-Cost Reconfigurable Nonlinear Core for Embedded DNN Applications**

![](https://img.shields.io/badge/IEEE-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ICFPT51103.2020.00014-sandybrown?style=flat-square)](https://doi.org/10.1109/ICFPT51103.2020.00014)


**A Multi-Neural Network Acceleration Architecture**

![](https://img.shields.io/badge/ACM/IEEE-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ISCA45697.2020.00081-sandybrown?style=flat-square)](https://doi.org/10.1109/ISCA45697.2020.00081)


**A Primer in BERTology: What We Know About How BERT Works**

![](https://img.shields.io/badge/MIt%20Press-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1162/tacl_a_00349-sandybrown?style=flat-square)](https://doi.org/10.1162/tacl_a_00349)


**A Reconfigurable DNN Training Accelerator on FPGA**

![](https://img.shields.io/badge/IEEE-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/SiPS50750.2020.9195234-sandybrown?style=flat-square)](https://doi.org/10.1109/SiPS50750.2020.9195234)


**A^3: Accelerating Attention Mechanisms in Neural Networks with Approximation**

![](https://img.shields.io/badge/IEEE-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/HPCA47549.2020.00035-sandybrown?style=flat-square)](https://doi.org/10.1109/HPCA47549.2020.00035)


**Emerging Neural Workloads and Their Impact on Hardware**

![](https://img.shields.io/badge/IEEE-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.23919/DATE48585.2020.9116435-sandybrown?style=flat-square)](https://doi.org/10.23919/DATE48585.2020.9116435)


**Accelerating event detection with DGCNN and FPGAS**

![](https://img.shields.io/badge/MDPI-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.3390/electronics9101666-sandybrown?style=flat-square)](https://doi.org/10.3390/electronics9101666)


**An Empirical Analysis of BERT Embedding for Automated Essay Scoring**

![](https://img.shields.io/badge/TheSAI-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.14569/ijacsa.2020.0111027-sandybrown?style=flat-square)](https://doi.org/10.14569/ijacsa.2020.0111027)


**An investigation on different underlying quantization schemes for pre-trained language models**

![](https://img.shields.io/badge/Springer-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1007/978--3--030--60450--9_29-sandybrown?style=flat-square)](https://doi.org/10.1007/978-3-030-60450-9_29)


**ATT: A Fault-Tolerant ReRAM Accelerator for Attention-based Neural Networks**

![](https://img.shields.io/badge/IEEE-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ICCD50377.2020.00047-sandybrown?style=flat-square)](https://doi.org/10.1109/ICCD50377.2020.00047)


**Binarybert: Pushing the limit of bert quantization**

![](https://img.shields.io/badge/arXiv-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2012.15701-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2012.15701)


**Capuchin: Tensor-based GPU Memory Management for Deep Learning**

![](https://img.shields.io/badge/ACM-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3373376.3378505-sandybrown?style=flat-square)](https://doi.org/10.1145/3373376.3378505)


**CATBERT: Context-Aware Tiny BERT for Detecting Social Engineering Emails**

![](https://img.shields.io/badge/arXiv-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2010.03484-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2010.03484)


**CatBERT: Context-Aware Tiny BERT for Detecting Targeted Social Engineering Emails**

![](https://img.shields.io/badge/arXiv-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2010.03484-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2010.03484)


**ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT**

![](https://img.shields.io/badge/ACM-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3397271.3401075-sandybrown?style=flat-square)](https://doi.org/10.1145/3397271.3401075)


**Combining Feature Selection Methods with BERT: An In-depth Experimental Study of Long Text Classification**

![](https://img.shields.io/badge/Springer-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1007/978--3--030--67537--0_34-sandybrown?style=flat-square)](https://doi.org/10.1007/978-3-030-67537-0_34)


**Comparison of Deep Learning Models and Various Text Pre-Processing Techniques for the Toxic Comments Classification**

![](https://img.shields.io/badge/MDPI-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.3390/app10238631-sandybrown?style=flat-square)](https://doi.org/10.3390/app10238631)


**Compressing BERT: Studying the Effects of Weight Pruning on Transfer Learning**

![](https://img.shields.io/badge/arXiv-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2002.08307-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2002.08307)


**Compression of deep learning models for NLP**

![](https://img.shields.io/badge/ACM-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3340531.3412171-sandybrown?style=flat-square)](https://doi.org/10.1145/3340531.3412171)


**Deep Learning Acceleration with Neuron-to-Memory Transformation**

![](https://img.shields.io/badge/IEEE-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/HPCA47549.2020.00011-sandybrown?style=flat-square)](https://doi.org/10.1109/HPCA47549.2020.00011)


**Earlybert: Efficient bert training via early-bird lottery tickets**

![](https://img.shields.io/badge/arXiv-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2101.00063-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2101.00063)


**Efficient algorithms and hardware for natural language processing**

![](https://img.shields.io/badge/MIT-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://hdl.handle.net/1721.1/127440-sandybrown?style=flat-square)](https://hdl.handle.net/1721.1/127440)


**Efficient transformer-based large scale language representations using hardware-friendly block structured pruning**

![](https://img.shields.io/badge/arXiv-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2009.08065-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2009.08065)


**FARM: A flexible accelerator for recurrent and memory augmented neural networks**

![](https://img.shields.io/badge/Springer-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1007/s11265--020--01555--w-sandybrown?style=flat-square)](https://doi.org/10.1007/s11265-020-01555-w)


**Fastformers: Highly efficient transformer models for natural language understanding**

![](https://img.shields.io/badge/arXiv-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2010.13382-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2010.13382)


**FTRANS: energy-efficient acceleration of transformers using FPGA**

![](https://img.shields.io/badge/ACM/IEEE-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3370748.3406567-sandybrown?style=flat-square)](https://doi.org/10.1145/3370748.3406567)


**Hardware accelerator for multi-head attention and position-wise feed-forward in the transformer**

![](https://img.shields.io/badge/IEEE-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/SOCC49529.2020.9524802-sandybrown?style=flat-square)](https://doi.org/10.1109/SOCC49529.2020.9524802)


**Improving Accuracy and Speeding Up Document Image Classification Through Parallel Systems**

![](https://img.shields.io/badge/Springer-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1007/978--3--030--50417--5_29-sandybrown?style=flat-square)](https://doi.org/10.1007/978-3-030-50417-5_29)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7302855/pdf/978-3-030-50417-5_Chapter_29.pdf)

**Improving post training neural quantization: Layer-wise calibration and integer programming**

![](https://img.shields.io/badge/arXiv-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2006.10518-sandybrown?style=flat-square)](https://arxiv.org/abs/2006.10518)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2006.10518.pdf)

**Integer quantization for deep learning inference: Principles and empirical evaluation**

![](https://img.shields.io/badge/arXiv-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2004.09602-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2004.09602)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2004.09602.pdf)

**Ladabert: Lightweight adaptation of bert through hybrid model compression**

![](https://img.shields.io/badge/arXiv-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2004.04124-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2004.04124)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2004.04124.pdf)

**Load What You Need: Smaller Versions of Multilingual BERT**

![](https://img.shields.io/badge/arXiv-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2010.05609-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2010.05609)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2010.05609.pdf)

**Look-Up Table based Energy Efficient Processing in Cache Support for Neural Network Acceleration**

![](https://img.shields.io/badge/IEEE/ACM-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/MICRO50266.2020.00020-sandybrown?style=flat-square)](https://doi.org/10.1109/MICRO50266.2020.00020)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://www.microarch.org/micro53/papers/738300a088.pdf)

**MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers**

![](https://img.shields.io/badge/ACM-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://dl.acm.org/doi/abs/10.5555/3495724.3496209-sandybrown?style=flat-square)](https://dl.acm.org/doi/abs/10.5555/3495724.3496209)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://dl.acm.org/doi/pdf/10.5555/3495724.3496209)

**Movement Pruning: Adaptive Sparsity by Fine-Tuning**

![](https://img.shields.io/badge/NeurIPS-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2005.07683-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2005.07683)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://proceedings.neurips.cc/paper/2020/file/eae15aabaa768ae4a5993a8a4f4fa6e4-Paper.pdf)

**MSP: an FPGA-specific mixed-scheme, multi-precision deep neural network quantization framework**

![](https://img.shields.io/badge/arXiv-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2009.07460-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2009.07460)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2009.07460.pdf)

**Poor Man's BERT: Smaller and Faster Transformer Models**

![](https://img.shields.io/badge/Arxiv-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2004.03844-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2004.03844)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2004.03844v1)

**PoWER-BERT: Accelerating BERT Inference via Progressive Word-vector Elimination**

![](https://img.shields.io/badge/PMLR-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://proceedings.mlr.press/v119/goyal20a.html-sandybrown?style=flat-square)](https://proceedings.mlr.press/v119/goyal20a.html)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](http://proceedings.mlr.press/v119/goyal20a/goyal20a.pdf)

**Pruning Redundant Mappings in Transformer Models via Spectral-Normalized Identity Prior**

![](https://img.shields.io/badge/Arxiv-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2010.01791-sandybrown?style=flat-square)](https://arxiv.org/abs/2010.01791)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2010.01791.pdf)

**Q-BERT: Hessian Based Ultra Low Precision Quantization of BERT**

![](https://img.shields.io/badge/AAAI-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1609/aaai.v34i05.6409-sandybrown?style=flat-square)](https://doi.org/10.1609/aaai.v34i05.6409)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://ojs.aaai.org/index.php/AAAI/article/view/6409/6265)

**ReTransformer: ReRAM-based processing-in-memory architecture for transformer acceleration**

![](https://img.shields.io/badge/ACM-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3400302.3415640-sandybrown?style=flat-square)](https://doi.org/10.1145/3400302.3415640)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://dl.acm.org/doi/pdf/10.1145/3400302.3415640)

**SqueezeBERT: What can computer vision teach NLP about efficient neural networks?**

![](https://img.shields.io/badge/Arxiv-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2006.11316-sandybrown?style=flat-square)](https://arxiv.org/abs/2006.11316)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2006.11316.pdf)

**TernaryBERT: Distillation-aware Ultra-low Bit BERT**

![](https://img.shields.io/badge/Arxiv-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2009.12812-sandybrown?style=flat-square)](https://arxiv.org/abs/2009.12812)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2009.12812)

**Training Large Neural Networks with Constant Memory using a New Execution Algorithm**

![](https://img.shields.io/badge/Arxiv-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2002.05645-sandybrown?style=flat-square)](https://arxiv.org/abs/2002.05645)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2002.05645.pdf)

**Ultron-AutoML: An open-source, distributed, scalable framework for efficient hyper-parameter optimization**

![](https://img.shields.io/badge/IEEE-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/BigData50022.2020.9378071-sandybrown?style=flat-square)](https://doi.org/10.1109/BigData50022.2020.9378071)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://ashish-gupta03.github.io/files/Ultron.pdf)

**Towards Fully 8-bit Integer Inference for the Transformer Model**

![](https://img.shields.io/badge/Arixv-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2009.08034-sandybrown?style=flat-square)](https://arxiv.org/abs/2009.08034)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2009.08034.pdf)

**TopicBERT for energy efficient document classification**

![](https://img.shields.io/badge/Arxiv-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2010.16407-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2010.16407)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2010.16407.pdf)

---
### 2021
**A Framework for Area-efficient Multi-task BERT Execution on ReRAM-based Accelerators**

![](https://img.shields.io/badge/IEEE/ACM-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ICCAD51958.2021.9643471-sandybrown?style=flat-square)](https://doi.org/10.1109/ICCAD51958.2021.9643471)


**A Full-Stack Search Technique for Domain Optimized Deep Learning Accelerators**

![](https://img.shields.io/badge/arXiv-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3503222.3507767-sandybrown?style=flat-square)](https://doi.org/10.1145/3503222.3507767)


**A Microcontroller is All You Need: Enabling Transformer Execution on Low-Power IoT Endnodes**

![](https://img.shields.io/badge/IEEE-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/COINS51742.2021.9524173-sandybrown?style=flat-square)](https://doi.org/10.1109/COINS51742.2021.9524173)


**A Quantitative Survey of Communication Optimizations in Distributed Deep Learning**

![](https://img.shields.io/badge/IEEE-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/MNET.011.2000530-sandybrown?style=flat-square)](https://doi.org/10.1109/MNET.011.2000530)


**A Study on Token Pruning for ColBERT**

![](https://img.shields.io/badge/arXiv-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2112.06540-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2112.06540)


**A White Paper on Neural Network Quantization**

![](https://img.shields.io/badge/arXiv-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2106.08295-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2106.08295)


**Accelerated Device Placement Optimization with Contrastive Learning**

![](https://img.shields.io/badge/ACM-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3472456.3472523-sandybrown?style=flat-square)](https://doi.org/10.1145/3472456.3472523)


**Accelerating bandwidth-bound deep learning inference with main-memory accelerators**

![](https://img.shields.io/badge/ACM-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3458817.3476146-sandybrown?style=flat-square)](https://doi.org/10.1145/3458817.3476146)


**Accelerating Emerging Neural Workloads**

![](https://img.shields.io/badge/Purdue%20University-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.25394/pgs.17139038.v1-sandybrown?style=flat-square)](https://doi.org/10.25394/pgs.17139038.v1)


**Accelerating Framework of Transformer by Hardware Design and Model Compression Co-Optimization**

![](https://img.shields.io/badge/IEEE/ACM-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ICCAD51958.2021.9643586-sandybrown?style=flat-square)](https://doi.org/10.1109/ICCAD51958.2021.9643586)


**Accelerating Transformer-based Deep Learning Models on FPGAs using Column Balanced Block Pruning**

![](https://img.shields.io/badge/IEEE-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ISQED51717.2021.9424344-sandybrown?style=flat-square)](https://doi.org/10.1109/ISQED51717.2021.9424344)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://wangshusen.github.io/papers/ISQED2021.pdf)

**Accommodating Transformer onto FPGA: Coupling the Balanced Model Compression and FPGA-Implementation Optimization**

![](https://img.shields.io/badge/ACM-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3453688.3461739-sandybrown?style=flat-square)](https://doi.org/10.1145/3453688.3461739)


**Adapting by pruning: A case study on BERT**

![](https://img.shields.io/badge/arXiv-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2105.03343-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2105.03343)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2105.03343.pdf)

**Adaptive Inference through Early-Exit Networks: Design, Challenges and Directions**

![](https://img.shields.io/badge/ACM-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3469116.3470012-sandybrown?style=flat-square)](https://doi.org/10.1145/3469116.3470012)


**Adaptive Spatio-Temporal Graph Enhanced Vision-Language Representation for Video QA**

![](https://img.shields.io/badge/IEEE-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/TIP.2021.3076556-sandybrown?style=flat-square)](https://doi.org/10.1109/TIP.2021.3076556)


**Algorithm-hardware Co-design of Attention Mechanism on FPGA Devices**

![](https://img.shields.io/badge/ACM-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3477002-sandybrown?style=flat-square)](https://doi.org/10.1145/3477002)


**Aquabolt-XL: Samsung HBM2-PIM with in-memory processing for ML accelerators and beyond**

![](https://img.shields.io/badge/IEEE-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/HCS52781.2021.9567191-sandybrown?style=flat-square)](https://doi.org/10.1109/HCS52781.2021.9567191)


**AUBER: Automated BERT regularization**

![](https://img.shields.io/badge/PlosOne-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1371/journal.pone.0253241-sandybrown?style=flat-square)](https://doi.org/10.1371/journal.pone.0253241)


**Automatic Mixed-Precision Quantization Search of BERT**

![](https://img.shields.io/badge/arXiv-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.24963/ijcai.2021/472-sandybrown?style=flat-square)](https://doi.org/10.24963/ijcai.2021/472)


**BERMo: What can BERT learn from ELMo?**

![](https://img.shields.io/badge/arXiv-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2110.15802-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2110.15802)


**BERT Model for Classification of Fake News using the Cloud Processing Capacity**

![](https://img.shields.io/badge/IEEE-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/R10--HTC53172.2021.9641632-sandybrown?style=flat-square)](https://doi.org/10.1109/R10-HTC53172.2021.9641632)


**Bertinho: Galician BERT representations**

![](https://img.shields.io/badge/arXiv-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2103.13799-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2103.13799)


**BERxiT: Early exiting for BERT with better fine-tuning and extension to regression**

![](https://img.shields.io/badge/ACL-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-http://dx.doi.org/10.18653/v1/2021.--eacl--main.8-sandybrown?style=flat-square)](http://dx.doi.org/10.18653/v1/2021.eacl-main.8)


**Beyond preserved accuracy: Evaluating loyalty and robustness of BERT compression**

![](https://img.shields.io/badge/arXiv-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2109.03228-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2109.03228)


**Binary Complex Neural Network Acceleration on FPGA : (Invited Paper)**

![](https://img.shields.io/badge/IEEE-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ASAP52443.2021.00021-sandybrown?style=flat-square)](https://doi.org/10.1109/ASAP52443.2021.00021)


**Biomedical Named Entity Recognition at Scale**

![](https://img.shields.io/badge/Springer-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1007/978--3--030--68763--2_48-sandybrown?style=flat-square)](https://doi.org/10.1007/978-3-030-68763-2_48)


**Block pruning for faster transformers**

![](https://img.shields.io/badge/arXiv-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2109.04838-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2109.04838)


**Compressing Large-Scale Transformer-Based Models: A Case Study on BERT**

![](https://img.shields.io/badge/MIT%20Press-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1162/tacl_a_00413-sandybrown?style=flat-square)](https://doi.org/10.1162/tacl_a_00413)


**DAP-BERT: Differentiable Architecture Pruning of BERT**

![](https://img.shields.io/badge/Springer-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1007/978--3--030--92185--9_30-sandybrown?style=flat-square)](https://doi.org/10.1007/978-3-030-92185-9_30)


**Demystifying BERT: Implications for Accelerator Design**

![](https://img.shields.io/badge/arXiv-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2104.08335-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2104.08335)


**Dynamic-TinyBERT: Boost TinyBERT's Inference Efficiency by Dynamic Sequence Length**

![](https://img.shields.io/badge/arXiv-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2111.09645-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2111.09645)


**EAGLE: Expedited Device Placement with Automatic Grouping for Large Models**

![](https://img.shields.io/badge/IEEE-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/IPDPS49936.2021.00068-sandybrown?style=flat-square)](https://doi.org/10.1109/IPDPS49936.2021.00068)


**EBERT: Efficient BERT Inference with Dynamic Structured Pruning**

![](https://img.shields.io/badge/ACL-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-http://dx.doi.org/10.18653/v1/2021.findings--acl.425-sandybrown?style=flat-square)](http://dx.doi.org/10.18653/v1/2021.findings-acl.425)


**EdgeBERT: Sentence-Level Energy Optimizations for Latency-Aware Multi-Task NLP Inference**

![](https://img.shields.io/badge/IEEE/ACM-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3466752.3480095-sandybrown?style=flat-square)](https://doi.org/10.1145/3466752.3480095)


**ELSA: Hardware-Software co-design for efficient, lightweight self-attention mechanism in neural networks**

![](https://img.shields.io/badge/IEEE-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ISCA52012.2021.00060-sandybrown?style=flat-square)](https://doi.org/10.1109/ISCA52012.2021.00060)


**Enabling energy-efficient DNN training on hybrid GPU-FPGA accelerators**

![](https://img.shields.io/badge/ACM-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3447818.3460371-sandybrown?style=flat-square)](https://doi.org/10.1145/3447818.3460371)


**Enabling One-Size-Fits-All Compilation Optimization for Inference Across Machine Learning Computers**

![](https://img.shields.io/badge/IEEE-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/TC.2021.3128266-sandybrown?style=flat-square)](https://doi.org/10.1109/TC.2021.3128266)


**Energy efficiency boost in the AI-infused POWER10 processor**

![](https://img.shields.io/badge/IEEE-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ISCA52012.2021.00012-sandybrown?style=flat-square)](https://doi.org/10.1109/ISCA52012.2021.00012)


**Fixed-point Quantization for Vision Transformer**

![](https://img.shields.io/badge/IEEE-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/CAC53003.2021.9728246-sandybrown?style=flat-square)](https://doi.org/10.1109/CAC53003.2021.9728246)


**FlexACC: A Programmable Accelerator with Application-Specific ISA for Flexible Deep Neural Network Inference**

![](https://img.shields.io/badge/IEEE-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ASAP52443.2021.00046-sandybrown?style=flat-square)](https://doi.org/10.1109/ASAP52443.2021.00046)


**Gemmini: Enabling systematic deep-learning architecture evaluation via full-stack integration**

![](https://img.shields.io/badge/IEEE-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/DAC18074.2021.9586216-sandybrown?style=flat-square)](https://doi.org/10.1109/DAC18074.2021.9586216)


**Gobo: Quantizing attention-based nlp models for low latency and energy efficient inference**

![](https://img.shields.io/badge/IEEE/ACM-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/MICRO50266.2020.00071-sandybrown?style=flat-square)](https://doi.org/10.1109/MICRO50266.2020.00071)


**Hardware Acceleration of Fully Quantized BERT for Efficient Natural Language Processing**

![](https://img.shields.io/badge/IEEE-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.23919/DATE51398.2021.9474043-sandybrown?style=flat-square)](https://doi.org/10.23919/DATE51398.2021.9474043)


**Hardware acceleration of sparse and irregular tensor computations of ml models: A survey and insights**

![](https://img.shields.io/badge/IEEE-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/JPROC.2021.3098483-sandybrown?style=flat-square)](https://doi.org/10.1109/JPROC.2021.3098483)


**HMC-TRAN: A Tensor-core Inspired Hierarchical Model Compression for Transformer-based DNNs on GPU**

![](https://img.shields.io/badge/ACM-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3453688.3461740-sandybrown?style=flat-square)](https://doi.org/10.1145/3453688.3461740)


**HoloFormer: Deep Compression of Pre-Trained Transforms via Unified Optimization of N: M Sparsity and Integer Quantization**

![](https://img.shields.io/badge/None-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-None-sandybrown?style=flat-square)](None)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://openreview.net/pdf?id=eAEcdRkcMHh)

**How Deep Learning Model Architecture and Software Stack Impacts Training Performance in the Cloud**

![](https://img.shields.io/badge/Springer-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/978--3--030--89385--9-sandybrown?style=flat-square)](https://doi.org/978-3-030-89385-9)


**How to Train BERT with an Academic Budget**

![](https://img.shields.io/badge/arXiv-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2104.07705-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2104.07705)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2104.07705.pdf)

**I-BERT: Integer-only BERT Quantization**

![](https://img.shields.io/badge/PMLR-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2101.01321-sandybrown?style=flat-square)](https://proceedings.mlr.press/v139/kim21d.html)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](http://proceedings.mlr.press/v139/kim21d/kim21d.pdf)

**Improving the efficiency of transformers for resource-constrained devices**

![](https://img.shields.io/badge/IEEE-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/DSD53832.2021.00074-sandybrown?style=flat-square)](https://doi.org/10.1109/DSD53832.2021.00074)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2106.16006.pdf)

**KAISA: An adaptive second-order optimizer framework for deep neural networks**

![](https://img.shields.io/badge/ACM-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3458817.3476152-sandybrown?style=flat-square)](https://doi.org/10.1145/3458817.3476152)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2107.01739.pdf)

**KDLSQ-BERT: A Quantized Bert Combining Knowledge Distillation with Learned Step Size Quantization**

![](https://img.shields.io/badge/arXiv-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2101.05938-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2101.05938)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2101.05938.pdf)

**Kunlun: A 14nm High-Performance AI Processor for Diversified Workloads**

![](https://img.shields.io/badge/IEEE-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ISSCC42613.2021.9366056-sandybrown?style=flat-square)](https://doi.org/10.1109/ISSCC42613.2021.9366056)


**Layerweaver: Maximizing Resource Utilization of Neural Processing Units via Layer-Wise Scheduling**

![](https://img.shields.io/badge/IEEE-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/HPCA51647.2021.00056-sandybrown?style=flat-square)](https://doi.org/10.1109/HPCA51647.2021.00056)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://taejunham.github.io/data/layerweaver_hpca21.pdf)

**Learning Light-Weight Translation Models from Deep Transformer**

![](https://img.shields.io/badge/AAAI-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1609/aaai.v35i15.17561-sandybrown?style=flat-square)](https://doi.org/10.1609/aaai.v35i15.17561)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://ojs.aaai.org/index.php/AAAI/article/view/17561/17368)

**M2M: Learning to Enhance Low-Light Image from Model to Mobile FPGA**

![](https://img.shields.io/badge/Springer-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1007/978--3--030--89029--2_22-sandybrown?style=flat-square)](https://doi.org/10.1007/978-3-030-89029-2_22)


**NAS-BERT: Task-Agnostic and Adaptive-Size BERT Compression with Neural Architecture Search**

![](https://img.shields.io/badge/ACM-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3447548.3467262-sandybrown?style=flat-square)](https://doi.org/10.1145/3447548.3467262)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2105.14444.pdf)

**NeuralScale: A RISC-V Based Neural Processor Boosting AI Inference in Clouds**

![](https://img.shields.io/badge/CARRV-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://carrv.github.io/2021/-sandybrown?style=flat-square)](https://carrv.github.io/2021/)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://carrv.github.io/2021/papers/CARRV2021_paper_67_Zhan.pdf)

**NLP-Fast: A Fast, Scalable, and Flexible System to Accelerate Large-Scale Heterogeneous NLP Models**

![](https://img.shields.io/badge/IEEE-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/PACT52795.2021.00013-sandybrown?style=flat-square)](https://doi.org/10.1109/PACT52795.2021.00013)


**NPE: An FPGA-based Overlay Processor for Natural Language Processing**

![](https://img.shields.io/badge/ACM/SIGDA-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3431920.3439477-sandybrown?style=flat-square)](https://doi.org/10.1145/3431920.3439477)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2104.06535.pdf)

**Predicting Efficiency/Effectiveness Trade-offs for Dense vs. Sparse Retrieval Strategy Selection**

![](https://img.shields.io/badge/ACM-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3459637.3482159-sandybrown?style=flat-square)](https://doi.org/10.1145/3459637.3482159)


**PTQ4ViT: Post-Training Quantization Framework for Vision Transformers with Twin Uniform Quantization**

![](https://img.shields.io/badge/Arxiv-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2111.12293-sandybrown?style=flat-square)](https://arxiv.org/abs/2111.12293)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2111.12293)

**Randomly Wired Network Based on RoBERTa and Dialog History Attention for Response Selection**

![](https://img.shields.io/badge/IEEE-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/TASLP.2021.3077119-sandybrown?style=flat-square)](https://doi.org/10.1109/TASLP.2021.3077119)


**Re2PIM: A Reconfigurable ReRAM-Based PIM Design for Variable-Sized Vector-Matrix Multiplication**

![](https://img.shields.io/badge/ACM-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3453688.3461494-sandybrown?style=flat-square)](https://doi.org/10.1145/3453688.3461494)


**RISC-VTF: RISC-V Based Extended Instruction Set for Transformer**

![](https://img.shields.io/badge/IEEE-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/SMC52423.2021.9658643-sandybrown?style=flat-square)](https://doi.org/10.1109/SMC52423.2021.9658643)


**RMSMP: A Novel Deep Neural Network Quantization Framework with Row-wise Mixed Schemes and Multiple Precisions**

![](https://img.shields.io/badge/None-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-xx-sandybrown?style=flat-square)](None)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://openaccess.thecvf.com/content/ICCV2021/papers/Chang_RMSMP_A_Novel_Deep_Neural_Network_Quantization_Framework_With_Row-Wise_ICCV_2021_paper.pdf)

**Sanger: A Co-Design Framework for Enabling Sparse Attention using Reconfigurable Architecture**

![](https://img.shields.io/badge/IEEE/ACM-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3466752.3480125-sandybrown?style=flat-square)](https://doi.org/10.1145/3466752.3480125)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://dl.acm.org/doi/pdf/10.1145/3466752.3480125)

**Simplified TinyBERT: Knowledge Distillation for Document Retrieval**

![](https://img.shields.io/badge/Springer-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1007/978--3--030--72240--1_21-sandybrown?style=flat-square)](https://doi.org/10.1007/978-3-030-72240-1_21)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2009.07531.pdf)

**SmaQ: Smart Quantization for DNN Training by Exploiting Value Clustering**

![](https://img.shields.io/badge/IEEE-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/LCA.2021.3108505-sandybrown?style=flat-square)](https://doi.org/10.1109/LCA.2021.3108505)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://hparch.gatech.edu/papers/nima_2021_cal.pdf)

**Softermax: Hardware/Software Co-Design of an Efficient Softmax for Transformers**

![](https://img.shields.io/badge/ACM/IEEE-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/DAC18074.2021.9586134-sandybrown?style=flat-square)](https://doi.org/10.1109/DAC18074.2021.9586134)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2103.09301.pdf)

**SpAtten: Efficient Sparse Attention Architecture with Cascade Token and Head Pruning**

![](https://img.shields.io/badge/IEEE-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/HPCA51647.2021.00018-sandybrown?style=flat-square)](https://doi.org/10.1109/HPCA51647.2021.00018)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2012.09852.pdf)

**SQuAT: Sharpness- and Quantization-Aware Training for BERT**

![](https://img.shields.io/badge/Arxiv-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2210.07171-sandybrown?style=flat-square)](https://arxiv.org/abs/2210.07171)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2210.07171.pdf)

**Stochastic precision ensemble: self-knowledge distillation for quantized deep neural networks**

![](https://img.shields.io/badge/AAAI-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1609/aaai.v35i8.16839-sandybrown?style=flat-square)](https://doi.org/10.1609/aaai.v35i8.16839)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://ojs.aaai.org/index.php/AAAI/article/view/16839/16646)

**Talos: A Weighted Speedup-Aware Device Placement of Deep Learning Models**

![](https://img.shields.io/badge/IEEE-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ASAP52443.2021.00023-sandybrown?style=flat-square)](https://doi.org/10.1109/ASAP52443.2021.00023)


**TR-BERT: Dynamic Token Reduction for Accelerating BERT Inference**

![](https://img.shields.io/badge/Arxiv-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2105.11618-sandybrown?style=flat-square)](https://arxiv.org/abs/2105.11618)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2105.11618.pdf)

**Training with Quantization Noise for Extreme Model Compression**

![](https://img.shields.io/badge/Arxiv-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2004.07320-sandybrown?style=flat-square)](https://arxiv.org/abs/2004.07320)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2004.07320.pdf)

**Transformer Acceleration with Dynamic Sparse Attention**

![](https://img.shields.io/badge/Arxiv-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2110.11299-sandybrown?style=flat-square)](https://arxiv.org/abs/2110.11299)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2110.11299)

**Understanding and Overcoming the Challenges of Efficient Transformer Quantization**

![](https://img.shields.io/badge/Arxiv-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2109.12948-sandybrown?style=flat-square)](https://arxiv.org/abs/2109.12948)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2109.12948.pdf)

**Vis-TOP: Visual Transformer Overlay Processor**

![](https://img.shields.io/badge/Arxiv-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2110.10957-sandybrown?style=flat-square)](https://arxiv.org/abs/2110.10957)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2110.10957.pdf)

**Elbert: Fast albert with confidence-window based early exit**

![](https://img.shields.io/badge/IEEE-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ICASSP39728.2021.9414572-sandybrown?style=flat-square)](https://doi.org/10.1109/ICASSP39728.2021.9414572)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2107.00175.pdf)

**Ghostbert: Generate more features with cheap operations for BERT**

![](https://img.shields.io/badge/ACL-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-http://dx.doi.org/10.18653/v1/2021.acl--long.509-sandybrown?style=flat-square)](http://dx.doi.org/10.18653/v1/2021.acl-long.509)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://aclanthology.org/2021.acl-long.509.pdf)

**ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning**

![](https://img.shields.io/badge/IEEE-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/TPAMI.2021.3095381-sandybrown?style=flat-square)](https://doi.org/10.1109/TPAMI.2021.3095381)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9477085)

**Prune once for all: Sparse pre-trained language models**

![](https://img.shields.io/badge/Arxiv-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2111.05754-sandybrown?style=flat-square)](https://arxiv.org/abs/2111.05754)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2111.05754.pdf)

**ROSITA: Refined BERT cOmpreSsion with InTegrAted techniques**

![](https://img.shields.io/badge/None-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1609/aaai.v35i10.17056-sandybrown?style=flat-square)](https://doi.org/10.1609/aaai.v35i10.17056)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://ojs.aaai.org/index.php/AAAI/article/download/17056/16863)

**VS-Quant: Per-vector Scaled Quantization for Accurate Low-Precision Neural Network Inference**

![](https://img.shields.io/badge/MLSys-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://proceedings.mlsys.org/paper_files/paper/2021-sandybrown?style=flat-square)](https://proceedings.mlsys.org/paper_files/paper/2021)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://proceedings.mlsys.org/paper_files/paper/2021/file/48a6431f04545e11919887748ec5cb52-Paper.pdf)

---
### 2022
**A 28nm 27.5TOPS/W Approximate-Computing-Based Transformer Processor with Asymptotic Sparsity Speculating and Out-of-Order Computing**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ISSCC42614.2022.9731686-sandybrown?style=flat-square)](https://doi.org/10.1109/ISSCC42614.2022.9731686)


**A 40nm 5.6TOPS/W 239GOPS/mm2 Self-Attention Processor with Sign Random Projection-based Approximation**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ESSCIRC55480.2022.9911343-sandybrown?style=flat-square)](https://doi.org/10.1109/ESSCIRC55480.2022.9911343)


**A Dual-Mode Similarity Search Accelerator based on Embedding Compression for Online Cross-Modal Image-Text Retrieval**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/FCCM53951.2022.9786159-sandybrown?style=flat-square)](https://doi.org/10.1109/FCCM53951.2022.9786159)


**A Fast and Flexible FPGA-based Accelerator for Natural Language Processing Neural Networks**

![](https://img.shields.io/badge/ACM-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3564606-sandybrown?style=flat-square)](https://doi.org/10.1145/3564606)


**A Fast Post-Training Pruning Framework for Transformers**

![](https://img.shields.io/badge/arXiv-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2204.09656-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2204.09656)


**A Framework for Accelerating Transformer-Based Language Model on ReRAM-Based Architecture**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/TCAD.2021.3121264-sandybrown?style=flat-square)](https://doi.org/10.1109/TCAD.2021.3121264)


**A length adaptive algorithm-hardware co-design of transformer on FPGA through sparse attention and dynamic pipelining**

![](https://img.shields.io/badge/ACM/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3489517.3530585-sandybrown?style=flat-square)](https://doi.org/10.1145/3489517.3530585)


**A Lite Romanian BERT: ALR-BERT**

![](https://img.shields.io/badge/MDPI-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.3390/computers11040057-sandybrown?style=flat-square)](https://doi.org/10.3390/computers11040057)


**A Resource-Saving Energy-Efficient Reconfigurable Hardware Accelerator for BERT-based Deep Neural Network Language Models using FFT Multiplication**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ISCAS48785.2022.9937531-sandybrown?style=flat-square)](https://doi.org/10.1109/ISCAS48785.2022.9937531)


**A Self-Attention Network for Deep JSCCM: The Design and FPGA Implementation**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/GLOBECOM48099.2022.10001518-sandybrown?style=flat-square)](https://doi.org/10.1109/GLOBECOM48099.2022.10001518)


**Accelerating attention mechanism on fpgas based on efficient reconfigurable systolic array**

![](https://img.shields.io/badge/ACM-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3549937-sandybrown?style=flat-square)](https://doi.org/10.1145/3549937)


**Accelerating attention through gradient-based learned runtime pruning**

![](https://img.shields.io/badge/ACM-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3470496.3527423-sandybrown?style=flat-square)](https://doi.org/10.1145/3470496.3527423)


**Accelerating NLP Tasks on FPGA with Compressed BERT and a Hardware-Oriented Early Exit Method**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ISVLSI54635.2022.00092-sandybrown?style=flat-square)](https://doi.org/10.1109/ISVLSI54635.2022.00092)


**Accelerating Transformer Networks through Recomposing Softmax Layers**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/IISWC55918.2022.00018-sandybrown?style=flat-square)](https://doi.org/10.1109/IISWC55918.2022.00018)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](http://scale.snu.ac.kr/papers/2022-11-Conference-IISWC-Softmax-recomposition.pdf)

**Achieving the Performance of All-Bank In-DRAM PIM With Standard Memory Interface: Memory-Computation Decoupling**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ACCESS.2022.3203051-sandybrown?style=flat-square)](https://doi.org/10.1109/ACCESS.2022.3203051)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9870805)

**Adaptable Butterfly Accelerator for Attention-based NNs via Hardware and Algorithm Co-design**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/MICRO56248.2022.00050-sandybrown?style=flat-square)](https://doi.org/10.1109/MICRO56248.2022.00050)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2209.09570.pdf)

**AlphaTuning: Quantization-Aware Parameter-Efficient Adaptation of Large-Scale Pre-Trained Language Models**

![](https://img.shields.io/badge/arXiv-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2210.03858-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2210.03858)


**Alternative non-BERT model choices for the textual classification in low-resource languages and environments**

![](https://img.shields.io/badge/ACL-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-http://dx.doi.org/10.18653/v1/2022.deeplo--1.20-sandybrown?style=flat-square)](http://dx.doi.org/10.18653/v1/2022.deeplo--1.20)


**An Algorithm-Hardware Co-Optimized Framework for Accelerating N:M Sparse Transformers**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/TVLSI.2022.3197282-sandybrown?style=flat-square)](https://doi.org/10.1109/TVLSI.2022.3197282)


**An Automatic and Efficient BERT Pruning for Edge AI Systems**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ISQED54688.2022.9806197-sandybrown?style=flat-square)](https://doi.org/10.1109/ISQED54688.2022.9806197)


**An Efficient Hardware Accelerator for Sparse Transformer Neural Networks**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ISCAS48785.2022.9937659-sandybrown?style=flat-square)](https://doi.org/10.1109/ISCAS48785.2022.9937659)


**An Energy-Efficient Transformer Processor Exploiting Dynamic Weak Relevances in Global Attention**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/JSSC.2022.3213521-sandybrown?style=flat-square)](https://doi.org/10.1109/JSSC.2022.3213521)


**An FPGA-Based Transformer Accelerator Using Output Block Stationary Dataflow for Object Recognition Applications**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/TCSII.2022.3196055-sandybrown?style=flat-square)](https://doi.org/10.1109/TCSII.2022.3196055)


**Analog-memory-based 14nm Hardware Accelerator for Dense Deep Neural Networks including Transformers**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ISCAS48785.2022.9937292-sandybrown?style=flat-square)](https://doi.org/10.1109/ISCAS48785.2022.9937292)


**Answer Fast: Accelerating BERT on the Tensor Streaming Processor**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ASAP54787.2022.00022-sandybrown?style=flat-square)](https://doi.org/10.1109/ASAP54787.2022.00022)


**ANT: Exploiting Adaptive Numerical Data Type for Low-bit Deep Neural Network Quantization**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/MICRO56248.2022.00095-sandybrown?style=flat-square)](https://doi.org/10.1109/MICRO56248.2022.00095)


**APT: The master-copy-free training method for quantised neural network on edge devices**

![](https://img.shields.io/badge/Elsevier-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1016/j.jpdc.2022.04.005-sandybrown?style=flat-square)](https://doi.org/10.1016/j.jpdc.2022.04.005)


**Auto-ViT-Acc: An FPGA-Aware Automatic Acceleration Framework for Vision Transformer with Mixed-Scheme Quantization**

![](https://img.shields.io/badge/arXiv-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2208.05163-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2208.05163)


**Balance Multi-Head Attention based on Software and Hardware Co-design**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/CSCloud--EdgeCom54986.2022.00018-sandybrown?style=flat-square)](https://doi.org/10.1109/CSCloud-EdgeCom54986.2022.00018)


**BEBERT: Efficient and robust binary ensemble BERT**

![](https://img.shields.io/badge/arXiv-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2210.15976-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2210.15976)


**BERT model optimization methods for inference: a comparative study of five alternative BERT-model implementations**

![](https://img.shields.io/badge/LUT%20University-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://urn.fi/URN:NBN:fi--fe2022121270782-sandybrown?style=flat-square)](https://urn.fi/URN:NBN:fi-fe2022121270782)


**BERT on a Data Diet: Finding Important Examples by Gradient-Based Pruning**

![](https://img.shields.io/badge/arXiv-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2211.05610-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2211.05610)


**BERTPerf: Inference Latency Predictor for BERT on ARM big.LITTLE Multi-Core Processors**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/SiPS55645.2022.9919203-sandybrown?style=flat-square)](https://doi.org/10.1109/SiPS55645.2022.9919203)


**BiBERT: Accurate Fully Binarized BERT**

![](https://img.shields.io/badge/arXiv-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2203.06390-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2203.06390)


**Bigger&Faster: Two-stage Neural Architecture Search for Quantized Transformer Models**

![](https://img.shields.io/badge/arXiv-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2209.12127-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2209.12127)


**BiT: Robustly Binarized Multi-distilled Transformer**

![](https://img.shields.io/badge/arXiv-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2205.13016-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2205.13016)


**Boosting Distributed Training Performance of the Unpadded BERT Model**

![](https://img.shields.io/badge/arXiv-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2208.08124-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2208.08124)


**Compact Token Representations with Contextual Quantization for Efficient Document Re-ranking**

![](https://img.shields.io/badge/arXiv-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2203.15328-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2203.15328)


**Compressing Pre-trained Transformers via Low-Bit NxM Sparsity for Natural Language Understanding**

![](https://img.shields.io/badge/arXiv-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2206.15014-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2206.15014)


**Compression of Generative Pre-trained Language Models via Quantization**

![](https://img.shields.io/badge/arXiv-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2203.10705-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2203.10705)


**CONNA: Configurable Matrix Multiplication Engine for Neural Network Acceleration**

![](https://img.shields.io/badge/MDPI-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.3390/electronics11152373-sandybrown?style=flat-square)](https://doi.org/10.3390/electronics11152373)


**CPSAA: Accelerating Sparse Attention using Crossbar-based Processing-In-Memory Architecture**

![](https://img.shields.io/badge/arXiv-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2210.06696-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2210.06696)


**Demystifying BERT: System Design Implications**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/IISWC55918.2022.00033-sandybrown?style=flat-square)](https://doi.org/10.1109/IISWC55918.2022.00033)


**DFX: A Low-latency Multi-FPGA Appliance for Accelerating Transformer-based Text Generation**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/MICRO56248.2022.00051-sandybrown?style=flat-square)](https://doi.org/10.1109/MICRO56248.2022.00051)


**DiVIT: Algorithm and architecture co-design of differential attention in vision transformer**

![](https://img.shields.io/badge/Elsevier-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1016/j.sysarc.2022.102520-sandybrown?style=flat-square)](https://doi.org/10.1016/j.sysarc.2022.102520)


**DOTA: Detect and Omit Weak Attentions for Scalable Transformer Acceleration**

![](https://img.shields.io/badge/ACM-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3503222.3507738-sandybrown?style=flat-square)](https://doi.org/10.1145/3503222.3507738)


**DQ-BART: Efficient Sequence-to-Sequence Model via Joint Distillation and Quantization**

![](https://img.shields.io/badge/arXiv-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2203.11239-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2203.11239)


**DTQAtten: Leveraging Dynamic Token-based Quantization for Efficient Attention Architecture**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.23919/DATE54114.2022.9774692-sandybrown?style=flat-square)](https://doi.org/10.23919/DATE54114.2022.9774692)


**Dynamic Precision Analog Computing for Neural Networks**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/JSTQE.2022.3218019-sandybrown?style=flat-square)](https://doi.org/10.1109/JSTQE.2022.3218019)


**EFA-Trans: An Efficient and Flexible Acceleration Architecture for Transformers**

![](https://img.shields.io/badge/MDPI-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.3390/electronics11213550-sandybrown?style=flat-square)](https://doi.org/10.3390/electronics11213550)


**Efficient Document Retrieval by End-to-End Refining and Quantizing BERT Embedding with Contrastive Product Quantization**

![](https://img.shields.io/badge/arXiv-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2210.17170v1-sandybrown?style=flat-square)](https://arxiv.org/abs/2210.17170v1)


**Elastic Processing and Hardware Architectures for Machine Learning**

![](https://img.shields.io/badge/ProQuest-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-3e9f91ca96ba3320587da2bbec561a2b/-sandybrown?style=flat-square)](https://www.proquest.com/openview/3e9f91ca96ba3320587da2bbec561a2b/)


**Empirical Evaluation of Post-Training Quantization Methods for Language Tasks**

![](https://img.shields.io/badge/arXiv-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2210.16621-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2210.16621)


**Enabling and Accelerating Dynamic Vision Transformer Inference for Real-Time Applications**

![](https://img.shields.io/badge/arXiv-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2212.02687-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2212.02687)


**Enabling Efficient Large-Scale Deep Learning Training with Cache Coherent Disaggregated Memory Systems**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/HPCA53966.2022.00018-sandybrown?style=flat-square)](https://doi.org/10.1109/HPCA53966.2022.00018)


**Enabling Energy-Efficient Inference for Self-Attention Mechanisms in Neural Networks**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/AICAS54282.2022.9869924-sandybrown?style=flat-square)](https://doi.org/10.1109/AICAS54282.2022.9869924)


**Enabling fast uncertainty estimation: accelerating bayesian transformers via algorithmic and hardware optimizations**

![](https://img.shields.io/badge/ACM-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3489517.3530451-sandybrown?style=flat-square)](https://doi.org/10.1145/3489517.3530451)


**Enabling Fast Uncertainty Estimation: Exploiting Structured Sparsity in Bayesian Transformers**

![](https://img.shields.io/badge/ACM-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3489517.3530451-sandybrown?style=flat-square)](https://spiral.imperial.ac.uk/bitstream/10044/1/96226/2/dac22hf3_final_bayesatt.pdf)


**Ensemble Model Compression for Fast and Energy-Efficient Ranking on FPGAs**

![](https://img.shields.io/badge/Springer-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1007/978--3--030--99736--6_18-sandybrown?style=flat-square)](https://doi.org/10.1007/978-3-030-99736-6_18)


**Extending the ONNX Runtime Framework for the Processing-in-Memory Execution**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ICEIC54506.2022.9748444-sandybrown?style=flat-square)](https://doi.org/10.1109/ICEIC54506.2022.9748444)


**Extreme Compression for Pre-trained Transformers Made Simple and Efficient**

![](https://img.shields.io/badge/arXiv-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2206.01859-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2206.01859)


**Fast Heterogeneous Task Mapping for Reducing Edge DNN Latency**

![](https://img.shields.io/badge/None-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ASAP54787.2022.00020-sandybrown?style=flat-square)](https://doi.org/10.1109/ASAP54787.2022.00020)


**FILM-QNN: Efficient FPGA Acceleration of Deep Neural Networks with Intra-Layer, Mixed-Precision Quantization**

![](https://img.shields.io/badge/ACM/SIGDA-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3490422.3502364-sandybrown?style=flat-square)](https://doi.org/10.1145/3490422.3502364)


**Fine-and Coarse-Granularity Hybrid Self-Attention for Efficient BERT**

![](https://img.shields.io/badge/arXiv-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2203.09055-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2203.09055)


**FPGA-aware automatic acceleration framework for vision transformer with mixed-scheme quantization: late breaking results**

![](https://img.shields.io/badge/ACM/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3489517.3530618-sandybrown?style=flat-square)](https://doi.org/10.1145/3489517.3530618)


**FPGA-based design and implementation of the location attention mechanism in neural networks**

![](https://img.shields.io/badge/IOS%20Press-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.3233/JIFS--212273-sandybrown?style=flat-square)](https://doi.org/10.3233/JIFS-212273)


**From dense to sparse: Contrastive pruning for better pre-trained language model compression**

![](https://img.shields.io/badge/AAAI-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1609/aaai.v36i10.21408-sandybrown?style=flat-square)](https://doi.org/10.1609/aaai.v36i10.21408)


**Future Scaling of Memory Hierarchy for Tensor Cores and Eliminating Redundant Shared Memory Traffic Using Inter-Warp Multicastin**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/TC.2022.3207134-sandybrown?style=flat-square)](https://doi.org/10.1109/TC.2022.3207134)


**Greedy-layer pruning: Speeding up transformer models for natural language processing**

![](https://img.shields.io/badge/Elsevier-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1016/j.patrec.2022.03.023-sandybrown?style=flat-square)](https://doi.org/10.1016/j.patrec.2022.03.023)


**GuardNN: secure accelerator architecture for privacy-preserving deep learning**

![](https://img.shields.io/badge/ACM/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3489517.3530439-sandybrown?style=flat-square)](https://doi.org/10.1145/3489517.3530439)


**Handling heavy-tailed input of transformer inference on GPUs**

![](https://img.shields.io/badge/ACM-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3524059.3532372-sandybrown?style=flat-square)](https://doi.org/10.1145/3524059.3532372)


**Hardware Acceleration of Transformer Networks using FPGAs**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/PACET56979.2022.9976354-sandybrown?style=flat-square)](https://doi.org/10.1109/PACET56979.2022.9976354)


**Hardware and Software Co-design for Soft Switch in ViT Variants Processing Unit**

![](https://img.shields.io/badge/Springer-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1007/978--3--031--10989--8_55-sandybrown?style=flat-square)](https://doi.org/10.1007/978-3-031-10989-8_55)


**Hardware and Software Co-optimization for Windows Attention**

![](https://img.shields.io/badge/None-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1007/978--3--031--10989--8_52-sandybrown?style=flat-square)](https://doi.org/10.1007/978-3-031-10989-8_52)


**Improving Oversubscribed GPU Memory Performance in the PyTorch Framework**

![](https://img.shields.io/badge/Springer-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1007/s10586--022--03805--x-sandybrown?style=flat-square)](https://doi.org/10.1007/s10586-022-03805-x)


**Integer Fine-tuning of Transformer-based Models**

![](https://img.shields.io/badge/arXiv-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2209.09815-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2209.09815)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2209.09815.pdf)

**Learned Token Pruning in Contextualized Late Interaction over BERT (ColBERT)**

![](https://img.shields.io/badge/ACM-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3477495.3531835-sandybrown?style=flat-square)](https://doi.org/10.1145/3477495.3531835)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://web.Arxiv.org/web/20220713100651id_/https://dl.acm.org/doi/pdf/10.1145/3477495.3531835)

**Lightweight Composite Re-Ranking for Efficient Keyword Search with BERT**

![](https://img.shields.io/badge/ACM-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3488560.3498495-sandybrown?style=flat-square)](https://doi.org/10.1145/3488560.3498495)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://dl.acm.org/doi/pdf/10.1145/3488560.3498495)

**Lightweight Transformers for Conversational AI**

![](https://img.shields.io/badge/ACL-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-http://dx.doi.org/10.18653/v1/2022.naacl--industry.25-sandybrown?style=flat-square)](http://dx.doi.org/10.18653/v1/2022.naacl-industry.25)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://aclanthology.org/2022.naacl-industry.25.pdf)

**LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale**

![](https://img.shields.io/badge/arXiv-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2208.07339-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2208.07339)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2208.07339)

**Low-Bit Quantization of Transformer for Audio Speech Recognition**

![](https://img.shields.io/badge/Springer-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1007/978--3--031--19032--2_12-sandybrown?style=flat-square)](https://doi.org/10.1007/978-3-031-19032-2_12)


**Low-Precision Quantization Techniques for Hardware-Implementation-Friendly BERT Models**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ISQED54688.2022.9806238-sandybrown?style=flat-square)](https://doi.org/10.1109/ISQED54688.2022.9806238)


**MKQ-BERT: Quantized BERT with 4-bits Weights and Activations**

![](https://img.shields.io/badge/arXiv-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2203.13483-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2203.13483)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2203.13483.pdf)

**Mokey: enabling narrow fixed-point inference for out-of-the-box floating-point transformer models**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3470496.3527438-sandybrown?style=flat-square)](https://doi.org/10.1145/3470496.3527438)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2203.12758.pdf)

**Mr. BiQ: Post-Training Non-Uniform Quantization Based on Minimizing the Reconstruction Error**

![](https://img.shields.io/badge/IEEE/CVF-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/CVPR52688.2022.01201-sandybrown?style=flat-square)](https://doi.org/10.1109/CVPR52688.2022.01201)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://openaccess.thecvf.com/content/CVPR2022/papers/Jeon_Mr.BiQ_Post-Training_Non-Uniform_Quantization_Based_on_Minimizing_the_Reconstruction_Error_CVPR_2022_paper.pdf)

**Near-Optimal Sparse Allreduce for Distributed Deep Learning**

![](https://img.shields.io/badge/ACM-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3503221.3508399-sandybrown?style=flat-square)](https://doi.org/10.1145/3503221.3508399)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2201.07598.pdf)

**Nebula: A Scalable and Flexible Accelerator for DNN Multi-Branch Blocks on Embedded Systems**

![](https://img.shields.io/badge/MDPI-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.3390/electronics11040505-sandybrown?style=flat-square)](https://doi.org/10.3390/electronics11040505)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://www.mdpi.com/2079-9292/11/4/505/pdf)

**NEEBS: Nonexpert large-scale environment building system for deep neural network**

![](https://img.shields.io/badge/Wiley-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1002/cpe.7499-sandybrown?style=flat-square)](https://doi.org/10.1002/cpe.7499)


**Optimal Brain Compression: A framework for accurate post-training quantization and pruning**

![](https://img.shields.io/badge/NeurIPS-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2208.11580-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2208.11580)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2208.11580.pdf)

**PipeBERT: High-throughput BERT Inference for ARM Big.LITTLE Multi-core Processors**

![](https://img.shields.io/badge/Springer-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1007/s11265--022--01814--y-sandybrown?style=flat-square)](https://doi.org/10.1007/s11265-022-01814-y)


**Post-Training Quantization for Longformer with Chunkwise Quantization Granularity and Optimized Percentile**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ICCCS55155.2022.9846198-sandybrown?style=flat-square)](https://doi.org/10.1109/ICCCS55155.2022.9846198)


**Pre-trained Language Model with Feature Reduction and No Fine-Tuning**

![](https://img.shields.io/badge/Springer-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1007/978--981--19--3923--5_59-sandybrown?style=flat-square)](https://doi.org/10.1007/978-981-19-3923-5_59)


**Privacy-Preserving Text Classification on BERT Embeddings with Homomorphic Encryption**

![](https://img.shields.io/badge/Arxiv-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2210.02574-sandybrown?style=flat-square)](https://arxiv.org/abs/2210.02574)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2210.02574.pdf)

**ProSE: the architecture and design of a protein discovery engine**

![](https://img.shields.io/badge/ACM-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3503222.3507722-sandybrown?style=flat-square)](https://doi.org/10.1145/3503222.3507722)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://par.nsf.gov/servlets/purl/10394954)

**QDrop: Randomly Dropping Quantization for Extremely Low-bit Post-Training Quantization**

![](https://img.shields.io/badge/Arxiv-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2203.05740-sandybrown?style=flat-square)](https://arxiv.org/abs/2203.05740)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2203.05740)

**QuaLA-MiniLM: a Quantized Length Adaptive MiniLM**

![](https://img.shields.io/badge/Arxiv-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2210.17114-sandybrown?style=flat-square)](https://arxiv.org/abs/2210.17114)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2210.17114)

**RCT: Resource Constrained Training for Edge AI**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/TNNLS.2022.3190451-sandybrown?style=flat-square)](https://doi.org/10.1109/TNNLS.2022.3190451)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2103.14493.pdf)

**ReAAP: A Reconfigurable and Algorithm-Oriented Array Processor With Compiler-Architecture Co-Design**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/TC.2022.3213177-sandybrown?style=flat-square)](https://doi.org/10.1109/TC.2022.3213177)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://ieeexplore.ieee.org/iel7/12/4358213/09914609.pdf)

**Row-wise Accelerator for Vision Transformer**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/AICAS54282.2022.9869928-sandybrown?style=flat-square)](https://doi.org/10.1109/AICAS54282.2022.9869928)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2205.03998.pdf)

**S4: a High-sparsity, High-performance AI Accelerator**

![](https://img.shields.io/badge/Arxive-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2207.08006-sandybrown?style=flat-square)](https://arxiv.org/abs/2207.08006)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2207.08006)

**SALO: an efficient spatial accelerator enabling hybrid sparse attention mechanisms for long sequences**

![](https://img.shields.io/badge/ACM-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3489517.3530504-sandybrown?style=flat-square)](https://doi.org/10.1145/3489517.3530504)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2206.14550.pdf)

**Searching for memory-lighter architectures for OCR-augmented image captioning**

![](https://img.shields.io/badge/None-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.3233/JIFS--219230-sandybrown?style=flat-square)](https://doi.org/10.3233/JIFS-219230)


**SensiMix: Sensitivity-Aware 8-bit index & 1-bit value mixed precision quantization for BERT compression**

![](https://img.shields.io/badge/PLOSONE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1371/journal.pone.0265621-sandybrown?style=flat-square)](https://doi.org/10.1371/journal.pone.0265621)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0265621&type=printable)

**Sentiment Analysis Using Pre-Trained Language Model With No Fine-Tuning and Less Resource**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ACCESS.2022.3212367-sandybrown?style=flat-square)](https://doi.org/10.1109/ACCESS.2022.3212367)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9912410)

**Software and Hardware Fusion Multi-Head Attention**

![](https://img.shields.io/badge/Springer-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-http://dx.doi.org/10.1007/978--3--031--10989--8_51-sandybrown?style=flat-square)](http://dx.doi.org/10.1007/978-3-031-10989-8_51)


**Sparse Attention Acceleration with Synergistic In-Memory Pruning and On-Chip Recomputation**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/MICRO56248.2022.00059-sandybrown?style=flat-square)](https://doi.org/10.1109/MICRO56248.2022.00059)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2209.00606.pdf)

**SwiftPruner: Reinforced Evolutionary Pruning for Efficient Ad Relevance**

![](https://img.shields.io/badge/ACM-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3511808.3557139-sandybrown?style=flat-square)](https://doi.org/10.1145/3511808.3557139)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2209.00625.pdf)

**T-OPU: An FPGA-based Overlay Processor for Natural Language Processing**

![](https://img.shields.io/badge/UCLA-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://escholarship.org/uc/item/9r46v693-sandybrown?style=flat-square)](https://escholarship.org/uc/item/9r46v693)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://escholarship.org/content/qt9r46v693/qt9r46v693.pdf)

**The Optimal BERT Surgeon: Scalable and Accurate Second-Order Pruning for Large Language Models**

![](https://img.shields.io/badge/Arxiv-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2203.07259-sandybrown?style=flat-square)](https://arxiv.org/abs/2203.07259)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2203.07259.pdf)

**Towards efficient post-training quantization of pre-trained language models**

![](https://img.shields.io/badge/NeurIPS-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-link-sandybrown?style=flat-square)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/096347b4efc264ae7f07742fea34af1f-Abstract-Conference.html)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://proceedings.neurips.cc/paper_files/paper/2022/file/096347b4efc264ae7f07742fea34af1f-Paper-Conference.pdf)

**Train Flat, Then Compress: Sharpness-Aware Minimization Learns More Compressible Models**

![](https://img.shields.io/badge/Arxiv-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2205.12694-sandybrown?style=flat-square)](https://arxiv.org/abs/2205.12694)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2205.12694.pdf)

**TranCIM: Full-Digital Bitline-Transpose CIM-based Sparse Transformer Accelerator With Pipeline/Parallel Reconfigurable Modes**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/JSSC.2022.3213542-sandybrown?style=flat-square)](https://doi.org/10.1109/JSSC.2022.3213542)


**TransPIM: A Memory-based Acceleration via Software-Hardware Co-Design for Transformer**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/HPCA53966.2022.00082-sandybrown?style=flat-square)](https://doi.org/10.1109/HPCA53966.2022.00082)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://par.nsf.gov/servlets/purl/10345536)

**VAQF: Fully Automatic Software-Hardware Co-Design Framework for Low-Bit Vision Transformer**

![](https://img.shields.io/badge/Arxiv-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2201.06618-sandybrown?style=flat-square)](https://arxiv.org/abs/2201.06618)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2201.06618.pdf)

**Varuna: Scalable, Low-cost Training of Massive Deep Learning Models**

![](https://img.shields.io/badge/ACM-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3492321.3519584-sandybrown?style=flat-square)](https://doi.org/10.1145/3492321.3519584)


**ViA: A Novel Vision-Transformer Accelerator Based on FPGA**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/TCAD.2022.3197489-sandybrown?style=flat-square)](https://doi.org/10.1109/TCAD.2022.3197489)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](None)

**Work-in-Progress: Utilizing latency and accuracy predictors for efficient hardware-aware NAS**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/CODES-ISSS55005.2022.00014-sandybrown?style=flat-square)](https://doi.org/10.1109/CODES-ISSS55005.2022.00014)


**XTC: Extreme Compression for Pre-trained Transformers Made Simple and Efficient**

![](https://img.shields.io/badge/None-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-None-sandybrown?style=flat-square)](None)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://proceedings.neurips.cc/paper_files/paper/2022/file/1579d5d8edacd85ac1a86aea28bdf32d-Paper-Conference.pdf)

**ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers**

![](https://img.shields.io/badge/None-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-None-sandybrown?style=flat-square)](None)


**Fully Unsupervised Machine Translation Using Context-Aware Word Translation and Denoising Autoencoder**

![](https://img.shields.io/badge/None-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1080/08839514.2022.2031817-sandybrown?style=flat-square)](https://doi.org/10.1080/08839514.2022.2031817)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://www.tandfonline.com/doi/pdf/10.1080/08839514.2022.2031817)

**DistilHuBERT: Speech representation learning by layer-wise distillation of hidden-unit BERT**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ICASSP43922.2022.9747490-sandybrown?style=flat-square)](https://doi.org/10.1109/ICASSP43922.2022.9747490)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2110.01900.pdf)

**Data Movement Reduction for DNN Accelerators: Enabling Dynamic Quantization Through an eFPGA**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ISVLSI54635.2022.00082-sandybrown?style=flat-square)](https://doi.org/10.1109/ISVLSI54635.2022.00082)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://scholar.archive.org/work/hd53tmr62rhn3mxplkjzhrnvw4/access/wayback/https://publikationen.bibliothek.kit.edu/1000151937/149523013)

**Hardware-friendly compression and hardware acceleration for transformer: A survey**

![](https://img.shields.io/badge/AIMPress-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://www.aimspress.com/article/doi/10.3934/era.2022192-sandybrown?style=flat-square)](https://www.aimspress.com/article/doi/10.3934/era.2022192)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://www.aimspress.com/aimspress-data/era/2022/10/PDF/era-30-10-192.pdf)

**Hardware/Software Co-Design of Edge DNN Accelerators with TFLite**

![](https://img.shields.io/badge/HiPEAC-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1007/978--3--030--87208--8_5-sandybrown?style=flat-square)](https://https://eprints.gla.ac.uk/280378/)


**Workload-Balanced Graph Attention Network Accelerator with Top-K Aggregation Candidates**

![](https://img.shields.io/badge/ACM-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3508352.3549343-sandybrown?style=flat-square)](https://doi.org/10.1145/3508352.3549343)


---
### 2023
**An Efficient Transformer Inference Engine on DSP**

![](https://img.shields.io/badge/Springer-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1007/978--3--031--22677--9_29-sandybrown?style=flat-square)](https://doi.org/10.1007/978-3-031-22677-9_29)


**CHARM: Composing Heterogeneous Accelerators for Matrix Multiply on Versal ACAP Architecture**

![](https://img.shields.io/badge/arXiv-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2301.02359-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2301.02359)


**DTATrans: Leveraging Dynamic Token-Based Quantization With Accuracy Compensation Mechanism for Efficient Transformer Architecture**

![](https://img.shields.io/badge/IEEE-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/TCAD.2022.3181541-sandybrown?style=flat-square)](https://doi.org/10.1109/TCAD.2022.3181541)


**ENEX-FP: A BERT-Based Address Recognition Model**

![](https://img.shields.io/badge/MDPI-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.3390/electronics12010209-sandybrown?style=flat-square)](https://doi.org/10.3390/electronics12010209)


**HAMMER: Hardware-friendly Approximate Computing for Self-attention with Mean-redistribution and Linearization**

![](https://img.shields.io/badge/IEEE-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/LCA.2022.3233832-sandybrown?style=flat-square)](https://doi.org/10.1109/LCA.2022.3233832)


**SECDA-TFLite: A toolkit for efficient development of FPGA-based DNN accelerators for edge inference**

![](https://img.shields.io/badge/Elsevier-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1016/j.jpdc.2022.11.005-sandybrown?style=flat-square)](https://doi.org/10.1016/j.jpdc.2022.11.005)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://www.sciencedirect.com/science/article/pii/S0743731522002301/pdfft?md5=444fdc7e73724f5d9881d162bed2a735&pid=1-s2.0-S0743731522002301-main.pdf)

**SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models**

![](https://img.shields.io/badge/Arxiv-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2211.10438-sandybrown?style=flat-square)](https://arxiv.org/abs/2211.10438)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2211.10438.pdf)

**Sparse*BERT: Sparse Models Generalize To New tasks and Domains**

![](https://img.shields.io/badge/Arxiv-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2205.12452-sandybrown?style=flat-square)](https://arxiv.org/abs/2205.12452)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2205.12452)

**Teacher Intervention: Improving Convergence of Quantization Aware Training for Ultra-Low Precision Transformers**

![](https://img.shields.io/badge/Arxiv-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2302.11812-sandybrown?style=flat-square)](https://arxiv.org/abs/2302.11812)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2302.11812)

**TiC-SAT: Tightly-Coupled Systolic Accelerator for Transformers**

![](https://img.shields.io/badge/ACM-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3566097.3567867-sandybrown?style=flat-square)](https://doi.org/10.1145/3566097.3567867)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://infoscience.epfl.ch/record/298067/files/TiC_SAT_ASPDAC-preprint.pdf)

**ViTALiTy: Unifying Low-rank and Sparse Approximation for Vision Transformer Acceleration with a Linear Taylor Attention**

![](https://img.shields.io/badge/IEEE-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/HPCA56546.2023.10071081-sandybrown?style=flat-square)](https://doi.org/10.1109/HPCA56546.2023.10071081)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2211.05109.pdf)

**ViTA: A Vision Transformer Inference Accelerator for Edge Applications**

![](https://img.shields.io/badge/Arxiv-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2302.09108-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2302.09108)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2302.09108.pdf)

**Trends in AI inference energy consumption: Beyond the performance-vs-parameter laws of deep learning**

![](https://img.shields.io/badge/Elsevier-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1016/j.suscom.2023.100857-sandybrown?style=flat-square)](https://doi.org/10.1016/j.suscom.2023.100857)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://www.sciencedirect.com/science/article/pii/S2210537923000124/pdfft?md5=4bec2735c1586b935287e6afea9e63a2&pid=1-s2.0-S2210537923000124-main.pdf)

**TRON: Transformer Neural Network Acceleration with Non-Coherent Silicon Photonics**

![](https://img.shields.io/badge/Arxiv-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2303.12914-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2303.12914)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2303.12914.pdf)

**TransCODE: Co-design of Transformers and Accelerators for Efficient Training and Inference**

![](https://img.shields.io/badge/Arxiv-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2303.14882-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2303.14882)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2303.14882)

**TinyVers: A Tiny Versatile System-on-chip with State-Retentive eMRAM for ML Inference at the Extreme Edge**

![](https://img.shields.io/badge/IEEE-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/JSSC.2023.3236566-sandybrown?style=flat-square)](https://doi.org/10.1109/JSSC.2023.3236566)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2301.03537.pdf)

**Architecting High Performance Silicon Systems for Accurate and Efficient On-Chip Deep Learning**

![](https://img.shields.io/badge/HarwardLibrary-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://nrs.harvard.edu/URN--3:HUL.INSTREPOS:37375806-sandybrown?style=flat-square)](https://nrs.harvard.edu/URN-3:HUL.INSTREPOS:37375806)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://dash.harvard.edu/bitstream/handle/1/37375806/Final_Draft_PhD_Dissertation_Thierry_Tambe.pdf)

**Hardware-efficient Softmax Approximation for Self-Attention Networks**

![](https://img.shields.io/badge/IEEE-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ISCAS46773.2023.10181465-sandybrown?style=flat-square)](https://doi.org/10.1109/ISCAS46773.2023.10181465)


**Fast Prototyping Next-Generation Accelerators for New ML Models using MASE: ML Accelerator System Exploration**

![](https://img.shields.io/badge/Arxiv-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2307.15517-sandybrown?style=flat-square)](https://arxiv.org/abs/2307.15517)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2307.15517.pdf)

**Advances in Electromagnetics Empowered by Artificial Intelligence and Deep Learning**

![](https://img.shields.io/badge/Wiley-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-ISBN:9781119853893-sandybrown?style=flat-square)](https://books.google.com/books?id=rlPNEAAAQBAJ)


**A Scalable GPT-2 Inference Hardware Architecture on FPGA**

![](https://img.shields.io/badge/IEEE-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/IJCNN54540.2023.10191067-sandybrown?style=flat-square)](https://doi.org/10.1109/IJCNN54540.2023.10191067)


**BL-PIM: Varying the Burst Length to Realize the All-Bank Performance and Minimize the Multi-Workload Interference for in-DRAM PIM**

![](https://img.shields.io/badge/IEEE-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ACCESS.2023.3300893-sandybrown?style=flat-square)](https://doi.org/10.1109/ACCESS.2023.3300893)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10198428)

**Integrated Transformers Inference Framework for Multiple Tenants on GPU**

![](https://img.shields.io/badge/SydneyDigital-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://hdl.handle.net/2123/31606-sandybrown?style=flat-square)](https://hdl.handle.net/2123/31606)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://ses.library.usyd.edu.au/bitstream/handle/2123/31606/Thesis__Yuning_Zhang%20%281%29.pdf?sequence=2&isAllowed=y)

**Embedded Deep Learning Accelerators: A Survey on Recent Advances**

![](https://img.shields.io/badge/IEEE-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/TAI.2023.3311776-sandybrown?style=flat-square)](https://doi.org/10.1109/TAI.2023.3311776)


**Collective Communication Enabled Transformer Acceleration on Heterogeneous Clusters**

![](https://img.shields.io/badge/TSpace-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://hdl.handle.net/1807/130585-sandybrown?style=flat-square)](https://hdl.handle.net/1807/130585)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://tspace.library.utoronto.ca/bitstream/1807/130585/3/Gao_Yu_202311_MAS_thesis.pdf)

**FET-OPU: A Flexible and Efficient FPGA-Based Overlay Processor for Transformer Networks**

![](https://img.shields.io/badge/IEEE-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ICCAD57390.2023.10323752-sandybrown?style=flat-square)](https://doi.org/10.1109/ICCAD57390.2023.10323752)


**Racism and Hate Speech Detection on Twitter: A QAHA-Based Hybrid Deep Learning Approach Using LSTM-CNN**

![](https://img.shields.io/badge/ACADLore-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.56578/ijkis010202-sandybrown?style=flat-square)](https://doi.org/10.56578/ijkis010202)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://library.acadlore.com/IJKIS/2023/1/2/IJKIS_01.02_02.pdf)

**22.9 A 12nm 18.1TFLOPs/W Sparse Transformer Processor with Entropy-Based Early Exit, Mixed-Precision Predication and Fine-Grained Power Management**

![](https://img.shields.io/badge/IEEE-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ISSCC42615.2023.10067817-sandybrown?style=flat-square)](https://doi.org/10.1109/ISSCC42615.2023.10067817)


**P^3 ViT: A CIM-Based High-Utilization Architecture With Dynamic Pruning and Two-Way Ping-Pong Macro for Vision Transformer**

![](https://img.shields.io/badge/IEEE-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/TCSI.2023.3315060-sandybrown?style=flat-square)](https://doi.org/10.1109/TCSI.2023.3315060)


**I-ViT: Integer-only Quantization for Efficient Vision Transformer Inference**

![](https://img.shields.io/badge/IEEE/CVF-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2207.01405-sandybrown?style=flat-square)](https://arxiv.org/abs/2207.01405)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2207.01405)

**Enabling efficient edge intelligence: a hardware-software codesign approach**

![](https://img.shields.io/badge/NanyangTechnologicalUniversity-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://dr.ntu.edu.sg/handle/10356/172499-sandybrown?style=flat-square)](https://dr.ntu.edu.sg/handle/10356/172499)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://dr.ntu.edu.sg/bitstream/10356/172499/2/Thesis_Final_HUAISHUO.pdf)

**Automatic Kernel Generation for Large Language Models on Deep Learning Accelerators**

![](https://img.shields.io/badge/IEEE-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ICCAD57390.2023.10323944-sandybrown?style=flat-square)](https://doi.org/10.1109/ICCAD57390.2023.10323944)


**A Low-Latency and Lightweight FPGA-Based Engine for Softmax and Layer Normalization Acceleration**

![](https://img.shields.io/badge/IEEE-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ICCE--Asia59966.2023.10326397-sandybrown?style=flat-square)](https://doi.org/10.1109/ICCE-Asia59966.2023.10326397)


**PP-Transformer: Enable Efficient Deployment of Transformers Through Pattern Pruning**

![](https://img.shields.io/badge/IEEE-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ICCAD57390.2023.10323836-sandybrown?style=flat-square)](https://doi.org/10.1109/ICCAD57390.2023.10323836)


**DEAP: Design Space Exploration for DNN Accelerator Parallelism**

![](https://img.shields.io/badge/Arxive-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2312.15388-sandybrown?style=flat-square)](https://arxiv.org/abs/2312.15388)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2312.15388.pdf)

**Understanding the Potential of FPGA-Based Spatial Acceleration for Large Language Model Inference**

![](https://img.shields.io/badge/Arxive-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2312.15159-sandybrown?style=flat-square)](https://arxiv.org/abs/2312.15159)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2312.15159.pdf)

**An RRAM-Based Computing-in-Memory Architecture and Its Application in Accelerating Transformer Inference**

![](https://img.shields.io/badge/IEEE-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.ieeecomputersociety.org/10.1109/TVLSI.2023.3345651-sandybrown?style=flat-square)](https://doi.ieeecomputersociety.org/10.1109/TVLSI.2023.3345651)


**Mobile Transformer Accelerator Exploiting Various Line Sparsity and Tile-Based Dynamic Quantization**

![](https://img.shields.io/badge/IEEE-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/TCAD.2023.3347291-sandybrown?style=flat-square)](https://doi.org/10.1109/TCAD.2023.3347291)


**A Lightweight Transformer Model using Neural ODE for FPGAs**

![](https://img.shields.io/badge/IEEE-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/IPDPSW59300.2023.00029-sandybrown?style=flat-square)](https://doi.org/10.1109/IPDPSW59300.2023.00029)


**TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings**

![](https://img.shields.io/badge/ACM-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3579371.3589350-sandybrown?style=flat-square)](https://doi.org/10.1145/3579371.3589350)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://dl.acm.org/doi/pdf/10.1145/3579371.3589350)

**FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU**

![](https://img.shields.io/badge/PMLR-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://proceedings.mlr.press/v202/sheng23a-sandybrown?style=flat-square)](https://proceedings.mlr.press/v202/sheng23a)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://proceedings.mlr.press/v202/sheng23a/sheng23a.pdf)

**ITA: An Energy-Efficient Attention and Softmax Accelerator for Quantized Transformers**

![](https://img.shields.io/badge/Arxiv-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2307.03493-sandybrown?style=flat-square)](https://arxiv.org/abs/2307.03493)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2307.03493)

**X-Former: In-Memory Acceleration of Transformers**

![](https://img.shields.io/badge/Arxiv-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2303.07470-sandybrown?style=flat-square)](https://arxiv.org/abs/2303.07470)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2303.07470)

**GPT4AIGChip: Towards Next-Generation AI Accelerator Design Automation via Large Language Models**

![](https://img.shields.io/badge/Arxiv-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2309.10730-sandybrown?style=flat-square)](https://arxiv.org/abs/2309.10730)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2309.10730)

**HeatViT: Hardware-Efficient Adaptive Token Pruning for Vision Transformers**

![](https://img.shields.io/badge/IEEE-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/HPCA56546.2023.10071047-sandybrown?style=flat-square)](https://doi.org/10.1109/HPCA56546.2023.10071047)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2211.08110)

**ViTCoD: Vision Transformer Acceleration via Dedicated Algorithm and Accelerator Co-Design**

![](https://img.shields.io/badge/IEEE-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2210.09573-sandybrown?style=flat-square)](https://arxiv.org/abs/2210.09573)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2210.09573)

**AccelTran: A Sparsity-Aware Accelerator for Dynamic Inference with Transformers**

![](https://img.shields.io/badge/IEEE-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/TCAD.2023.3273992-sandybrown?style=flat-square)](https://doi.org/10.1109/TCAD.2023.3273992)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2302.14705)

**Streaming Tensor Programs: A Programming Abstraction for Streaming Dataflow Accelerators**

![](https://img.shields.io/badge/?-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-NO_DATA-sandybrown?style=flat-square)](False)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://cgyurgyik.github.io/files/pubs/step-yarch.pdf)

**Lightening-Transformer: A Dynamically-operated Optically-interconnected Photonic Transformer Accelerator**

![](https://img.shields.io/badge/Arxiv-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2305.19533-sandybrown?style=flat-square)](https://arxiv.org/abs/2305.19533)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2305.19533)

**Chameleon: a Heterogeneous and Disaggregated Accelerator System for Retrieval-Augmented Language Models**

![](https://img.shields.io/badge/Arxiv-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2310.09949-sandybrown?style=flat-square)](https://arxiv.org/abs/2310.09949)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2310.09949)

**An Efficient FPGA-Based Accelerator for Swin Transformer**

![](https://img.shields.io/badge/Arxiv-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2308.13922-sandybrown?style=flat-square)](https://arxiv.org/abs/2308.13922)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2308.13922)

**Mixed-TD: Efficient Neural Network Accelerator with Layer-Specific Tensor Decomposition**

![](https://img.shields.io/badge/Arxiv-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2306.05021-sandybrown?style=flat-square)](https://arxiv.org/abs/2306.05021)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2306.05021)

---
### 2024
**A Cost-Efficient FPGA Implementation of Tiny Transformer Model using Neural ODE**

![](https://img.shields.io/badge/Arxive-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2401.02721-sandybrown?style=flat-square)](https://arxiv.org/abs/2401.02721)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2401.02721.pdf)

**FlightLLM: Efficient Large Language Model Inference with a Complete Mapping Flow on FPGAs**

![](https://img.shields.io/badge/Arxive-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2401.03868-sandybrown?style=flat-square)](https://arxiv.org/abs/2401.03868)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2401.03868.pdf)

**Accelerating Neural Networks for Large Language Models and Graph Processing with Silicon Photonics**

![](https://img.shields.io/badge/Arxive-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2401.06885-sandybrown?style=flat-square)](https://arxiv.org/abs/2401.06885)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2401.06885.pdf)

**Quantization and Hardware Architecture Co-Design for Matrix-Vector Multiplications of Large Language Models**

![](https://img.shields.io/badge/IEEE-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/TCSI.2024.3350661-sandybrown?style=flat-square)](https://doi.org/10.1109/TCSI.2024.3350661)


**RDCIM: RISC-V Supported Full-Digital Computing-in-Memory Processor With High Energy Efficiency and Low Area Overhead**

![](https://img.shields.io/badge/IEEE-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/TCSI.2024.3350664-sandybrown?style=flat-square)](https://doi.org/10.1109/TCSI.2024.3350664)


**A Survey on Hardware Accelerators for Large Language Models**

![](https://img.shields.io/badge/Arxive-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2401.09890-sandybrown?style=flat-square)](https://arxiv.org/abs/2401.09890)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2401.09890.pdf)

**BETA: Binarized Energy-Efficient Transformer Accelerator at the Edge**

![](https://img.shields.io/badge/Arxive-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2401.11851-sandybrown?style=flat-square)](https://arxiv.org/abs/2401.11851)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/abs/2401.11851.pdf)

**AttentionLego: An Open-Source Building Block For Spatially-Scalable Large Language Model Accelerator With Processing-In-Memory Technology**

![](https://img.shields.io/badge/Arxive-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2401.11459-sandybrown?style=flat-square)](https://arxiv.org/abs/2401.11459)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/abs/2401.11459.pdf)

**SSR: Spatial Sequential Hybrid Architecture for Latency Throughput Tradeoff in Transformer Acceleration**

![](https://img.shields.io/badge/Arxive-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2401.10417-sandybrown?style=flat-square)](https://arxiv.org/abs/2401.10417)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/abs/2401.10417.pdf)

**CIM-MLC: A Multi-level Compilation Stack for Computing-In-Memory Accelerators**

![](https://img.shields.io/badge/Arxiv-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2401.12428-sandybrown?style=flat-square)](https://arxiv.org/abs/2401.12428)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2401.12428.pdf)

**CIM-MLC: A Multi-level Compilation Stack for Computing-In-Memory Accelerators**

![](https://img.shields.io/badge/Arxiv-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2401.12428-sandybrown?style=flat-square)](https://arxiv.org/abs/2401.12428)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2401.12428.pdf)

**The Era of Generative Artificial Intelligence: In-Memory Computing Perspective**

![](https://img.shields.io/badge/IEEE-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/IEDM45741.2023.10413786-sandybrown?style=flat-square)](https://doi.org/10.1109/IEDM45741.2023.10413786)


**Hydragen: High-Throughput LLM Inference with Shared Prefixes**

![](https://img.shields.io/badge/Arxiv-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2402.05099-sandybrown?style=flat-square)](https://arxiv.org/abs/2402.05099)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2402.05099.pdf)

**A Survey on Transformer Compression**

![](https://img.shields.io/badge/Arxiv-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2402.05964.pdf-sandybrown?style=flat-square)](https://arxiv.org/abs/2402.05964.pdf)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2402.05964.pdf)

**SLEB: Streamlining LLMs through Redundancy Verification and Elimination of Transformer Blocks**

![](https://img.shields.io/badge/Arxiv-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2402.09025-sandybrown?style=flat-square)](https://arxiv.org/abs/2402.09025)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2402.09025.pdf)

**Stochastic Spiking Attention: Accelerating Attention with Stochastic Computing in Spiking Networks**

![](https://img.shields.io/badge/Arxiv-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2402.09109-sandybrown?style=flat-square)](https://arxiv.org/abs/2402.09109)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2402.09109.pdf)

**Reusing Softmax Hardware Unit for GELU Computation in Transformers**

![](https://img.shields.io/badge/Arxiv-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2402.10118-sandybrown?style=flat-square)](https://arxiv.org/abs/2402.10118)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/abs/2402.10118.pdf)

**ConSmax: Hardware-Friendly Alternative Softmax with Learnable Parameters**

![](https://img.shields.io/badge/Arxiv-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2402.10930-sandybrown?style=flat-square)](https://arxiv.org/abs/2402.10930)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2402.10930.pdf)

**Speculative Streaming: Fast LLM Inference without Auxiliary Models**

![](https://img.shields.io/badge/Arxiv-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2402.11131-sandybrown?style=flat-square)](https://arxiv.org/abs/2402.11131)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2402.11131.pdf)

**H3D-Transformer: A Heterogeneous 3D (H3D) Computing Platform for Transformer Model Acceleration on Edge Devices**

![](https://img.shields.io/badge/ACM-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://dl.acm.org/doi/10.1145/3649219-sandybrown?style=flat-square)](https://dl.acm.org/doi/10.1145/3649219)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://dl.acm.org/doi/pdf/10.1145/3649219)

**NeuPIMs: NPU-PIM Heterogeneous Acceleration for Batched LLM Inferencing**

![](https://img.shields.io/badge/Arxiv-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2403.00579-sandybrown?style=flat-square)](https://arxiv.org/abs/2403.00579)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2403.00579.pdf)

**Cerberus: Triple Mode Acceleration of Sparse Matrix and Vector Multiplication**

![](https://img.shields.io/badge/ACM-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3653020-sandybrown?style=flat-square)](https://doi.org/10.1145/365302000579)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://dl.acm.org/doi/pdf/10.1145/3653020)

**DEFA: Efficient Deformable Attention Acceleration via Pruning-Assisted Grid-Sampling and Multi-Scale Parallel Processing**

![](https://img.shields.io/badge/Arxiv-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2403.10913-sandybrown?style=flat-square)](https://arxiv.org/abs/2403.10913)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2403.10913.pdf)

**FastDecode: High-Throughput GPU-Efficient LLM Serving using Heterogeneous Pipelines**

![](https://img.shields.io/badge/Arxiv-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2403.11421-sandybrown?style=flat-square)](https://arxiv.org/abs/2403.11421)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2403.11421.pdf)

**Accelerating ViT Inference on FPGA through Static and Dynamic Pruning**

![](https://img.shields.io/badge/Arxiv-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2403.14047-sandybrown?style=flat-square)](https://arxiv.org/abs/2403.14047)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2403.14047.pdf)

**Allspark: Workload Orchestration for Visual Transformers on Processing In-Memory Systems**

![](https://img.shields.io/badge/Arxiv-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2403.15069-sandybrown?style=flat-square)](https://arxiv.org/abs/2403.15069)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2403.15069.pdf)

**Impact of High-Level-Synthesis on Reliability of Artificial Neural Network Hardware Accelerators**

![](https://img.shields.io/badge/IEEE-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/TNS.2024.3377596-sandybrown?style=flat-square)](https://doi.org/10.1109/TNS.2024.3377596)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://inria.hal.science/hal-04514579/file/TNS2024_HLS.pdf)

**An FPGA-Based Reconfigurable Accelerator for Convolution-Transformer Hybrid EfficientViT**

![](https://img.shields.io/badge/Arxiv-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2403.20230-sandybrown?style=flat-square)](https://arxiv.org/abs/2403.20230)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2403.20230.pdf)

**TransFRU: Efficient Deployment of Transformers on FPGA with Full Resource Utilization**

![](https://img.shields.io/badge/IEEE-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ASP--DAC58780.2024.10473976-sandybrown?style=flat-square)](https://doi.org/10.1109/ASP-DAC58780.2024.10473976)


**PRIMATE: Processing in Memory Acceleration for Dynamic Token-pruning Transformers**

![](https://img.shields.io/badge/IEEE-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ASP--DAC58780.2024.10473968-sandybrown?style=flat-square)](https://doi.org/10.1109/ASP-DAC58780.2024.10473968)


**SWAT: An Efficient Swin Transformer Accelerator Based on FPGA**

![](https://img.shields.io/badge/IEEE-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ASP--DAC58780.2024.10473931-sandybrown?style=flat-square)](https://doi.org/10.1109/ASP-DAC58780.2024.10473931)


**VTR: An Optimized Vision Transformer for SAR ATR Acceleration on FPGA**

![](https://img.shields.io/badge/Arxiv-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2404.04527v1-sandybrown?style=flat-square)](https://arxiv.org/abs/2404.04527v1)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2404.04527v1)

**Workload-Aware Hardware Accelerator Mining for Distributed Deep Learning Training**

![](https://img.shields.io/badge/Arxiv-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2404.14632v1-sandybrown?style=flat-square)](https://arxiv.org/abs/2404.14632v1)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2404.14632v1)

**QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving**

![](https://img.shields.io/badge/Arxiv-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2405.04532-sandybrown?style=flat-square)](https://arxiv.org/abs/2405.04532)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2405.04532)

**NeuPIMs: NPU-PIM Heterogeneous Acceleration for Batched LLM Inferencing**

![](https://img.shields.io/badge/ACM-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://dl.acm.org/doi/abs/10.1145/3620666.3651380-sandybrown?style=flat-square)](https://dl.acm.org/doi/abs/10.1145/3620666.3651380)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://dl.acm.org/doi/pdf/10.1145/3620666.3651380)

**VITA: ViT Acceleration for Efficient 3D Human Mesh Recovery via Hardware-Algorithm Co-Design**

![](https://img.shields.io/badge/ACM-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-False-sandybrown?style=flat-square)](False)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://www.crcv.ucf.edu/chenchen/2024_DAC_VITA_Final.pdf)

**HLSTransform: Energy-Efficient Llama 2 Inference on FPGAs Via High Level Synthesis**

![](https://img.shields.io/badge/Arxiv-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2405.00738-sandybrown?style=flat-square)](https://arxiv.org/abs/2405.00738)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2405.00738)

**SCAR: Scheduling Multi-Model AI Workloads on Heterogeneous Multi-Chiplet Module Accelerators**

![](https://img.shields.io/badge/Arxiv-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2405.00790-sandybrown?style=flat-square)](https://arxiv.org/abs/2405.00790)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2405.00790)

**Trio-ViT: Post-Training Quantization and Acceleration for Softmax-Free Efficient Vision Transformer**

![](https://img.shields.io/badge/Arxiv-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2405.03882-sandybrown?style=flat-square)](https://arxiv.org/abs/2405.03882)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2405.03882)

**SambaNova SN40L: Scaling the AI Memory Wall with Dataflow and Composition of Experts**

![](https://img.shields.io/badge/Arxiv-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2405.07518-sandybrown?style=flat-square)](https://arxiv.org/abs/2405.07518)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2405.07518)

**TensorMap: A Deep RL-Based Tensor Mapping Framework for Spatial Accelerators**

![](https://img.shields.io/badge/IEEE-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.ieeecomputersociety.org/10.1109/TC.2024.3398424-sandybrown?style=flat-square)](https://doi.ieeecomputersociety.org/10.1109/TC.2024.3398424)


**JIT-Q: Just-in-time Quantization with Processing-In-Memory for Efficient ML Training**

![](https://img.shields.io/badge/MLSys-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2311.05034-sandybrown?style=flat-square)](https://arxiv.org/abs/2311.05034)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://proceedings.mlsys.org/paper_files/paper/2024/file/136b9a13861308c8948cd308ccd02658-Paper-Conference.pdf)

**DCT-ViT: High-Frequency Pruned Vision Transformer with Discrete Cosine Transform**

![](https://img.shields.io/badge/IEEE-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ACCESS.2024.3410231-sandybrown?style=flat-square)](https://doi.org/10.1109/ACCESS.2024.3410231)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10549904)

**TransAxx: Efficient Transformers with Approximate Computing**

![](https://img.shields.io/badge/Arxiv-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2402.07545-sandybrown?style=flat-square)](https://arxiv.org/abs/2402.07545)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2402.07545)

**ITA: An Energy-Efficient Attention and Softmax Accelerator for Quantized Transformers**

![](https://img.shields.io/badge/Arxiv-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2307.03493-sandybrown?style=flat-square)](https://arxiv.org/abs/2307.03493)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2307.03493)

**CGRA4ML: A Framework to Implement Modern Neural Networks for Scientific Edge Computing**

![](https://img.shields.io/badge/Arxiv-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2408.15561-sandybrown?style=flat-square)](https://arxiv.org/abs/2408.15561)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2408.15561)

**ProTEA: Programmable Transformer Encoder Acceleration on FPGA**

![](https://img.shields.io/badge/Arxiv-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2409.13975-sandybrown?style=flat-square)](https://arxiv.org/abs/2409.13975)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2409.13975)

**CAT: Customized Transformer Accelerator Framework on Versal ACAP**

![](https://img.shields.io/badge/Arxiv-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2409.09689-sandybrown?style=flat-square)](https://arxiv.org/abs/2409.09689)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2409.09689)

**Co-design of a TinyLLM using Programmable Logic and Software on an FPGA**

![](https://img.shields.io/badge/IEEE-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/MWSCAS60917.2024.10658754-sandybrown?style=flat-square)](https://doi.org/10.1109/MWSCAS60917.2024.10658754)


**BitShare: An Efficient Precision-Scalable Accelerator with Combining-Like-Terms GEMM**

![](https://img.shields.io/badge/IEEE-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-RESEARCH_GATE-sandybrown?style=flat-square)](https://www.researchgate.net/publication/381370829_BitShare_An_Efficient_Precision-Scalable_Accelerator_with_Combining-Like-Terms_GEMM)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://www.researchgate.net/profile/Junzhong-Shen/publication/381370829_BitShare_An_Efficient_Precision-Scalable_Accelerator_with_Combining-Like-Terms_GEMM/links/666a46cba54c5f0b94613261/BitShare-An-Efficient-Precision-Scalable-Accelerator-with-Combining-Like-Terms-GEMM.pdf)

**SDA: Low-Bit Stable Diffusion Acceleration on Edge FPGA**

![](https://img.shields.io/badge/IEEE-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-NO_DATA-sandybrown?style=flat-square)](https://github.com/Michaela1224/SDA_code)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://www.sfu.ca/~zhenman/files/C41-FPL2024-SDA.pdf)

**Hardware Accelerator for MobileViT Vision Transformer with Reconfigurable Computation**

![](https://img.shields.io/badge/IEEE-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ISCAS58744.2024.10558190-sandybrown?style=flat-square)](https://doi.org/10.1109/ISCAS58744.2024.10558190)


**In-Memory Transformer Self-Attention Mechanism Using Passive Memristor Crossbar**

![](https://img.shields.io/badge/IEEE-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ISCAS58744.2024.10558182-sandybrown?style=flat-square)](https://doi.org/10.1109/ISCAS58744.2024.10558182)


**A 3.55 mJ/frame Energy-efficient Mixed-Transformer based Semantic Segmentation Accelerator for Mobile Devices**

![](https://img.shields.io/badge/IEEE-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ISCAS58744.2024.10558649-sandybrown?style=flat-square)](https://doi.org/10.1109/ISCAS58744.2024.10558649)


**FLAG: Formula-LLM-Based Auto-Generator for Baseband Hardware**

![](https://img.shields.io/badge/IEEE-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ISCAS58744.2024.10558482-sandybrown?style=flat-square)](https://doi.org/10.1109/ISCAS58744.2024.10558482)


**CV-CIM: A Hybrid Domain Xor-Derived Similarity-Aware Computation-in-Memory Supporting Cost-Volume Construction**

![](https://img.shields.io/badge/IEEE-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/JSSC.2024.3421589-sandybrown?style=flat-square)](https://doi.org/10.1109/JSSC.2024.3421589)


**LPU: A Latency-Optimized and Highly Scalable Processor for Large Language Model Inference**

![](https://img.shields.io/badge/USENIX-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://www.usenix.org/conference/osdi24/presentation/zhuang-sandybrown?style=flat-square)](https://www.usenix.org/conference/osdi24/presentation/zhuang)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://www.usenix.org/system/files/osdi24-zhuang.pdf)

**ARTEMIS: A Mixed Analog-Stochastic In-DRAM Accelerator for Transformer Neural Networks**

![](https://img.shields.io/badge/Arxiv-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2407.12638-sandybrown?style=flat-square)](https://arxiv.org/abs/2407.12638)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2407.12638)

**CHOSEN: Compilation to Hardware Optimization Stack for Efficient Vision Transformer Inference**

![](https://img.shields.io/badge/Arxiv-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2407.12736-sandybrown?style=flat-square)](https://arxiv.org/abs/2407.12736)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2407.12736)

**Co-Designing Binarized Transformer and Hardware Accelerator for Efficient End-to-End Edge Deployment**

![](https://img.shields.io/badge/Arxiv-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2407.12070-sandybrown?style=flat-square)](https://arxiv.org/abs/2407.12070)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2407.12070)

**SPSA: Exploring Sparse-Packing Computation on Systolic Arrays From Scratch**

![](https://img.shields.io/badge/IEEE-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/TCAD.2024.3434359-sandybrown?style=flat-square)](https://doi.org/10.1109/TCAD.2024.3434359)


**SPSA: Exploring Sparse-Packing Computation on Systolic Arrays From Scratch**

![](https://img.shields.io/badge/IEEE-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/TCAD.2024.3434447-sandybrown?style=flat-square)](https://doi.org/10.1109/TCAD.2024.3434447)


**MECLA: Memory-Compute-Efficient LLM Accelerator with Scaling Sub-matrix Partition**

![](https://img.shields.io/badge/IEEE-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ISCA59077.2024.00079-sandybrown?style=flat-square)](https://doi.org/10.1109/ISCA59077.2024.00079)


**TCP: A Tensor Contraction Processor for AI Workloads Industrial Product**

![](https://img.shields.io/badge/IEEE-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ISCA59077.2024.00069-sandybrown?style=flat-square)](https://doi.org/10.1109/ISCA59077.2024.00069)


**A 109-GOPs/W FPGA-based Vision Transformer Accelerator with Weight-Loop Dataflow Featuring Data Reusing and Resource Saving**

![](https://img.shields.io/badge/IEEE-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/TCSVT.2024.3439600-sandybrown?style=flat-square)](https://doi.org/10.1109/TCSVT.2024.3439600)


**Klotski v2: Improved DNN Model Orchestration Framework for Dataflow Architecture Accelerators**

![](https://img.shields.io/badge/IEEE-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/TCAD.2024.3446858-sandybrown?style=flat-square)](https://doi.org/10.1109/TCAD.2024.3446858)


**Quartet: A Holistic Hybrid Parallel Framework for Training Large Language Models**

![](https://img.shields.io/badge/Springer-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1007/978--3--031--69766--1_29-sandybrown?style=flat-square)](https://doi.org/10.1007/978-3-031-69766-1_29)


**Inference with Transformer Encoders on ARM and RISC-V Multicore Processors**

![](https://img.shields.io/badge/Springer-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1007/978--3--031--69766--1_26-sandybrown?style=flat-square)](https://doi.org/10.1007/978-3-031-69766-1_26)


**Mentor: A Memory-Eficient Sparse-dense Matrix Multiplication Accelerator Based on Column-Wise Product**

![](https://img.shields.io/badge/ACM-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://dl.acm.org/doi/pdf/10.1145/3688612-sandybrown?style=flat-square)](https://dl.acm.org/doi/pdf/10.1145/3688612)


**Cost-Effective LLM Accelerator Using Processing in Memory Technology**

![](https://img.shields.io/badge/IEEE-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/VLSITechnologyandCir46783.2024.10631397-sandybrown?style=flat-square)](https://doi.org/10.1109/VLSITechnologyandCir46783.2024.10631397)


**A 28nm 4.35TOPS/mm2 Transformer Accelerator with Basis-vector Based Ultra Storage Compression, Decomposed Computation and Unified LUT-Assisted Cores**

![](https://img.shields.io/badge/IEEE-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/VLSITechnologyandCir46783.2024.10631311-sandybrown?style=flat-square)](https://doi.org/10.1109/VLSITechnologyandCir46783.2024.10631311)


**FPGA-Based Sparse Matrix Multiplication Accelerators: From State-of-the-art to Future Opportunities**

![](https://img.shields.io/badge/ACM-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3687480-sandybrown?style=flat-square)](https://doi.org/10.1145/3687480)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://dl.acm.org/doi/pdf/10.1145/3687480)

**LPU: A Latency-Optimized and Highly Scalable Processor for Large Language Model Inference**

![](https://img.shields.io/badge/IEEE-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.ieeecomputersociety.org/10.1109/MM.2024.3420728-sandybrown?style=flat-square)](https://doi.ieeecomputersociety.org/10.1109/MM.2024.3420728)


**Efficient Transformer Acceleration via Reconfiguration for Encoder and Decoder Models and Sparsity-Aware Algorithm Mapping**

![](https://img.shields.io/badge/ACM-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3665314.3670798-sandybrown?style=flat-square)](https://doi.org/10.1145/3665314.3670798)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://dl.acm.org/doi/pdf/10.1145/3665314.3670798)

**VisionAGILE: A Versatile Domain-Specific Accelerator for Computer Vision Tasks**

![](https://img.shields.io/badge/ACM-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.ieeecomputersociety.org/10.1109/TPDS.2024.3466891-sandybrown?style=flat-square)](https://doi.ieeecomputersociety.org/10.1109/TPDS.2024.3466891)


**Cambricon-LLM: A Chiplet-Based Hybrid Architecture for On-Device Inference of 70B LLM**

![](https://img.shields.io/badge/Arxive-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2409.15654-sandybrown?style=flat-square)](https://arxiv.org/abs/2409.15654)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2409.15654)

**FAMOUS: Flexible Accelerator for the Attention Mechanism of Transformer on UltraScale+ FPGAs**

![](https://img.shields.io/badge/Arxive-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2409.14023-sandybrown?style=flat-square)](https://arxiv.org/abs/2409.14023)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2409.14023)

**Hardware-oriented algorithms for softmax and layer normalization of large language models**

![](https://img.shields.io/badge/Arxive-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1007/s11432--024--4137--4-sandybrown?style=flat-square)](https://doi.org/10.1007/s11432-024-4137-4)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2409.14023)

**Optimizing DNN Inference on Multi-Accelerator SoCs at Training-time**

![](https://img.shields.io/badge/Arxive-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2409.18566-sandybrown?style=flat-square)](https://arxiv.org/abs/2409.18566)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2409.18566)

**Optimizing DNN Inference on Multi-Accelerator SoCs at Training-time**

![](https://img.shields.io/badge/Arxive-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2409.18566-sandybrown?style=flat-square)](https://arxiv.org/abs/2409.18566)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2409.18566)

**DSTC: Dual-Side Sparse Tensor Core for DNNs Acceleration on Modern GPU Architectures**

![](https://img.shields.io/badge/IEEE-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.ieeecomputersociety.org/10.1109/TC.2024.3475814-sandybrown?style=flat-square)](https://doi.ieeecomputersociety.org/10.1109/TC.2024.3475814)


**Power Efficient ASIC Design for Vision Transformer using Systolic Array Matrix Multiplier**

![](https://img.shields.io/badge/IEEE-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/VDAT63601.2024.10705728-sandybrown?style=flat-square)](https://doi.org/10.1109/VDAT63601.2024.10705728)


**M^2-ViT: Accelerating Hybrid Vision Transformers with Two-Level Mixed Quantization**

![](https://img.shields.io/badge/Arxive-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2410.09113-sandybrown?style=flat-square)](https://arxiv.org/abs/2410.09113)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2410.09113)

**A Cascaded ReRAM-based Crossbar Architecture for Transformer Neural Network Acceleration**

![](https://img.shields.io/badge/ACM-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3701034-sandybrown?style=flat-square)](https://doi.org/10.1145/3701034)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://dl.acm.org/doi/pdf/10.1145/3701034)

**OPASCA: Outer Product Based Accelerator With Unified Architecture for Sparse Convolution and Attention**

![](https://img.shields.io/badge/IEEE-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/TCAD.2024.3483092-sandybrown?style=flat-square)](https://doi.org/10.1109/TCAD.2024.3483092)


**HotaQ: Hardware Oriented Token Adaptive Quantization for Large Language Models**

![](https://img.shields.io/badge/IEEE-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/TCAD.2024.3487781-sandybrown?style=flat-square)](https://doi.org/10.1109/TCAD.2024.3487781)


**Analysis Towards Deployment and Acceleration for ViT on a Lightweight RISC- V Processor**

![](https://img.shields.io/badge/IEEE-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/IAI63275.2024.10730301-sandybrown?style=flat-square)](https://doi.org/10.1109/IAI63275.2024.10730301)


**Improving Transformer Inference Through Optimized Non-Linear Operations With Quantization-Approximation-Based Strategy**

![](https://img.shields.io/badge/IEEE-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/TCAD.2024.3488572-sandybrown?style=flat-square)](https://doi.org/10.1109/TCAD.2024.3488572)


**HyCTor: A Hybrid CNN-Transformer Network Accelerator With Flexible Weight/Output Stationary Dataflow and Multi-Core Extension**

![](https://img.shields.io/badge/IEEE-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/TCAD.2024.3490173-sandybrown?style=flat-square)](https://doi.org/10.1109/TCAD.2024.3490173)


**Shrinking the Giant : Quasi-Weightless Transformers for Low Energy Inference**

![](https://img.shields.io/badge/Arxive-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2411.01818-sandybrown?style=flat-square)](https://arxiv.org/abs/2411.01818)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2411.01818)

**Multilayer Dataflow based Butterfly Sparsity Orchestration to Accelerate Attention Workloads**

![](https://img.shields.io/badge/Arxive-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2411.00734-sandybrown?style=flat-square)](https://arxiv.org/abs/2411.00734)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2411.00734)

**TATAA: Programmable Mixed-Precision Transformer Acceleration with a Transformable Arithmetic Architecture**

![](https://img.shields.io/badge/Arxive-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2411.03697-sandybrown?style=flat-square)](https://arxiv.org/abs/2411.03697)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2411.03697)

**Exploring Approximation and Dataflow Co-Optimization for Scalable Transformer Inference Architecture on the Edge**

![](https://img.shields.io/badge/IEEE-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/SOCC62300.2024.10737793-sandybrown?style=flat-square)](https://doi.org/10.1109/SOCC62300.2024.10737793)


**Vision Transformer Inference on a CNN Accelerator**

![](https://img.shields.io/badge/IEEE-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ICCD63220.2024.00101-sandybrown?style=flat-square)](https://doi.org/10.1109/ICCD63220.2024.00101)


**Addressing Architectural Obstacles for Overlay with Stream Network Abstraction**

![](https://img.shields.io/badge/Arxive-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2411.17966-sandybrown?style=flat-square)](https://arxiv.org/abs/2411.17966)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2411.17966)

**A Runtime-Adaptive Transformer Neural Network Accelerator on FPGAs**

![](https://img.shields.io/badge/Arxive-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2411.18148-sandybrown?style=flat-square)](https://arxiv.org/abs/2411.18148)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2411.18148)

**A Dataflow Compiler for Efficient LLM Inference using Custom Microscaling Formats**

![](https://img.shields.io/badge/Arxive-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2307.15517-sandybrown?style=flat-square)](https://arxiv.org/abs/2307.15517)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2307.15517)

**MR-Transformer: FPGA Accelerated Deep Learning Attention Model for Modulation Recognition**

![](https://img.shields.io/badge/IEEE-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/TWC.2024.3506743-sandybrown?style=flat-square)](https://doi.org/10.1109/TWC.2024.3506743)


**ISOAcc: In-situ Shift Operation-based Accelerator For Efficient in-SRAM Multiplication**

![](https://img.shields.io/badge/ACM-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3707205-sandybrown?style=flat-square)](https://doi.org/10.1145/3707205)


**VGA: Hardware Accelerator for Scalable Long Sequence Model Inference**

![](https://img.shields.io/badge/ACM-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.ieeecomputersociety.org/10.1109/MICRO61859.2024.00106-sandybrown?style=flat-square)](https://doi.ieeecomputersociety.org/10.1109/MICRO61859.2024.00106)


**Hardware Accelerated Vision Transformer via Heterogeneous Architecture Design and Adaptive Dataflow Mapping**

![](https://img.shields.io/badge/ACM-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.ieeecomputersociety.org/10.1109/TC.2024.3517751-sandybrown?style=flat-square)](https://doi.ieeecomputersociety.org/10.1109/TC.2024.3517751)


**IMTP: Search-based Code Generation for In-memory Tensor Programs**

![](https://img.shields.io/badge/Arxiv-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2412.19630-sandybrown?style=flat-square)](https://arxiv.org/abs/2412.19630)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2412.19630)

**Debunking the CUDA Myth Towards GPU-based AI Systems**

![](https://img.shields.io/badge/Arxiv-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2501.00210-sandybrown?style=flat-square)](https://arxiv.org/abs/2501.00210)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2501.00210)

**A Fully-Integrated Energy-Scalable Transformer Accelerator for Language Understanding on Edge Devices**

![](https://img.shields.io/badge/IEEE-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/OJSSCS.2024.3524359-sandybrown?style=flat-square)](https://doi.org/10.1109/OJSSCS.2024.3524359)


**AxLaM: energy-efficient accelerator design for language models for edge computing**

![](https://img.shields.io/badge/RoyalSociety-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1098/rsta.2023.0395-sandybrown?style=flat-square)](https://doi.org/10.1098/rsta.2023.0395)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://doi.org/10.1098/rsta.2023.0395)

**FLARE: FP-Less PTQ and Low-ENOB ADC Based AMS-PiM for Error-Resilient, Fast, and Efficient Transformer Acceleration**

![](https://img.shields.io/badge/Arxiv-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2411.14733-sandybrown?style=flat-square)](https://arxiv.org/abs/2411.14733)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2411.14733)

**Optimized Spatial Architecture Mapping Flow for Transformer Accelerators**

![](https://img.shields.io/badge/Arxiv-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2410.07407-sandybrown?style=flat-square)](https://arxiv.org/abs/2410.07407)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2410.07407)

**OPAL: Outlier-Preserved Microscaling Quantization Accelerator for Generative Large Language Models**

![](https://img.shields.io/badge/Arxiv-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2409.05902-sandybrown?style=flat-square)](https://arxiv.org/abs/2409.05902)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2409.05902)

**Research on LLM Acceleration Using the High-Performance RISC-V Processor "Xiangshan" (Nanhu Version) Based on the Open-Source Matrix Instruction Set Extension (Vector Dot Product)**

![](https://img.shields.io/badge/Arxiv-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2409.00661-sandybrown?style=flat-square)](https://arxiv.org/abs/2409.00661)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2409.00661)

**Designing Efficient LLM Accelerators for Edge Devices**

![](https://img.shields.io/badge/Arxiv-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2408.00462-sandybrown?style=flat-square)](https://arxiv.org/abs/2408.00462)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2408.00462)

**SWAT: Scalable and Efficient Window Attention-based Transformers Acceleration on FPGAs**

![](https://img.shields.io/badge/Arxiv-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2405.17025-sandybrown?style=flat-square)](https://arxiv.org/abs/2405.17025)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2405.17025)

**ME-ViT: A Single-Load Memory-Efficient FPGA Accelerator for Vision Transformers**

![](https://img.shields.io/badge/Arxiv-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2402.09709-sandybrown?style=flat-square)](https://arxiv.org/abs/2402.09709)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2402.09709)

**IANUS: Integrated Accelerator based on NPU-PIM Unified Memory System**

![](https://img.shields.io/badge/Arxiv-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2410.15008-sandybrown?style=flat-square)](https://arxiv.org/abs/2410.15008)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2410.15008)

**LlamaF: An Efficient Llama2 Architecture Accelerator on Embedded FPGAs**

![](https://img.shields.io/badge/Arxiv-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2409.11424-sandybrown?style=flat-square)](https://arxiv.org/abs/2409.11424)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2409.11424)

**EdgeLLM: A Highly Efficient CPU-FPGA Heterogeneous Edge Accelerator for Large Language Models**

![](https://img.shields.io/badge/Arxiv-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2407.21325-sandybrown?style=flat-square)](https://arxiv.org/abs/2407.21325)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2407.21325)

**Integer-only Quantized Transformers for Embedded FPGA-based Time-series Forecasting in AIoT**

![](https://img.shields.io/badge/Arxiv-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2407.11041-sandybrown?style=flat-square)](https://arxiv.org/abs/2407.11041)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2407.11041)

**Torch2Chip: An End-to-end Customizable Deep Neural Network Compression and Deployment Toolkit for Prototype Hardware Accelerator Design**

![](https://img.shields.io/badge/Arxiv-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2405.01775-sandybrown?style=flat-square)](https://arxiv.org/abs/2405.01775)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2405.01775)

**An FPGA-Based Accelerator Enabling Efficient Support for CNNs with Arbitrary Kernel Sizes**

![](https://img.shields.io/badge/Arxiv-2024-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2402.14307-sandybrown?style=flat-square)](https://arxiv.org/abs/2402.14307)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2402.14307)

---
### 2025
**MixGCN: Scalable GCN Training by Mixture of Parallelism and Mixture of Accelerators**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2501.01951-sandybrown?style=flat-square)](https://arxiv.org/abs/2501.01951)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2501.01951)

**Ultra Memory-Efficient On-FPGA Training of Transformers via Tensor-Compressed Optimization**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2501.06663-sandybrown?style=flat-square)](https://arxiv.org/abs/2501.06663)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2501.06663)

**An Efficient Sparse Hardware Accelerator for Spike-Driven Transformer**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2501.07825-sandybrown?style=flat-square)](https://arxiv.org/abs/2501.07825)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2501.07825)

**MPTorch-FPGA: a Custom Mixed-Precision Framework for FPGA-based DNN Training**

![](https://img.shields.io/badge/IEEE-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://hal.science/hal--04882989v2-sandybrown?style=flat-square)](https://hal.science/hal-04882989v2)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://hal.science/hal-04882989v2)

**Atleus: Accelerating Transformers on the Edge Enabled by 3D Heterogeneous Manycore Architectures**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2501.09588-sandybrown?style=flat-square)](https://arxiv.org/abs/2501.09588)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2501.09588)

**Stream-HLS: Towards Automatic Dataflow Acceleration**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2501.09118-sandybrown?style=flat-square)](https://arxiv.org/abs/2501.09118)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2501.09118)

**An Efficient Window-Based Vision Transformer Accelerator via Mixed-Granularity Sparsity**

![](https://img.shields.io/badge/IEEE-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/TCSI.2025.3527541-sandybrown?style=flat-square)](https://doi.org/10.1109/TCSI.2025.3527541)


**Hardware Accelerator for Bidirectional Encoder Representations from Transformers (BERT)**

![](https://img.shields.io/badge/IEEE-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ICM63406.2024.10815822-sandybrown?style=flat-square)](https://doi.org/10.1109/ICM63406.2024.10815822)


**Fine-Grained Structured Sparse Computing for FPGA-Based AI Inference**

![](https://img.shields.io/badge/IEEE-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/TCAD.2024.3524356-sandybrown?style=flat-square)](https://doi.org/10.1109/TCAD.2024.3524356)


**An ultra-low-power CGRA for accelerating Transformers at the edge**

![](https://img.shields.io/badge/HAL-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://hal.science/hal--04914400v1-sandybrown?style=flat-square)](https://hal.science/hal-04914400v1)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://hal.science/hal-04914400v1)

**A Tensor-Train Decomposition based Compression of LLMs on Group Vector Systolic Accelerator**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2501.19135-sandybrown?style=flat-square)](https://arxiv.org/abs/2501.19135)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2501.19135)

**Towards Efficient LUT-based PIM: A Scalable and Low-Power Approach for Modern Workloads**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2502.02142-sandybrown?style=flat-square)](https://arxiv.org/abs/2502.02142)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2502.02142)

**Systolic Sparse Tensor Slices: FPGA Building Blocks for Sparse and Dense AI Acceleration**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2502.03763-sandybrown?style=flat-square)](https://arxiv.org/abs/2502.03763)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2502.03763)

**Hybrid Photonic-digital Accelerator for Attention Mechanism**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2501.11286-sandybrown?style=flat-square)](https://arxiv.org/abs/2501.11286)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2501.11286)

**HeteroLLM: Accelerating Large Language Model Inference on Mobile SoCs platform with Heterogeneous AI Accelerators**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2501.14794-sandybrown?style=flat-square)](https://arxiv.org/abs/2501.14794)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2501.14794)

**DRViT: A dynamic redundancy-aware vision transformer accelerator via algorithm and architecture co-design on FPGA**

![](https://img.shields.io/badge/Elsevier-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1016/j.jpdc.2025.105042-sandybrown?style=flat-square)](https://doi.org/10.1016/j.jpdc.2025.105042)


**StreamDCIM: A Tile-based Streaming Digital CIM Accelerator with Mixed-stationary Cross-forwarding Dataflow for Multimodal Transformer**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2502.05798-sandybrown?style=flat-square)](https://arxiv.org/abs/2502.05798)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2502.05798)

**StreamDCIM: A Tile-based Streaming Digital CIM Accelerator with Mixed-stationary Cross-forwarding Dataflow for Multimodal Transformer**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2502.04563-sandybrown?style=flat-square)](https://arxiv.org/abs/2502.04563)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2502.04563)

**Data-aware Dynamic Execution of Irregular Workloads on Heterogeneous Systems**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2502.06304-sandybrown?style=flat-square)](https://arxiv.org/abs/2502.06304)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2502.06304)

**UbiMoE: A Ubiquitous Mixture-of-Experts Vision Transformer Accelerator With Hybrid Computation Pattern on FPGA**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2502.05602-sandybrown?style=flat-square)](https://arxiv.org/abs/2502.05602)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2502.05602)

**A 28nm 0.22Î¼J/token memory-compute-intensity-aware CNN-Transformer accelerator with hybrid-attention-based layer-fusion and cascaded pruning for semantic-segmentation**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2512.17555-sandybrown?style=flat-square)](https://arxiv.org/abs/2512.17555)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2512.17555)

**SeVeDo: A Heterogeneous Transformer Accelerator for Low-Bit Inference via Hierarchical Group Quantization and SVD-Guided Mixed Precision**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2512.12930-sandybrown?style=flat-square)](https://arxiv.org/abs/2512.12930)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2512.12930)

**PD-Swap: Prefill-Decode Logic Swapping for End-to-End LLM Inference on Edge FPGAs via Dynamic Partial Reconfiguration**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2512.11550-sandybrown?style=flat-square)](https://arxiv.org/abs/2512.11550)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2512.11550)

**RIFT: A Scalable Methodology for LLM Accelerator Fault Assessment using Reinforcement Learning**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2512.09829-sandybrown?style=flat-square)](https://arxiv.org/abs/2512.09829)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2512.09829)

**DCO: Dynamic Cache Orchestration for LLM Accelerators through Predictive Management**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2512.07312-sandybrown?style=flat-square)](https://arxiv.org/abs/2512.07312)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2512.07312)

**BitStopper: An Efficient Transformer Attention Accelerator via Stage-fusion and Early Termination**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2512.06457-sandybrown?style=flat-square)](https://arxiv.org/abs/2512.06457)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2512.06457)

**P3-LLM: An Integrated NPU-PIM Accelerator for Edge LLM Inference Using Hybrid Numerical Formats**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2511.06838-sandybrown?style=flat-square)](https://arxiv.org/abs/2511.06838)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2511.06838)

**Low Power Vision Transformer Accelerator with Hardware-Aware Pruning and Optimized Dataflow**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2510.14393-sandybrown?style=flat-square)](https://arxiv.org/abs/2510.14393)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2510.14393)

**TeLLMe v2: An Efficient End-to-End Ternary LLM Prefill and Decode Accelerator with Table-Lookup Matmul on Edge FPGAs**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2510.15926-sandybrown?style=flat-square)](https://arxiv.org/abs/2510.15926)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2510.15926)

**SnipSnap: A Joint Compression Format and Dataflow Co-Optimization Framework for Efficient Sparse LLM Accelerator Design**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2509.17072-sandybrown?style=flat-square)](https://arxiv.org/abs/2509.17072)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2509.17072)

**CompAir: Synergizing Complementary PIMs and In-Transit NoC Computation for Efficient LLM Acceleration**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2509.13710-sandybrown?style=flat-square)](https://arxiv.org/abs/2509.13710)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2509.13710)

**StreamTensor: Make Tensors Stream in Dataflow Accelerators for LLMs**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2509.13694-sandybrown?style=flat-square)](https://arxiv.org/abs/2509.13694)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2509.13694)

**MCBP: A Memory-Compute Efficient LLM Inference Accelerator Leveraging Bit-Slice-enabled Sparsity and Repetitiveness**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2509.10372-sandybrown?style=flat-square)](https://arxiv.org/abs/2509.10372)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2509.10372)

**Opto-ViT: Architecting a Near-Sensor Region of Interest-Aware Vision Transformer Accelerator with Silicon Photonics**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2507.07044-sandybrown?style=flat-square)](https://arxiv.org/abs/2507.07044)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2507.07044)

**COBRA: Algorithm-Architecture Co-optimized Binary Transformer Accelerator for Edge Inference**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2504.16269-sandybrown?style=flat-square)](https://arxiv.org/abs/2504.16269)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2504.16269)

**TeLLMe: An Energy-Efficient Ternary LLM Accelerator for Prefilling and Decoding on Edge FPGAs**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2504.16266-sandybrown?style=flat-square)](https://arxiv.org/abs/2504.16266)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2504.16266)

**UniCAIM: A Unified CAM/CIM Architecture with Static-Dynamic KV Cache Pruning for Efficient Long-Context LLM Inference**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2504.07479-sandybrown?style=flat-square)](https://arxiv.org/abs/2504.07479)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2504.07479)

**PIM-LLM: A High-Throughput Hybrid PIM Architecture for 1-bit LLMs**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2504.01994-sandybrown?style=flat-square)](https://arxiv.org/abs/2504.01994)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2504.01994)

**VESTA: A Versatile SNN-Based Transformer Accelerator with Unified PEs for Multiple Computational Layers**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2503.20246-sandybrown?style=flat-square)](https://arxiv.org/abs/2503.20246)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2503.20246)

**An Efficient Data Reuse with Tile-Based Adaptive Stationary for Transformer Accelerators**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2503.19640-sandybrown?style=flat-square)](https://arxiv.org/abs/2503.19640)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2503.19640)

**NeCTAr: A Heterogeneous RISC-V SoC for Language Model Inference in Intel 16**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2503.14708-sandybrown?style=flat-square)](https://arxiv.org/abs/2503.14708)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2503.14708)

**A Review on Proprietary Accelerators for Large Language Models**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2503.09650-sandybrown?style=flat-square)](https://arxiv.org/abs/2503.09650)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2503.09650)

**T-REX: A 68-567 Î¼s/token, 0.41-3.95 Î¼J/token Transformer Accelerator with Reduced External Memory Access and Enhanced Hardware Utilization in 16nm FinFET**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2503.00322-sandybrown?style=flat-square)](https://arxiv.org/abs/2503.00322)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2503.00322)

**Vision Transformer Accelerator ASIC for Real-Time Low-Power Sleep Staging**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2502.16334-sandybrown?style=flat-square)](https://arxiv.org/abs/2502.16334)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2502.16334)

**LightMamba: Efficient Mamba Acceleration on FPGA with Quantization and Hardware Co-design**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2502.15260-sandybrown?style=flat-square)](https://arxiv.org/abs/2502.15260)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2502.15260)

**FPGA Co-Design for Efficient N:M Sparse and Quantized Model Inference**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2512.24713-sandybrown?style=flat-square)](https://arxiv.org/abs/2512.24713)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2512.24713)

**CXL-SpecKV: A Disaggregated FPGA Speculative KV-Cache for Datacenter LLM Serving**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2512.11920-sandybrown?style=flat-square)](https://arxiv.org/abs/2512.11920)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2512.11920)

**LUT-LLM: Efficient Large Language Model Inference with Memory-based Computations on FPGAs**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2511.06174-sandybrown?style=flat-square)](https://arxiv.org/abs/2511.06174)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2511.06174)

**FaRAccel: FPGA-Accelerated Defense Architecture for Efficient Bit-Flip Attack Resilience in Transformer Models**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2510.24985-sandybrown?style=flat-square)](https://arxiv.org/abs/2510.24985)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2510.24985)

**ISAAC: Intelligent, Scalable, Agile, and Accelerated CPU Verification via LLM-aided FPGA Parallelism**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2510.10225-sandybrown?style=flat-square)](https://arxiv.org/abs/2510.10225)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2510.10225)

**TrackCore-F: Deploying Transformer-Based Subatomic Particle Tracking on FPGAs**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2509.26335-sandybrown?style=flat-square)](https://arxiv.org/abs/2509.26335)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2509.26335)

**Real Time FPGA Based Transformers &amp; VLMs for Vision Tasks: SOTA Designs and Optimizations**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2509.04162-sandybrown?style=flat-square)](https://arxiv.org/abs/2509.04162)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2509.04162)

**Systolic Array-based Architecture for Low-Bit Integerized Vision Transformers**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2508.20334-sandybrown?style=flat-square)](https://arxiv.org/abs/2508.20334)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2508.20334)

**DiffAxE: Diffusion-driven Hardware Accelerator Generation and Design Space Exploration**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2508.10303-sandybrown?style=flat-square)](https://arxiv.org/abs/2508.10303)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2508.10303)

**Hummingbird: A Smaller and Faster Large Language Model Accelerator on Embedded FPGA**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2507.03308-sandybrown?style=flat-square)](https://arxiv.org/abs/2507.03308)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2507.03308)

**CoQMoE: Co-Designed Quantization and Computation Orchestration for Mixture-of-Experts Vision Transformer on FPGA**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2506.08496-sandybrown?style=flat-square)](https://arxiv.org/abs/2506.08496)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2506.08496)

**Dataflow &amp; Tiling Strategies in Edge-AI FPGA Accelerators: A Comprehensive Literature Review**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2505.08992-sandybrown?style=flat-square)](https://arxiv.org/abs/2505.08992)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2505.08992)

**Design and Implementation of an FPGA-Based Hardware Accelerator for Transformer**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2503.16731-sandybrown?style=flat-square)](https://arxiv.org/abs/2503.16731)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2503.16731)

**TerEffic: Highly Efficient Ternary LLM Inference on FPGA**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2502.16473-sandybrown?style=flat-square)](https://arxiv.org/abs/2502.16473)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2502.16473)

**Pushing up to the Limit of Memory Bandwidth and Capacity Utilization for Efficient LLM Decoding on Embedded FPGA**

![](https://img.shields.io/badge/Arxiv-2025-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2502.10659-sandybrown?style=flat-square)](https://arxiv.org/abs/2502.10659)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2502.10659)

---
### 2026
**A Fully Reconfigurable and Programmable VLIW Architecture for Nonlinear Function Acceleration**

![](https://img.shields.io/badge/IEEE-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/tvlsi.2026.3729817-sandybrown?style=flat-square)](https://doi.org/10.1109/tvlsi.2026.3729817)


**MorphAtt: A Neuromorphic Accelerator for Efficient Multi-Head Attention Processing in Spiking Vision Transformers**

![](https://img.shields.io/badge/Arxiv-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2609.33207-sandybrown?style=flat-square)](https://arxiv.org/abs/2609.33207)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2609.33207)

**From Acceleration to Accelerating Acceleration: Applications and Tools for Democratizing Hardware Design**

![](https://img.shields.io/badge/IEEE-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/fccm68464.2026.00082-sandybrown?style=flat-square)](https://doi.org/10.1109/fccm68464.2026.00082)


**Defending Side-Channel Attacks in FPGA-based Convolutional Layers and Multi-Head Attention Layers with Channel-Level Parallelization**

![](https://img.shields.io/badge/ACM-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3847663-sandybrown?style=flat-square)](https://doi.org/10.1145/3847663)


**HiS-CIM: A 72.65-TOPS/W Hetero-CIM-Based LLM Accelerator Exploiting Hierarchical Weight Compression and Multi-Granular Sparsity**

![](https://img.shields.io/badge/IEEE-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/jssc.2026.3734350-sandybrown?style=flat-square)](https://doi.org/10.1109/jssc.2026.3734350)


**A Novel Projection-Wise Quantization Method and A Custom Accelerator Design for Efficient Sub-4-Bit Large Language Model Inference**

![](https://img.shields.io/badge/Other-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-__no_data__-sandybrown?style=flat-square)](__no_data__)


**HeteroReason: Heterogeneous FPGA-GPU Acceleration for Disaggregated Speculative Reasoning**

![](https://img.shields.io/badge/ACM-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-__no_data__-sandybrown?style=flat-square)](__no_data__)


**Reinforcement Learning-Guided High-Level Synthesis Optimization for Energy-Efficient Operator Fusion in Systolic Arrays: An Analytical Evaluation**

![](https://img.shields.io/badge/Other-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-__no_data__-sandybrown?style=flat-square)](__no_data__)


**RVBSCop: A RISC-V-Based Coprocessor With 229.53 GOPS/W and 92.19 GOPS/mm2 for LLM Inference**

![](https://img.shields.io/badge/IEEE-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/tvlsi.2026.3733137-sandybrown?style=flat-square)](https://doi.org/10.1109/tvlsi.2026.3733137)


**A Comprehensive Analysis and Chip Implementation of Hybrid-Bonding-Based 3D-DRAM Process-Near-Memory Design for LLM Inference**

![](https://img.shields.io/badge/IEEE-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/jssc.2026.3733213-sandybrown?style=flat-square)](https://doi.org/10.1109/jssc.2026.3733213)


**GRADE-RTL: Evaluating LLM-Generated RTL Beyond Compilation**

![](https://img.shields.io/badge/Arxiv-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2609.25335-sandybrown?style=flat-square)](https://arxiv.org/abs/2609.25335)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2609.25335)

**LUT-DiT: A LUT-Based Diffusion Transformer Accelerator for Scalable LUT Inference**

![](https://img.shields.io/badge/IEEE-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/tvlsi.2026.3732472-sandybrown?style=flat-square)](https://doi.org/10.1109/tvlsi.2026.3732472)


**A Dual-Precision MX Datapath for Layer-Adaptive RLHF Quantization on FPGAs**

![](https://img.shields.io/badge/__no_data__-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-__no_data__-sandybrown?style=flat-square)](__no_data__)


**Lockstep: Bit-Exact, Verifiable Transformer Inference at Parity Across Devices, for Dense and Mixture-of-Experts Models**

![](https://img.shields.io/badge/__no_data__-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-__no_data__-sandybrown?style=flat-square)](__no_data__)


**Hermes+: Adaptive Memory-Efficient Pipeline Inference for Large Models on Edge Devices**

![](https://img.shields.io/badge/ACM-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3845607-sandybrown?style=flat-square)](https://doi.org/10.1145/3845607)


**Can Agents Design Better Chips with a Higher Level Abstraction?**

![](https://img.shields.io/badge/Arxiv-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2609.21157-sandybrown?style=flat-square)](https://arxiv.org/abs/2609.21157)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2609.21157)

**ViT-CIM: A Computing-in-Memory based Vision Transformer Accelerator Featuring Multi-Head Self Attention and Token-Level Optimization**

![](https://img.shields.io/badge/IEEE-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/isset70600.2026.11682578-sandybrown?style=flat-square)](https://doi.org/10.1109/isset70600.2026.11682578)


**HBQ: Hierarchical Scaling Block Quantization with Hardware-Efficiency-Aware Design for Accurate LLM Inference**

![](https://img.shields.io/badge/Arxiv-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2609.00450-sandybrown?style=flat-square)](https://arxiv.org/abs/2609.00450)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2609.00450)

**Memory-Efficient Acceleration for Emerging Applications via Hardware/Software Co-Design**

![](https://img.shields.io/badge/Other-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-__no_data__-sandybrown?style=flat-square)](__no_data__)


**Cerium: A Multi-GPU Framework for Terabyte-Scale Encrypted Inference**

![](https://img.shields.io/badge/ACM-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3830418.3843864-sandybrown?style=flat-square)](https://doi.org/10.1145/3830418.3843864)


**Silicon Photonics for Short-Reach Interconnects: Progress in High-Bandwidth and Energy-Efficient Optical I/O**

![](https://img.shields.io/badge/Wiley-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1002/lpor.71894-sandybrown?style=flat-square)](https://doi.org/10.1002/lpor.71894)


**The Particle Accelerator Principle in AI Research Forensics: Event Reconstruction, Epistemic Pile-Up, and Invariant Verification in Large Language Model Scientific Inferences**

![](https://img.shields.io/badge/OpenAlex-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.13140/rg.2.2.28406.56645-sandybrown?style=flat-square)](https://doi.org/10.13140/rg.2.2.28406.56645)


**balmame7-mld/fpga-transformer-ccontrol-flow: Release â€” FPGA Transformer accelerator artifact**

![](https://img.shields.io/badge/OpenAlex-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.5281/zenodo.21869283-sandybrown?style=flat-square)](https://doi.org/10.5281/zenodo.21869283)


**ShatterQuant: Breaking Uniform Precision with Block-Wise Mixed-Precision on a Systolic Transformer Hardware Accelerator**

![](https://img.shields.io/badge/Arxiv-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2610.00207-sandybrown?style=flat-square)](https://arxiv.org/abs/2610.00207)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2610.00207)

**FlexPosit: Tunable Fractional Precision for LLM Inference Accelerators**

![](https://img.shields.io/badge/Arxiv-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2609.04724-sandybrown?style=flat-square)](https://arxiv.org/abs/2609.04724)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2609.04724)

**AI Hardware Accelerators for Large Language Models: Architectures and the Memory Wall**

![](https://img.shields.io/badge/Arxiv-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2608.28048-sandybrown?style=flat-square)](https://arxiv.org/abs/2608.28048)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2608.28048)

**FSGen: Agile Fused and Sparse Accelerator Generator with Accurate Power Model for LLM Applications**

![](https://img.shields.io/badge/Arxiv-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2608.09252-sandybrown?style=flat-square)](https://arxiv.org/abs/2608.09252)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2608.09252)

**Interpolation of Non-Linear Functions for LLMs using Partial Reconfiguration in FPGAs**

![](https://img.shields.io/badge/Arxiv-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2608.03033-sandybrown?style=flat-square)](https://arxiv.org/abs/2608.03033)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2608.03033)

**MDTransformer: A Hardware-Software Co-Design of Mode-Division Photonic Transformer Accelerator with Inverse-Designed Coherent Crossbar**

![](https://img.shields.io/badge/Arxiv-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2607.26016-sandybrown?style=flat-square)](https://arxiv.org/abs/2607.26016)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2607.26016)

**Opto-ViT-v2: Noise-Resilient On-Chip Fine-Tuning for Photonic Near-Sensor Vision Transformer Accelerators**

![](https://img.shields.io/badge/Arxiv-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2607.19421-sandybrown?style=flat-square)](https://arxiv.org/abs/2607.19421)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2607.19421)

**ELiTeFormer: An Efficient Transformer for FPGAs**

![](https://img.shields.io/badge/Arxiv-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2607.03652-sandybrown?style=flat-square)](https://arxiv.org/abs/2607.03652)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2607.03652)

**DxPTA: An Architecture Design Space Exploration with Optical Dataflow-guided Strategy for HW/SW Co-Design of Photonic Transformer Accelerators**

![](https://img.shields.io/badge/Arxiv-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2606.06515-sandybrown?style=flat-square)](https://arxiv.org/abs/2606.06515)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2606.06515)

**CHIMERA: A Flexible and Scalable 3.1 TOPS/W AI-MCU with Transformer Accelerator and 563 Gb/s Shared-L2 Memory Subsystem with QoS Guarantees**

![](https://img.shields.io/badge/Arxiv-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2606.02358-sandybrown?style=flat-square)](https://arxiv.org/abs/2606.02358)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2606.02358)

**Implementation of an Adaptive Transformer Accelerator for Accurate Outdoor Localization with Massive MIMO**

![](https://img.shields.io/badge/Arxiv-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2605.13507-sandybrown?style=flat-square)](https://arxiv.org/abs/2605.13507)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2605.13507)

**31.1 A 14.08-to-135.69Token/s ReRAM-on-Logic Stacked Outlier-Free Large-Language-Model Accelerator with Block-Clustered Weight-Compression and Adaptive Parallel-Speculative-Decoding**

![](https://img.shields.io/badge/Arxiv-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2605.09375-sandybrown?style=flat-square)](https://arxiv.org/abs/2605.09375)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2605.09375)

**VitaLLM: A Versatile and Tiny Accelerator for Mixed-Precision LLM Inference on Edge Devices**

![](https://img.shields.io/badge/Arxiv-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2605.00320-sandybrown?style=flat-square)](https://arxiv.org/abs/2605.00320)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2605.00320)

**VitaLLM: A Versatile, Ultra-Compact Ternary LLM Accelerator with Dependency-Aware Scheduling**

![](https://img.shields.io/badge/Arxiv-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2604.27396-sandybrown?style=flat-square)](https://arxiv.org/abs/2604.27396)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2604.27396)

**RCW-CIM: A Digital CIM-based LLM Accelerator with Read-Compute/Write**

![](https://img.shields.io/badge/Arxiv-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2604.27384-sandybrown?style=flat-square)](https://arxiv.org/abs/2604.27384)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2604.27384)

**A Full-Stack Performance Evaluation Infrastructure for 3D-DRAM-based LLM Accelerators**

![](https://img.shields.io/badge/Arxiv-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2604.08044-sandybrown?style=flat-square)](https://arxiv.org/abs/2604.08044)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2604.08044)

**MXFormer: A Microscaling Floating-Point Charge-Trap Transistor Compute-in-Memory Transformer Accelerator**

![](https://img.shields.io/badge/Arxiv-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2602.12480-sandybrown?style=flat-square)](https://arxiv.org/abs/2602.12480)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2602.12480)

**FlexLLM: Composable HLS Library for Flexible Hybrid LLM Accelerator Design**

![](https://img.shields.io/badge/Arxiv-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2601.15710-sandybrown?style=flat-square)](https://arxiv.org/abs/2601.15710)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2601.15710)

**CD-PIM: A High-Bandwidth and Compute-Efficient LPDDR5-Based PIM for Low-Batch LLM Acceleration on Edge-Device**

![](https://img.shields.io/badge/Arxiv-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2601.12298-sandybrown?style=flat-square)](https://arxiv.org/abs/2601.12298)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2601.12298)

**The Particle Accelerator Principle in AI Research Forensics**

![](https://img.shields.io/badge/OpenAlex-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.5281/zenodo.23123163-sandybrown?style=flat-square)](https://doi.org/10.5281/zenodo.23123163)


**EdgeDAE: Acceleration of Diffusion Action Experts for Real-Time Physical AI with Tiny VLAs on Edge FPGA-GPU Systems**

![](https://img.shields.io/badge/OpenAlex-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arxiv.2610.00311-sandybrown?style=flat-square)](https://arxiv.org/abs/2610.00311)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2610.00311)

**AdaptViT: Runtime-Adaptive Vision Transformer Deployment on Custom RISC-V**

![](https://img.shields.io/badge/Arxiv-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2610.02288-sandybrown?style=flat-square)](https://arxiv.org/abs/2610.02288)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2610.02288)

**vidax: A Unified JAX Framework for Video Generative Models on Accelerator Meshes**

![](https://img.shields.io/badge/OpenAlex-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-__no_data__-sandybrown?style=flat-square)](https://arxiv.org/abs/2609.18077)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2609.18077)

**The World Model Hardware Accelerator**

![](https://img.shields.io/badge/OpenAlex-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arxiv.2609.16244-sandybrown?style=flat-square)](https://arxiv.org/abs/2609.16244)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2609.16244)

**Fengshui: Demystifying Chiplet Ecosystem and Bespoke Neural Network Accelerator Codesign**

![](https://img.shields.io/badge/OpenAlex-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arxiv.2609.10970-sandybrown?style=flat-square)](https://arxiv.org/abs/2609.10970)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2609.10970)

**Constraint-guided generative AI for gate-level VLSI design with early physical awareness**

![](https://img.shields.io/badge/OpenAlex-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1016/j.vlsi.2026.102888-sandybrown?style=flat-square)](https://doi.org/10.1016/j.vlsi.2026.102888)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://www.sciencedirect.com/science/article/pii/S0167926026002439/pdf)

**MeshKV: A Network-on-Chip KV Cache Fabric for Scalable Transformer Decoding Accelerators**

![](https://img.shields.io/badge/OpenAlex-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arxiv.2609.19207-sandybrown?style=flat-square)](https://arxiv.org/abs/2609.19207)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2609.19207)

**UNISON: A Co-Designed Near-Memory Scheduler of Session KV Residency for LLM Agents**

![](https://img.shields.io/badge/OpenAlex-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arxiv.2609.09643-sandybrown?style=flat-square)](https://arxiv.org/abs/2609.09643)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2609.09643)

**An energy-efficient FPGA accelerator for transformer inference using layer-wise precision optimization**

![](https://img.shields.io/badge/OpenAlex-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1016/j.rineng.2026.112807-sandybrown?style=flat-square)](https://doi.org/10.1016/j.rineng.2026.112807)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://www.sciencedirect.com/science/article/pii/S2590123026038211/pdf)

**AI-Assisted Design of a Post-Quantum Cryptographic Accelerator: A Deployed-Silicon Case Study**

![](https://img.shields.io/badge/OpenAlex-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arxiv.2609.04058-sandybrown?style=flat-square)](https://arxiv.org/abs/2609.04058)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2609.04058)

**Adaptive Memory-Centric Hardware Accelerator for Long-Context Transformer Inference Using Predictive Dataflow Optimization**

![](https://img.shields.io/badge/OpenAlex-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-__no_data__-sandybrown?style=flat-square)](https://fsrap.com/index.php/JRHAES/article/view/342)


**Gen-TAS: A Generative AI-Aided Hardware-Software Task Allocation Framework for FPGA-GPP Heterogeneous Systems**

![](https://img.shields.io/badge/OpenAlex-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arxiv.2608.28160-sandybrown?style=flat-square)](https://arxiv.org/abs/2608.28160)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2608.28160)

**Helios: Scalable Multi-accelerator FPGA Architecture for Efficient Inference of Transformer-Based Models**

![](https://img.shields.io/badge/OpenAlex-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1007/978-981-92-4805-6_23-sandybrown?style=flat-square)](https://doi.org/10.1007/978-981-92-4805-6_23)


**Survey of FPGA-based accelerators for large language models**

![](https://img.shields.io/badge/OpenAlex-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1016/j.jpdc.2026.105349-sandybrown?style=flat-square)](https://doi.org/10.1016/j.jpdc.2026.105349)


**On-chip non-volatile all-optical residual neural network accelerator**

![](https://img.shields.io/badge/OpenAlex-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1038/s41377-026-02325-2-sandybrown?style=flat-square)](https://doi.org/10.1038/s41377-026-02325-2)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://www.nature.com/articles/s41377-026-02325-2.pdf)

**Recent Developments in Transformer Inference Deployment on FPGA Platforms: A Survey**

![](https://img.shields.io/badge/Arxiv-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2609.01212-sandybrown?style=flat-square)](https://arxiv.org/abs/2609.01212)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2609.01212)

**ATLAS: Automated HLS for DL-Optimized FPGAs**

![](https://img.shields.io/badge/Arxiv-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2607.07643-sandybrown?style=flat-square)](https://arxiv.org/abs/2607.07643)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2607.07643)

**FlexViT: A Flexible FPGA-based Accelerator for Edge Vision Transformers**

![](https://img.shields.io/badge/Arxiv-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2606.31938-sandybrown?style=flat-square)](https://arxiv.org/abs/2606.31938)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2606.31938)

**Towards Autonomous Accelerator Design: FPGA Accelerator Generation with SECDA**

![](https://img.shields.io/badge/Arxiv-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2606.11117-sandybrown?style=flat-square)](https://arxiv.org/abs/2606.11117)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2606.11117)

**Triple-Hoisted Baby-Step Giant-Step Linear Transformation over CKKS Homomorphic Encryption and Hardware Accelerator**

![](https://img.shields.io/badge/Arxiv-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2605.17222-sandybrown?style=flat-square)](https://arxiv.org/abs/2605.17222)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2605.17222)

**LLM-Driven Design Space Exploration of FPGA-based Accelerators**

![](https://img.shields.io/badge/Arxiv-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2605.05920-sandybrown?style=flat-square)](https://arxiv.org/abs/2605.05920)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2605.05920)

**Design Conductor 2.0: An agent builds a TurboQuant inference accelerator in 80 hours**

![](https://img.shields.io/badge/Arxiv-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2605.05170-sandybrown?style=flat-square)](https://arxiv.org/abs/2605.05170)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2605.05170)

**ViM-Q: Scalable Algorithm-Hardware Co-Design for Vision Mamba Model Inference on FPGA**

![](https://img.shields.io/badge/Arxiv-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2605.01935-sandybrown?style=flat-square)](https://arxiv.org/abs/2605.01935)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2605.01935)

**GraphLeap: Decoupling Graph Construction and Convolution for Vision GNN Acceleration on FPGA**

![](https://img.shields.io/badge/Arxiv-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2604.21290-sandybrown?style=flat-square)](https://arxiv.org/abs/2604.21290)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2604.21290)

**TRINE: A Token-Aware, Runtime-Adaptive FPGA Inference Engine for Multimodal AI**

![](https://img.shields.io/badge/Arxiv-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2603.22867-sandybrown?style=flat-square)](https://arxiv.org/abs/2603.22867)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2603.22867)

**SkipOPU: An FPGA-based Overlay Processor for Large Language Models with Dynamically Allocated Computation**

![](https://img.shields.io/badge/Arxiv-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2603.14785-sandybrown?style=flat-square)](https://arxiv.org/abs/2603.14785)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2603.14785)

**A Persistent-State Dataflow Accelerator for Memory-Bound Linear Attention Decode on FPGA**

![](https://img.shields.io/badge/Arxiv-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2603.05931-sandybrown?style=flat-square)](https://arxiv.org/abs/2603.05931)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2603.05931)

**FAST-Prefill: FPGA Accelerated Sparse Attention for Long Context LLM Prefill**

![](https://img.shields.io/badge/Arxiv-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2602.20515-sandybrown?style=flat-square)](https://arxiv.org/abs/2602.20515)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2602.20515)

**LAAFD: LLM-based Agents for Accelerated FPGA Design**

![](https://img.shields.io/badge/Arxiv-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2602.06085-sandybrown?style=flat-square)](https://arxiv.org/abs/2602.06085)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2602.06085)

**HFRWKV: A High-Performance Fully On-Chip Hardware Accelerator for RWKV**

![](https://img.shields.io/badge/Arxiv-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2601.02135-sandybrown?style=flat-square)](https://arxiv.org/abs/2601.02135)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2601.02135)

**IMC-CLINIC: Coupled Loss-Informed Newton Iterations for Clipping in Analog In-Memory Computing**

![](https://img.shields.io/badge/OpenAlex-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arxiv.2609.35586-sandybrown?style=flat-square)](https://arxiv.org/abs/2609.35586)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2609.35586)

**SPIMOE: Exploiting Hybrid Sparsity for Reasoning MoE Inference on Heterogeneous PIM Architectures**

![](https://img.shields.io/badge/OpenAlex-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arxiv.2609.34612-sandybrown?style=flat-square)](https://arxiv.org/abs/2609.34612)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2609.34612)

**Torch-PIM: Automated Profile-Guided PIM Offloading for PyTorch**

![](https://img.shields.io/badge/OpenAlex-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-__no_data__-sandybrown?style=flat-square)](https://arxiv.org/abs/2609.34657)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2609.34657)

**PATTON: Enabling Commodity PIM for Production LLM Serving**

![](https://img.shields.io/badge/Arxiv-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2609.11392-sandybrown?style=flat-square)](https://arxiv.org/abs/2609.11392)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2609.11392)

**Heterogeneous LLM Serving with General-Purpose Processing-Near-Memory for Retrieval-Based Sparse Attention**

![](https://img.shields.io/badge/Arxiv-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2608.03555-sandybrown?style=flat-square)](https://arxiv.org/abs/2608.03555)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2608.03555)

**Automatic Model-Hardware Co-Adaptation for Heterogeneous AI Accelerators**

![](https://img.shields.io/badge/Arxiv-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2607.12875-sandybrown?style=flat-square)](https://arxiv.org/abs/2607.12875)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2607.12875)

**FastTPS: An Optimized Method for LLM Token Phase for AI accelerators**

![](https://img.shields.io/badge/Arxiv-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2607.11211-sandybrown?style=flat-square)](https://arxiv.org/abs/2607.11211)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2607.11211)

**Approximate Attention Weighting for Sustainable FPGA-Based Vision Transformer Inference**

![](https://img.shields.io/badge/Arxiv-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2607.01798-sandybrown?style=flat-square)](https://arxiv.org/abs/2607.01798)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2607.01798)

**HBM Is Not All You Need: Efficient Disaggregated LLM Serving across Memory-heterogeneous Accelerators**

![](https://img.shields.io/badge/Arxiv-2026-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://arxiv.org/abs/2606.29986-sandybrown?style=flat-square)](https://arxiv.org/abs/2606.29986)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2606.29986)

---
## Analysis

Publication counts by year (regenerated from `data/papers.yaml`):

<p align="center">
  <img src="./data/img/publications.png" width='360' alt="Publications by year"/>
</p>

<p align="center">
  <img src="./data/figs/publication_year.png" width='300' alt="Publication year chart"/>
</p>

---

## Links

- **Live website:** https://aliemo.github.io/transfomers-silicon-research/
- **GitHub:** https://github.com/aliemo/transfomers-silicon-research
- **Pipeline guide:** [`data/PIPELINE.md`](./data/PIPELINE.md)
- **Website docs:** [`website/README.md`](./website/README.md)
