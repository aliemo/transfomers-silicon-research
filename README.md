# BERT Model on Silicon
> **Research and Materials on Hardware implementation of BERT (Bidirectional Encoder Representations from Transformers) Model**

## BERT Model

### Description
BERT is a method of pre-training language representations, meaning that we train a general-purpose "language understanding" model on a large text corpus (like Wikipedia) and then use that model for downstream NLP tasks. BERT was created and published in 2018 by Jacob Devlin and his colleagues from Google. BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7 (5.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5% absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1% absolute improvement).

### Architecture

BERT is a Transformer-based model. The architecture of BERT is similar to the original Transformer model, except that BERT has two separate Transformer models: one for the left-to-right direction (the “encoder”) and one for the right-to-left direction (the “encoder”). The output of each model is the hidden state output by the final Transformer layer. The two models are pre-trained jointly on a large corpus of unlabeled text. The pre-training task is a simple and straightforward masked language modeling objective. The pre-trained BERT model can then be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.

--- 

## Reference Papers

**1. Attention Is All You Need**

![](https://img.shields.io/badge/arXiv-2017-skyblue?colorstyle=plastic) [![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.1706.03762-sandybrown?style=flat-square?&style=plastic)](https://arxiv.org/abs/1706.03762) [![PDF-Download](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/1706.03762.pdf) 

[![Code-Link](https://img.shields.io/badge/Code-PyTorch-red?style=plastic)](https://github.com/jadore801120/attention-is-all-you-need-pytorch) [![Code-Link](https://img.shields.io/badge/Code-TensorFlow-orange?style=plastic)](https://github.com/lsdefine/attention-is-all-you-need-keras)
 
<details>
<summary><img src="https://img.shields.io/badge/ABSTRACT-9575cd?&style=plastic"/></summary>
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 Englishto-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.
</details>

#

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

## Important Papers

**Distilling the Knowledge in a Neural Network**

![](https://img.shields.io/badge/arXiv-2015-skyblue?colorstyle=plastic) [![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.1503.02531-sandybrown?style=flat-square)](https://arxiv.org/abs/1503.02531) [![PDF-Download](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/1503.02531.pdf)


**Distilling Knowledge Learned in BERT for Text Generation**

![](https://img.shields.io/badge/arXiv-2020-skyblue?colorstyle=plastic) [![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.1911.03829-sandybrown?style=flat-square?&style=plastic)](https://arxiv.org/abs//1911.03829) [![PDF-Download](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/1911.03829.pdf)


**DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter**

![](https://img.shields.io/badge/arXiv-2019-skyblue?colorstyle=plastic) [![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.1910.01108-sandybrown?style=flat-square?&style=plastic)](https://arxiv.org/abs/1910.01108) [![PDF-Download](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/1910.01108.pdf)

**TinyBERT: Distilling BERT for Natural Language Understanding**

![](https://img.shields.io/badge/arXiv-2020-skyblue?colorstyle=plastic) [![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.1909.10351-sandybrown?style=flat-square?&style=plastic)](https://arxiv.org/abs/1909.10351) [![PDF-Download](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/1909.10351.pdf)

**Distilling the Knowledge in a Neural Network**

![](https://img.shields.io/badge/arXiv-2015-skyblue?colorstyle=plastic) [![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.1503.02531-sandybrown?style=flat-square?&style=plastic)](https://arxiv.org/abs/1503.02531) [![PDF-Download](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/1503.02531.pdf)

**FastBERT: a Self-distilling BERT with Adaptive Inference Time**

![](https://img.shields.io/badge/arXiv-2020-skyblue?colorstyle=plastic) [![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2004.02178-sandybrown?style=flat-square?&style=plastic)](https://arxiv.org/abs/2004.02178) [![PDF-Download](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2004.02178.pdf)

**Distilling Task-Specific Knowledge from BERT into Simple Neural Networks**

![](https://img.shields.io/badge/arXiv-2019-skyblue?colorstyle=plastic) [![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.1903.12136-sandybrown?style=flat-square?&style=plastic)](https://arxiv.org/abs/1903.12136) [![PDF-Download](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/1903.12136.pdf)

**Patient Knowledge Distillation for BERT Model Compression**

![](https://img.shields.io/badge/arXiv-2019-skyblue?colorstyle=plastic) [![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.1908.09355-sandybrown?style=flat-square?&style=plastic)](https://arxiv.org/abs/1908.09355) [![PDF-Download](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/1908.09355.pdf)

**MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices**

![](https://img.shields.io/badge/arXiv-2020-skyblue?colorstyle=plastic) [![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2004.02984-sandybrown?style=flat-square?&style=plastic)](https://arxiv.org/abs/2004.02984) [![PDF-Download](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2004.02984.pdf)

**Improving Multi-Task Deep Neural Networks via Knowledge Distillation for Natural Language Understanding**

![](https://img.shields.io/badge/arXiv-2019-skyblue?colorstyle=plastic) [![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.1904.09482-sandybrown?style=flat-square?&style=plastic)](https://arxiv.org/abs/1904.09482) [![PDF-Download](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/1904.09482.pdf)


## BERT on Silicon Papers

**A 28nm 27.5TOPS/W Approximate-Computing-Based Transformer Processor with Asymptotic Sparsity Speculating and Out-of-Order Computing**

![](https://img.shields.io/badge/None-2022-skyblue?colorstyle=flat-square)
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

**A Framework for Area-efficient Multi-task BERT Execution on ReRAM-based Accelerators**

![](https://img.shields.io/badge/IEEE/ACM-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ICCAD51958.2021.9643471-sandybrown?style=flat-square)](https://doi.org/10.1109/ICCAD51958.2021.9643471)

**A Full-Stack Search Technique for Domain Optimized Deep Learning Accelerators**

![](https://img.shields.io/badge/arXiv-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3503222.3507767-sandybrown?style=flat-square)](https://doi.org/10.1145/3503222.3507767)

**A length adaptive algorithm-hardware co-design of transformer on FPGA through sparse attention and dynamic pipelining**

![](https://img.shields.io/badge/ACM/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3489517.3530585-sandybrown?style=flat-square)](https://doi.org/10.1145/3489517.3530585)

**A Lite Romanian BERT: ALR-BERT**

![](https://img.shields.io/badge/MDPI-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.3390/computers11040057-sandybrown?style=flat-square)](https://doi.org/10.3390/computers11040057)

**A Low-Cost Reconfigurable Nonlinear Core for Embedded DNN Applications**

![](https://img.shields.io/badge/IEEE-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ICFPT51103.2020.00014-sandybrown?style=flat-square)](https://doi.org/10.1109/ICFPT51103.2020.00014)

**A Microcontroller is All You Need: Enabling Transformer Execution on Low-Power IoT Endnodes**

![](https://img.shields.io/badge/IEEE-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/COINS51742.2021.9524173-sandybrown?style=flat-square)](https://doi.org/10.1109/COINS51742.2021.9524173)

**A Multi-Neural Network Acceleration Architecture**

![](https://img.shields.io/badge/ACM/IEEE-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ISCA45697.2020.00081-sandybrown?style=flat-square)](https://doi.org/10.1109/ISCA45697.2020.00081)

**A Power Efficient Neural Network Implementation on Heterogeneous FPGA and GPU Devices**

![](https://img.shields.io/badge/IEEE-2019-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/IRI.2019.00040-sandybrown?style=flat-square)](https://doi.org/10.1109/IRI.2019.00040)

**A Primer in BERTology: What We Know About How BERT Works**

![](https://img.shields.io/badge/MIt%20Press-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1162/tacl_a_00349-sandybrown?style=flat-square)](https://doi.org/10.1162/tacl_a_00349)

**A Quantitative Survey of Communication Optimizations in Distributed Deep Learning**

![](https://img.shields.io/badge/IEEE-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/MNET.011.2000530-sandybrown?style=flat-square)](https://doi.org/10.1109/MNET.011.2000530)

**A Reconfigurable DNN Training Accelerator on FPGA**

![](https://img.shields.io/badge/IEEE-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/SiPS50750.2020.9195234-sandybrown?style=flat-square)](https://doi.org/10.1109/SiPS50750.2020.9195234)

**A Resource-Saving Energy-Efficient Reconfigurable Hardware Accelerator for BERT-based Deep Neural Network Language Models using FFT Multiplication**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ISCAS48785.2022.9937531-sandybrown?style=flat-square)](https://doi.org/10.1109/ISCAS48785.2022.9937531)

**A Self-Attention Network for Deep JSCCM: The Design and FPGA Implementation**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/GLOBECOM48099.2022.10001518-sandybrown?style=flat-square)](https://doi.org/10.1109/GLOBECOM48099.2022.10001518)

**A Simple and Effective Approach to Automatic Post-Editing with Transfer Learning**

![](https://img.shields.io/badge/arXiv-2019-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.1906.06253-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.1906.06253)

**A Study on Token Pruning for ColBERT**

![](https://img.shields.io/badge/arXiv-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2112.06540-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2112.06540)

**A White Paper on Neural Network Quantization**

![](https://img.shields.io/badge/arXiv-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2106.08295-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2106.08295)

**A^3: Accelerating Attention Mechanisms in Neural Networks with Approximation**

![](https://img.shields.io/badge/IEEE-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/HPCA47549.2020.00035-sandybrown?style=flat-square)](https://doi.org/10.1109/HPCA47549.2020.00035)

**Emerging Neural Workloads and Their Impact on Hardware**

![](https://img.shields.io/badge/IEEE-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.23919/DATE48585.2020.9116435-sandybrown?style=flat-square)](https://doi.org/10.23919/DATE48585.2020.9116435)

**Accelerated Device Placement Optimization with Contrastive Learning**

![](https://img.shields.io/badge/ACM-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3472456.3472523-sandybrown?style=flat-square)](https://doi.org/10.1145/3472456.3472523)

**Accelerating attention mechanism on fpgas based on efficient reconfigurable systolic array**

![](https://img.shields.io/badge/ACM-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3549937-sandybrown?style=flat-square)](https://doi.org/10.1145/3549937)

**Accelerating attention through gradient-based learned runtime pruning**

![](https://img.shields.io/badge/ACM-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3470496.3527423-sandybrown?style=flat-square)](https://doi.org/10.1145/3470496.3527423)

**Accelerating bandwidth-bound deep learning inference with main-memory accelerators**

![](https://img.shields.io/badge/ACM-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3458817.3476146-sandybrown?style=flat-square)](https://doi.org/10.1145/3458817.3476146)

**Accelerating Emerging Neural Workloads**

![](https://img.shields.io/badge/Purdue%20University-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.25394/pgs.17139038.v1-sandybrown?style=flat-square)](https://doi.org/10.25394/pgs.17139038.v1)

**Accelerating event detection with DGCNN and FPGAS**

![](https://img.shields.io/badge/MDPI-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.3390/electronics9101666-sandybrown?style=flat-square)](https://doi.org/10.3390/electronics9101666)

**Accelerating Framework of Transformer by Hardware Design and Model Compression Co-Optimization**

![](https://img.shields.io/badge/IEEE/ACM-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ICCAD51958.2021.9643586-sandybrown?style=flat-square)](https://doi.org/10.1109/ICCAD51958.2021.9643586)

**Accelerating NLP Tasks on FPGA with Compressed BERT and a Hardware-Oriented Early Exit Method**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ISVLSI54635.2022.00092-sandybrown?style=flat-square)](https://doi.org/10.1109/ISVLSI54635.2022.00092)

**Accelerating Transformer Networks through Recomposing Softmax Layers**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/IISWC55918.2022.00018-sandybrown?style=flat-square)](https://doi.org/10.1109/IISWC55918.2022.00018)

**Accelerating Transformer-based Deep Learning Models on FPGAs using Column Balanced Block Pruning**

![](https://img.shields.io/badge/IEEE-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ISQED51717.2021.9424344-sandybrown?style=flat-square)](https://doi.org/10.1109/ISQED51717.2021.9424344)

**Accommodating Transformer onto FPGA: Coupling the Balanced Model Compression and FPGA-Implementation Optimization**

![](https://img.shields.io/badge/ACM-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3453688.3461739-sandybrown?style=flat-square)](https://doi.org/10.1145/3453688.3461739)

**Achieving the Performance of All-Bank In-DRAM PIM With Standard Memory Interface: Memory-Computation Decoupling**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ACCESS.2022.3203051-sandybrown?style=flat-square)](https://doi.org/10.1109/ACCESS.2022.3203051)

**Adaptable Butterfly Accelerator for Attention-based NNs via Hardware and Algorithm Co-design**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/MICRO56248.2022.00050-sandybrown?style=flat-square)](https://doi.org/10.1109/MICRO56248.2022.00050)

**Adapting by pruning: A case study on BERT**

![](https://img.shields.io/badge/arXiv-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2105.03343-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2105.03343)

**Adaptive Inference through Early-Exit Networks: Design, Challenges and Directions**

![](https://img.shields.io/badge/ACM-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3469116.3470012-sandybrown?style=flat-square)](https://doi.org/10.1145/3469116.3470012)

**Adaptive Spatio-Temporal Graph Enhanced Vision-Language Representation for Video QA**

![](https://img.shields.io/badge/IEEE-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/TIP.2021.3076556-sandybrown?style=flat-square)](https://doi.org/10.1109/TIP.2021.3076556)

**Algorithm-hardware Co-design of Attention Mechanism on FPGA Devices**

![](https://img.shields.io/badge/ACM-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3477002-sandybrown?style=flat-square)](https://doi.org/10.1145/3477002)

**Algorithm-Hardware Co-Design of Single Shot Detector for Fast Object Detection on FPGAs**

![](https://img.shields.io/badge/IEEE/ACM-2018-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3240765.3240775-sandybrown?style=flat-square)](https://doi.org/10.1145/3240765.3240775)

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

**An Efficient Transformer Inference Engine on DSP**

![](https://img.shields.io/badge/Springer-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1007/978--3--031--22677--9_29-sandybrown?style=flat-square)](https://doi.org/10.1007/978-3-031-22677-9_29)

**An Empirical Analysis of BERT Embedding for Automated Essay Scoring**

![](https://img.shields.io/badge/TheSAI-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.14569/ijacsa.2020.0111027-sandybrown?style=flat-square)](https://doi.org/10.14569/ijacsa.2020.0111027)

**An Energy-Efficient Transformer Processor Exploiting Dynamic Weak Relevances in Global Attention**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/JSSC.2022.3213521-sandybrown?style=flat-square)](https://doi.org/10.1109/JSSC.2022.3213521)

**An Evaluation of Transfer Learning for Classifying Sales Engagement Emails at Large Scale**

![](https://img.shields.io/badge/IEEE-2019-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/CCGRID.2019.00069-sandybrown?style=flat-square)](https://doi.org/10.1109/CCGRID.2019.00069)

**An FPGA-Based Transformer Accelerator Using Output Block Stationary Dataflow for Object Recognition Applications**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/TCSII.2022.3196055-sandybrown?style=flat-square)](https://doi.org/10.1109/TCSII.2022.3196055)

**An investigation on different underlying quantization schemes for pre-trained language models**

![](https://img.shields.io/badge/Springer-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1007/978--3--030--60450--9_29-sandybrown?style=flat-square)](https://doi.org/10.1007/978-3-030-60450-9_29)

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

