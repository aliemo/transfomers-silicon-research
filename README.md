# BERT Model on Silicon
> **Research and Materials on Hardware implementation of BERT (Bidirectional Encoder Representations from Transformers) Model**

<center><img src="https://img.shields.io/badge/Status-WIP-ff69b4?style=flat-square"/></center>
<center><img src="https://img.shields.io/badge/Progress-%2599-ef6c00?labelColor=1565c0&style=flat-square"/></center>

<p align="center">
  <img src="./data/img/BERT-on-Silicon.png" />
</p>

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

---


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

---


## BERT on Silicon


### 2018
**Algorithm-Hardware Co-Design of Single Shot Detector for Fast Object Detection on FPGAs**

![](https://img.shields.io/badge/IEEE/ACM-2018-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3240765.3240775-sandybrown?style=flat-square)](https://doi.org/10.1145/3240765.3240775)


---
### 2019
**An Evaluation of Transfer Learning for Classifying Sales Engagement Emails at Large Scale**

![](https://img.shields.io/badge/IEEE-2019-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/CCGRID.2019.00069-sandybrown?style=flat-square)](https://doi.org/10.1109/CCGRID.2019.00069)


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


**A^3: Accelerating Attention Mechanisms in Neural Networks with Approximation**

![](https://img.shields.io/badge/IEEE-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/HPCA47549.2020.00035-sandybrown?style=flat-square)](https://doi.org/10.1109/HPCA47549.2020.00035)


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


**Deep Learning Acceleration with Neuron-to-Memory Transformation**

![](https://img.shields.io/badge/IEEE-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/HPCA47549.2020.00011-sandybrown?style=flat-square)](https://doi.org/10.1109/HPCA47549.2020.00011)


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

**Look-Up Table based Energy Efficient Processing in Cache Support for Neural Network Acceleration**

![](https://img.shields.io/badge/IEEE/ACM-2020-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/MICRO50266.2020.00020-sandybrown?style=flat-square)](https://doi.org/10.1109/MICRO50266.2020.00020)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://www.microarch.org/micro53/papers/738300a088.pdf)

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

**Kunlun: A 14nm High-Performance AI Processor for Diversified Workloads**

![](https://img.shields.io/badge/IEEE-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ISSCC42613.2021.9366056-sandybrown?style=flat-square)](https://doi.org/10.1109/ISSCC42613.2021.9366056)


**Layerweaver: Maximizing Resource Utilization of Neural Processing Units via Layer-Wise Scheduling**

![](https://img.shields.io/badge/IEEE-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/HPCA51647.2021.00056-sandybrown?style=flat-square)](https://doi.org/10.1109/HPCA51647.2021.00056)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://taejunham.github.io/data/layerweaver_hpca21.pdf)

**M2M: Learning to Enhance Low-Light Image from Model to Mobile FPGA**

![](https://img.shields.io/badge/Springer-2021-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1007/978--3--030--89029--2_22-sandybrown?style=flat-square)](https://doi.org/10.1007/978-3-030-89029-2_22)


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


**A Framework for Accelerating Transformer-Based Language Model on ReRAM-Based Architecture**

![](https://img.shields.io/badge/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/TCAD.2021.3121264-sandybrown?style=flat-square)](https://doi.org/10.1109/TCAD.2021.3121264)


**A length adaptive algorithm-hardware co-design of transformer on FPGA through sparse attention and dynamic pipelining**

![](https://img.shields.io/badge/ACM/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3489517.3530585-sandybrown?style=flat-square)](https://doi.org/10.1145/3489517.3530585)


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


**Elastic Processing and Hardware Architectures for Machine Learning**

![](https://img.shields.io/badge/ProQuest-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-3e9f91ca96ba3320587da2bbec561a2b/-sandybrown?style=flat-square)](https://www.proquest.com/openview/3e9f91ca96ba3320587da2bbec561a2b/)


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


**Fast Heterogeneous Task Mapping for Reducing Edge DNN Latency**

![](https://img.shields.io/badge/None-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/ASAP54787.2022.00020-sandybrown?style=flat-square)](https://doi.org/10.1109/ASAP54787.2022.00020)


**FILM-QNN: Efficient FPGA Acceleration of Deep Neural Networks with Intra-Layer, Mixed-Precision Quantization**

![](https://img.shields.io/badge/ACM/SIGDA-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3490422.3502364-sandybrown?style=flat-square)](https://doi.org/10.1145/3490422.3502364)


**FPGA-aware automatic acceleration framework for vision transformer with mixed-scheme quantization: late breaking results**

![](https://img.shields.io/badge/ACM/IEEE-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3489517.3530618-sandybrown?style=flat-square)](https://doi.org/10.1145/3489517.3530618)


**FPGA-based design and implementation of the location attention mechanism in neural networks**

![](https://img.shields.io/badge/IOS%20Press-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.3233/JIFS--212273-sandybrown?style=flat-square)](https://doi.org/10.3233/JIFS-212273)


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


**LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale**

![](https://img.shields.io/badge/arXiv-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2208.07339-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2208.07339)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2208.07339)

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

**Near-Optimal Sparse Allreduce for Distributed Deep Learning**

![](https://img.shields.io/badge/ACM-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1145/3503221.3508399-sandybrown?style=flat-square)](https://doi.org/10.1145/3503221.3508399)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2201.07598.pdf)

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

**Hardware-friendly compression and hardware acceleration for transformer: A survey**

![](https://img.shields.io/badge/AIMPress-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://www.aimspress.com/article/doi/10.3934/era.2022192-sandybrown?style=flat-square)](https://www.aimspress.com/article/doi/10.3934/era.2022192)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://www.aimspress.com/aimspress-data/era/2022/10/PDF/era-30-10-192.pdf)

**Hardware/Software Co-Design of Edge DNN Accelerators with TFLite**

![](https://img.shields.io/badge/HiPEAC-2022-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1007/978--3--030--87208--8_5-sandybrown?style=flat-square)](https://https://eprints.gla.ac.uk/280378/)


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


**HAMMER: Hardware-friendly Approximate Computing for Self-attention with Mean-redistribution and Linearization**

![](https://img.shields.io/badge/IEEE-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1109/LCA.2022.3233832-sandybrown?style=flat-square)](https://doi.org/10.1109/LCA.2022.3233832)


**ViTA: A Vision Transformer Inference Accelerator for Edge Applications**

![](https://img.shields.io/badge/Arxiv-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2302.09108-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2302.09108)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2302.09108.pdf)

**TRON: Transformer Neural Network Acceleration with Non-Coherent Silicon Photonics**

![](https://img.shields.io/badge/Arxiv-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2303.12914-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2303.12914)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2303.12914.pdf)

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

**Trends in AI inference energy consumption: Beyond the performance-vs-parameter laws of deep learning**

![](https://img.shields.io/badge/Elsevier-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.1016/j.suscom.2023.100857-sandybrown?style=flat-square)](https://doi.org/10.1016/j.suscom.2023.100857)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://www.sciencedirect.com/science/article/pii/S2210537923000124/pdfft?md5=4bec2735c1586b935287e6afea9e63a2&pid=1-s2.0-S2210537923000124-main.pdf)

**TransCODE: Co-design of Transformers and Accelerators for Efficient Training and Inference**

![](https://img.shields.io/badge/Arxiv-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://doi.org/10.48550/arXiv.2303.14882-sandybrown?style=flat-square)](https://doi.org/10.48550/arXiv.2303.14882)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://arxiv.org/pdf/2303.14882)

**Architecting High Performance Silicon Systems for Accurate and Efficient On-Chip Deep Learning**

![](https://img.shields.io/badge/HarwardLibrary-2023-skyblue?colorstyle=flat-square)
[![DOI-Link](https://img.shields.io/badge/DOI-https://nrs.harvard.edu/URN--3:HUL.INSTREPOS:37375806-sandybrown?style=flat-square)](https://nrs.harvard.edu/URN-3:HUL.INSTREPOS:37375806)
[![PDF-Link](https://img.shields.io/badge/PDF-Download-darkgreen?logoColor=red&&style=flat-square&logo=adobe)](https://dash.harvard.edu/bitstream/handle/1/37375806/Final_Draft_PhD_Dissertation_Thierry_Tambe.pdf)

---
