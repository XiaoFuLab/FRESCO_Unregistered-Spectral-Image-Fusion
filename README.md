## **FRESCO**: **F**actorized **R**epresentation for **E**nhanced **S**uper-resolution using latent **C**omponent-adversarial **O**ptimization

<p align="center">
  <img src="assets/un_reg_objective.png" width="850">
</p>

This repository corresponds to the paper:

> **Unregistered Spectral Image Fusion: Unmixing, Adversarial Learning, and Recoverability** Jiahui Song, Sagar Shrestha, Xiao Fu  

<p>
  <a href="https://arxiv.org/abs/2603.21510">
    <b>[ arXiv ]</b>
  </a>
</p>

## Overview

**FRESCO** addresses the unregistered hyperspectral and multispectral image fusion problem using a two-stage framework:

<p align="center">
  <img src="assets/MSR_HSR.png" width="850">
</p>

1. **MSR stage**: Coupled spectral unmixing is used to recover the M-SRI.

<p align="center">
  <img src="assets/MSR.png" width="750">
</p>

2. **HSR stage**: Latent-space adversarial learning is used to recover the H-SRI by matching abundance patch distributions.

<p align="center">
  <img src="assets/distribution_matching.png" width="850">
</p>


The method does not require **paired training data** or **spatial co-registration**.


## Installation

This algorithm was implemented with **Python 3.12.8**. To set up the environment, run:

```bash
conda create -n fresco python=3.12.8
conda activate fresco
pip install -r requirements.txt
```

## Semi-real Experiments

**MSR Results**

<p align="center">
  <img src="assets/pavia_MSR.png" width="850">
</p>

**HSR Results**

<p align="center">
  <img src="assets/pavia_HSR.png" width="850">
</p>


## Real Applications

<p align="center">
  <img src="assets/real_application.png" width="850">
</p>
