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

Hyperspectral images provide high spectral resolution but low spatial resolution, while multispectral images provide higher spatial resolution but fewer spectral bands. Existing HSI-MSI fusion methods usually assume spatial co-registration, which can be difficult to guarantee in practice.

**FRESCO** (**F**actorized **R**epresentation for **E**nhanced **S**uper-resolution using latent **C**omponent-adversarial **O**ptimization) addresses this unregistered setting using a two-stage framework:

1. **MSR stage**: Coupled spectral unmixing is used to recover the super-resolved MSI-side spectral image.


$$
\begin{aligned}
\min_{\{S_r^{(H)}, c_r^{(H)}, S_r^{(M)}\}_{r=1}^{R}}
&\left\|Y^{(H)} - \sum_{r=1}^{R} S_r^{(H)} \circ c_r^{(H)} \right\|_F^2
+
\left\|Y^{(M)} - \sum_{r=1}^{R} S_r^{(M)} \circ \left(P^{(M)} c_r^{(H)}\right) \right\|_F^2 \\
\text{s.t.}\quad
&\operatorname{rank}\left(S_r^{(M)}\right) \leq L_r^{(M)}, \quad
\operatorname{rank}\left(S_r^{(H)}\right) \leq L_r^{(H)}, \quad \forall r \in [R], \\
&\sum_{r=1}^{R} S_r^{(M)} = \mathbf{1}\mathbf{1}^{\top}, \quad
\sum_{r=1}^{R} S_r^{(H)} = \mathbf{1}\mathbf{1}^{\top}, \\
&S_r^{(M)}, S_r^{(H)}, c_r^{(H)} \geq 0, \quad \forall r \in [R].
\end{aligned}
$$


2. **HSR stage**: Latent-space adversarial learning is used to recover the super-resolved HSI-side spectral image by matching abundance patch distributions.

The method does not require paired training data or spatial co-registration.

