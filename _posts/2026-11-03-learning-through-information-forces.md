---
layout: post
title: "Learning Through Information Forces: Five Visual ITL Experiments"
date: 2026-11-03
categories: [machine-learning, information-theory]
tags: [ITL, Renyi-entropy, kernels, mutual-information, divergence]
---

How can a kernel function turn information-theoretic quantities into forces that guide learning? In a new interactive article, I explore the mathematics of **Information-Theoretic Learning (ITL)**, drawing on José C. Principe's *Information Theoretic Learning: Rényi's Entropy and Kernel Perspectives* (2010).

The article presents five computational experiments:

1. **Gaussian potential fields:** how kernel bandwidth changes local information landscapes in one and two dimensions.
2. **Information-force fields:** gradients of kernel potentials, numerical gradient checks, and the connection with mean shift.
3. **Particle attraction and repulsion:** how optimizing quadratic Rényi entropy changes the configuration of data points.
4. **Distribution matching:** minimizing Euclidean divergence between kernel density estimates and comparing with Cauchy–Schwarz divergence.
5. **Quadratic mutual information:** learning a one-dimensional projection that preserves the information in two class labels.

<div class="imgcap">
<img src="/assets/Information Forces_ A Visual Scientific Journey.png">
<div class="thecap" style="text-align:justify"> A Neural Network with two inputs, one hidden layer with two neurons and one two-neuron output layer. 
</div>
  
The demonstrations include derivations, figures, optimization trajectories, and results from an executable Python/Colab notebook. In the distribution-matching example, estimated Euclidean divergence decreased from **0.4269 to 0.0304**. In the feature-learning example, estimated quadratic mutual information increased from **0.00738 to 0.11835**. These values describe illustrative synthetic experiments, not benchmark performance.

**[Read the full interactive article →]({{ '/itl-visual-experiments/' | relative_url }})**

The central idea is simple: by estimating information potentials from pairwise kernel evaluations and differentiating them, we can turn entropy, divergence, and mutual information into optimization objectives with interpretable gradient forces.
