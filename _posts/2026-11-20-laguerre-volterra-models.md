---
layout: post
title: "Laguerre–Volterra Models for Nonlinear Dynamics"
date: 2026-11-20
categories: [mathematics, robotics]
tags: [laguerre-basis, volterra-series, system-identification, antmaze]
---

How can a dynamics model use a long input history with only a few features? This interactive article shows how **discrete-time Laguerre filters** summarize past inputs and how a **Volterra model** uses products of those features to capture nonlinear interactions.

Adjust the Laguerre pole and lag window to see how the first four basis functions change. A second-order example explains how pairs of delayed inputs contribute to the output. The article closes with an AntMaze example that uses filtered joint commands to help predict the next state increment.

**[Explore the interactive Laguerre–Volterra article →]({{ '/laguerre-volterra.html' | relative_url }})**
