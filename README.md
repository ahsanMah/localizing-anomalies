---
title: Ano Edm Gr
emoji: 👀
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 4.36.1
app_file: hfapp.py
pinned: false
---

# Anomaly Localization with Score-Based Diffusion Models

Detect anomalies in natural images!

### Background
This project aims to add anomaly detection capabilites to score-based diffusion models. The main idea is to learn the distribution of the typical score vectors for each patch position. One can train a postion-condiditoned normalizing flow model to estimate the likelihood of the outputs of the score vector. The upshot is that you can visualize the per-patch likelihood and inspect the resulting heatmap of anomalies! This idea comes from work done as part of my doctoral dissertation, with the underlying methodology described in (this paper)[https://arxiv.org/abs/2010.13132]. 


The code builds upon the excellent (EDM2 repository)[https://github.com/NVlabs/edm2/] by NVLabs. Please refer to the original work form more information on the diffusion models.

### Caveats
As the underlying models are trained on Imagenet-1k, you may have better success when the subject belongs to one of the 1000 classes it was trianed on.
