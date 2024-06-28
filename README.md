---
title: Localizing Anomalies
sdk: gradio
sdk_version: 4.36.1
app_file: hfapp.py
pinned: false
---

# Anomaly Localization with Score-Based Diffusion Models
<p align="middle">
    <img src="samples/duckelephant.jpeg" alt="input elephant" width="32%"/>
    <img src="assets/heatmap.webp" alt="heatmap image" width="32%"/>
</p>

## Introduction

This project aims to add anomaly detection capabilities to score-based diffusion models, enabling the detection of anomalies in natural images. The core idea is to learn the distribution of typical score vectors for each patch position in an image. By training a position-conditioned normalizing flow model, we can estimate the likelihood of the score vectors' outputs. This allows for visualizing per-patch likelihoods and generating heatmaps of anomalies.

## Background

The methodology for this project is based on research conducted as part of my doctoral dissertation. The underlying concepts are detailed in the following paper: [Multiscale Score Matching for Out-of-Distribution Detection](https://arxiv.org/abs/2010.13132).

This code builds upon the excellent [EDM2 repository](https://github.com/NVlabs/edm2) by NVLabs. For more information on the diffusion models, please refer to the original work.

## Features

- **Anomaly Detection**: Identify and localize anomalies in natural images.
- **Score-Based Diffusion Models**: Utilizes state-of-the-art (non-latent) diffusion models.
- **Heatmap Visualization**: Generate heatmaps to visualize anomaly likelihoods.

## Installation

You can run the gradio app locally. To get started, clone the project and install the dependencies.

```bash
   git clone https://github.com/ahsanMah/localizing-anomalies.git
   cd localizing-anomalies
   pip install -r requirements.txt
```

The model will work without a GPU but may take 15-30 seconds given your resources. If you have a NVIDIA GPU, install PyTorch with CUDA.

## Usage
```bash
DNNLIB_CACHE_DIR=/tmp/ python app.py
```
Then go to [http://localhost:7860](localhost:7860)

Note that running this app will download a pickle of the EDM models by NVLabs available [here](https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/). The models will be saved in the `DNNLIB_CACHE_DIR` directory you specify, defaulting to `/tmp/` if not.

## Best Practices and Usage Tips

To get the best results from our anomaly localization model, consider the following recommendations:

- **Image Content**: As the underlying models are trained on ImageNet-1k, you may have better success when the subject belongs to one of the 1000 classes it was trained on.

- **Subject Positioning**: It helps to have the subject centered in the middle of the image. This is due to the resize-center-crop that was used to train the score models.

- **Image Aspect**: For optimal performance, try to use square asopect ratios. In my testing it does not matter as much as the subject positioning.

- **Model Selection**: Choose the small, medium, large model preset (e.g., `edm2-img64-s-fid`) based on your available computational resources: 
- **Fine-tuning**: For domain-specific applications, consider tuning the model on a dataset more closely related to your target domain using the information below

## Training the Model

This section outlines the steps to train the model locally. Make sure you have the required dependencies installed and a GPU with CUDA support for optimal performance.

### 1. Prepare the Dataset

First, download a dataset such as Imagenette and unzip it into a folder. Then, prepare the dataset using the `dataset_tool.py` script:

```bash
python dataset_tool.py convert --source=/path/to/imagenette/ --dest=/path/to/output/img64/ --resolution=64x64 --transform=center-crop-dhariwal
```

This command will process the images, resizing them to 64x64 pixels and applying the center-crop-dhariwal transform that was used to train the backbone score-based diffusion model

### 2. Train the Flow Model
To train the flow model, use the following command:
```bash
DNNLIB_CACHE_DIR=/path/to/model/cache CUDA_VISIBLE_DEVICES=0 python msma.py train-flow --outdir models/ --dataset_path /path/to/prepared/dataset --preset edm2-img64-s-fid --num_flows 8 --epochs 20 --batch_size 32
```
Options:

- --outdir: Directory to save the trained model and logs
- --dataset_path: Path to the prepared dataset
- --preset: Configuration preset (default: "edm2-img64-s-fid")
- --num_flows: Number of normalizing flow functions in the PatchFlow model (default: 4)
- --epochs: Number of training epochs (default: 10)
- --batch_size: batch size for training (default: 128)

Note that the evaluation step will use twice the number of batches as training

### 3. Cache Score Norms
While the flow model is training (or after, if GPU resources are limited), cache the score norms:
```bash
DNNLIB_CACHE_DIR=/path/to/model/cache CUDA_VISIBLE_DEVICES=1 python msma.py cache-scores --outdir models/ --dataset_path /path/to/prepared/dataset --preset edm2-img64-s-fid --batch_size=128
```
Options:
- --outdir: Directory to save the cached scores
- --dataset_path: Path to the prepared dataset
- --preset: Configuration preset (should match the flow training preset)
- --batch_size: Number of samples per batch (default: 64)


### 4. Train the Gaussian Mixture Model (GMM)
Finally, train the GMM using the cached score norms:

```bash
DNNLIB_CACHE_DIR=/path/to/model/cache python msma.py train-gmm --outdir models/ --preset edm2-img64-s-fid
```

Options:

- --outdir: Directory to load the cached scores and save the trained GMM (should match the previous step)
- --preset: Configuration preset (should match previous steps)
- --gridsearch: (Optional) Use grid search to find the best number of components otherwise 7 are used (default: False)

### Notes

- Adjust the paths in the commands according to your directory structure.
- The `DNNLIB_CACHE_DIR` environment variable sets the cache directory for pre-trained models.
- `CUDA_VISIBLE_DEVICES` allows you to specify which GPU to use for training.
- You can run the flow training and score norm caching concurrently if you have multiple GPUs available.
- The `--preset` option should be consistent across all steps to ensure compatibility.

For more detailed information on each command and its options, refer to the script documentation or run python msma.py [command] --help.





## Acknowledgements

Special thanks to NVLabs for their [EDM2 repository](https://github.com/NVlabs/edm2).