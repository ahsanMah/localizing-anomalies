import json
import os
from functools import cache
from pickle import load

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from msma import (
    ScoreFlow,
    build_model_from_config,
    build_model_from_pickle,
    config_presets,
)


@cache
def load_model(modeldir, preset="edm2-img64-s-fid", device="cpu"):
    modeldir = f"{modeldir}/{preset}"
    with open(f"{modeldir}/config.json", "rb") as f:
        model_params = json.load(f)
    scorenet = build_model_from_pickle(preset=preset)
    model = ScoreFlow(scorenet, **model_params['PatchFlow'])
    model.flow.load_state_dict(torch.load(f"{modeldir}/flow.pt"))
    print("Loaded:", model_params)
    return model.to(device)


@cache
def load_model_from_hub(preset, device):
    cache_dir = "/tmp/"
    if 'DNNLIB_CACHE_DIR' in os.environ:
        cache_dir = os.environ["DNNLIB_CACHE_DIR"]


    for fname in ['config.json', 'gmm.pkl', 'refscores.npz', 'model.safetensors' ]:
        cached_fname = hf_hub_download(
            repo_id="ahsanMah/localizing-edm",
            subfolder=preset,
            filename=fname,
            cache_dir=cache_dir,
        )
    modeldir = os.path.dirname(cached_fname)
    print("HF Cache Dir:", modeldir)

    with open(f"{modeldir}/config.json", "rb") as f:
        model_params = json.load(f)
        print("Loaded:", model_params)

    hf_checkpoint = f"{modeldir}/model.safetensors"
    model = build_model_from_config(model_params)
    model.load_state_dict(load_file(hf_checkpoint), strict=True)
    model = model.eval().requires_grad_(False)
    model.to(device)
    return model, modeldir


@cache
def load_reference_scores(model_dir):
    with np.load(f"{model_dir}/refscores.npz", "rb") as f:
        ref_nll = f["arr_0"]
    return ref_nll


def compute_gmm_likelihood(x_score, model_dir):


    with open(f"{model_dir}/gmm.pkl", "rb") as f:
        clf = load(f)
        nll = -clf.score(x_score)

    ref_nll = load_reference_scores(model_dir)
    percentile = (ref_nll < nll).mean() * 100

    return nll, percentile, ref_nll


def plot_against_reference(nll, ref_nll):
    fig, ax = plt.subplots()
    ax.hist(ref_nll, label="Reference Scores", bins=25)
    ax.axvline(nll, label="Image Score", c="red", ls="--")
    plt.legend()
    fig.tight_layout()
    return fig


def plot_heatmap(img: Image, heatmap: np.array):
    fig, ax = plt.subplots()
    cmap = plt.get_cmap("gist_heat")
    h = -heatmap[0, 0].copy()
    qmin, qmax = np.quantile(h, 0.8), np.quantile(h, 0.999)
    h = np.clip(h, a_min=qmin, a_max=qmax)
    h = (h - h.min()) / (h.max() - h.min())
    h = cmap(h, bytes=True)[:, :, :3]
    h = Image.fromarray(h).resize(img.size, resample=Image.Resampling.BILINEAR)
    im = Image.blend(img, h, alpha=0.6)
    return im

@torch.no_grad
def run_inference(model, img):
    img = torch.nn.functional.interpolate(img, size=64, mode="bilinear")
    score_norms = model.scorenet(img)
    score_norms = score_norms.square().sum(dim=(2, 3, 4)) ** 0.5
    img_likelihood = model(img).cpu().numpy()
    score_norms = score_norms.cpu().numpy()
    return img_likelihood, score_norms

def localize_anomalies(input_img, preset="edm2-img64-s-fid", load_from_hub=False):

    orig_size = input_img.size
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # img = center_crop_imagenet(64, img)
    input_img = input_img.resize(size=(64, 64), resample=Image.Resampling.LANCZOS)

    with torch.inference_mode():
        img = np.array(input_img)
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        img = img.float().to(device)
        if load_from_hub:
            model, modeldir = load_model_from_hub(preset=preset, device=device)
        else:
            model = load_model(modeldir="models", preset=preset, device=device)
            modeldir = f"models/{preset}"
        img_likelihood, score_norms = run_inference(model, img)
        nll, pct, ref_nll = compute_gmm_likelihood(
            score_norms, model_dir=modeldir
        )

    outstr = f"Anomaly score: {nll:.3f} / {pct:.2f} percentile"
    histplot = plot_against_reference(nll, ref_nll)
    heatmapplot = plot_heatmap(input_img, img_likelihood)
    heatmapplot = heatmapplot.resize(orig_size)

    return outstr, heatmapplot, histplot

def build_demo(inference_fn):

    demo = gr.Interface(
        fn=inference_fn,
        inputs=[
            gr.Image(type="pil", label="Input Image"),
            gr.Dropdown(
                choices=config_presets.keys(),
                label="Score Model Preset",
                info="The preset of the underlying score estimator. These are the EDM2 diffusion models from Karras et.al.",
            ),
            gr.Checkbox(
                label="HuggingFace Hub",
                value=True,
                info="Load a pretrained model from HuggingFace. Uncheck to use a model from `models`  directory.",
            ),
        ],
        outputs=[
            gr.Text(
                label="Estimated global outlier scores - Percentiles with respect to Imagenette Scores"
            ),
            gr.Image(label="Anomaly Heatmap", min_width=160),
            gr.Plot(label="Comparing to Imagenette"),
        ],
        examples=[
            ["samples/duckelephant.jpeg", "edm2-img64-s-fid", True],
            ["samples/sharkhorse.jpeg", "edm2-img64-s-fid", True],
            ["samples/goldfish.jpeg", "edm2-img64-s-fid", True],
        ],
    )

    return demo

demo = build_demo(localize_anomalies)
if __name__ == "__main__":
    demo.launch()
