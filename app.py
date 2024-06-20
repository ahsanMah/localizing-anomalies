from functools import cache
from pickle import load

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch

from msma import ScoreFlow, config_presets


@cache
def load_model(modeldir, preset="edm2-img64-s-fid", device='cpu', outdir=None):
    model = ScoreFlow(preset, device=device)
    model.flow.load_state_dict(torch.load(f"{modeldir}/{preset}/flow.pt"))
    return model

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
    ax.hist(ref_nll, label="Reference Scores")
    ax.axvline(nll, label='Image Score', c='red', ls="--")
    plt.legend()
    fig.tight_layout()
    return fig


def plot_heatmap(heatmap):
    fig, ax = plt.subplots()
    im = heatmap[0,0]
    ax.imshow(im, cmap='gist_heat')
    fig.tight_layout()
    return fig

# def compute_scores    


def run_inference(img, preset="edm2-img64-s-fid", device="cuda"):

    with torch.inference_mode():
        img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0)
        img = torch.nn.functional.interpolate(img, size=64, mode='bilinear')
        img = img.to(device)
        model = load_model(modeldir='models', preset=preset, device=device)
        x = model.scorenet(img)
        x = x.square().sum(dim=(2, 3, 4)) ** 0.5
        img_likelihood = model(img).cpu().numpy()
        nll, pct, ref_nll = compute_gmm_likelihood(x.cpu(), model_dir=f"models/{preset}")
    
    outstr = f"Anomaly score: {nll:.3f} / {pct:.2f} percentile"
    histplot = plot_against_reference(nll, ref_nll)
    heatmapplot = plot_heatmap(img_likelihood)

    return outstr, heatmapplot, histplot


demo = gr.Interface(
    fn=run_inference,
    inputs=["image"],
    outputs=["text",
             gr.Plot(label="Anomaly Heatmap"),
             gr.Plot(label="Comparing to Imagenette"),
            ],
)

if __name__ == "__main__":
    demo.launch()
