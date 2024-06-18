from functools import cache
from pickle import load

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch

from scorer import build_model, config_presets


@cache
def load_model(preset="edm2-img64-s-fid", device='cpu'):
    return build_model(preset, device)

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


def run_inference(img, preset="edm2-img64-s-fid", device="cuda"):
    img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0)
    img = torch.nn.functional.interpolate(img, size=64, mode='bilinear')
    model = load_model(preset=preset, device=device)
    x = model(img.cuda())
    x = x.square().sum(dim=(2, 3, 4)) ** 0.5
    nll, pct, ref_nll = compute_gmm_likelihood(x.cpu(), model_dir=f"models/{preset}")

    plot = plot_against_reference(nll, ref_nll)

    outstr = f"Anomaly score: {nll:.3f} / {pct:.2f} percentile"
    return outstr, plot


demo = gr.Interface(
    fn=run_inference,
    inputs=["image"],
    outputs=["text", gr.Plot(label="Comparing to Imagenette")],
)

if __name__ == "__main__":
    demo.launch()
