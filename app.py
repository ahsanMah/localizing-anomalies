from functools import cache
from pickle import load

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch

from scorer import build_model


@cache
def load_model(device):
    return build_model(device=device)

@cache
def load_reference_scores(gmmdir='models'):
    with np.load(f"{gmmdir}/refscores.npz", "rb") as f:
        ref_nll = f["arr_0"]
    return ref_nll

def compute_gmm_likelihood(x_score, gmmdir='models'):
    with open(f"{gmmdir}/gmm.pkl", "rb") as f:
        clf = load(f)
        nll = -clf.score(x_score)

    ref_nll = load_reference_scores(gmmdir)
    percentile = (ref_nll < nll).mean() * 100

    return nll, percentile

def plot_against_reference(nll):
    ref_nll = load_reference_scores()
    print(ref_nll.shape)
    fig, ax = plt.subplots()
    ax.hist(ref_nll)
    ax.axvline(nll, label='Image Score', c='red', ls="--")
    plt.legend()
    fig.tight_layout()
    return fig

def run_inference(img, device='cuda'):
    img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0)
    img = torch.nn.functional.interpolate(img, size=64, mode='bilinear')
    model = load_model(device=device)
    x = model(img.cuda())
    x = x.square().sum(dim=(2, 3, 4)) ** 0.5
    nll, pct = compute_gmm_likelihood(x.cpu())

    plot = plot_against_reference(nll)
    print(plot)
    outstr = f"Anomaly score: {nll:.3f} -> {pct:.2f} percentile"
    return outstr, plot


demo = gr.Interface(
    fn=run_inference,
    inputs=["image"],
    outputs=["text", gr.Plot()],
)

if __name__ == "__main__":
    demo.launch()
