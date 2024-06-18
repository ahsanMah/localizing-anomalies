from pickle import load

import gradio as gr
import numpy as np
import torch

from scorer import build_model


def compute_gmm_likelihood(x_score, gmmdir='models'):
    with open(f"{gmmdir}/gmm.pkl", "rb") as f:
        clf = load(f)
        nll = -clf.score(x_score)

    with np.load(f"{gmmdir}/refscores.npz", "wb") as f:
        ref_nll = f["arr_0"]
        percentile = (ref_nll < nll).mean() * 100

    return nll, percentile

def run_inference(img):
    img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0)
    img = torch.nn.functional.interpolate(img, size=64, mode='bilinear')
    model = build_model(device='cuda')
    x = model(img.cuda())
    x = x.square().sum(dim=(2, 3, 4)) ** 0.5
    nll, pct = compute_gmm_likelihood(x.cpu())

    return f"Image of shape: {img.shape} -> {nll:.3f}@{pct:.2f}"


demo = gr.Interface(
    fn=run_inference,
    inputs=["image"],
    outputs=["text"],
)

demo.launch()
