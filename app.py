from functools import cache
from pickle import load

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
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

def plot_heatmap(img: Image, heatmap: np.array):
    fig, ax = plt.subplots()
    cmap = plt.get_cmap("gist_heat")
    h = heatmap[0,0].copy()
    qmin, qmax = np.quantile(h, 0.5), np.quantile(h, 0.999)
    h = np.clip(h, a_min=qmin, a_max=qmax)
    h = (h-h.min()) / (h.max() - h.min())
    h = cmap(h, bytes=True)[:,:,:3]
    h = Image.fromarray(h).resize(img.size, resample=Image.Resampling.BILINEAR)
    im = Image.blend(img, h, alpha=0.6)
    im = ax.imshow(np.array(im))
    # fig.colorbar(im)
    # plt.grid(False)
    # plt.axis("off")
    fig.tight_layout()
    return fig

def run_inference(input_img, preset="edm2-img64-s-fid", device="cuda"):

    # img = center_crop_imagenet(64, img)
    input_img = input_img.resize(size=(64, 64), resample=Image.Resampling.LANCZOS)

    with torch.inference_mode():
        img = np.array(input_img)
        img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0)
        img = img.float().to(device)
        model = load_model(modeldir='models', preset=preset, device=device)
        img_likelihood = model(img).cpu().numpy()

        img = torch.nn.functional.interpolate(img, size=64, mode='bilinear')
        x = model.scorenet(img)
        x = x.square().sum(dim=(2, 3, 4)) ** 0.5
        nll, pct, ref_nll = compute_gmm_likelihood(x.cpu(), model_dir=f"models/{preset}")

    outstr = f"Anomaly score: {nll:.3f} / {pct:.2f} percentile"
    histplot = plot_against_reference(nll, ref_nll)
    heatmapplot = plot_heatmap(input_img, img_likelihood)

    return outstr, heatmapplot, histplot


demo = gr.Interface(
    fn=run_inference,
    inputs=[gr.Image(type='pil', label="Input Image")],
    outputs=["text",
             gr.Plot(label="Anomaly Heatmap"),
             gr.Plot(label="Comparing to Imagenette"),
            ],
)

if __name__ == "__main__":
    demo.launch()
