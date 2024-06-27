import numpy as np
import PIL.Image as Image
import spaces
import torch

from app import (
    build_demo,
    compute_gmm_likelihood,
    load_model_from_hub,
    plot_against_reference,
    plot_heatmap,
)


@spaces.GPU
def run_inference(model, img):
    print("model on cuda:", next(model.scorenet.net.parameters()).is_cuda)
    img = torch.nn.functional.interpolate(img, size=64, mode="bilinear")
    score_norms = model.scorenet(img)
    score_norms = score_norms.square().sum(dim=(2, 3, 4)) ** 0.5
    img_likelihood = model(img).cpu().numpy()
    score_norms = score_norms.cpu().numpy()
    return img_likelihood, score_norms


def localize_anomalies(input_img, preset="edm2-img64-s-fid", load_from_hub=False):
    print("NEW LOCALIZE")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # img = center_crop_imagenet(64, img)
    input_img = input_img.resize(size=(64, 64), resample=Image.Resampling.LANCZOS)

    with torch.inference_mode():
        img = np.array(input_img)
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        img = img.float().to(device)
        model = load_model_from_hub(preset=preset, device=device)
        img_likelihood, score_norms = run_inference(model, img)
        nll, pct, ref_nll = compute_gmm_likelihood(
            score_norms, model_dir=f"models/{preset}"
        )

    outstr = f"Anomaly score: {nll:.3f} / {pct:.2f} percentile"
    histplot = plot_against_reference(nll, ref_nll)
    heatmapplot = plot_heatmap(input_img, img_likelihood)

    return outstr, heatmapplot, histplot


demo = build_demo(localize_anomalies)
if __name__ == "__main__":
    demo.launch()
