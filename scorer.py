import os
import pickle
from pickle import dump, load

import numpy as np
import PIL.Image
import torch
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import dnnlib


class EDMScorer(torch.nn.Module):
    def __init__(
        self,
        net,
        stop_ratio=0.8,  # Maximum ratio of noise levels to compute
        num_steps=10,  # Number of noise levels to evaluate.
        use_fp16=False,  # Execute the underlying model at FP16 precision?
        sigma_min=0.002,  # Minimum supported noise level.
        sigma_max=80,  # Maximum supported noise level.
        sigma_data=0.5,  # Expected standard deviation of the training data.
        rho=7,  # Time step discretization.
        device=torch.device("cpu"),  # Device to use.
    ):
        super().__init__()
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.net = net.eval()

        # Adjust noise levels based on how far we want to accumulate
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max * stop_ratio

        step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        print("Using steps:", t_steps)

        self.register_buffer("sigma_steps", t_steps.to(torch.float64))

    @torch.inference_mode()
    def forward(
        self,
        x,
        force_fp32=False,
    ):
        x = x.to(torch.float32)

        batch_scores = []
        for sigma in self.sigma_steps:
            xhat = self.net(x, sigma, force_fp32=force_fp32)
            c_skip = self.net.sigma_data**2 / (sigma**2 + self.net.sigma_data**2)
            score = xhat - (c_skip * x)

            # score_norms = score.mean(1)
            # score_norms = score.square().sum(dim=(1, 2, 3)) ** 0.5
            batch_scores.append(score)
        batch_scores = torch.stack(batch_scores, axis=1)

        return batch_scores


def build_model(netpath=f"edm2-img64-s-1073741-0.075.pkl", device="cpu"):
    model_root = "https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions"
    netpath = f"{model_root}/{netpath}"
    with dnnlib.util.open_url(netpath, verbose=1) as f:
        data = pickle.load(f)
    net = data["ema"]
    model = EDMScorer(net, num_steps=20).to(device)
    return model


def train_gmm(score_path, outdir="out/msma/"):
    X = torch.load(score_path)

    gm = GaussianMixture(n_components=5, random_state=42)
    clf = Pipeline([("scaler", StandardScaler()), ("GMM", gm)])
    clf.fit(X)
    inlier_nll = -clf.score_samples(X)

    with open(f"{outdir}/refscores.npz", "wb") as f:
        np.savez_compressed(f, inlier_nll)

    with open(f"{outdir}/gmm.pkl", "wb") as f:
        dump(clf, f, protocol=5)


def compute_gmm_likelihood(x_score, gmmdir):
    with open(f"{gmmdir}/gmm.pkl", "rb") as f:
        clf = load(f)
        nll = -clf.score_samples(x_score)

    with np.load(f"{gmmdir}/refscores.npz", "wb") as f:
        ref_nll = f["arr_0"]
        percentile = (ref_nll < nll).mean()

    return nll, percentile


def test_runner(device="cpu"):
    f = "goldfish.JPEG"
    image = (PIL.Image.open(f)).resize((64, 64), PIL.Image.Resampling.LANCZOS)
    image = np.array(image)
    image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
    x = torch.from_numpy(image).unsqueeze(0).to(device)
    model = build_model(device=device)
    scores = model(x)
    return scores


def runner(dataset_path, device="cpu"):
    dsobj = ImageFolderDataset(path=dataset_path, resolution=64)
    refimg, reflabel = dsobj[0]
    print(refimg.shape, refimg.dtype, reflabel)
    dsloader = torch.utils.data.DataLoader(
        dsobj, batch_size=48, num_workers=4, prefetch_factor=2
    )

    model = build_model(device=device)
    score_norms = []

    for x, _ in tqdm(dsloader):
        s = model(x.to(device))
        s = s.square().sum(dim=(2, 3, 4)) ** 0.5
        score_norms.append(s.cpu())

    score_norms = torch.cat(score_norms, dim=0)

    os.makedirs("out/msma", exist_ok=True)
    with open("out/msma/imagenette64_score_norms.pt", "wb") as f:
        torch.save(score_norms, f)

    print(f"Computed score norms for {score_norms.shape[0]} samples")


if __name__ == "__main__":
    # runner("/GROND_STOR/amahmood/datasets/img64/", device="cuda")
    train_gmm("out/msma/imagenette64_score_norms.pt")
    s = test_runner(device="cuda")
    s = s.square().sum(dim=(2, 3, 4)) ** 0.5
    s = s.to("cpu").numpy()
    nll, pct = compute_gmm_likelihood(s, gmmdir="out/msma/")
    print(f"Anomaly score for image: {nll[0]:.3f} @ {pct*100:.2f} percentile")
