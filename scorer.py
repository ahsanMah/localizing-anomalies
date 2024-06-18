import os
import pickle
from pickle import dump, load

import numpy as np
import PIL.Image
import torch
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import dnnlib

model_root = "https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions"

config_presets = {
    "edm2-img64-s-fid": f"{model_root}/edm2-img64-s-1073741-0.075.pkl",  # fid = 1.58
    "edm2-img64-m-fid": f"{model_root}/edm2-img64-m-2147483-0.060.pkl",  # fid = 1.43
    "edm2-img64-l-fid": f"{model_root}/edm2-img64-l-1073741-0.040.pkl",  # fid = 1.33
}


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
        self.sigma_min = 1e-1
        self.sigma_max = sigma_max * stop_ratio

        step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
        t_steps = (
            self.sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (self.sigma_min ** (1 / rho) - self.sigma_max ** (1 / rho))
        ) ** rho
        # print("Using steps:", t_steps)

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
            batch_scores.append(score)
        batch_scores = torch.stack(batch_scores, axis=1)

        return batch_scores


def build_model(preset="edm2-img64-s-fid", device="cpu"):
    netpath = config_presets[preset]
    with dnnlib.util.open_url(netpath, verbose=1) as f:
        data = pickle.load(f)
    net = data["ema"]
    model = EDMScorer(net, num_steps=20).to(device)
    return model


def train_gmm(score_path, outdir):
    def quantile_scorer(gmm, X, y=None):
        return np.quantile(gmm.score_samples(X), 0.1)

    X = torch.load(score_path)

    gm = GaussianMixture(init_params="kmeans", covariance_type="full", max_iter=100000)
    clf = Pipeline([("scaler", StandardScaler()), ("GMM", gm)])
    clf.fit(X)
    inlier_nll = -clf.score_samples(X)

    param_grid = dict(
        GMM__n_components=range(2, 11, 2),
    )

    grid = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        cv=10,
        n_jobs=2,
        verbose=1,
        scoring=quantile_scorer,
    )

    grid_result = grid.fit(X)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    print("-----" * 15)
    means = grid_result.cv_results_["mean_test_score"]
    stds = grid_result.cv_results_["std_test_score"]
    params = grid_result.cv_results_["params"]
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    clf = grid.best_estimator_

    os.makedirs(outdir, exist_ok=True)
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
    # f = "doge.jpg"
    f = "goldfish.JPEG"
    image = (PIL.Image.open(f)).resize((64, 64), PIL.Image.Resampling.LANCZOS)
    image = np.array(image)
    image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
    x = torch.from_numpy(image).unsqueeze(0).to(device)
    model = build_model(device=device)
    scores = model(x)
    return scores


def runner(preset, dataset_path, device="cpu"):
    dsobj = ImageFolderDataset(path=dataset_path, resolution=64)
    refimg, reflabel = dsobj[0]
    print(refimg.shape, refimg.dtype, reflabel)
    dsloader = torch.utils.data.DataLoader(
        dsobj, batch_size=48, num_workers=4, prefetch_factor=2
    )

    model = build_model(preset=preset, device=device)
    score_norms = []

    for x, _ in tqdm(dsloader):
        s = model(x.to(device))
        s = s.square().sum(dim=(2, 3, 4)) ** 0.5
        score_norms.append(s.cpu())

    score_norms = torch.cat(score_norms, dim=0)

    os.makedirs("out/msma", exist_ok=True)
    with open(f"out/msma/{preset}_imagenette_score_norms.pt", "wb") as f:
        torch.save(score_norms, f)

    print(f"Computed score norms for {score_norms.shape[0]} samples")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preset = "edm2-img64-s-fid"
    # runner(
    #     preset=preset,
    #     dataset_path="/GROND_STOR/amahmood/datasets/img64/",
    #     device="cuda",
    # )
    train_gmm(
        f"out/msma/{preset}_imagenette_score_norms.pt", outdir=f"out/msma/{preset}"
    )
    s = test_runner(device=device)
    s = s.square().sum(dim=(2, 3, 4)) ** 0.5
    s = s.to("cpu").numpy()
    nll, pct = compute_gmm_likelihood(s, gmmdir=f"out/msma/{preset}")
    print(f"Anomaly score for image: {nll[0]:.3f} @ {pct*100:.2f} percentile")
