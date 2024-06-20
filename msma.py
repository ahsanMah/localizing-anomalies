import os
import pickle
from functools import partial
from pickle import dump, load

import click
import numpy as np
import PIL.Image
import torch
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Subset
from tqdm import tqdm

import dnnlib
from dataset import ImageFolderDataset
from flowutils import PatchFlow

model_root = "https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions"

config_presets = {
    "edm2-img64-s-fid": f"{model_root}/edm2-img64-s-1073741-0.075.pkl",  # fid = 1.58
    "edm2-img64-m-fid": f"{model_root}/edm2-img64-m-2147483-0.060.pkl",  # fid = 1.43
    "edm2-img64-l-fid": f"{model_root}/edm2-img64-l-1073741-0.040.pkl",  # fid = 1.33
}


class StandardRGBEncoder:
    def __init__(self):
        super().__init__()

    def encode(self, x):  # raw pixels => final pixels
        return x.to(torch.float32) / 127.5 - 1

    def decode(self, x):  # final latents => raw pixels
        return (x.to(torch.float32) * 127.5 + 128).clip(0, 255).to(torch.uint8)


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
        self.encoder = StandardRGBEncoder()

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
        x = self.encoder.encode(x).to(torch.float32)

        batch_scores = []
        for sigma in self.sigma_steps:
            xhat = self.net(x, sigma, force_fp32=force_fp32)
            c_skip = self.net.sigma_data**2 / (sigma**2 + self.net.sigma_data**2)
            score = xhat - (c_skip * x)
            batch_scores.append(score)
        batch_scores = torch.stack(batch_scores, axis=1)

        return batch_scores


class ScoreFlow(torch.nn.Module):
    def __init__(
        self,
        preset,
        device="cpu",
    ):
        super().__init__()

        scorenet = build_model(preset)
        h = w = scorenet.net.img_resolution
        c = scorenet.net.img_channels
        num_sigmas = len(scorenet.sigma_steps)
        self.flow = PatchFlow((num_sigmas, c, h, w))

        self.flow = self.flow.to(device)
        self.scorenet = scorenet.to(device).requires_grad_(False)
        self.flow.init_weights()

    def forward(self, x, **score_kwargs):
        x_scores = self.scorenet(x, **score_kwargs)
        return self.flow(x_scores)


def build_model(preset="edm2-img64-s-fid", device="cpu"):
    netpath = config_presets[preset]
    with dnnlib.util.open_url(netpath, verbose=1) as f:
        data = pickle.load(f)
    net = data["ema"]
    model = EDMScorer(net, num_steps=20).to(device)
    return model


def quantile_scorer(gmm, X, y=None):
    return np.quantile(gmm.score_samples(X), 0.1)


def train_gmm(score_path, outdir, grid_search=False):
    X = torch.load(score_path)

    gm = GaussianMixture(
        n_components=7, init_params="kmeans", covariance_type="full", max_iter=100000
    )
    clf = Pipeline([("scaler", StandardScaler()), ("GMM", gm)])

    if grid_search:
        param_grid = dict(
            GMM__n_components=range(2, 11, 1),
        )

        grid = GridSearchCV(
            estimator=clf,
            param_grid=param_grid,
            cv=5,
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

    clf.fit(X)
    inlier_nll = -clf.score_samples(X)

    os.makedirs(outdir, exist_ok=True)
    with open(f"{outdir}/refscores.npz", "wb") as f:
        np.savez_compressed(f, inlier_nll)

    with open(f"{outdir}/gmm.pkl", "wb") as f:
        dump(clf, f, protocol=5)


def compute_gmm_likelihood(x_score, gmmdir):
    with open(f"{gmmdir}/gmm.pkl", "rb") as f:
        clf = load(f)
        nll = -clf.score_samples(x_score)

    with np.load(f"{gmmdir}/refscores.npz", "rb") as f:
        ref_nll = f["arr_0"]
        percentile = (ref_nll < nll).mean()

    return nll, percentile


def cache_score_norms(preset, dataset_path, outdir, device="cpu"):
    dsobj = ImageFolderDataset(path=dataset_path, resolution=64)
    refimg, reflabel = dsobj[0]
    print(f"Loading dataset from {dataset_path}")
    print(
        f"Number of Samples: {len(dsobj)} - shape: {refimg.shape}, dtype: {refimg.dtype}, labels {reflabel}"
    )
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

    os.makedirs(f"{outdir}/{preset}/", exist_ok=True)
    with open(f"{outdir}/{preset}/imagenette_score_norms.pt", "wb") as f:
        torch.save(score_norms, f)

    print(f"Computed score norms for {score_norms.shape[0]} samples")


def train_flow(dataset_path, preset, outdir, device="cuda"):
    dsobj = ImageFolderDataset(path=dataset_path, resolution=64)
    refimg, reflabel = dsobj[0]
    print(f"Loaded {len(dsobj)} samples from {dataset_path}")

    # Subset of training dataset
    val_ratio = 0.1
    train_len = int((1 - val_ratio) * len(dsobj))
    val_len = len(dsobj) - train_len

    print(
        f"Generating train/test split with ratio={val_ratio} -> {train_len}/{val_len}..."
    )
    train_ds = Subset(dsobj, range(train_len))
    val_ds = Subset(dsobj, range(train_len, train_len + val_len))

    trainiter = torch.utils.data.DataLoader(
        train_ds, batch_size=48, num_workers=4, prefetch_factor=2
    )
    testiter = torch.utils.data.DataLoader(
        val_ds, batch_size=48, num_workers=4, prefetch_factor=2
    )

    model = ScoreFlow(preset, device=device)
    opt = torch.optim.AdamW(model.flow.parameters(), lr=3e-4, weight_decay=1e-5)
    train_step = partial(
        PatchFlow.stochastic_step,
        flow_model=model.flow,
        opt=opt,
        train=True,
        n_patches=64,
        device=device,
    )
    eval_step = partial(
        PatchFlow.stochastic_step,
        flow_model=model.flow,
        train=False,
        n_patches=128,
        device=device,
    )

    os.makedirs(f"{outdir}/{preset}", exist_ok=True)
    pbar = tqdm(trainiter, desc="Train Loss: ? - Val Loss: ?")
    step = 0

    for x, _ in tqdm(trainiter):
        x = x.to(device)
        scores = model.scorenet(x)

        if step == 0:
            with torch.inference_mode():
                val_loss = eval_step(scores, x)

        train_loss = train_step(scores, x)

        if (step + 1) % 10 == 0:

            with torch.inference_mode():
                val_loss = 0.0
                for i, (x, _) in enumerate(testiter):
                    x = x.to(device)
                    scores = model.scorenet(x)
                    val_loss += eval_step(scores, x)
                    break
                val_loss /= i + 1

        pbar.set_description(
            f"Step: {step:d} - Train: {train_loss:.3f} - Val: {val_loss:.3f}"
        )
        step += 1
    
    torch.save(model.flow.state_dict(), f"{outdir}/{preset}/flow.pt")


@torch.inference_mode
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


def test_flow_runner(preset, device="cpu", load_weights=None):
    # f = "doge.jpg"
    f = "goldfish.JPEG"
    image = (PIL.Image.open(f)).resize((64, 64), PIL.Image.Resampling.LANCZOS)
    image = np.array(image)
    image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
    x = torch.from_numpy(image).unsqueeze(0).to(device)

    score_flow = ScoreFlow(preset, device=device)

    if load_weights is not None:
        score_flow.flow.load_state_dict(torch.load(load_weights))

    heatmap = score_flow(x)
    print(heatmap.shape)

    heatmap = score_flow(x).detach().cpu().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) * 255
    im = PIL.Image.fromarray(heatmap[0, 0])
    im.convert("RGB").save(
        "heatmap.png",
    )

    return


@click.command()

# Main options.
@click.option(
    "--run",
    help="Which function to run",
    type=click.Choice(
        ["cache-scores", "train-flow", "train-gmm"], case_sensitive=False
    ),
)
@click.option(
    "--outdir",
    help="Where to load/save the results",
    metavar="DIR",
    type=str,
    required=True,
)
@click.option(
    "--preset",
    help="Configuration preset",
    metavar="STR",
    type=str,
    default="edm2-img64-s-fid",
    show_default=True,
)
@click.option(
    "--data", help="Path to the dataset", metavar="ZIP|DIR", type=str, default=None
)
def cmdline(run, outdir, **opts):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preset = opts["preset"]
    dataset_path = opts["data"]

    if run in ["cache-scores", "train-flow"]:
        assert opts["data"] is not None, "Provide path to dataset"
    
    if run == "cache-scores":
        cache_score_norms(
            preset=preset, dataset_path=dataset_path, outdir=outdir, device=device
        )

    if run == "train-gmm":
        train_gmm(
            score_path=f"{outdir}/{preset}/imagenette_score_norms.pt",
            outdir=f"{outdir}/{preset}",
            grid_search=True,
        )
    
    if run == "train-flow":
        train_flow(dataset_path, outdir=outdir, preset=preset, device=device)
        test_flow_runner(preset, device=device, load_weights=f"{outdir}/{preset}/flow.pt")

    # train_flow(imagenette_path, preset, device)

    # cache_score_norms(
    #     preset=preset,
    #     dataset_path="/GROND_STOR/amahmood/datasets/img64/",
    #     device="cuda",
    # )
    # train_gmm(
    #     f"out/msma/{preset}_imagenette_score_norms.pt", outdir=f"out/msma/{preset}"
    # )
    # s = test_runner(device=device)
    # s = s.square().sum(dim=(2, 3, 4)) ** 0.5
    # s = s.to("cpu").numpy()
    # nll, pct = compute_gmm_likelihood(s, gmmdir=f"out/msma/{preset}/")
    # print(f"Anomaly score for image: {nll[0]:.3f} @ {pct*100:.2f} percentile")


if __name__ == "__main__":
    cmdline()
