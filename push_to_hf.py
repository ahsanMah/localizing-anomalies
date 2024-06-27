import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import click
import torch
from huggingface_hub import HfApi
from safetensors.torch import save_file

from msma import EDMScorer, ScoreFlow, build_model_from_pickle


@click.command
@click.option(
    "--basedir",
    help="Directory holding the model weights and logs",
    type=str,
    required=True,
)
@click.option(
    "--preset", help="Preset of the score model used", type=str, required=True
)
def main(basedir, preset):
    basedir = Path(basedir)
    modeldir = basedir / preset

    net = build_model_from_pickle(preset)
    with open(modeldir / "config.json", "rb") as f:
        model_params = json.load(f)

    model = ScoreFlow(
        net,
        device="cpu",
        **model_params["PatchFlow"],
    )
    model.flow.load_state_dict(torch.load(modeldir / "flow.pt"))

    api = HfApi()
    repo_name = "ahsanMah/localizing-edm"

    # Create repo if not existing yet and get the associated repo_id
    repo_id = api.create_repo(repo_name, exist_ok=True).repo_id

    # Save all files in a temporary directory and push them in a single commit
    with TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Save weights
        save_file(model.state_dict(), tmpdir / "model.safetensors")

        # save config
        (tmpdir / "config.json").write_text(
            json.dumps(model.config, sort_keys=True, indent=4)
        )

        # save gmm and cached score norms
        shutil.copyfile(modeldir / "gmm.pkl", tmpdir / "gmm.pkl")
        shutil.copyfile(modeldir / "refscores.npz", tmpdir / "refscores.npz")

        # Generate model card
        # card = generate_model_card(model)
        # (tmpdir / "README.md").write_text(card)

        # Save logs
        shutil.copytree(modeldir / "logs", tmpdir / "logs")

        # Push to hub
        api.upload_folder(repo_id=repo_id, path_in_repo=preset, folder_path=tmpdir)


if __name__ == "__main__":
    main()
