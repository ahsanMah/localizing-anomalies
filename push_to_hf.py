import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from huggingface_hub import HfApi
from safetensors.torch import save_file

from msma import ScoreFlow

basedir = Path("models/condgauss")
preset = "edm2-img64-s-fid"
modeldir = basedir / preset

model = ScoreFlow(preset)
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

    # Generate model card
    # card = generate_model_card(model)
    # (tmpdir / "README.md").write_text(card)

    # Save logs
    shutil.copytree(modeldir / "logs", tmpdir / "logs")
    # Save figures
    # Save evaluation metrics
    # ...

    # Push to hub
    api.upload_folder(repo_id=repo_id, folder_path=tmpdir)
