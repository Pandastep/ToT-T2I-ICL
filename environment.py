# environment.py
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the local SEED / CoBSAT model directory.
# By default, the code expects: <repo_root>/models/SEED
# The directory itself is not included in this repository.
SEED_PROJECT_ROOT = os.environ.get(
    "SEED_PROJECT_ROOT",
    os.path.join(ROOT_DIR, "models", "SEED")
)

# Optional Hugging Face cache directory.
TRANSFORMER_CACHE = os.environ.get(
    "TRANSFORMER_CACHE",
    os.path.join(ROOT_DIR, ".cache", "huggingface")
)
