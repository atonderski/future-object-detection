import os


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

config = {
    "visualization_path": f"{REPO_ROOT}/visualization",
    "checkpoint_path": f"{REPO_ROOT}/checkpoints",
    "nuscenes_path": f"{REPO_ROOT}/data/nuscenes",
    "nuimages_path": f"{REPO_ROOT}/data/nuimages",
}
