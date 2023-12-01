import os
from pathlib import Path


def get_shapenet_dir():
    return Path(os.environ["SHAPE_RETRIEVAL"]) / "shapenetcore-gltf"
