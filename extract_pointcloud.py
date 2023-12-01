import os, sys

sys.path.append(os.environ["SHAPE_RETRIEVAL"])

import numpy as np
import trimesh

from utils.path_utils import get_shapenet_dir

# TODO:
# 1. load the dataset - and visualize the point cloud


def load_shapenet_data(category="chair"):
    category_dir = get_shapenet_dir() / category
    fnames = list(category_dir.glob("*.gltf"))
    print(f"Number of chairs {len(fnames)}")
    chosen_mesh = 0

    print(f"Mesh used: {fnames[chosen_mesh]}")
    import pdb

    pdb.set_trace()
    mesh = trimesh.load(fnames[chosen_mesh], file_type="gltf")


if __name__ == "__main__":
    load_shapenet_data(category="chair")
