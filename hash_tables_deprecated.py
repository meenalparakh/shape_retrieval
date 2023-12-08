import os, sys

sys.path.append(os.environ["SHAPE_RETRIEVAL"])
# sys.path.append(os.environ["SPARSE_LSH"])

import numpy as np
import trimesh
import plotly.express as px
from scipy.sparse import csr_matrix

from utils.visualizer import VizServer
from src.feature_functions import FeatureFunctions, D2
from utils.path_utils import get_shapenet_dir
from utils.image_utils import compare
from src.extract_shape_features import compute_shape_features

from sparselsh import LSH

import random

random.seed(0)
np.random.seed(0)


def visualize_feature_vectors(
    feature_func: FeatureFunctions,
    viz_server: VizServer = None,
    category: str = "chair",
) -> None:
    category_dir = get_shapenet_dir() / category
    fnames = sorted(list(category_dir.glob("*.gltf")))
    print(f"Number of {category}: {len(fnames)}")

    for chosen_mesh in range(2):
        print(f"Mesh used ({category}): {fnames[chosen_mesh]}")
        mesh = trimesh.load(fnames[chosen_mesh], file_type="gltf", force="mesh")
        features, bins = compute_shape_features(
            mesh,
            feature_func,
            visualize_pcd=True,
            viz_server=viz_server,
            plot_histogram=True,
            obj_category=category + f"_{chosen_mesh}",
        )


def sparse_lsh_trial(categories):
    feature_func = D2("D2")
    feature_size = 100
    hash_size = 10
    category_features = []
    category_names = []
    for category in categories:
        category_dir = get_shapenet_dir() / category
        fnames = sorted(list(category_dir.glob("*.gltf")))

        for chosen_mesh in range(3):
            mesh = trimesh.load(fnames[chosen_mesh], file_type="gltf", force="mesh")
            features, bins = compute_shape_features(
                mesh,
                feature_func,
                max_num_samples=1000,
                hist_resolution=feature_size,
                visualize_pcd=False,
                # viz_server=viz_server,
                plot_histogram=False,
                obj_category=category + f"_{chosen_mesh}",
            )

            assert (len(bins) - 1) == feature_size
            category_features.append(features)
            category_names.append(category)

    category_features_NB = np.array(category_features)
    # assert category_features_NB.shape == (len(categories), feature_size)

    X = csr_matrix(category_features_NB[1:, :])

    lsh = LSH(
        hash_size=hash_size,
        input_dim=feature_size,
        num_hashtables=1,
        storage_config={"dict": None},
    )

    lsh.index(X, extra_data=category_names[1:])
    # an example retrieval - using the same vector as input
    # Build a 1-D (single row) sparse matrix
    X_sim = csr_matrix(category_features_NB[:1, :])
    # find the point in X nearest to X_sim
    points = lsh.query(X_sim, num_results=1)
    # split up the first result into its parts
    print(f"points are: {points}")
    (point, label), dist = points[0]
    print(f"Retrieved: {label}, \npoint: {point}, \ndistance: {dist}")  # 'last'


if __name__ == "__main__":
    categories = ["can", "airplane", "laptop", "chair", "mug"]
    sparse_lsh_trial(categories)

    # feature_func = D2("D2")
    # viz_server = VizServer()

    # for category in categories:
    #    visualize_feature_vectors(feature_func, viz_server, category)
    #    print("\n")

    # compare(categories, 2)
