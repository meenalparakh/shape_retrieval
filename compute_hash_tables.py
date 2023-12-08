import os, sys

sys.path.append(os.environ["SHAPE_RETRIEVAL"])
# sys.path.append(os.environ["SPARSE_LSH"])

import numpy as np
import trimesh
import pickle
import plotly.express as px
from scipy.sparse import csr_matrix

from utils.visualizer import VizServer
from src.feature_functions import FeatureFunctions, D2
from utils.path_utils import get_shapenet_dir, get_runs_dir
from utils.image_utils import compare
from src.extract_shape_features import compute_shape_features
from src.set_hyperparams import compute_params

# from SparseLSH.sparselsh import LSH
from src.lsh import MyLSH

import random

random.seed(0)
np.random.seed(0)

# For large scale object storage and retrieval:
# 0. divide the dataset into objects for storage and those for query - done
# 1. Compute features for all storage shapes - done
# 2. index those points into the hash tables - done
# 3. query points


def divide_objects(categories, run_dirname="run_1", ratio=0.01):
    """
    ratio is for number of objects to keep for
    query within each category
    """
    storage_fnames = {}
    query_fnames = {}
    for category in categories:
        object_dir = get_shapenet_dir() / category
        fnames = os.listdir(object_dir)
        fnames = [fn for fn in fnames if fn.endswith(".gltf")]
        random.shuffle(fnames)
        num_storage = len(fnames) - max(1, int(len(fnames) * ratio))
        storage_fnames[category] = fnames[:num_storage]
        query_fnames[category] = fnames[num_storage:]

    (get_runs_dir() / run_dirname).mkdir(parents=True, exist_ok=True)

    # store the object list for storage
    fname = get_runs_dir() / run_dirname / "storage_fnames.pkl"
    with open(fname, "wb") as f:
        pickle.dump(storage_fnames, f)

    # store the query object list
    fname = get_runs_dir() / run_dirname / "query_fnames.pkl"
    with open(fname, "wb") as f:
        pickle.dump(query_fnames, f)

    return fname


def save_mappings(
    feature_func,
    run_dirname,
    input_dim,
    total_inputs,
    projection_dist="cauchy",
    bin_param_r=3.0,
    r1=1.0,
    c=2,
    batch_size=100,
    bin_function="multiple",
):
    fname = get_runs_dir() / run_dirname / "storage_fnames.pkl"
    with open(fname, "rb") as f:
        storage_fnames = pickle.load(f)

    p_norm = 1 if projection_dist == "cauchy" else 2
    p1, p2, l, k, gamma = compute_params(
        total_inputs, p_norm, bin_param_r=bin_param_r, r1=r1, c=c
    )

    # feature_func = D2("D2")
    lsh = MyLSH(
        hash_size=k,
        input_dim=input_dim,
        num_hashtables=l,
        projection_dist=projection_dist,
        bin_function=bin_function,
        bin_param_r=bin_param_r,
        feature_func=feature_func,
    )

    count = 0
    input_buffer = []
    extra_data_buffer = []
    for cat, cat_fnames in storage_fnames.items():
        for fname in cat_fnames:
            mesh_fname = get_shapenet_dir() / cat / fname
            mesh = trimesh.load(mesh_fname, file_type="gltf", force="mesh")
            features, bins = compute_shape_features(
                mesh,
                feature_func,
                visualize_pcd=False,
                viz_server=None,
                plot_histogram=False,
                obj_category=None,
            )
            input_buffer.append(features)
            extra_data_buffer.append(f"{cat}_{fname}")
            count += 1
            if count == batch_size:
                lsh.index(input_buffer, extra_data_buffer)
                input_buffer = []
                extra_data_buffer = []

    # index the remaining objects (less than batch_size)
    lsh.index(input_buffer, extra_data_buffer)

    lsh_pickle = get_runs_dir() / run_dirname / "lsh.pkl"
    with open(lsh_pickle, "wb") as f:
        pickle.dump(lsh, f)

    return lsh_pickle


def query_objects(run_dirname, r1, r2):
    lsh_pickle = get_runs_dir() / run_dirname / "lsh.pkl"
    with open(lsh_pickle, "rb") as f:
        lsh: MyLSH = pickle.load(f)

    fname = get_runs_dir() / run_dirname / "query_fnames.pkl"
    with open(fname, "rb") as f:
        query_fnames = pickle.load(f)

    for cat, cat_fnames in query_fnames.items():
        for fname in cat_fnames:
            mesh_fname = get_shapenet_dir() / cat / fname
            mesh = trimesh.load(mesh_fname, file_type="gltf", force="mesh")
            features, bins = compute_shape_features(
                mesh,
                lsh.feature_func,
                visualize_pcd=False,
                viz_server=None,
                plot_histogram=False,
                obj_category=None,
            )
            result = lsh.query(features.reshape((1, -1)), r1=r1, r2=r2)
            if result is None:
                print(f"Queried: {cat}, model: {fname[:5]}..., Result: Fail")
            else:
                dist, (_, extra_data) = result
                extra_data: str = extra_data
                retrieved_category, retrieved_model = extra_data.split("_", maxsplit=1)
                print(
                    f"Queried: {cat}, model: {fname[:5]}..., \n"
                    f"Retrieved: \n\t{retrieved_category}, "
                    f"\n\tmodel: {retrieved_model}\n\tdistance: {dist}"
                )
            print()


if __name__ == "__main__":
    categories = ["can", "airplane"]

    # feature_func = D2("D2")
    # viz_server = VizServer()

    # for category in categories:
    #    visualize_feature_vectors(feature_func, viz_server, category)
    #    print("\n")

    # compare(categories, 2)
