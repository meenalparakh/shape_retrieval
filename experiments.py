import os
import sys

from compute_hash_tables import divide_objects, query_objects, save_mappings

sys.path.append(os.environ["SHAPE_RETRIEVAL"])

import typing as T
import trimesh
import random
import numpy as np

random.seed(0)
np.random.seed(0)
import logging

logging.basicConfig(level=logging.INFO)

from utils.path_utils import get_shapenet_dir
from utils.image_utils import compare, perf_time_inp_dim, performance_time_bargraph
from src.feature_functions import Dist, Angle, Area, Volume, FeatureFunctions
from src.extract_shape_features import compute_shape_features


def visualize_feature_vectors(
    feature_func: FeatureFunctions,
    categories: T.List[str],
    num_images: int = 2,
):
    for cat in categories:
        category_dir = get_shapenet_dir() / cat
        print(f"Category: {cat}")
        fnames = sorted(list(category_dir.glob("*.gltf")))[:num_images]
        for idx, fn in enumerate(fnames):
            print(f"\t{str(fn).split(os.sep)[-1]}")
            mesh = trimesh.load(fn, file_type="gltf", force="mesh")
            features, bins = compute_shape_features(
                mesh=mesh,
                feature_func=feature_func,
                pcd_size=0.5,
                max_num_samples=5000,
                hist_resolution=100,
                normalize=True,
                visualize_pcd=False,
                plot_histogram=True,
                obj_category=f"{cat}_{idx}",
            )
    compare(categories, num_images=num_images, name=f"{feature_func.name}")


if __name__ == "__main__":
    max_hashed = 50
    max_queries = 5
    skip_hashing = True
    categories = ["can", "airplane", "bowl"]
    input_dims_lst = [100, 500, 1000]
    num_samples_lst = [int(1e3), int(1e4), int(1e5)]
    projections_lst = ["cauchy", "gaussian"]

    feature_funcs_lst = [
        Dist("D1", ord=1),
        Dist("D2", ord=2),
        Angle("Angle"),
        Area("Area"),
        Volume("Volume"),
    ]
    feature_func_names_lst = [f.name for f in feature_funcs_lst]

    # dividing objects into train and test - to be used uniformly
    divide_objects(categories, ratio=0.01)

    # experiments across varying input_dims, keeping others fixed
    num_samples = num_samples_lst[0]
    projection = projections_lst[0]
    feature_func = feature_funcs_lst[0]

    performance = {cat: [] for cat in categories}
    query_time = []
    for inp_dim in input_dims_lst:
        
        logging.info(f"STARTING EXPERIMENT FOR INPUT DIM {inp_dim}...............")
        run_dirname = f"input_dim_{inp_dim}"
        if not skip_hashing:
            lsh, lsh_pickle = save_mappings(
                feature_func=feature_func,
                run_dirname=run_dirname,
                input_dim=inp_dim,
                projection_dist=projection,
                bin_param_r=100.0,
                r1=1.0,
                c=100.0,
                batch_size=100,
                pcd_size=0.5,
                pdf_resolution=num_samples,
                bin_function="binary",
                max_objects_hashed=max_hashed,
            )

        result_dict = query_objects(
            run_dirname, pdf_resolution=num_samples, max_queries=max_queries
        )
        for cat in categories:
            performance[cat].append(result_dict[cat])
        query_time.append(result_dict["average_query_time"])

    perf_time_inp_dim(input_dims_lst, performance, query_time)

    # experiments for performance vs query time for brute force search and hashing
    logging.info(f"STARTING EXPERIMENT FOR LSH vs BRUTE-FORCE ...............")

    inp_dim = input_dims_lst[0]
    num_samples = num_samples_lst[0]
    run_dirname = f"input_dim_{inp_dim}"
    methods = ["LSH", "brute-force"]
    performance = {}

    result_dict = query_objects(
        run_dirname, pdf_resolution=num_samples, max_queries=max_queries
    )
    for cat in categories:
        performance[cat] = [result_dict[cat]]
    query_time = [result_dict["average_query_time"]]

    result_dict = query_objects(
        run_dirname,
        pdf_resolution=num_samples,
        brute_force=True,
        max_queries=max_queries,
    )
    for cat in categories:
        performance[cat].append(result_dict[cat])
    query_time.append(result_dict["average_query_time"])

    performance_time_bargraph(methods, performance, query_time)
