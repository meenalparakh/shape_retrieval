import os, sys

sys.path.append(os.environ["SHAPE_RETRIEVAL"])
# sys.path.append(os.environ["SPARSE_LSH"])

import numpy as np
import trimesh
import pickle
import plotly.express as px
from scipy.sparse import csr_matrix
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)

from utils.visualizer import VizServer
from src.feature_functions import FeatureFunctions, Dist
from utils.path_utils import get_shapenet_dir, get_runs_dir, get_class_pickle
from utils.image_utils import compare, visualize_tables
from src.extract_shape_features import compute_shape_features
from src.set_hyperparams import compute_params

# from SparseLSH.sparselsh import LSH
from src.lsh import MyLSH

import random

random.seed(0)
np.random.seed(0)

# add logging statements everywhere
# For large scale object storage and retrieval:
# TODO:
# 4. start writing the report and collecting results
# check how many queries are there


def divide_objects(categories, force=True, ratio=0.05):
    """
    ratio is for number of objects to keep for
    query within each category
    """

    fname = get_runs_dir() / "storage_fnames.pkl"
    if fname.exists() and (not force):
        logging.info(
            f"Storage and query division possibly exists already - skipping this step!"
        )
        return

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
        logging.info(f"Number of Queries: {len(query_fnames[category])}") 
        logging.info(f"Number of storage: {len(storage_fnames[category])}")
        # logging.info(f"")

    get_runs_dir().mkdir(parents=True, exist_ok=True)

    # store the object list for storage
    fname = get_runs_dir() / "storage_fnames.pkl"
    with open(fname, "wb") as f:
        pickle.dump(storage_fnames, f)
    logging.info(f"Object list for hashing saved!")

    # store the query object list
    fname = get_runs_dir() / "query_fnames.pkl"
    with open(fname, "wb") as f:
        pickle.dump(query_fnames, f)
    logging.info(f"Object list for querying saved!")


def save_mappings(
    feature_func,
    run_dirname,
    input_dim,
    # total_inputs,
    projection_dist="cauchy",
    bin_param_r=3.0,
    r1=1.0,
    c=2,
    batch_size=100,
    pcd_size=0.5,
    pdf_resolution=1000,
    bin_function="multiple",
    max_objects_hashed=0,
):
    fname = get_runs_dir() / "storage_fnames.pkl"
    with open(fname, "rb") as f:
        storage_fnames = pickle.load(f)

    logging.info(f"Object list for hashing loaded!")

    p_norm = 1 if projection_dist == "cauchy" else 2
    total_inputs = sum([len(cat_fnames) for _, cat_fnames in storage_fnames.items()])
    p1, p2, l, k, gamma = compute_params(
        total_inputs, p_norm, bin_param_r=bin_param_r, r1=r1, c=c
    )
    logging.info(
        f"Parameters computed for n({total_inputs}): p1({p1:0.2f}), p2({p2:0.2f}), l({l}), k({k}), gamma({gamma:0.2f})"
    )
    logging.info(f"Initializing LSH class!")

    lsh = MyLSH(
        hash_size=k,
        input_dim=input_dim,
        num_hashtables=l,
        projection_dist=projection_dist,
        bin_function=bin_function,
        bin_param_r=bin_param_r,
        feature_func=feature_func,
        r1=r1,
        c=c,
    )

    logging.info(f"Starting hashing objects with batch size {batch_size}")
    count = 0
    input_buffer = []
    extra_data_buffer = []
    for cat, cat_fnames in storage_fnames.items():
        num_objects = len(cat_fnames)
        if max_objects_hashed > 0:
            n = min(num_objects, max_objects_hashed)
            cat_fnames = cat_fnames[:n]

        logging.info(f"\Hashing {len(cat_fnames)} objects from {cat} category")
        logging.basicConfig(level=logging.ERROR)
        for fname in tqdm(cat_fnames):
            mesh_fname = get_shapenet_dir() / cat / fname
            mesh = trimesh.load(mesh_fname, file_type="gltf", force="mesh")
            features, bins = compute_shape_features(
                mesh,
                feature_func,
                pcd_size=0.5,
                max_num_samples=pdf_resolution,
                hist_resolution=input_dim,
                normalize=True,
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

        logging.basicConfig(level=logging.INFO)

    # index the remaining objects (less than batch_size)
    lsh.index(input_buffer, extra_data_buffer)

    logging.info(f"Saving the LSH class")
    (get_runs_dir() / run_dirname).mkdir(exist_ok=True, parents=True)
    lsh_pickle = get_runs_dir() / run_dirname / "lsh.pkl"
    with open(lsh_pickle, "wb") as f:
        pickle.dump(lsh, f)

    return lsh, lsh_pickle


def query_objects(run_dirname, pdf_resolution, max_queries=0, brute_force=False):
    logging.info(f"Loading the LSH class ...")
    lsh_pickle = get_runs_dir() / run_dirname / "lsh.pkl"
    with open(lsh_pickle, "rb") as f:
        lsh: MyLSH = pickle.load(f)

    logging.info(f"Loading objects for query ...")
    fname = get_runs_dir() / "query_fnames.pkl"
    with open(fname, "rb") as f:
        query_fnames = pickle.load(f)

    cat_success_rates = {}
    cat_average_distance = {}
    cat_fails = {}

    total_query_time = []
    for cat, cat_fnames in query_fnames.items():
        n = len(cat_fnames)
        if max_queries > 0:
            n = min(max_queries, n)

        success_count = 0
        failed_count = 0
        distances = []
        for fname in cat_fnames[:n]:
            mesh_fname = get_shapenet_dir() / cat / fname
            mesh = trimesh.load(mesh_fname, file_type="gltf", force="mesh")
            features, bins = compute_shape_features(
                mesh,
                lsh.feature_func,
                pcd_size=0.5,
                max_num_samples=pdf_resolution,
                hist_resolution=lsh.input_dim,
                normalize=True,
                visualize_pcd=False,
                viz_server=None,
                plot_histogram=False,
                obj_category=None,
            )
            # query one input and take the result for that query

            if brute_force:
                results, query_time = lsh.brute_force_query(
                    features.reshape((1, -1)), return_time=True
                )
            else:
                results, query_time = lsh.query(
                    features.reshape((1, -1)), return_time=True
                )

            total_query_time.append(query_time)
            result = results[0]
            if result is None:
                failed_count += 1
                # print(f"Queried: {cat}, model: {fname[:5]}..., Result: Fail")
            else:
                dist, (_, extra_data) = result
                extra_data: str = extra_data
                retrieved_category, retrieved_model = extra_data.split("_", maxsplit=1)
                # print(
                #     f"Queried: {cat}, model: {fname[:5]}..., \n"
                #     f"Retrieved: \n\t{retrieved_category}, "
                #     f"\n\tmodel: {retrieved_model}\n\tdistance: {dist}"
                # )
                if retrieved_category == cat:
                    success_count += 1
                distances.append(dist)

            # print()
        cat_success_rates[cat] = (success_count, n)
        cat_average_distance[cat] = np.mean(distances)
        cat_fails[cat] = failed_count / n

    print("__________________________________________________")
    print(f"SUCCESS RATES:")
    for cat, val in cat_success_rates.items():
        print(f"\t{cat} ({val[1]} objects): {val[0]/val[1]:.2f}")
    total_success = sum([val[0] for _, val in cat_success_rates.items()])
    total_objects = sum([val[1] for _, val in cat_success_rates.items()])
    print(f"\tTotal: {total_success / total_objects:.2f}")
    print("__________________________________________________")

    queries_results = {}
    queries_results['cat_success_rate'] = {cat: (val[0] / val[1]) for cat, val in cat_success_rates.items()}
    queries_results["total_success_rate"] = total_success / total_objects
    queries_results["average_query_time"] = np.mean(total_query_time)
    queries_results['distance'] = cat_average_distance
    queries_results['failed_count'] = cat_fails

    with open(get_runs_dir() / run_dirname / "query_results.pkl", 'wb') as f:
        pickle.dump(queries_results, f)

    return queries_results


if __name__ == "__main__":
    categories = ["can", "airplane", "chair"]
    run_dirname = "run_1"
    feature_func = Dist("D2", ord=2)
    input_dim = 100

    rerun = False

    if rerun:
        divide_objects(categories, ratio=0.01)
        lsh, lsh_pickle = save_mappings(
            feature_func=feature_func,
            run_dirname=run_dirname,
            input_dim=input_dim,
            projection_dist="cauchy",
            bin_param_r=100.0,
            r1=1.0,
            c=100.0,
            batch_size=100,
            pcd_size=0.2,
            pdf_resolution=1000,
            bin_function="binary",
            max_objects_hashed=200,
        )
    else:
        logging.info(f"Loadint the previously saved class")
        lsh_pickle = get_class_pickle(run_dirname)
        with open(lsh_pickle, "rb") as f:
            lsh: MyLSH = pickle.load(f)

    visualize_tables(lsh.hash_tables, plot_type="scatter", max_tables=10)
    query_objects(run_dirname, pdf_resolution=1000, max_queries=20)

    # viz_server = VizServer()

    # for category in categories:
    #    visualize_feature_vectors(feature_func, viz_server, category)
    #    print("\n")

    # compare(categories, 2)
