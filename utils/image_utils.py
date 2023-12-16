import os
import random
import typing as T
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from sklearn.decomposition import PCA
import logging

import trimesh
from src.extract_shape_features import compute_shape_features
from src.feature_functions import FeatureFunctions
from utils.path_utils import get_results_dir, get_runs_dir, get_shapenet_dir
from src.lsh import MyLSH
import pickle
import plotly.express as px

def compare(categories, num_images, name="compare"):
    num_cols = len(categories)
    num_rows = num_images

    plt.figure(figsize=(3 * 3, 4 * 2))
    for col_idx, category in enumerate(categories):
        histogram = get_results_dir() / f"histogram_{category}.png"
        img = plt.imread(histogram)

        plt.subplot(2, 3, 1 + col_idx)
        plt.imshow(img)
        plt.axis("off")

    plt.subplots_adjust(
        left=0.00, bottom=0.01, right=1.0, top=0.99, wspace=0.01, hspace=0.01
    )
    plt.show()
    # plt.savefig(get_results_dir() / f"{name}.png")


def visualize_feature_vectors(
    feature_func: FeatureFunctions,
    categories: T.List[str],
    num_images: int = 2,
):
    # collect all image pairs
    im_fnames = {}
    num_cat = len(categories)
    for cat in categories:
        category_dir = get_shapenet_dir() / cat
        print(f"Category: {cat}")
        fnames = (list(category_dir.glob("*.gltf")))
        random.shuffle(fnames)
        fnames = fnames[:2]
        im_fnames[cat] = {}
        im_fnames[cat]['fnames'] = fnames
        im_fnames[cat]['features'] = []
        
        for idx, fn in enumerate(fnames):
            print(f"\t{str(fn).split(os.sep)[-1]}")
            mesh = trimesh.load(fn, file_type="gltf", force="mesh")
            features, bins = compute_shape_features(
                mesh,
                feature_func,
                pcd_size=0.5,
                max_num_samples=int(1e3),
                hist_resolution=100,
                normalize=True,
                visualize_pcd=False,
                viz_server=None,
                plot_histogram=False,
                obj_category=None,
            )
            im_fnames[cat]['features'].append(features)

    distances = np.zeros((num_cat, num_cat))
    for idx, cat in enumerate(categories):
        for idx2, cat2 in enumerate(categories):
            f1 = im_fnames[cat]['features'][0]
            f2 = im_fnames[cat2]['features'][1]
            dist = np.linalg.norm(f1 - f2, ord=2)
            distances[idx, idx2] = dist

    distances = distances / np.max(distances)
    
    fig = px.imshow(distances,
                    x=[f"{cat}_1" for cat in categories], 
                    y=[f"{cat}_2" for cat in categories])
    
    fig.write_image('distance_categories.png')
    

def visualize_feature_vectors_diff_funcs(
    feature_functions: FeatureFunctions,
    cat,
):
    # collect all image pairs
    category_dir = get_shapenet_dir() / cat
    print(f"Category: {cat}")
    fnames = (list(category_dir.glob("*.gltf")))
    random.shuffle(fnames)
    fname = fnames[0]
            
    for feature_func in feature_functions:
        mesh = trimesh.load(fname, file_type="gltf", force="mesh")
        features, bins = compute_shape_features(
                mesh,
                feature_func,
                pcd_size=0.5,
                max_num_samples=int(1e3),
                hist_resolution=100,
                normalize=True,
                visualize_pcd=False,
                viz_server=None,
                plot_histogram=True,
                obj_category=f"{feature_func.name}",
            )
    compare([fn.name for fn in feature_functions], 1, 'compare_func')

def visualize_tables(dirname, plot_type="bar", max_tables=0, name='table'):
    logging.info(f"Loading the LSH class ...")
    lsh_pickle = get_runs_dir() / dirname / "lsh.pkl"
    with open(lsh_pickle, "rb") as f:
        lsh: MyLSH = pickle.load(f)

    
    l = len(lsh.hash_tables)
    if max_tables > 0:
        l = min(max_tables, l)

    for table_idx, table in enumerate(lsh.hash_tables[:l]):
        keys = sorted(table.keys())
        keys_cat_counts = {}
        for idx, k in enumerate(keys):
            entries = table.get_val(k)
            cats = [e[1].split("_", maxsplit=1)[0] for e in entries]
            unique_cats, counts = np.unique(cats, return_counts=True)

            for cat, count in zip(unique_cats, counts):
                if cat not in keys_cat_counts:
                    keys_cat_counts[cat] = np.zeros(len(keys))
                keys_cat_counts[cat][idx] = count
        # if plot_type == "bar":
        if True:
            table_bar_graph(f"{name}_{table_idx}", keys, keys_cat_counts)
            # else:
            # table_pca_scatter_plot(table_idx, keys, keys_cat_counts)


def table_bar_graph(table_idx, keys, keys_cat_counts):
    all_categories = sorted(list(keys_cat_counts.keys()))
    # key_names = [str(k) for k in range(len(keys))]
    key_names = [f"{k}" for k, val in enumerate(keys)]

    fig = go.Figure()

    for cat in all_categories:
        cat_counts = keys_cat_counts[cat]
        fig.add_trace(go.Bar(x=key_names, y=cat_counts, name=cat))

    fig.update_layout(barmode="group", xaxis={"categoryorder": "category ascending"})
    # fig.show()
    fig.write_image(get_results_dir() / f"table_bar_graph_{table_idx}.png")


def table_pca_scatter_plot(table_idx, keys, keys_cat_counts, scale=0.50):
    all_categories = sorted(list(keys_cat_counts.keys()))
    keys = np.array(keys)
    if len(keys[0]) == 1:
        fig = go.Figure()
        for idx, cat in enumerate(all_categories):
            cat_counts = keys_cat_counts[cat]
            fig.add_trace(
                go.Scatter(
                    x=keys[:, 0],
                    y=[idx + 1] * len(keys),
                    name=cat,
                    mode="markers",
                    marker=dict(size=cat_counts * scale),
                )
            )
    else:
        pca = PCA(n_components=2)
        pca.fit(keys)
        key_components = pca.transform(keys)

        fig = go.Figure()

        for idx, cat in enumerate(all_categories):
            cat_counts = keys_cat_counts[cat]
            fig.add_trace(
                go.Scatter(
                    x=key_components[:, 0] + idx + 1.0,
                    y=key_components[:, 1],
                    name=cat,
                    mode="markers",
                    marker=dict(size=cat_counts * scale),
                )
            )

    # fig.show()
    fig.write_image(get_results_dir() / f"table_scatter_{table_idx}.png")


def performance_time_graph(
    methods: T.List[str],
    performance: T.Dict[str, T.List[float]],
    query_time: T.Dict[str, T.List[float]],
):
    categories = sorted(list(performance.keys()))
    num_cats = len(categories)

    # different colors correspond to different categories
    colors = list(range(num_cats))

    # different markers correspond to different methods
    markers = ["circle", "diamond", "asterik"]

    method_data = {method: {"time": [], "performance": []} for method in methods}
    for idx, method in enumerate(methods):
        for cat in categories:
            method_data[method]["performance"].append(performance[cat][idx])
            method_data[method]["time"].append(query_time[idx])

    fig = go.Figure()

    for method_idx, method in enumerate(methods):
        fig.add_trace(
            go.Scatter(
                mode="markers",
                x=method_data[method]["time"],
                y=method_data[method]["performance"],
                marker=dict(
                    color=colors,
                    symbol=markers[method_idx],
                    size=10.0
                ),
                name=method,
            )
        )
    fig.update_xaxes(type="log")
    fig.write_image(get_results_dir() / f"perf_time_method.png")


def performance_time_bargraph(
    methods: T.List[str],
    performance: T.Dict[str, T.List[float]],
    query_time: T.List[float],
):
    categories = sorted(list(performance.keys()))
    num_cats = len(categories)

    # different colors correspond to different categories
    colors = list(range(num_cats))

    fig = go.Figure()
    for idx, method in enumerate(methods):
        fig.add_trace(
            go.Bar(
                x=categories,
                y=[performance[cat][idx] for cat in categories],
                name=f"{method} ({query_time[idx]:.1E} s)",
            )
        )
    fig.update_layout(
        xaxis_title=f"Object Categories",
        yaxis_title=f"Success Rate",
    )
    fig.write_image(get_results_dir() / f"perf_time_method_bar.png")

def time_distance_graph(
    methods: T.List[str],
    distances: T.Dict[str, T.List[float]],
    query_time: T.List[float],
):
    categories = sorted(list(distances.keys()))
    num_cats = len(categories)

    # different colors correspond to different categories
    colors = list(range(num_cats))

    fig = go.Figure()
    for idx, method in enumerate(methods):
        fig.add_trace(
            go.Bar(
                x=categories,
                y=[distances[cat][idx] for cat in categories],
                name=f"{method} ({query_time[idx]:.1E} s)",
            )
        )
    fig.update_layout(
        xaxis_title=f"Object Categories",
        yaxis_title=f"Distance",
    )
    fig.write_image(get_results_dir() / f"dist_time_method_bar.png")

def perf_time_variable(
    variables: T.List,
    performance: T.Dict[str, T.List[float]],
    query_time: T.List[float],
    x_var: str,
    y_var: str,
    yaxis_range=None,
    log_x=False,
):
    categories = sorted(list(performance.keys()))
    num_cats = len(categories)

    fig = go.Figure()

    for idx, cat in enumerate(categories):
        fig.add_trace(
            go.Scatter(
                mode="lines+markers",
                x=variables,
                y=performance[cat],
                name=cat,
                marker=dict(color=idx),
            )
        )
    if log_x:
        fig.update_xaxes(type="log")
    fig.update_layout(
        xaxis_title=f"{x_var}",
        yaxis_title=f"{y_var}",
    )
    if yaxis_range:
        fig.update_layout(
            yaxis_range=yaxis_range,
        )
    fig.write_image(get_results_dir() / f"{y_var}_{x_var}.png")

    fig = go.Figure(
        go.Scatter(x=variables, y=query_time, mode="lines+markers")
    )
    if log_x:
        fig.update_xaxes(type="log")
    fig.update_layout(
        xaxis_title=f"{x_var}",
        yaxis_title="query time (s)",
    )
    fig.write_image(get_results_dir() / f"time_{x_var}.png")


def perf_time_inp_dim(
    input_dims: T.List[int],
    performance: T.Dict[str, T.List[float]],
    query_time: T.List[float],
):
    perf_time_variable(
        input_dims, performance, query_time, x_var="input_dim", y_var="performance", log_x=False
    )


def perf_time_num_samples(
    num_samples: T.List[int],
    performance: T.Dict[str, T.List[float]],
    query_time: T.List[float],
):
    perf_time_variable(
        num_samples, performance, query_time, variable_name="num_samples", log_x=True
    )


def perf_types(
    choices: T.List[str], 
    performance: T.Dict[str, T.List[float]], 
    query_time: T.List[float],
    choice_name: str,
    perf_name: str,
):
    categories = list(performance.keys())
    num_cats = len(categories)

    fig = go.Figure()
    for idx, choice in enumerate(choices):
        fig.add_trace(
            go.Bar(
                x=categories,
                y=[performance[cat][idx] for cat in categories],
                name=f"{choice} ({query_time[idx]:.1E} s)",
            )
        )
        
    fig.update_layout(
        xaxis_title=f"{choice_name}",
        yaxis_title=f"{perf_name}",
    )
    if perf_name == "Success Rate" or perf_name == "success rate":
        fig.update_layout(
            yaxis_range=[0.0, 1.2],
        )
    # for cat in categories:
    # fig.add_trace(
    #     go.Bar(
    #         x=choices,
    #         y=performance[cat],
    #         name=cat,
    #     )
    # )
    fig.write_image(get_results_dir() / f"{perf_name}_{choice_name}.png")


def perf_feature_funcs(feature_functions, performance):
    perf_types(feature_functions, performance, "feature_funcs")


def perf_hashing(projections, performance):
    perf_types(projections, performance, choice_name="hash_type")
