import typing as T
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from sklearn.decomposition import PCA

from utils.path_utils import get_results_dir


def compare(categories, num_images, name="compare"):
    num_cols = len(categories)
    num_rows = num_images

    plt.figure(figsize=(3 * num_cols, 4 * num_images))
    for col_idx, category in enumerate(categories):
        for row_idx in range(num_rows):
            histogram = get_results_dir() / f"histogram_{category}_{row_idx}.png"
            img = plt.imread(histogram)

            plt.subplot(num_rows, num_cols, 1 + col_idx + num_cols * row_idx)
            plt.imshow(img)
            plt.axis("off")

    plt.subplots_adjust(
        left=0.00, bottom=0.01, right=1.0, top=0.99, wspace=0.01, hspace=0.01
    )
    plt.savefig(get_results_dir() / f"{name}.png")


def visualize_tables(hash_tables, plot_type="bar", max_tables=0):
    l = len(hash_tables)
    if max_tables > 0:
        l = min(max_tables, l)

    for table_idx, table in enumerate(hash_tables[:l]):
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
            table_bar_graph(table_idx, keys, keys_cat_counts)
            # else:
            table_pca_scatter_plot(table_idx, keys, keys_cat_counts)


def table_bar_graph(table_idx, keys, keys_cat_counts):
    all_categories = sorted(list(keys_cat_counts.keys()))
    # key_names = [str(k) for k in range(len(keys))]
    key_names = [f"{k}_{val}" for k, val in enumerate(keys)]

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
            method_data[method]["time"].append(query_time[cat][idx])

    fig = go.Figure()

    for method_idx, method in enumerate(methods):
        fig.add_trace(
            data=go.Scatter(
                mode="markers",
                x=method_data[method]["performance"],
                y=method_data[method]["time"],
                log_y=True,
                marker=dict(
                    color=colors,
                    symbol=markers[method_idx],
                    name=method,
                    text=categories,
                ),
            )
        )
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
                name=f"{method} ({query_time[idx]} s)",
            )
        )
    fig.write_image(get_results_dir() / f"perf_time_method_bar.png")

def time_distance_graph(
    methods: T.List[str],
    distances: T.Dict[str, T.List[float]],
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
                name=f"{method} ({query_time[idx]} s)",
            )
        )
    fig.write_image(get_results_dir() / f"perf_time_method_bar.png")

def perf_time_variable(
    variables: T.List,
    performance: T.Dict[str, T.List[float]],
    query_time: T.List[float],
    variable_name: str,
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

    fig.write_image(get_results_dir() / f"perf_{variable_name}.png")

    fig = go.Figure(
        go.Scatter(x=variables, y=query_time, mode="lines+markers")
    )
    if log_x:
        fig.update_xaxes(type="log")
    fig.write_image(get_results_dir() / f"time_{variable_name}.png")


def perf_time_inp_dim(
    input_dims: T.List[int],
    performance: T.Dict[str, T.List[float]],
    query_time: T.List[float],
):
    perf_time_variable(
        input_dims, performance, query_time, variable_name="input_dim", log_x=False
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
    choices: T.List[str], performance: T.Dict[str, T.List[float]], choice_name: str
):
    categories = list(performance.keys())
    num_cats = len(categories)

    fig = go.Figure()
    for idx, choice in enumerate(choices):
        fig.add_trace(
            go.Bar(
                x=categories,
                y=[performance[cat][idx] for cat in categories],
                name=choice,
            )
        )
    # for cat in categories:
    # fig.add_trace(
    #     go.Bar(
    #         x=choices,
    #         y=performance[cat],
    #         name=cat,
    #     )
    # )
    fig.write_image(get_results_dir() / f"perf_{choice_name}.png")


def perf_feature_funcs(feature_functions, performance):
    perf_types(feature_functions, performance, "feature_funcs")


def perf_hashing(projections, performance):
    perf_types(projections, performance, choice_name="hash_type")
