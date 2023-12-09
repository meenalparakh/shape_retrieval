import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from sklearn.decomposition import PCA

from utils.path_utils import get_results_dir


def compare(categories, num_images, name="compare"):
    num_cols = len(categories)
    num_rows = num_images

    plt.figure(figsize=(3 * num_cols, 4*num_images))
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
    all_categories = list(keys_cat_counts.keys())
    # key_names = [str(k) for k in range(len(keys))]
    key_names = [f"{k}_{val}" for k, val in enumerate(keys)]

    first_cat = all_categories[0]
    first_cat_counts = keys_cat_counts[first_cat]
    fig = go.Figure(go.Bar(x=key_names, y=first_cat_counts, name=first_cat))

    for cat in all_categories[1:]:
        cat_counts = keys_cat_counts[cat]
        fig.add_trace(go.Bar(x=key_names, y=cat_counts, name=cat))

    fig.update_layout(barmode="group", xaxis={"categoryorder": "category ascending"})
    # fig.show()
    fig.write_image(get_results_dir() / f"table_bar_graph_{table_idx}.png")


def table_pca_scatter_plot(table_idx, keys, keys_cat_counts, scale=.50):
    all_categories = list(keys_cat_counts.keys())
    keys = np.array(keys)
    if len(keys[0]) == 1:
        first_cat = all_categories[0]
        first_cat_counts = keys_cat_counts[first_cat]
        fig = go.Figure(
            data=go.Scatter(
                x=keys[:, 0],
                y=[0] * len(keys),
                name=first_cat,
                mode="markers",
                marker=dict(size=first_cat_counts*scale),
            )
        )
        for idx, cat in enumerate(all_categories[1:]):
            cat_counts = keys_cat_counts[cat]
            fig.add_trace(
                go.Scatter(
                    x=keys[:, 0],
                    y=[idx+1] * len(keys),
                    name=cat,
                    mode="markers",
                    marker=dict(size=cat_counts*scale),
                )
            )
    else:

        pca = PCA(n_components=2)
        pca.fit(keys)
        key_components = pca.transform(keys)

        first_cat = all_categories[0]
        first_cat_counts = keys_cat_counts[first_cat]

        fig = go.Figure(
            data=go.Scatter(
                x=key_components[:, 0],
                y=key_components[:, 1],
                name=first_cat,
                mode="markers",
                marker=dict(size=first_cat_counts*scale),
            )
        )
        for idx, cat in enumerate(all_categories[1:]):
            cat_counts = keys_cat_counts[cat]
            fig.add_trace(
                go.Scatter(
                    x=key_components[:, 0] + idx + 1.0,
                    y=key_components[:, 1],
                    name=cat,
                    mode="markers",
                    marker=dict(size=cat_counts*scale),
                )
            )

    # fig.show()
    fig.write_image(get_results_dir() / f"table_scatter_{table_idx}.png")
