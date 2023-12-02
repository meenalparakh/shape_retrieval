import numpy as np
import trimesh
from trimesh.base import Trimesh
import typing as T
import plotly.express as px

from utils.visualizer import VizServer
from src.feature_functions import FeatureFunctions
from utils.path_utils import get_results_dir

def compute_shape_features(
    mesh: Trimesh,
    feature_func: FeatureFunctions,
    pcd_size: T.Union[float, int] = 0.5,
    max_num_samples: int = 1000,
    hist_resolution: int = 50,
    normalize: bool = True,
    visualize_pcd: bool = False,
    viz_server: T.Optional[VizServer] = None,
    plot_histogram=False,
    obj_category=None,
) -> T.Tuple[np.array, np.array]:
    
    if isinstance(pcd_size, float):
        num_vertices = len(mesh.vertices)
        num_points = int(num_vertices * pcd_size)
    elif isinstance(pcd_size, int):
        num_points = pcd_size

    pcd, _ = trimesh.sample.sample_surface_even(mesh, num_points, radius=0.005)
    pcd = np.asarray(pcd)
    print(f"PCD shape is {pcd.shape}, expected number of points {num_points}")

    if normalize:
        min_xyz = np.min(pcd, axis=0)
        max_xyz = np.max(pcd, axis=0)
        diagonal_length = np.linalg.norm(max_xyz - min_xyz)
        pcd = pcd * 1.0 / diagonal_length

    if visualize_pcd:
        if viz_server is None:
            viz_server = VizServer()
        viz_server.view_pcd(pts=pcd)

    feature_vector, bins = feature_func.pcd_features(pcd, 
                                                     num_samples=max_num_samples, 
                                                     resolution=hist_resolution)

    if plot_histogram:
        # create the bins
        bins = 0.5 * (bins[:-1] + bins[1:])
        fig = px.bar(
            x=bins,
            y=feature_vector,
            labels={"x": feature_func.name, "y": "PDF"},
            title=f"Density histogram for {obj_category}",
        )
        # fig.show()
        fig.write_image(get_results_dir() / f"histogram_{obj_category}.png")
    return feature_vector, bins