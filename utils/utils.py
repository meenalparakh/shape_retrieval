from typing import Any
import trimesh
import numpy as np
import typing as T
import random

from utils.visualizer import VizServer

class FeatureFunctions:
    def __init__(self, name):
        self.name = name
        
    def __call__(self, points_lst):
        pass
    
class D2(FeatureFunctions):

    def __init__(self, name):
        super().__init__(name)
        self.num_points = 2

    def __call__(self, points_lst):
        assert len(points_lst) == self.num_points
        return np.linalg.norm(points_lst[0], points_lst[1])
    
    def pcd_features(self, points_lst, num_samples):
        pts_idx1 = np.random.randint(0, len(points_lst), num_samples)
        
        pt_lst1 = random.sample(points_lst, k=num_samples)
        
        pt_lst2 = random.sample

def compute_shape_features(
    mesh,
    func,
    func_idx: int,
    pcd_size: T.Union[float, int] = 0.5,
    normalize: bool = True,
    visualize_pcd: bool = False,
    viz_server: T.Optional[VizServer] = None,
):
    if isinstance(pcd_size, float):
        num_vertices = len(mesh.vertices)
        num_points = int(num_vertices * pcd_size)
    elif isinstance(pcd_size, int):
        num_points = pcd_size

    pcd = trimesh.sample.sample_surface_even(mesh, num_points)
    assert pcd.shape == (num_points, 3)

    if normalize:
        min_xyz = np.min(pcd, axis=0)
        max_xyz = np.max(pcd, axis=0)
        diagonal_length = np.linalg.norm(max_xyz - min_xyz)
        pcd = pcd * 1.0 / diagonal_length

    if visualize_pcd:
        assert viz_server is not None, "Please provide a Viz server"
        viz_server.view_pcd(pts=pcd)
        
    
    