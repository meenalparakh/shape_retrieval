import os
import sys
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
from utils.image_utils import compare
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
    
    categories = ["chair", "can", "airplane", "mug", "bowl", "bottle"]
    feature_fun1 = Dist("D1", ord=1)
    feature_fun2 = Dist("D2", ord=2)
    feature_fun3 = Angle("Angle")
    feature_fun4 = Area("Area")
    feature_fun5 = Volume("Volume")
    visualize_feature_vectors(feature_fun5, categories)
    # visualize_feature_vectors(feature_fun2, categories)
    # visualize_feature_vectors(feature_fun3, caegories)