import os
from pathlib import Path


def get_shapenet_dir():
    return Path(os.environ["SHAPE_RETRIEVAL"]) / "shapenetcore-gltf"


def get_results_dir():
    return Path(os.environ["SHAPE_RETRIEVAL"]) / "images"


def get_runs_dir():
    return Path(os.environ["SHAPE_RETRIEVAL"]) / "runs"

def get_class_pickle(run_dirname):
    return Path(os.environ["SHAPE_RETRIEVAL"]) / "runs" / run_dirname / "lsh.pkl"
