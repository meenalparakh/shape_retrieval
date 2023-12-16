import os
from pathlib import Path


def get_shapenet_dir():
    return Path(os.environ["SHAPE_RETRIEVAL"]) / "shapenetcore-gltf"


def get_results_dir():
    return Path(os.environ["SHAPE_RETRIEVAL"]) / "images"


def get_runs_dir():
    return Path(os.environ["SHAPE_RETRIEVAL"]) / "runs"

def get_features_dir():
    return Path(os.environ["SHAPE_RETRIEVAL"]) / "features"


def get_features_dir_d1():
    return Path(os.environ["SHAPE_RETRIEVAL"]) / "features_d1"

def get_features_dir_angle():
    return Path(os.environ["SHAPE_RETRIEVAL"]) / "features_angle"

def get_features_dir_area():
    return Path(os.environ["SHAPE_RETRIEVAL"]) / "features_area"

def get_features_dir_volume():
    return Path(os.environ["SHAPE_RETRIEVAL"]) / "features_vol"


def get_class_pickle(run_dirname):
    return Path(os.environ["SHAPE_RETRIEVAL"]) / "runs" / run_dirname / "lsh.pkl"
