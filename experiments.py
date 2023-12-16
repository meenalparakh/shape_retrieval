import os
import sys

from compute_hash_tables import divide_objects, query_objects, save_mappings, visualize_feature_vectors_diff_funcs

sys.path.append(os.environ["SHAPE_RETRIEVAL"])
import pickle
import typing as T
import trimesh
import random
import numpy as np

random.seed(0)
np.random.seed(0)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)

from utils.path_utils import get_runs_dir, get_shapenet_dir
from utils.image_utils import compare, perf_time_inp_dim, perf_types, performance_time_bargraph, performance_time_graph
from utils.image_utils import time_distance_graph, visualize_tables, perf_time_variable
from src.feature_functions import Dist, Angle, Area, Volume, FeatureFunctions
from src.extract_shape_features import compute_shape_features
from compute_hash_tables import visualize_feature_vectors

# def visualize_feature_vectors(
#     feature_func: FeatureFunctions,
#     categories: T.List[str],
#     num_images: int = 2,
# ):
#     for cat in categories:
#         category_dir = get_shapenet_dir() / cat
#         print(f"Category: {cat}")
#         fnames = sorted(list(category_dir.glob("*.gltf")))[:num_images]
#         for idx, fn in enumerate(fnames):
#             print(f"\t{str(fn).split(os.sep)[-1]}")
#             mesh = trimesh.load(fn, file_type="gltf", force="mesh")
#             features, bins = compute_shape_features(
#                 mesh=mesh,
#                 feature_func=feature_func,
#                 pcd_size=0.5,
#                 max_num_samples=5000,
#                 hist_resolution=100,
#                 normalize=True,
#                 visualize_pcd=False,
#                 plot_histogram=True,
#                 obj_category=f"{cat}_{idx}",
#             )
#     compare(categories, num_images=num_images, name=f"{feature_func.name}")


if __name__ == "__main__":
    max_hashed = 400
    max_queries = 20
    skip_hashing = True
    categories = ["airplane", "bowl", "can", "chair"]
    input_dims_lst = [100, 500, 1000]
    # input_dims_lst = [100]
    bin_param_lst = [10.0, 25.0, 50.0, 75.0, 100.0]
    num_samples_lst = [int(1e3), int(1e4), int(1e5)]
    projections_lst = ["cauchy", "gaussian"]
    bin_function_lst = ['binary', 'multiple']

    feature_funcs_lst = [
        Dist("D1", ord=1),
        Dist("D2", ord=2),
        Angle("Angle"),
        Area("Area"),
        Volume("Volume"),
    ]
    feature_func_names_lst = [f.name for f in feature_funcs_lst]

    # dividing objects into train and test - to be used uniformly
    # divide_objects(categories, ratio=0.05, force=False)

    # experiments across varying input_dims, keeping others fixed
    num_samples = num_samples_lst[0]
    projection = projections_lst[1]
    feature_func = feature_funcs_lst[1]
    bin_function = bin_function_lst[1]
    inp_dim = input_dims_lst[0]

    # performance = {cat: [] for cat in categories}
    # distance = {cat: [] for cat in categories}
    # query_time = []
    # # for inp_dim in input_dims_lst:
    # for bin_param in bin_param_lst:
        
    #     random.seed(0)
    #     np.random.seed(0)
    #     logging.info(f"STARTING EXPERIMENT FOR BIN PARAM {bin_param}...............")
    #     run_dirname = f"bin_param_{int(bin_param)}"
    #     if (not (get_runs_dir() / 'runs' / run_dirname).exists()):
    #         lsh, lsh_pickle = save_mappings(
    #             feature_func=feature_func,
    #             run_dirname=run_dirname,
    #             input_dim=inp_dim,
    #             projection_dist=projection,
    #             bin_param_r=bin_param,
    #             r1=5.0,
    #             c=4.0,
    #             batch_size=500,
    #             pcd_size=0.5,
    #             pdf_resolution=num_samples,
    #             bin_function=bin_function,
    #             max_objects_hashed=max_hashed,
    #         )
    #     else:
    #         print("found..................")

    #     result_dict = query_objects(
    #         run_dirname, pdf_resolution=num_samples, max_queries=max_queries
    #     )
    #     result_dict_path = get_runs_dir() / run_dirname / 'query_results.pkl'
    #     with open(result_dict_path, 'rb') as f:
    #         result_dict = pickle.load(f)

    #     for cat in categories:
    #         performance[cat].append(result_dict['cat_success_rate'][cat])
    #         distance[cat].append(result_dict["distance"][cat])
 
    #     query_time.append(result_dict["average_query_time"])

    #     # visualize_tables(run_dirname, max_tables=10, name=bin_param)

    # perf_time_variable(bin_param_lst, performance, query_time, "bin size", "success rate", [0.0, 1.2])
    # perf_time_variable(bin_param_lst, distance, query_time, "bin size", "distance")
    # # experiments for performance vs query time for brute force search and hashing
    # # exit()
    
    # logging.info(f"STARTING EXPERIMENT FOR LSH vs BRUTE-FORCE ...............")


    # bin_param = bin_param_lst[-2]
    # run_dirname = f"bin_param_{int(bin_param)}"
    # methods = ["LSH", "brute-force"]
    # performance = {}
    # distance = {}

    # result_dict_path = get_runs_dir() / "bin_param_75" / 'query_results.pkl'
    # with open(result_dict_path, 'rb') as f:
    #     result_dict = pickle.load(f)
    
    # for cat in categories:
    #     performance[cat] = [result_dict['cat_success_rate'][cat]]
    #     distance[cat] = [result_dict['distance'][cat]]
    # query_time = [result_dict["average_query_time"]]

    # result_dict = query_objects(
    #     run_dirname,
    #     pdf_resolution=num_samples,
    #     brute_force=True,
    #     max_queries=max_queries,
    # )
    # result_dict_path = get_runs_dir() / "bin_param_75" / 'query_results_brute.pkl'
    # with open(result_dict_path, 'rb') as f:
    #     result_dict = pickle.load(f)
        
    # for cat in categories:
    #     performance[cat].append(result_dict['cat_success_rate'][cat])
    #     distance[cat].append(result_dict['distance'][cat])
        
    # query_time.append(result_dict["average_query_time"])

    # performance_time_bargraph(methods, performance, query_time)
    # # performance_time_graph(methods, performance, query_time)
    # time_distance_graph(methods, distance, query_time)
    # # experiments for distance vs query time for brute force search and hashing
    # #####################################################################################
    
    # # experiments for projection
    # performance = {cat: [] for cat in categories}
    # distance = {cat: [] for cat in categories}
    # query_time = []

    # print("bin function", bin_function)
    # bin_param = bin_param_lst[-2]
    # for proj in projections_lst[:1]:
    #     random.seed(0)
    #     np.random.seed(0)
    #     logging.info(f"STARTING EXPERIMENT FOR INPUT DIM {inp_dim}...............")
    #     run_dirname = f"projection_{proj}"
    #     lsh, lsh_pickle = save_mappings(
    #         feature_func=feature_func,
    #         run_dirname=run_dirname,
    #         input_dim=inp_dim,
    #         projection_dist=proj,
    #         bin_param_r=bin_param * 5.0,
    #         r1=5.0 * 5,
    #         c=4.0,
    #         batch_size=500,
    #         pcd_size=0.5,
    #         pdf_resolution=num_samples,
    #         bin_function=bin_function,
    #         max_objects_hashed=max_hashed,
    #     )
    #     result_dict = query_objects(
    #             run_dirname, pdf_resolution=num_samples, max_queries=max_queries
    #         )
    
    # for run_dir in ['projection_cauchy', 'bin_param_75']:
        
    #     result_dict_path = get_runs_dir() / run_dir / 'query_results.pkl'
    #     with open(result_dict_path, 'rb') as f:
    #         result_dict = pickle.load(f)

    #     for cat in categories:
    #         performance[cat].append(result_dict['cat_success_rate'][cat])
    #         distance[cat].append(result_dict["distance"][cat])
 
    #     query_time.append(result_dict["average_query_time"])

    #     # visualize_tables(run_dirname, max_tables=10, name=bin_param)

    # perf_types(projections_lst, performance, "Projection Type", "success rate")

    ###################################################################################
    
    # experiments for projection
    # performance = {cat: [] for cat in categories}
    # distance = {cat: [] for cat in categories}
    # query_time = []

    # print("bin function", bin_function)
    # projection = projections_lst[0]
    # for bin_param in bin_param_lst:
    #     random.seed(0)
    #     np.random.seed(0)
    #     logging.info(f"STARTING EXPERIMENT FOR INPUT DIM {inp_dim}...............")
    #     run_dirname = f"projection_{projection}_bin_param_{int(bin_param)}"
    #     lsh, lsh_pickle = save_mappings(
    #         feature_func=feature_func,
    #         run_dirname=run_dirname,
    #         input_dim=inp_dim,
    #         projection_dist=projection,
    #         bin_param_r=bin_param * 5.0,
    #         r1=5.0 * 5,
    #         c=4.0,
    #         batch_size=500,
    #         pcd_size=0.5,
    #         pdf_resolution=num_samples,
    #         bin_function=bin_function,
    #         max_objects_hashed=max_hashed,
    #     )
    #     result_dict = query_objects(
    #             run_dirname, pdf_resolution=num_samples, max_queries=max_queries
    #         )
    
    # for bin_param in bin_param_lst:
    #     run_dir = f"projection_cauchy_bin_param_{int(bin_param)}"
        
    #     result_dict_path = get_runs_dir() / run_dir / 'query_results.pkl'
    #     with open(result_dict_path, 'rb') as f:
    #         result_dict = pickle.load(f)

    #     for cat in categories:
    #         performance[cat].append(result_dict['cat_success_rate'][cat])
    #         distance[cat].append(result_dict["distance"][cat])
 
    #     query_time.append(result_dict["average_query_time"])

    #     # visualize_tables(run_dirname, max_tables=10, name=bin_param)

    # # perf_types(projections_lst, performance, "Projection Type", "success rate")
    # perf_time_variable(np.array(bin_param_lst) * 5.0, performance, query_time, "bin size - C", "success rate", [0.0, 1.2])
    # perf_time_variable(np.array(bin_param_lst) * 5.0, distance, query_time, "bin size - C", "distance")



    # logging.info(f"STARTING EXPERIMENT FOR LSH vs BRUTE-FORCE ...............")


    # bin_param = bin_param_lst[-2]
    # name = int(bin_param)
    # run_dirname = f"projection_cauchy_bin_param_{name}"
    # methods = ["LSH", "brute-force"]
    # performance = {}
    # distance = {}

    # result_dict_path = get_runs_dir() / run_dirname / 'query_results.pkl'
    # with open(result_dict_path, 'rb') as f:
    #     result_dict = pickle.load(f)
    
    # for cat in categories:
    #     performance[cat] = [result_dict['cat_success_rate'][cat]]
    #     distance[cat] = [result_dict['distance'][cat]]
    # query_time = [result_dict["average_query_time"]]

    # result_dict = query_objects(
    #     run_dirname,
    #     pdf_resolution=num_samples,
    #     brute_force=True,
    #     max_queries=max_queries,
    # )
    # result_dict_path = get_runs_dir() / run_dirname / 'query_results_brute.pkl'
    # with open(result_dict_path, 'rb') as f:
    #     result_dict = pickle.load(f)
        
    # for cat in categories:
    #     performance[cat].append(result_dict['cat_success_rate'][cat])
    #     distance[cat].append(result_dict['distance'][cat])
        
    # query_time.append(result_dict["average_query_time"])

    # performance_time_bargraph(methods, performance, query_time)
    # # performance_time_graph(methods, performance, query_time)
    # time_distance_graph(methods, distance, query_time)
    
    
    # method_names = ['LSH (L2)', 'LSH (L1)', 'Brute (L2)', 'Brute (L1)']
    # result_dict_paths = [
    #     'bin_param_75',
    #     'projection_cauchy_bin_param_75',
    # ]
    
    # performance = {cat: [] for cat in categories}
    # query_time = []
    # for fname in ['query_results.pkl', 'query_results_brute.pkl']:
    #     for run_dirname in result_dict_paths:
    #         result_dict_path = get_runs_dir() / run_dirname / fname
    #         with open(result_dict_path, 'rb') as f:
    #             result_dict = pickle.load(f)
    #         for cat in categories:
    #             performance[cat].append(result_dict['cat_success_rate'][cat])
    #         query_time.append(result_dict['average_query_time'])        
    # performance_time_bargraph(method_names, performance, query_time)


    # experiments for projection
    bin_param = bin_param_lst[-2]
    projection = projections_lst[0]

    # for feature_name, feature_func in zip(feature_func_names_lst, feature_funcs_lst):
    #     if feature_name == "D2":
    #         continue
    #     random.seed(0)
    #     np.random.seed(0)
    #     logging.info(f"STARTING EXPERIMENT FOR INPUT DIM {feature_name}...............")
    #     run_dirname = f"feature_func_{feature_name}"
    #     lsh, lsh_pickle = save_mappings(
    #         feature_func=feature_func,
    #         run_dirname=run_dirname,
    #         input_dim=inp_dim,
    #         projection_dist=projection,
    #         bin_param_r=bin_param * 5.0,
    #         r1=5.0 * 5,
    #         c=4.0,
    #         batch_size=500,
    #         pcd_size=0.5,
    #         pdf_resolution=num_samples,
    #         bin_function=bin_function,
    #         max_objects_hashed=max_hashed,
    #     )
    #     result_dict = query_objects(
    #             run_dirname, pdf_resolution=num_samples, max_queries=max_queries, feature_dir=feature_name
    #         )
    

    # performance = {cat: [] for cat in categories}
    # query_time = []
    
  
    # for feature_name, feature_func in zip(feature_func_names_lst, feature_funcs_lst):
    #     if feature_name == "D2":
    #         run_dir = 'projection_cauchy_bin_param_75'
    #     else:
    #         run_dir = f"feature_func_{feature_name}"
            
    #     result_dict_path = get_runs_dir() / run_dir / 'query_results.pkl'
    #     with open(result_dict_path, 'rb') as f:
    #         result_dict = pickle.load(f)

    #     for cat in categories:
    #         performance[cat].append(result_dict['cat_success_rate'][cat]) 
    #     query_time.append(result_dict["average_query_time"])

    #     # visualize_tables(run_dirname, max_tables=10, name=bin_param)
    # perf_types(feature_func_names_lst, performance, query_time, "Feature Function", "Success Rate")

    # # perf_types(projections_lst, performance, "Projection Type", "success rate")

    # visualize_feature_vectors(feature_funcs_lst[1], categories, 2)
    # visualize_feature_vectors_diff_funcs(feature_funcs_lst, categories[0])
    dirname = "projection_cauchy_bin_param_75"
    visualize_tables(dirname, plot_type="bar", max_tables=5, name='final_table')
