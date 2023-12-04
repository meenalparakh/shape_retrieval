#!/usr/bin/env python
from __future__ import print_function

import os
from operator import itemgetter
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, issparse, vstack
from scipy.spatial.distance import hamming
from sklearn.metrics.pairwise import cosine_distances

from sparselsh.storage import storage, serialize, deserialize


class MyLSH(object):
    def __init__(
        self,
        input_dim,
        hash_size=32,
        num_hashtables=1,
        projection_dist="gaussian",
        bin_function="multiple",
        bin_param_r=3.0,
        r1=1.0,
        c=2.0,
        feature_func=None,
        storage_config=None,
    ):
        # assert hash_size >= 1, "hash_size must be a positive integer"
        assert input_dim >= 1, "input_dim must be a positive integer"
        assert num_hashtables >= 1, "num_hashtables must be a positive integer"

        # note that if bin_function is binary, then hash_size is relevant
        # and if bin_function is multiple, bin_param_r is relevant

        self.hash_size = hash_size
        self.input_dim = input_dim
        self.num_hashtables = num_hashtables
        self.bin_function = bin_function
        self.bin_param_r = bin_param_r
        self.feature_func = feature_func
        self.r2 = c * r1

        self.projection_dist = projection_dist
        print(f"Projection Distribution: {projection_dist}\n"
              f"Number of Tables: {num_hashtables}\n"
              f"Input Dimension: {input_dim}\n"
              f"Hash Key Dimension: {hash_size}\n"
              f"Size of bins: {bin_param_r}\n"
              f"Query threshold: {self.r2}\n"
              f"Features: {feature_func.name}\n")

        self.projection_planes = self.get_hash_function(projection_dist)

        if storage_config is None:
            storage_config = {"dict": None}
        self.storage_config = storage_config

        self._init_hashtables()

    def get_hash_function(self, name):
        if name == "gaussian":
            if self.bin_function == "binary":
                return np.random.rand(
                    self.num_hashtables, self.input_dim, self.hash_size
                )
            else:
                return np.random.rand(
                    self.num_hashtables, self.input_dim, self.hash_size
                ), np.random.uniform(0, self.bin_param_r, (self.num_hashtables, self.hash_size))

        if name == "cauchy":
            if self.bin_function == "binary":
                return np.random.standard_cauchy(
                    (self.num_hashtables, self.input_dim, self.hash_size)
                )
            else:
                return np.random.standard_cauchy(
                    (self.num_hashtables, self.input_dim, self.hash_size)
                ), np.random.uniform(0, self.bin_param_r, (self.num_hashtables, self.hash_size))

    def _init_hashtables(self):
        """Initialize the hash tables such that each record will be in the
        form of "[storage1, storage2, ...]" """

        self.hash_tables = [
            storage(self.storage_config, i) for i in range(self.num_hashtables)
        ]

    def _hash(self, input_points):
        N = input_points.shape[0]
        assert input_points.shape[1] == self.input_dim
        input_points_1ND = input_points[None, ...]

        if self.bin_function == "binary":
            # projection_planes has shape TDH: num_tables x input_dim x hash_size
            projections = input_points_1ND @ self.projection_planes
            assert projections.shape == (self.num_hashtables, N, self.hash_size)
            keys = np.packbits((projections > 0), axis=-1)
            # assert keys.shape == (self.num_hashtables, N)
            return keys
        else:
            # resulting projections has shape TNH
            projections = input_points_1ND @ self.projection_planes[0]
            keys = np.floor_divide(
                projections
                + self.projection_planes[1][:, np.newaxis, :],
                self.bin_param_r,
            )
            if np.any(keys > 255) or np.any(keys < 0):
                print(f"WARNING: keys are outside 0-255 range. "
                      f"the min and max are: {keys.min()} and {keys.max()}")
            return keys.astype(np.uint8)

    def _bytes_string_to_array(self, hash_key):
        """Takes a hash key (bytes string) and turn it
        into a numpy matrix we can do calculations with.

        :param hash_key
        """
        return np.array(list(hash_key))

    def index(self, input_points, extra_data=None):
        """
        Index input points by adding them to the selected storage.
        """

        N = len(input_points)
        assert (
            input_points.shape[1] == self.input_dim
        ), "input_points wrong 2nd dimension"
        assert N == 1 or (
            N > 1 and (extra_data is None or len(extra_data) == N)
        ), "input_points dimension needs to match extra data size"

        keys = self._hash(input_points)
        for table_idx, table in enumerate(self.hash_tables):
            for entry_idx in range(N):
                val = (
                    (input_points[entry_idx],)
                    if extra_data is None
                    else (input_points[entry_idx], extra_data[entry_idx])
                )
                key = tuple(keys[table_idx, entry_idx])
                table.append_val(key, val)

        
    def query(
        self, query_points,
    ):
        
        M = len(query_points)
        assert query_points.shape[1] == self.input_dim
        keys = self._hash(query_points)
        max_checks = 4 * self.num_hashtables
        close_points = [[] for _ in M]
        stop_flag = [False for _ in M]
        
        # collect all close points for all queries, and stop if 4xl points obtained
        for table_idx, table in enumerate(self.hash_tables):
            for query_idx in range(M):
                if stop_flag[query_idx]:
                    continue
                key = tuple(keys[table_idx, query_idx])
                same_bin_points = table.get_list(key)
                close_points[query_idx].extend(same_bin_points)
                if len(close_points[query_idx]) >= max_checks:
                    close_points[query_idx] = close_points[query_idx][:max_checks]
                    stop_flag[query_idx] = True
                    
        # check distance from each of the collected points for each query
        # return the closest if distance is less than r_2
        for query_idx in range(M):
            close_inputs = [val[0] for val in close_points[query_idx]]
            if self.projection_dist == "gaussian":
                distances = np.linalg.norm(np.array(close_inputs) - query_points[query_idx:query_idx+1, :], axis=1)
            elif self.projection_dist == "cauchy":
                distances = np.linalg.norm(np.array(close_inputs) - query_points[query_idx:query_idx+1, :], axis=1, ord=1)
                
            if np.min(distances) < self.r2:
                idx = np.argmin(distances)
                return distances[idx], close_points[query_idx][idx]
            else:
                return None

    ### distance functions
    @staticmethod
    def hamming_dist(x, Y):
        dists = np.zeros(Y.shape[0])
        for ix, y in enumerate(Y):
            dists[ix] = hamming(x, y)
        return dists

    @staticmethod
    def euclidean_dist(x, Y):
        # repeat x as many times as the number of rows in Y
        xx = csr_matrix(np.ones([Y.shape[0], 1]) * x)
        diff = Y - xx
        dists = np.sqrt(diff.dot(diff.T).diagonal()).reshape((1, -1))
        return dists[0]

    @staticmethod
    def euclidean_dist_square(x, Y):
        # repeat x as many times as the number of rows in Y
        xx = csr_matrix(np.ones([Y.shape[0], 1]) * x)
        diff = Y - xx
        if diff.nnz == 0:
            dists = np.zeros((1, Y.shape[0]))
        else:
            if Y.shape[0] > 1:
                dists = diff.dot(diff.T).diagonal().reshape((1, -1))
            else:
                dists = diff.dot(diff.T).toarray()
        return dists[0]

    @staticmethod
    def l1norm_dist(x, Y):
        # repeat x as many times as the number of rows in Y
        xx = csr_matrix(np.ones([Y.shape[0], 1]) * x)
        dists = np.asarray(abs(Y - xx).sum(axis=1).reshape((1, -1)))
        return dists[0]

    @staticmethod
    def cosine_dist(x, Y):
        dists = cosine_distances(x, Y)
        return dists[0]
