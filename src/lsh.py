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
        hash_size,
        input_dim,
        num_hashtables=1,
        projection_dist="gaussian",
        bin_function="binary",
        bin_param_r=3.0,
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

        self.projection_dist = projection_dist
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
                    self.num_hashtables, self.input_dim, 1
                ), np.random.uniform(0, self.bin_param_r, self.num_hashtables)

        if name == "cauchy":
            if self.bin_function == "binary":
                return np.random.standard_cauchy(
                    (self.num_hashtables, self.input_dim, self.hash_size)
                )
            else:
                return np.random.standard_cauchy(
                    (self.num_hashtables, self.input_dim, 1)
                ), np.random.uniform(0, self.bin_param_r, self.num_hashtables)

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
            keys = np.packbits((projections > 0), axis=-1)[:, :, 0]
            assert keys.shape == (self.num_hashtables, N)
            return keys
        else:
            # resulting projections has shape TN1
            projections = input_points_1ND @ self.projection_planes[0]
            keys = np.floor_divide(
                projections
                + self.projection_planes[1].reshape(self.num_hashtables, 1, 1),
                self.bin_param_r,
            )[:, :, 0]
            assert keys.shape == (self.num_hashtables, N)
            return keys

    def _bytes_string_to_array(self, hash_key):
        """Takes a hash key (bytes string) and turn it
        into a numpy matrix we can do calculations with.

        :param hash_key
        """
        return np.array(list(hash_key))

    def index(self, input_points, extra_data=None):
        """Index input points by adding them to the selected storage.

        If `extra_data` is provided, it will become the value of the dictionary
        {input_point: extra_data}, which in turn will become the value of the
        hash table.

        :param input_points:
            A sparse CSR matrix. The dimension needs to be N x `input_dim`, N>0.
        :param extra_data:
            (optional) A list of values to associate with the points. Commonly
            this is a target/class-value of some type.
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
                table.append_val(keys[table_idx, entry_idx], val)
                
    def query(
        self, query_points, r1=1.0, r2=3.0,
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
                same_bin_points = table.get_list(keys[table_idx, query_idx])
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
                
            if np.min(distances) < r2:
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
