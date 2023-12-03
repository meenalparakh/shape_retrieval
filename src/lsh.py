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
        hash_function="gaussian",
        bin_function="binary",
        bin_param_r=3.0,
        storage_config=None,
        matrices_filename=None,
        overwrite=False,
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

        self.projection_planes = self.get_hash_function(hash_function)

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
        self, query_points, distance_func=None, dist_threshold=None, num_results=None
    ):
        assert issparse(query_points), "query_points needs to be sparse"
        assert (
            query_points.shape[1] == self.input_dim
        ), "query_points wrong 2nd dimension"
        assert (
            num_results is None or num_results >= 1
        ), "num_results needs to be a positive integer"

        if distance_func is None or distance_func == "euclidean":
            d_func = LSH.euclidean_dist_square
        elif distance_func == "true_euclidean":
            d_func = LSH.euclidean_dist
        elif distance_func == "cosine":
            d_func = LSH.cosine_dist
        elif distance_func == "l1norm":
            d_func = LSH.l1norm_dist
        elif distance_func == "hamming":
            d_func = LSH.hamming_dist
        else:
            raise ValueError("The distance function %s is invalid." % distance_func)
        if dist_threshold and (
            dist_threshold <= 0 or (distance_func == "cosine" and dist_threshold > 1.0)
        ):
            raise ValueError("The distance threshold %s is invalid." % dist_threshold)

        # Create a list of lists of candidate neighbors
        # NOTE: Currently this only does exact matching on hash key, the
        # previous version also got the 2 most simlilar hashes and included
        # the contents of those as well. Not sure if this should actually
        # be done since we can change hash size or add more tables to get
        # better accuracy
        candidates = []
        for i, table in enumerate(self.hash_tables):
            # get hashes of query points for the specific plane
            keys = self._hash(self.uniform_planes[i], query_points)
            for j in range(keys.shape[0]):
                # TODO: go through each hashkey in table and do the following
                # if the key is more similar to hashkey than hamming dist < 2

                # Create a sublist of candidate neighbors for each query point
                if len(candidates) <= j:
                    candidates.append([])
                new_candidates = table.get_list(keys[j].tobytes())
                if new_candidates is not None and len(new_candidates) > 0:
                    candidates[j].extend(new_candidates)

        # Create a ranked list of lists of candidate neighbors
        ranked_candidates = []

        # for each query point ...
        # Create a sublist of ranked candidate neighbors for each query point
        for j in range(query_points.shape[0]):
            point_results = []
            # hash candidates from above for jth query point
            row_candidates = candidates[j]

            # Remove extra info from candidates to convert them into a matrix
            cands = []
            extra_datas = []
            for row in row_candidates:
                cands.append(row[0])
                if len(row) == 2:
                    extra_datas.append(row[1])

            if not cands:
                ranked_candidates.append(point_results)
                continue

            cand_csr = vstack(cands)
            distances = d_func(query_points[j], cand_csr)
            if dist_threshold is not None:
                accepted = np.unique(np.where(distances < dist_threshold)[0])
                cand_csr = cand_csr[accepted, :]
                distances = distances[accepted]

            # Rank candidates by distance function, this has to
            # support having empty cand_csr and distances
            indices = np.argsort(np.array(distances))
            neighbors_sorted = cand_csr[indices]
            dists_sorted = distances[indices]

            # if we have extra data
            if extra_datas:
                # Sort extra_data by ranked distances
                try:
                    extra_data_sorted = itemgetter(*list(indices))(extra_datas)
                # we have no results, so no extra_datas
                except TypeError:
                    extra_data_sorted = []

                neigh_extra_tuples = zip(neighbors_sorted, extra_data_sorted)
                for ix, (neigh, ext) in enumerate(neigh_extra_tuples):
                    if num_results is not None and ix >= num_results:
                        break
                    dist = dists_sorted[ix]
                    point_results.append(((neigh, ext), dist))

            else:
                for ix, neigh in enumerate(neighbors_sorted):
                    if num_results is not None and ix >= num_results:
                        break
                    dist = dists_sorted[ix]
                    point_results.append(((neigh,), dist))

            ranked_candidates.append(point_results)

        if query_points.shape[0] == 1:
            # Backwards compat fix
            return ranked_candidates[0]

        return ranked_candidates

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
