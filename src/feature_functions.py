import numpy as np
import typing as T


class FeatureFunctions:
    def __init__(self, name):
        self.name = name

    def __call__(self, points_lst: np.array):
        raise NotImplementedError

    def pcd_features(
        self, pcd: np.array, num_samples: int, resolution: int
    ) -> T.Tuple[np.array, np.array]:
        raise NotImplementedError


class D2(FeatureFunctions):
    def __init__(self, name):
        super().__init__(name)
        self.num_points = 2

    def __call__(self, points_lst):
        assert len(points_lst) == self.num_points
        return np.linalg.norm(points_lst[0], points_lst[1])

    def pcd_features(
        self, pcd: np.array, num_samples: int, resolution: int = 100
    ) -> T.Tuple[np.array, np.array]:
        # num_pairs = math.comb(len(points_lst), self.num_points)
        pts_idx1 = np.random.randint(len(pcd), size=num_samples)
        pts_idx2 = np.random.randint(len(pcd), size=num_samples)
        same_pts = pts_idx1 == pts_idx2
        additional_pairs = np.random.choice(
            len(pcd), size=(np.sum(same_pts), 2), replace=False
        )
        pts_idx1[same_pts] = additional_pairs[:, 0]
        pts_idx2[same_pts] = additional_pairs[:, 1]

        pts1 = pcd[pts_idx1]
        pts2 = pcd[pts_idx2]
        d2 = np.linalg.norm(pts1 - pts2, axis=1)

        # range is at most 1.5 because models have been normalized
        # to obtain the diagonal length 1.0
        hist, bin_edges = np.histogram(
            d2, bins=resolution, density=True, range=(0.0, 1.0)
        )
        return hist, bin_edges
