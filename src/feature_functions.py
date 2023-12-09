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


class Dist(FeatureFunctions):
    def __init__(self, name, ord=2):
        super().__init__(name)
        self.num_points = 2
        self.ord = ord

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
        indices = np.nonzero(same_pts.astype(np.int8))
        # print(indices)
        for idx in indices:
            sample = np.random.choice(len(pcd), size=2, replace=False)
            pts_idx1[idx] = sample[0]
            pts_idx2[idx] = sample[1]

        assert np.sum(pts_idx1 == pts_idx2) == 0
        pts1 = pcd[pts_idx1]
        pts2 = pcd[pts_idx2]
        d2 = np.linalg.norm(pts1 - pts2, axis=1, ord=self.ord)

        # range is at most 1.5 because models have been normalized
        # to obtain the diagonal length 1.0
        if self.ord == 2:
            r = (0.0, 1.0)
        elif self.ord == 1:
            r = (0.0, 3.0)
        hist, bin_edges = np.histogram(
            d2, bins=resolution, density=True, range=r
        )
        return hist, bin_edges


class Angle(FeatureFunctions):
    def __init__(self, name):
        super().__init__(name)
        self.num_points = 3

    def pcd_features(
        self, pcd: np.array, num_samples: int, resolution: int = 100
    ) -> T.Tuple[np.array, np.array]:
        # num_pairs = math.comb(len(points_lst), self.num_points)
        pts_idx1 = np.random.randint(len(pcd), size=num_samples)
        pts_idx2 = np.random.randint(len(pcd), size=num_samples)
        pts_idx3 = np.random.randint(len(pcd), size=num_samples)

        same_pts_2 = (pts_idx2 == pts_idx1)
        same_pts_3 = (pts_idx3 == pts_idx1)
        same_pts_1 = (pts_idx2 == pts_idx3)
        
        same_pts = same_pts_1 | same_pts_2 | same_pts_3
        indices = np.nonzero(same_pts.astype(np.int8))
        # print(indices)
        for idx in indices:
            sample = np.random.choice(len(pcd), size=3, replace=False)
            pts_idx1[idx] = sample[0]
            pts_idx2[idx] = sample[1]
            pts_idx3[idx] = sample[2]
        
        assert np.sum(((pts_idx1 == pts_idx2) | (pts_idx3 == pts_idx2) | (pts_idx1 == pts_idx3))) == 0
        
        pts1 = pcd[pts_idx1]
        pts2 = pcd[pts_idx2]
        pts3 = pcd[pts_idx3]
        
        direction1 = pts2 - pts1
        direction2 = pts3 - pts1
        
        direction1 = direction1 / np.linalg.norm(direction1, axis=1, keepdims=True)
        direction2 = direction2 / np.linalg.norm(direction2, axis=1, keepdims=True)
        
        cos_thetas = np.sum(direction1 * direction2, axis=1)
        
        # range is at most 1.5 because models have been normalized
        # to obtain the diagonal length 1.0
        hist, bin_edges = np.histogram(
            cos_thetas, bins=resolution, density=True, range=(-1.0, 1.0)
        )
        return hist, bin_edges
    
    
def normal(triangles):
    # The cross product of two sides is a normal vector
    return np.cross(triangles[:,1] - triangles[:,0], 
                    triangles[:,2] - triangles[:,0], axis=1)

def area(triangles):
    # The norm of the cross product of two sides is twice the area
    return np.linalg.norm(normal(triangles), axis=1) / 2


class Area(FeatureFunctions):
    def __init__(self, name):
        super().__init__(name)
        self.num_points = 3

    def pcd_features(
        self, pcd: np.array, num_samples: int, resolution: int = 100
    ) -> T.Tuple[np.array, np.array]:
        # num_pairs = math.comb(len(points_lst), self.num_points)
        pts_idx1 = np.random.randint(len(pcd), size=num_samples)
        pts_idx2 = np.random.randint(len(pcd), size=num_samples)
        pts_idx3 = np.random.randint(len(pcd), size=num_samples)

        same_pts_2 = (pts_idx2 == pts_idx1)
        same_pts_3 = (pts_idx3 == pts_idx1)
        same_pts_1 = (pts_idx2 == pts_idx3)
        
        same_pts = same_pts_1 | same_pts_2 | same_pts_3
        indices = np.nonzero(same_pts.astype(np.int8))
        # print(indices)
        for idx in indices:
            sample = np.random.choice(len(pcd), size=3, replace=False)
            pts_idx1[idx] = sample[0]
            pts_idx2[idx] = sample[1]
            pts_idx3[idx] = sample[2]
        
        assert np.sum(((pts_idx1 == pts_idx2) | (pts_idx3 == pts_idx2) | (pts_idx1 == pts_idx3))) == 0
        
        pts1 = pcd[pts_idx1]
        pts2 = pcd[pts_idx2]
        pts3 = pcd[pts_idx3]
        
        direction1 = pts2 - pts1
        direction2 = pts3 - pts1
        area = 0.5 * np.linalg.norm(np.cross(direction1, direction2, axis=1),
                                    axis=1)
        area = np.sqrt(area)
                
        # range is at most 1.5 because models have been normalized
        # to obtain the diagonal length 1.0
        hist, bin_edges = np.histogram(
            area, bins=resolution, density=True, range=(0.0, 0.5)
        )
        return hist, bin_edges
    
def check_same(pts):
    n = len(pts)
    size = len(pts[0])
    same_results = np.zeros(size, dtype=bool)
    
    for i in range(n-1):
        for j in range(i + 1, n):
            same_results = same_results | (pts[i] == pts[j])
    return same_results            
    

class Volume(FeatureFunctions):
    def __init__(self, name):
        super().__init__(name)
        self.num_points = 4

    def pcd_features(
        self, pcd: np.array, num_samples: int, resolution: int = 100
    ) -> T.Tuple[np.array, np.array]:
        # num_pairs = math.comb(len(points_lst), self.num_points)
        pts = np.random.randint(len(pcd), size=(self.num_points, num_samples))
        same_pts = check_same(pts)
        indices = np.nonzero(same_pts.astype(np.int8))
        # print(indices)
        for idx in indices:
            sample = np.random.choice(len(pcd), size=self.num_points, replace=False)
            for j in range(self.num_points):
                pts[j, idx] = sample[j]
        
        assert np.sum(check_same(pts)) == 0
        
        p3 = pcd[pts[3]]
        a = pcd[pts[0]] - p3
        b = pcd[pts[1]] - p3
        c = pcd[pts[2]] - p3
        tmp_val = np.sum(a * np.cross(b, c, axis=1), axis=1) 
        volume = np.abs(tmp_val) / 6.0
        volume = volume ** (1/3.0)
        # range is at most 1.5 because models have been normalized
        # to obtain the diagonal length 1.0
        hist, bin_edges = np.histogram(
            volume, bins=resolution, density=True, range=(0.0, .50)
        )
        return hist, bin_edges