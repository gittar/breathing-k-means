# 
# breathing k-means reference implementation 
# (C) 2021 Bernd Fritzke
#
# common parameters:
# X: data set
# C: centroids

import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import math

class BKMeans(KMeans):
    def __init__(self, m=5, theta=1.1, **kwargs):
        """ m: breathing depth
            theta: neighborhood freezing radius
            kwargs: arguments for scikit-learn's KMeans
        """
        self.m = m
        self.theta = theta
        super().__init__(**kwargs)
        assert m <= self.n_clusters, f"m({m}) exceeds n_clusters({self.n_clusters})!"

    @staticmethod
    def get_error_and_utility(X, C):
        """compute error and utility per centroid"""
        n = len(X)
        k = len(C)
        dist = cdist(X, C, metric="sqeuclidean")
        dist_srt_idx = np.argsort(dist)
        # distances to nearest centroid
        d1 = dist[np.arange(n), dist_srt_idx[:, 0]]
        # distances to 2nd-nearest centroid
        d2 = dist[np.arange(n), dist_srt_idx[:, 1]]
        # utility
        util = d2-d1
        # aggregate error and utility for each centroid
        errs = {i: 0 for i in range(k)}
        utils = {i: 0 for i in range(k)}
        for center, e, u in zip(dist_srt_idx[:, 0], d1, util):
            errs[center] += e  # aggregate error for each centroid
            utils[center] += u  # aggregate utility for each centroid
        return np.array([*errs.values()]), np.array([*utils.values()])

    def fit(self, X):
        """ compute k-means clustering via breathing k-means (if m > 0) """
        # run k-means++ (unless differently specified via 'init' parameter)
        super().fit(X)
        # handle trivial case k=1
        if self.n_clusters == 1:
            return self
        m = self.m
        # store best error and codebook so far
        E_best = self.inertia_
        C_best = self.cluster_centers_ + 0
        # no multiple trials from here on
        self.n_init = 1
        tmp = self.init
        while m > 0:
            # add m centroids ("breathe in")
            self.init = self._breathe_in(X, self.cluster_centers_, m)
            self.n_clusters = len(self.init)
            super().fit(X)  # run k-means (on enlarged codebook)

            # delete m centroids ("breathe out")
            self.init = self._breathe_out(X, self.cluster_centers_, m)
            self.n_clusters = len(self.init)
            super().fit(X)  # run k-means

            if self.inertia_ < E_best*(1-self.tol):
                # improvement! memorize best result so far
                E_best = self.inertia_
                C_best = self.cluster_centers_ + 0
            else:
                m -= 1  # no improvement: reduce "breathing depth"
        self.init = tmp # restore for compatibility with sklearn
        self.inertia_ = E_best
        self.cluster_centers_ = C_best
        return self

    def _breathe_in(self, X, C, m):
        """ add m centroids near centroids with large error"""
        E, _ = self.get_error_and_utility(X, C)  # get error
        eps = math.sqrt(np.sum(E)/len(X))*0.01   # offset factor
        # indices of max error centroids
        max_e_i = (-E).argsort()[:m]
        Dplus = C[max_e_i]+(np.random.rand(C.shape[1])-0.5)*eps
        return np.concatenate([C, Dplus])

    def _breathe_out(self, X, C, m):
        """ remove m centroids while avoiding large error increase"""
        _, U = self.get_error_and_utility(X, C)  # get utility
        useless_sorted_idx = U.argsort()
        k = len(C)
        # mutual distances among centroids (kxk-matrix)
        c_dist = cdist(C, C, metric="euclidean")
        c_dist_srt_idx = np.argsort(c_dist)
        mean_dist_to_nearest = np.sort(c_dist)[:, 1].mean()
        # boolean kxk-matrix for closeness info
        is_close = c_dist < mean_dist_to_nearest*self.theta
        Dminus = set()   # index set of centroids to remove
        Frozen = set()   # index set of frozen centroids
        for useless_idx in useless_sorted_idx:
            # ensure that current centroid is not frozen
            if useless_idx not in Frozen:  
                # register current centroid for removal
                Dminus.add(useless_idx)
                # number of close neighbors
                n_neighbors = is_close[useless_idx].sum()-1
                # freeze close neighbors while enough unfrozen remain
                for neighbor in c_dist_srt_idx[useless_idx]\
                    [1:n_neighbors+1]:
                    if len(Frozen) + m < k:
                        # still enough unfrozen to find m to remove
                        Frozen.add(neighbor) # freeze this neighbor
                if len(Dminus) == m:
                    # found m centroids to remove
                    break 
        # return reduced codebook
        return C[list(set(range(k))-Dminus)]

