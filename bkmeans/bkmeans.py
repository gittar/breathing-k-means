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
__version__="V1.1"
class BKMeans(KMeans):
    def get_version():
        return __version__
    def __init__(self, m=5, n_init=1, **kwargs):
        """ m: breathing depth
            n_init: number of times k-means++ is run initially
            kwargs: arguments for scikit-learns KMeans """
        super().__init__(n_init=n_init, **kwargs)
        self.m = min(m,self.n_clusters) # ensure m <= k

    def get_error(self, X, C):
        """compute error per centroid"""
        # squared distances between data and centroids
        dist = cdist(X, C, metric="sqeuclidean")
        # indices to nearest centroid
        dist_min = np.argmin(dist,axis=1)
        # distances to nearest centroid
        d1 = dist[np.arange(len(X)), dist_min]
        # aggregate error for each centroid
        return np.array([np.sum(d1[dist_min==i]) for i in range(len(C))])

    def get_utility(self, X, C):
        """compute utility per centroid"""
        # squared distances between data and centroids
        dist = cdist(X, C, metric="sqeuclidean")
        # indices to nearest and 2nd-nearest centroid
        dist_srt = dist.argpartition(kth=1)[:,:2]
        # squared distances to nearest centroid
        d1 = dist[np.arange(len(X)), dist_srt[:, 0]]
        # squared distances to 2nd-nearest centroid
        d2 = dist[np.arange(len(X)), dist_srt[:, 1]]
        # utility
        util = d2-d1
        # aggregate utility for each centroid
        return np.array([np.sum(util[dist_srt[:, 0]==i]) for i in range(len(C))])

    def _lloyd(self,C,X):
        """perform Lloyd's algorithm"""
        self.init = C # set cluster centers
        self.n_clusters = len(C) # set k-value
        super().fit(X) # Lloyd's algorithm, sets self.inertia_ (a.k.a. phi)

    def fit(self, X):
        """ compute k-means clustering via breathing k-means (if m > 0) """
        # run k-means++ (unless 'init' parameter specifies differently)
        super().fit(X) # requires self.n_clusters >= 1
        # handle trivial case k=1
        if self.n_clusters == 1:
            return self
        # assertion: self.n_clusters >= 2
        m = self.m
        # memorize best error and codebook so far
        E_best = self.inertia_
        C_best = self.cluster_centers_
        tmp = self.n_init, self.init # store for compatibility with sklearn
        # no multiple trials from here on
        self.n_init = 1
        while m > 0:
            # add m centroids ("breathe in") and run Lloyd's algorithm
            self._lloyd(self._breathe_in(X, self.cluster_centers_, m),X)
            # delete m centroids ("breathe out") and run Lloyd's algorithm
            self._lloyd(self._breathe_out(X, self.cluster_centers_, m),X)
            if self.inertia_ < E_best*(1-self.tol):
                # improvement! update memorized best error and codebook so far
                E_best = self.inertia_
                C_best = self.cluster_centers_
            else:
                m -= 1  # no improvement: reduce "breathing depth"
        self.n_init, self.init = tmp # restore for compatibility with sklearn
        self.inertia_ = E_best
        self.cluster_centers_ = C_best
        return self

    def _breathe_in(self, X, C, m):
        """ add m centroids near centroids with large error"""
        E = self.get_error(X, C)  # per centroid
        # indices of max error centroids
        max_err = (-E).argpartition(kth=m-1)[:m]
        # multiplicative small constant for offset vectors
        eps = 0.01
        # root-mean-square error
        RMSE=math.sqrt(np.sum(E)/len(X))
        # m new centroids created near max error centroids
        Dplus = C[max_err]+eps*RMSE*(np.random.rand(m,C.shape[1])-0.5)
        # return enlarged codebook
        return np.concatenate([C, Dplus])

    def _breathe_out(self, X, C, m):
        """ remove m centroids while avoiding large error increase"""
        U = self.get_utility(X, C)  # per centroid
        useless_sorted = U.argsort()
        # mutual distances among centroids (kxk-matrix)
        c_dist = cdist(C, C, metric="sqeuclidean")
        # index of nearest neighbor for each centroid
        nearest_neighbor=c_dist.argpartition(kth=1)[:,1]
        Dminus = set()   # index set of centroids to remove
        Frozen = set()   # index set of frozen centroids
        for useless in useless_sorted:
            # ensure that current centroid is not frozen
            if useless not in Frozen:  
                # register current centroid for removal
                Dminus.add(useless)
                nn=nearest_neighbor[useless]
                if len(Frozen) + m < len(C):
                    # freeze nearest neighbor centroid
                    Frozen.add(nn)
                if len(Dminus) == m:
                    # found m centroids to remove
                    break 
        # return reduced codebook
        return C[list(set(range(len(C)))-Dminus)]
