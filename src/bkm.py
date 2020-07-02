#
# reference implementation of breathing k-means
# (C) 2020 Bernd Fritzke
#
# common parameters
# X: data
# C: centroids
# m: breathing depth (number of centroids added/removed during breathing)

import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import math

class BKMeans(KMeans):
    def __init__(self,n_clusters=8, m=5, init="random", n_init=1, **kwargs):
        assert n_clusters >= m, f"breathing depth m ({m}) larger n_clusters ({n_clusters})"
        self.m=m
        super().__init__(n_clusters=n_clusters, init=init, n_init=n_init, **kwargs)

    def fit(self,X):
        """ Compute k-means clustering."""
        super().fit(X) # standard k-means
        m=self.m
        Phi_best=self.inertia_
        C_best=self.cluster_centers_
        while m > 0:
            # add m centroids ("breathe in")
            self.init = self._breathe_in(X=X,C=self.cluster_centers_,m=m)
            self.n_clusters=self.init.shape[0]
            super().fit(X) # standard k-means on enlarged codebook

            # delete m centroids ("breathe out")
            self.init = self._breathe_out(X=X,C=self.cluster_centers_,m=m)
            self.n_clusters=self.init.shape[0]
            super().fit(X) # standard k-means

            # memorize best result so far
            if self.inertia_ < Phi_best*(1-0.0001):
                Phi_best=self.inertia_
                C_best=self.cluster_centers_
            else:
                m-=1 # no improvement: reduce "breathing depth"

        self.inertia_ = Phi_best
        self.cluster_centers_= C_best
        return self

    def get_error_and_utility(self,X,C):
        """compute error and utility per centroid"""
        n=X.shape[0]
        k=C.shape[0]
        dist=cdist(X,C,metric="sqeuclidean")
        dist_srt_idx=np.argsort(dist)
        # distances of each data point to closest (d1) and 2nd-closest (d2) centroid
        d1=dist[np.arange(n),dist_srt_idx[:,0]]
        d2=dist[np.arange(n),dist_srt_idx[:,1]]
        # utility
        util=d2-d1
        # aggregate error and utility for each centroid
        errs = {i:0 for i in range(k)}
        utils= {i:0 for i in range(k)}
        for center, e, u in zip(dist_srt_idx[:,0], d1, util):
            errs[center]+=e
            utils[center]+=u
        return np.array([*errs.values()]), np.array([*utils.values()])

    def _breathe_out(self,X,C,m):
        """ remove m centroids while avoiding large error increase"""
        _,U = self.get_error_and_utility(X,C) # get utility
        useless_sorted_idx=U.argsort()
        k = C.shape[0]
        # distances among centroids
        c_dist=cdist(C,C,metric="euclidean")  
        c_dist_srt_idx = np.argsort(c_dist)
        mean_dist_to_nearest = np.sort(c_dist)[:,1].mean()
        is_close= c_dist < mean_dist_to_nearest*1.1
        D=set()        # index set of centroids to remove
        Frozen=set()   # index set of frozen centroids
        for useless_idx in useless_sorted_idx:
            if useless_idx in Frozen:
                continue
            else:
                D.add(useless_idx)
                n_neighbors = is_close[useless_idx].sum()-1
                for neighbor in c_dist_srt_idx[useless_idx][1:n_neighbors+1]:
                    if len(Frozen) + m < k:
                        # still enough unfrozen to find m centroids for removal
                        Frozen.add(neighbor)
                if len(D)==m:
                    break
        return C[list(set(range(k))-D)]

    def _breathe_in(self,X,C,m):
        """ add m centroids near centroids with large error"""
        E,_ = self.get_error_and_utility(X,C)    # get error
        eps=math.sqrt(np.sum(E)/X.shape[0])*0.01 # offset factor
        max_e_i=(-E).argsort()[:m]         # indices of max error centroids
        D=C[max_e_i]+(np.random.rand(C.shape[1])-0.5)*eps
        return np.concatenate([C,D])

    def fit_predict(self,X):
        """Compute cluster centers and predict cluster index for each sample.
        """
        self.fit(X)
        return self.predict(X)

if __name__ == "__main__":
    from demo import f_demo
    f_demo()