#
# this program generates data from a Gaussian mixture distribution and runs
# * k-means++ (a.k.a. k-means with k-means++ initialization)
# * breathing k-means with random initialization (the default)
# * breathing k-means with k-means++ initialization
#

from time import time
import numpy as np
from sklearn.cluster import KMeans
from bkm import BKMeans
def f_demo():
    # data from d-dim. Gaussian mixture distribution
    g=50 # clusters in the data
    k=g*2  # k-value
    n=10000 # number of data points
    sig=0.01 # standard deviation of Gaussians
    d=2 # dimensionality of the data
    tol=sig*sig # tolerance
    X = np.concatenate([np.random.random(d)+np.random.normal(size=(n//g,d),scale=0.01) for _ in range(g)])

    
    print(f"\nk={k}, data set: {X.shape[0]} points drawn from mixture of {g} Gaussians in R^{d} with cov={sig}^2*I{d}\n")
    print("algorithm         | init      | n_init | SSE                   | t")
    print("__________________|___________|________|_______________________|_______________________")
    # k-means++ (scikit-learn)
    t0=time()
    km=KMeans(n_clusters=k,tol=tol)
    km.fit(X)
    t_km=time()-t0
    print(f"k-means           | k-means++ | 10     | {km.inertia_:.4f}                |{t_km:5.2f}s ")
    
    # breathing k-means
    t0=time()
    bkm=BKMeans(n_clusters=k,tol=tol)
    bkm.fit(X)
    t_bkm=time()-t0
    imp = 1-bkm.inertia_/km.inertia_
    overhead = t_bkm/t_km -1
    print(f"breathing k-means | random    |  1     | {bkm.inertia_:.4f} ({imp:.2%} better) |{t_bkm:5.2f}s ({overhead:+06.2%} overhead)")

    # breathing k-means with k-means++ initialization
    t0=time()
    bkm=BKMeans(n_clusters=k, init="k-means++",tol=tol)
    bkm.fit(X)
    t_bkm=time()-t0
    imp = 1-bkm.inertia_/km.inertia_
    overhead = t_bkm/t_km -1
    print(f"breathing k-means | k-means++ |  1     | {bkm.inertia_:.4f} ({imp:.2%} better) |{t_bkm:5.2f}s ({overhead:+06.2%} overhead)")
 

