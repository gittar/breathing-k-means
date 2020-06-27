import numpy as np
import matplotlib.pyplot as plt
from time import time
from mydataset import MyDataSet
from sklearn.cluster import KMeans # KMeans class from scikit-lean
from aux import plot
from bkm import BKMeans 

def run_models(X,k,opt=None,dotext=True):
    """for dataset X and given k-value run k-means++ and breathing k-means"""
    # k-means++ (scikit-learn)
    if dotext: 
        print("k-means++ ...           ", end="")
    t0=time()
    km=KMeans(n_clusters=k)
    km.fit(X)
    t_km=time()-t0
    if dotext:
        if opt is None: 
            print(f" SSE={km.inertia_:.3f}                 t={t_km:6.2f}s ")
        else:
            print(f" SSE=Opt+{km.inertia_/opt-1:6.2%}, t={t_km:6.2f}s ")

    # breathing k-means
    if dotext: 
        print("breathing k-means ...   ", end="")
    t0=time()
    bkm=BKMeans(n_clusters=k)
    bkm.fit(X)
    t_bkm=time()-t0
    # SSE improvement
    imp = 1-bkm.inertia_/km.inertia_
    if dotext: 
        overhead = t_bkm/t_km -1
        if opt is None: 
            print(f" SSE={bkm.inertia_:.3f} ({imp:6.2%} lower)  t={t_bkm:6.2f}s", end="")
        else:
            print(f" SSE=Opt+{bkm.inertia_/opt-1:6.2%}, t={t_bkm:6.2f}s ", end="")
        print(f" ({overhead:6.2%} overhead)")
        
    return km,bkm,imp

def do_fig(X,k,n,km,bkm):
    """3-part figure for problems without known optimum
    statistics: improvement over k-means++"""
    imp = 1-bkm.inertia_/km.inertia_
    fig,ax=plt.subplots(1,3,figsize=(21,6))
    plot(X,ax=ax[0],title=f"data set\n n={n} ")
    plot(X,ax=ax[1],C=km.cluster_centers_,title=f"k-means++\nk={k}, SSE={km.inertia_:.3f}")
    plot(X,ax=ax[2],C=bkm.cluster_centers_,title=f"breathing k-means \nk={k}, SSE={bkm.inertia_:.3f}, {imp:.2%} lower")
    plt.show()

def do_fig_opt(X,k,n,km,bkm,D,opt2):
    """3-part figure for problems with known optimum
    statistics: percent deviation from optimum"""
    if opt2:
        opt = D.get_opt2_sse()
        Copt= D.get_optimum2()
    else:
        opt = D.get_opt_sse()
        Copt= D.get_optimum()
    fig,ax=plt.subplots(1,3,figsize=(21,7))
    plot(X,ax=ax[0],C=Copt,title=f"data set with optimal solution (red)\n n={n}, k={k} ")
    plot(X,ax=ax[1],C=km.cluster_centers_,title=f"k-means++\nk={k}, SSE=Opt+{km.inertia_/opt-1:.2%}")
    plot(X,ax=ax[2],C=bkm.cluster_centers_,title=f"breathing k-means \nk={k}, SSE=Opt+{bkm.inertia_/opt-1:.2%},")
    plt.show()

def run_on_stored_dataset(prefix, dir="../data", k=50, doplot=True, dotext=True, retval=False):
    """data sets drawn from Gaussian mixture distributions
    k:   number of cluster centers
    doplot: show plot?
    dotext: show text output?
    retval: return data set and algorithm objects?
    """
    D = MyDataSet(prefix=prefix,dir=dir)
    X=D.get_data().astype(np.float64)
    n=D.get_n()
    km,bkm,imp=run_models(X=X,k=k,dotext=dotext)
    if doplot:
        do_fig(X,k,n,km,bkm)
    if retval:
        return X,km,bkm

def run_on_stored_dataset_with_opt(prefix,dir="../data", doplot=True, dotext=True, retval=False):
    """stored data sets with known optimum for a particular k-value
    prefix: prefix of data,optimum, and inf-files in dir
    dir: directory with data files
    doplot: show plot?
    dotext: show text output?
    retval: return data set and algorithm objects?
    """
    D = MyDataSet(prefix=prefix,dir=dir)
    X=D.get_data()#.astype(np.float32)
    n = D.get_n()
    # use the k for which the optimum is known
    k=D.get_opt_k()
    opt=D.get_opt_sse()
    km,bkm,imp=run_models(X=X,k=k,opt=opt,dotext=dotext)
    if doplot:
        do_fig_opt(X=X,k=k,n=n,km=km,bkm=bkm,D=D,opt2=False)
    if retval:
        return X,km,bkm

def run_on_stored_dataset_with_opt2(prefix,dir="../data", doplot=True, dotext=True, retval=False):
    """stored data sets with known optimum and k==n/2
    prefix: prefix of data,optimum, and inf-files in dir
    dir: directory with data files
    doplot: show plot?
    dotext: show text output?
    retval: return data set and algorithm objects?
    """
    D = MyDataSet(prefix=prefix,dir=dir)
    X=D.get_data()#.astype(np.float32)
    # use the k for which the optimum2 is known
    k=D.get_opt2_k()
    n = D.get_n()
    opt=D.get_opt2_sse()
    km,bkm,imp=run_models(X=X,k=k,opt=opt,dotext=dotext)
    if doplot:
        do_fig_opt(X=X,k=k,n=n,km=km,bkm=bkm,D=D,opt2=True)
    if retval:
        return X,km,bkm

def run_on_gaussian_mixture(g=20,k=60,sig=0.01, n=10000, doplot=True, dotext=True, retval=False):
    """data sets drawn from newly generated Gaussian mixture distributions
    g:   number of Gaussians
    k:   number of cluster centers
    sig: standard deviation of Gaussian (in each direction)
    n:   number of data points
    doplot: show plot?
    dotext: show text output?
    retval: return data set and algorithm objects?
    """
    d=2
    X = np.concatenate([np.random.random(d)+np.random.normal(size=(n//g,d),scale=sig) for _ in range(g)])
    km,bkm,imp=run_models(X=X,k=k,dotext=dotext)
    if doplot:
        do_fig(X,k,n,km,bkm)
    if retval:
        return X,km,bkm

def run_on_csv_file(filename, dir="../data", k=50,  doplot=True, dotext=True, retval=False):
    """general data sets stored in csv-files
    filename: name of csv-file
    dir: dir for csv-file
    k:   number of cluster centers
    doplot: show plot?
    dotext: show text output?
    retval: return data set and algorithm objects?
    """
    D=MyDataSet(filename=filename,dir=dir)
    X=D.get_data()
    n = D.get_n()
    km,bkm,imp=run_models(X=X,k=k,dotext=dotext)    
    if doplot:
        do_fig(X,k,n,km,bkm)
    if retval:
        return X,km,bkm

def run_on_array(X, k=50, doplot=True, dotext=True, retval=False):
    """data sets given as numpy array X
    k:   number of cluster centers
    doplot: show plot?
    dotext: show text output?
    retval: return data set and algorithm objects?
    """
    D=MyDataSet(X=X)
    n = D.get_n()
    km,bkm,imp=run_models(X=X,k=k,dotext=dotext)
    if doplot:
        do_fig(X,k,n,km,bkm)
    if retval:
        return X,km,bkm
