import math
import numpy as np
import matplotlib.pyplot as plt
import dataset


def isstuff(x):
    return (x is not None) and \
        not (isinstance(x, int) and x == 0) and \
        not (isinstance(x, str) and x == "") and \
        not (isinstance(x, bool) and not x)

def noticks(ax):
    """modify graph to have no axis ticks"""
    ax.tick_params(length=0)
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])


def fmt(x, maxdig=6, maxpost=3):
    """format float possibly in E-format"""
    X = math.fabs(x)
    l10 = math.floor(math.log10(X))
    if l10 > 5:
        return f"{x:.{maxdig-3}E}"
    else:
        return f"{x:.{min(maxpost,max(0,maxdig-l10))}f}"

def do_fig(km, bkm, D=None, X=None, show=1,  # NOSONAR
           figsize=(21, 9), tightlayout=1, pad=1, **kwargs):
    """3-part figure for k-means problems
    statistics: improvement over k-means++

    km: KMeans object
    bkm: breathing k-means object
    D: DataSet object
    X: data array
    """

    assert not(D is None) == (X is None)  # exactly one must be set
    if D is None:  # and isinstance(X,np.array):
        D = dataset.DataSet.from_array(X, normalize=False)

    if not isinstance(D, dataset.DataSet):
        print("D is no dataset", type(D))
        if isinstance(D, np.ndarray):
            print("numpyarr")
            D = dataset.DataSet.from_array(D)

    if "showdim" not in kwargs:
        kwargs["showdim"] = 0
    if "ticks" not in kwargs:
        kwargs["ticks"] = 0
    if "voro" not in kwargs:
        kwargs["voro"] = 0
    if "fontsize" not in kwargs:
        kwargs["fontsize"] = 25
    if "unitsquare" not in kwargs:
        kwargs["unitsquare"] = 1
    showdim = D.comp_showdim(kwargs["showdim"])

    X = D.get_data()
    k = km.n_clusters

    assert k == bkm.n_clusters
    imp = 1 - bkm.inertia_ / km.inertia_
    fig, axs = plt.subplots(1, 3, figsize=figsize)

    # plot dataset
    if D.get_d() > 2:
        title = f"Data set, n={D.get_n()}\ndims: {showdim} of {D.get_d()}"
    else:
        title = f"Data set, n={D.get_n()}"

    D.plot(ax=axs[0], title=title, **kwargs)

    # plot k-means++ result
    inert = fmt(km.inertia_, 6, 3)
    D.plot(ax=axs[1], C=km.cluster_centers_,
            title=f"k-means++\nSSE={inert}",
            subtitle="", **kwargs)
    if imp >= 0:
        subtitle = f"{imp:.2%} lower SSE"
    else:
        subtitle = f"{-imp:.2%} higher(!) SSE"

    # plot breathing k-means result
    inert = fmt(bkm.inertia_, 6, 3)
    D.plot(ax=axs[2], C=bkm.cluster_centers_,
            title=f"breathing k-means\nSSE={inert}",
            subtitle=subtitle, **kwargs)

    if tightlayout:
        plt.tight_layout(pad=pad)
    if show:
        plt.show()
    return fig

def frmt(x):
    if math.fabs(x) < 90:
        return f"{x:.3f}"
    elif math.fabs(x) < 900:
        return f"{x:.2f}"
    elif math.fabs(x) < 5000:
        return f"{x:.1f}"
    else:
        return f"{x:.0f}"


def eqlim(axs, xlim=None, ylim=None, unitpad=None, X=None):
    """n a list of axes set all dims to that of the first axis
       or to the dimensions given in the parameters"""
    assert ylim is None or len(ylim) == 2
    assert xlim is None or len(xlim) == 2
    try:
        axs = axs.flat
    except AttributeError as _:
        # here we likely have just one Axes object
        # put it in a list ....
        axs = [axs]
    a0 = axs[0]
    if unitpad is not None:
        # take unitsquare with padding of unitpad
        if X is None:
            xlim = (0 - unitpad, 1 + unitpad)
            ylim = xlim
        else:
            # check how extended the data is, dont cut anything
            xmin = np.min(X[:, 0])
            xmax = np.max(X[:, 0])
            ymin = np.min(X[:, 1])
            ymax = np.max(X[:, 1])
            xlim = (min(0, xmin) - unitpad, max(1, xmax) + unitpad)
            ylim = (min(xlim[0], ymin), max(xlim[1], ymax))
    else:
        # take limits of first ax if not specified otherwise
        if xlim is None:
            xlim = a0.get_xlim()
        if ylim is None:
            ylim = a0.get_ylim()
    # give same limits to all axes
    for ax in axs:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect(1.0)
