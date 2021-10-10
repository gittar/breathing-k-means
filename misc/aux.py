import numpy as np
import matplotlib.pyplot as plt
import dataset

def noticks(ax):
    """modify graph to have no axis ticks"""
    ax.tick_params(length=0)
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])


def do_fig(km, bkm, D=None, X=None, show=1,  # NOSONAR
           figsize=(21, 9),
           tightlayout=1, pad=1, show_k=0, **kwargs):
    """3-part figure for k-means problems
    statistics: improvement over k-means++

    km: KMeans object
    bkm: breathing k-means object
    D: DataSet object
    X: data array
    """
    assert not(D is None) == (X is None)  # exactly one must be set
    if D is None:  
        D = dataset.DataSet.from_array(X, normalize=False)

    if (not (isinstance(D, dataset.DataSet) 
        )):
        if isinstance(D, np.ndarray):
            D = dataset.DataSet.from_array(D)
        else:
            raise Exception("no dataset provided"+str(type(D)))

    if "showdim" not in kwargs:
        kwargs["showdim"] = 0
    if "ticks" not in kwargs:
        kwargs["ticks"] = 0
    if "voro" not in kwargs:
        kwargs["voro"] = 0
    if "showdata" not in kwargs:
        kwargs["showdata"] = 1
    if "fontsize" not in kwargs:
        kwargs["fontsize"] = 25
    if "unitsquare" not in kwargs:
        kwargs["unitsquare"] = 0
    if "subtitle_color" not in kwargs:
        kwargs["subtitle_color"] = "black"
    if "subtitle_alpha" not in kwargs:
        kwargs["subtitle_alpha"] = 1.0
    showdim = D.comp_showdim(kwargs["showdim"])

    X = D.get_data()
    k = km.n_clusters
    assert k == bkm.n_clusters
    imp = 1 - bkm.inertia_ / km.inertia_
    fig, axs = plt.subplots(1, 3, figsize=figsize)
    if imp >= 0:
        subtitle = f"{imp:.2%} lower SSE"
    else:
        subtitle = f"{-imp:.2%} higher SSE"
    #
    # plot dataset
    #
    if D.get_d() > 2:
        title = f"Data set, n={D.get_n()}\ndims: {showdim} of {D.get_d()}"
    else:
        title = f"Data set, n={D.get_n()}"
        if show_k:
            title += f", k={km.n_clusters}"
    D.plot(ax=axs[0], title=title, **kwargs)
    #
    # plot k-means++ result
    #
    inert = f"{km.inertia_:.3e}"
    D.plot(ax=axs[1], C=km.cluster_centers_,
            title=f"k-means++\nSSE={inert}",
            subtitle="", 
        **kwargs)
    #
    # plot breathing k-means result
    #
    if imp >= 0:
        addendum = f" ({imp:.2%} lower)"
    else:
        addendum = f" ({-imp:.2%} higher!)"
    inert = f"{bkm.inertia_:.3e}"
    D.plot(ax=axs[2], C=bkm.cluster_centers_,
            title=f"breathing k-means\nSSE={inert}",
            subtitle=subtitle,
             **kwargs)
    if tightlayout:
        plt.tight_layout(pad=pad)
    if show:
        plt.show()
    return fig

def eqlim(axs, xlim=None, ylim=None, unitsquare=0, unitpad=None, X=None):
    """n a list of axes set all dims to that of the first axis
       or to the dimensions given in the parameters"""
    assert ylim is None or len(ylim) == 2
    assert xlim is None or len(xlim) == 2
    try:
        axs = axs.flat
    except AttributeError:
        # here we likely have just one Axes object
        # put it in a list ....
        axs = [axs]
    a0 = axs[0]
    if unitpad is not None:
        # take unitsquare with padding of unitpad
        if X is None and unitsquare:
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
