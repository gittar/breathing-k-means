import random
import os
import pandas as pd
import numpy as np
import functools
import matplotlib.pyplot as plt
import scipy.spatial

# DataSet  class

def arr2txt(x):
    """replace arrays by the string 'array'"""
    if isinstance(x, np.ndarray):
        return "array"
    else:
        return x

# wrapper to save the current call for documentation purposes
def store_call(func, *args, **kwargs):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):

        if "name" in kwargs:
            # name parameter is handled on its own and not forwarded so that not all functions need one
            #print("Name found", kwargs["name"])
            thename = kwargs["name"]
            del kwargs["name"]
        else:
            thename = None
        # create the Dataset Object D
        obj = func(*args, **kwargs)
        if obj.name is None:
            obj.name = thename
        #
        # reconstruct the call ....
        # actual arrays are replaced by "array"
        #
        mycall = func.__name__ + "("
        # args
        if len(args) > 1:
            mycall += ", ".join(map(lambda x: "\"" +
                                    str(arr2txt(x)) + "\"", args[1:])) + ", "
        # kwargs
        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                kwargs[k] = "(array)"
        mycall += ", ".join([f"{k}={v}" for k, v in kwargs.items()])
        if obj._seed is not None:
            mycall += f",seed={obj.seed}"
        mycall += ")"
        obj._called_as = func.__qualname__.split(".")[0] + "." + mycall
        return obj
    return wrapper

class UnknownInitException(Exception):
    pass

class DataSet:
    """
administers test datasets for clustering algorithms
    """
    _num_type = np.float64

    def __init__(self, n=None, d=None):
        """constructor, sets variable which are needed to create the dataset later
        to actually generate data, method .... needs to be called
        """
        self.version = "1.0.1"
        self.set_name(None)
        self.n = n  # number of data points
        self.d = d  # dimensionality of data
        self._called_as = None  # how is this dataset generated?
        self._seed = None

    def __repr__(self):
        # for print(), only show basecall
        return f"""DataSet Object, name="{self.name}" n={self.get_n()}, d={self.get_d()}, {self._called_as} """

    def read_csv(self, filename, dir=None):
        if dir is None:
            dir = "."
        csvfile = os.path.join(dir, filename)
        assert os.path.isfile(csvfile), f"file not found: {csvfile}"
        df = pd.read_csv(csvfile, header=None)
        self.data = df.to_numpy()
        self.set_description("no details available")
        self._called_as = "(no info)"

    def set_description(self, desc):
        self.description = desc

    def set_name(self, name):
        self.name = name

    def get_name(self):
        if self.name is None:
            return "(unnamed)"
        else:
            return self.name

    @classmethod
    @store_call
    def from_array(cls, array, normalize=False):
        """from numpy array
        if normalize == True: rescale to unit square, keeping aspect ratio
        """
        data = array
        n, d = data.shape
        obj = cls(n=n, d=d)
        obj.set_data(data)
        if normalize:
            obj.normalize()
        obj.description = f"data from array, dim={d}, n={n}"
        return obj

    @classmethod
    @store_call
    def from_csv_file(
            cls,
            filename,
            dim=None,
            normalize=False):
        """from csv-file
        if normalize == True: rescale to unit square, keeping aspect ratio
        """
        df = pd.read_csv(filename, header=None)
        data = df.to_numpy()
        if dim is not None:
            data = data[:, :dim]
        n, d = data.shape
        obj = cls(n=n, d=d)
        obj.set_data(data)
        if normalize:
            obj.normalize()
        obj.description = f"data from csv-file {filename}, dim={d}, n={n}"
        return obj

    @classmethod
    @store_call
    def from_txt_file(cls, filename, dim=2, normalize=False):
        """textfile with data points one per line, space-separated
        if normalize == True: rescale to unit square, keeping aspect ratio
        """
        data = np.loadtxt(filename)
        data = data[:, :dim]  # only 2D #TODO reinstate has_label
        n, d = data.shape
        obj = cls(n=n, d=d)
        obj.set_data(data)
        if normalize:
            obj.normalize()

        obj.description = f"data from txt-file {filename}, dim={d}, n={n}"
        return obj

    def normalize(self, vertical="center", horizontal="center"):
        """ scale data such that it fits into the unit square
        (only applicable for 2D-data)
        vertical: vertical alignment (center or top, else bottom)
        horizontal: horiontal alignment: (center or right, else left)
        """
        X = self.get_data()
        _, d = X.shape
        if d == 2:
            yrange = np.max(X[:, 1]) - np.min(X[:, 1])
            xrange = np.max(X[:, 0]) - np.min(X[:, 0])
            maxrange = max(xrange, yrange)
            scale = 1 / maxrange
            # scale to length/wid 1
            X = X * scale  # scaling
            # shift to have lower-left on origin
            offset = - np.asarray([np.min(X[:, 0]), np.min(X[:, 1])])
            X = X + offset
            # left, bottom of unit square
            uyrange = yrange / maxrange
            uxrange = xrange / maxrange

            # shift data if it does not fill whole unit square in one direction
            if vertical == "center":
                dy = (1 - uyrange) / 2
            elif vertical == "top":
                dy = (1 - uyrange)
            else:
                dy = 0

            if horizontal == "center":
                dx = (1 - uxrange) / 2
            elif horizontal == "right":
                dx = (1 - uxrange)
            else:
                dx = 0

            shift = np.asarray([dx, dy])
            X += shift
            self.data = X

    @staticmethod
    def set_random_seed(seed):
        """Set random seed for tensorflow, python and numpy

        :param seed (int): random seed"""
        if seed is None:
            # do nothing
            # 2**32=4294967296 = maximum seed value + 1
            # seed = int((time()*1000)%4294967296)
            pass
        else:
            # Python
            random.seed(seed)
            # numpy
            np.random.seed(seed)

    def get_error_of(self, C):
        """compute summed squared error of given codebook for this dataset"""
        X = self.get_data()
        dist=scipy.spatial.distance.cdist(X,C,metric="sqeuclidean")
        return np.sum(np.min(dist,axis=1))

    def get_d(self):
        """return number of features (d)"""
        return self.get_data().shape[1]

    def get_n(self):
        """return number of samples"""
        return len(self.get_data())

    def shuffle(self):
        """ randomize data order
        """
        np.random.shuffle(self.data)

    def get_data(self):
        """return dataset"""
        return self.data

    def set_data(self, X):
        """set dataset"""
        self.data = X

    def comp_showdim(self, showdim):
        # determine the two dimensions shown for high-D data
        if isinstance(showdim, int):
            showdim = (showdim, (showdim + 1) % self.get_d())
        elif isinstance(showdim, tuple) or isinstance(showdim, list):
            assert len(showdim) == 2
        else:
            assert 0 == 1
        assert showdim[0] < self.get_d(), f"showdim {showdim[0]} too large"
        assert showdim[1] < self.get_d(), f"showdim {showdim[1]} too large"
        return showdim

    #
    # relative font size depends on "fontsize" and figsize or size of ax
    #
    # Absolute sizes:
    # * fontsize
    # * centersize
    # * dotsize
    def plot(self,  # NOSONAR
             ax=None,
             X=None,
             C=None,
             title=None,  # title string, font size from rc
             subtitle=None,  # additional text shown *under* the figure
             subtitle_alpha = 0.5,
             subtitle_color = "black",#"green",#"red",
             subtitle_face = None,

             dotsize=1,  # size of data points
             dotcolor="green",  # color of datapoints

             centersize=6,  # sice of centroids
             centermarker="o",  # marker symbol used for centroids
             centerrim=True,  # have white rime around centers
             centercolor="#ff6666",

             ticks=False,  # have ticks
             fontsize=30,#20,  # fontsize for the title and subtitle string
             return_fig=False,  # return figure
             showdim=0,  # (int) for high-dim. data show this col and next or
             # (2-tuple) the given 2 columns, e.g. (2,7)
             figsize=(6, 6),  # figsize if no ax is given
             voro=False,  # show voronoi diagram
             delaunay=False, # show delaunay triangulation
             showdata=True,
             # show data points (usually just one of this or showboxes is true)
             showaxis=True,  # show axis
             plotfacecolor="white",

             line_width=1,  # voronoi lines
             line_colors="blue",  # voronoi line color

             show=True,
             unitsquare=False,  # only show the unit square unless data would be cut
             forceunitsquare=0,  # force showing only the unit square
             unitpad=0.02,  # padding around unit square
             square=False,  # ensure square form (not necessarily unit square)
             ):
        """
        general plotting function
        """
        #
        # axis font size is determined by:
        #    plt.rcParams['font.size']=xx
        #
        showdim = self.comp_showdim(showdim)
        D = self
        if forceunitsquare:
            unitsquare = 1
        if square:
            # if drawing area should be square assume not unitsquare
            unitsquare = 0
            forceunitsquare = 0
        doshow = False # show figure?
        if ax is None:
            # standalone figure
            doshow = show
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            fig.patch.set_facecolor(plotfacecolor)
        else:
            # plot on axis object
            fig = plt.gcf()
            fig.patch.set_facecolor(plotfacecolor)
            fig = None

        if istrue(C):
            # no valid C provided, just truth value
            C=0

        # get data
        if X is None and D is not None:
            X = D.get_data()

        # reduce high-dim data
        if X is not None and len(X.shape) == 2 and X.shape[1] == 1:
            # 1-D - make 2D
            X = make_2D(X, 0.0)
        X = X[:, showdim]
        assert X.shape[1] == 2
  
        # reduce high-dim codebook
        if isSomething(C):
            if len(C.shape) == 2 and C.shape[1] == 1:
                # blow up C to 2 dimensions
                C = make_2D(C, 0.0)
            # select 2 dimensions to display
            C = C[:, showdim]

        # remove ticks if not desired
        if not ticks:
            noticks(ax)

        # Voronoi diagram wrt centers (# of centers must be > 2)
        if voro and isSomething(C) and len(C) > 2:
            try:
                vor = scipy.spatial.Voronoi(C)
                scipy.spatial.voronoi_plot_2d(
                    vor,
                    ax=ax,
                    show_vertices=False,
                    show_points=True,
                    line_colors=line_colors,
                    line_width=line_width,
                    zorder=7)

            except scipy.spatial.qhull.QhullError:
                print("QhullError: no voronoi for this data ....")

        # Delaunay wrt centers (# of centers must be > 2)
        if delaunay and isSomething(C) and len(C) > 2:
            try:
                tri = scipy.spatial.Delaunay(C)
                ax.triplot(C[:,0], C[:,1],tri.simplices, color="orange")
            except scipy.spatial.qhull.QhullError:
                print("QhullError: no voronoi for this data ....")

        # data range
        minx = np.min(X[:, 0])
        maxx = np.max(X[:, 0])
        dx = maxx - minx
        miny = np.min(X[:, 1])
        maxy = np.max(X[:, 1])
        dy = maxy - miny
        dmax = max(dx, dy)

        if (dmax > 1) and (forceunitsquare or unitsquare):
            # do not do for data extending larger than 1 in any direction
            print("Warning: Data is cut out due to unitsquare option. data extension:",dmax,"force:",forceunitsquare,"unit",unitsquare)
            #unitsquare = 0

        if unitsquare:
            # limit plot area to unit square plus small margin
            xlim = (0 - unitpad, 1 + unitpad)
            ylim = xlim
            setlim(ax, xlim=xlim, ylim=ylim)
        elif square:
            # enforce square canvas
            xlim = (minx, minx + dmax)
            ylim = (miny, miny + dmax)

            setlim(ax, xlim=xlim, ylim=ylim, unitpad=unitpad,X=X)
        else:  # no scaling here, but apply padding
            if unitpad is None:
                omnipad = 0
            else:
                omnipad = unitpad
            # make limits dependent on data
            xlim = (np.min(X[:, 0]) - omnipad * dx,
                    np.max(X[:, 0]) + omnipad * dx)
            if X.shape[1] == 1:
                # one-D data
                ylim = (0, 0.5)
            else:
                ylim = (np.min(X[:, 1]) - omnipad * dy,
                        np.max(X[:, 1]) + omnipad * dy)
            setlim(ax, xlim=xlim, ylim=ylim,X=X)

        # same scale for all dimensions
        ax.set_aspect(1.0)

        zz = -2

        # data
        if showdata:
            scat(X, s=dotsize, c=dotcolor, ax=ax, zorder=zz + 4)

        # codebook with white rim
        if isSomething(C):
            if showdata:
                for i in range(len(C)):
                    if centerrim:
                        scat(C[i:i + 1], s=centersize * 1.4,
                             c="white", ax=ax, zorder=zz + 5)
                    scat(C[i:i + 1], s=centersize * 1, c=centercolor,
                         ax=ax, zorder=zz + 5, marker=centermarker)
            else:
                scat(C, s=centersize * 1, c=centercolor, ax=ax, zorder=zz + 6)

        # title #TODO: Overfontsize?
        ax.set_title(title, fontsize=fontsize)

        # transparent overlay
        if subtitle is not None:
            t=ax.text(
                0.5,
                -0.02,
                subtitle,
                horizontalalignment='center',
                verticalalignment='top',
                transform=ax.transAxes,
                fontsize=fontsize,
                fontweight="normal",
                alpha=subtitle_alpha,
                color=subtitle_color,
                zorder=12)
            if not subtitle_face in [None,"white"]:
                t.set_bbox(dict(facecolor=subtitle_face, alpha=1.0, edgecolor="white"))

        if not showaxis:
            ax.axis("off")

        if doshow:
            # pass
            if not return_fig:
                plt.show()
        else:
            if not return_fig and fig is not None:
                plt.close(fig)

        if return_fig:
            return fig

    def get_aspect(self):
        """ get height/width ratio
        """
        X = self.get_data()
        wid=max(X[:,0])-min(X[:,0])
        hgt=max(X[:,1])-min(X[:,1])
        return hgt/wid


def istrue(x):
    return not isinstance(x, np.ndarray) and x in [1, True]

def isSomething(x):
    return (x is not None) and \
        not (isinstance(x, int) and x == 0) and \
        not (isinstance(x, str) and x == "") and \
        not (isinstance(x, bool) and not x)

def noticks(ax):
    """modify graph o have no axis ticks"""
    ax.tick_params(length=0)
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])

def setlim(ax, xlim=None, ylim=None, unitsquare=False, unitpad=None, X=None):
    """set limits for an ax object"""
    assert ylim is None or len(ylim) == 2
    assert xlim is None or len(xlim) == 2
    if hasattr(ax,"dirty"):
        # do not apply further xlim/ylim setting here
        return
    else:
        ax.dirty=1
    if unitpad is not None:
        # take unitsquare with padding of unitpad
        if X is None and unitsquare:
            xlim = (0 - unitpad, 1 + unitpad)
            ylim = xlim
        else:
            # check how extended the data is, dont cut anything
            xmin = np.min(X[:, 0])
            xmax = np.max(X[:, 0])
            dx =xmax-xmin
            ymin = np.min(X[:, 1])
            ymax = np.max(X[:, 1])
            dy = ymax-ymin
            xlim = (min(xlim[0], xmin) - unitpad*dx, max(xlim[1], xmax) + unitpad*dx)
            ylim = (min(ylim[0], ymin)- unitpad*dy, max(ylim[1], ymax)+ unitpad*dy)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect(1.0)

def eqlim(axs, xlim=None, ylim=None, unitpad=None, X=None):
    """in a list of axes set all dims to that of the first axis
    or to the dimensions given in the parameters
    """
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
        # HACK unitpad used as a trigger! improve
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

def scat(P, s=1, c="r", ax=None, zorder=0, marker="o"):
    """scatter plot"""
    if P is None:
        # no data, ignore
        return
    if P.shape[1] == 1:
        # 1D, add second dimension
        x = np.zeros((len(P), 2))
        x[:, :-1] = P
        P = x
    if ax is None:
        ax = plt
    if P is not None:
        ax.plot(P[:, 0], P[:, 1], ms=s, c=c,
                marker=marker, ls="", zorder=zorder)

def make_2D(X, y=0):
    if (len(X.shape) == 2 and X.shape[1] == 1) or len(X.shape) == 1:
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        x = np.zeros((len(X), 2)) + y
        x[:, :-1] = X
        X = x
    return X

