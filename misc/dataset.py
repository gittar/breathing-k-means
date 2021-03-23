import random
import os
import json
import pandas as pd
import numpy as np
import functools
import matplotlib.pyplot as plt
import scipy.spatial
import matplotlib.patches as patches

# DataSet  class
# (the works)

np.seterr(all="raise")


def sort_dict(D):
    "sort dicts with numeric keys"
    return {k: v for k, v in sorted(D.items(), key=lambda item: int(item[0]))}

def simmi(x):
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
                                    str(simmi(x)) + "\"", args[1:])) + ", "
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


class InsufficientParamsException(Exception):
    pass


class NoKException(Exception):
    pass


class KTooLargeException(Exception):
    pass


class UnknownInitException(Exception):
    pass


class OldFileFormatException(Exception):
    pass


class NoDataException(Exception):
    pass


class DataSet:
    """
administers test datasets for clustering algorithms

some methods provide initalization of a codebook with different methods
    * randomly from the dataset without repetition: random_init()
    """

    _num_type = np.float64

    @classmethod
    def _use_32(cls):
        cls._num_type = np.float32

    @classmethod
    def _use_64(cls):
        cls._num_type = np.float64

    @staticmethod
    def _get_num_type():
        return DataSet._num_type

    @staticmethod
    def noptword():
        return "generic"

    def __init__(self, n=None, d=None):
        """constructor, sets variable which are needed to create the dataset later
        to actually generate data, method .... needs to be called
        """
        self.version = "1.0.1"
        self.set_name(None)
        self.deterministic = True
        self.n = n  # number of data points
        self.d = d  # dimensionality of data

        # codebook after initialization (random or k-means++)
        self.ibook = None
        self.k = None
        self.d2 = 1  # minimum dimension for ibook (for drawing purposes)
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

    def codebook_present(self):
        return self.ibook is not None

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

    def is_consistent(self):
        return True

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

    def normalize(self, vertical="center", horizontal="center", REFX=None):
        """ scale data such that it fits into the unit square
        (only applicable for 2D-data)
        vertical: vertical alignment (center or top, bottom nyi) # TODO
        horizontal: horiontal alignment: (center or right, left nyi) # TODO

        if aray REFX is given, determine scaling such that REFX fits into the unit square
        """
        X = self.get_data()
        _, d = X.shape
        if d == 2:
            if REFX is None:
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
            else:
                #
                # normalize wrt reference data
                #
                yrange = np.max(REFX[:, 1]) - np.min(REFX[:, 1])
                xrange = np.max(REFX[:, 0]) - np.min(REFX[:, 0])
                maxrange = max(xrange, yrange)
                scale = 1 / maxrange
                X = X * scale  # scaling
                REFX = REFX * scale
                offset = - np.asarray([np.min(REFX[:, 0]), np.min(REFX[:, 1])])
                X = X + offset
                # left, bottom of unit square
                uyrange = yrange / maxrange
                uxrange = xrange / maxrange

            # shift data if it does not fill whole unit square in one direction
            if vertical == "center":
                dy = (1 - uyrange) / 2
            elif vertical == "top":
                dy = (1 - uyrange)

            if horizontal == "center":
                dx = (1 - uxrange) / 2
            elif horizontal == "right":
                dx = (1 - uxrange)

            shift = np.asarray([dx, dy])
            X += shift
            if hasattr(self, "gamma"):
                # adjust gamma
                self.gamma *= scale

            self.data = X

    @staticmethod
    def set_random_seed(seed):
        """Set random seed for tensorflow, python and numpy

        :param seed (int): random seed"""
        if seed is not None:
            # Python
            random.seed(seed)
            # numpy
            np.random.seed(seed)

    def get_error_of(self, C):
        """compute summed squared error of given codebook for this dataset"""
        X = self.get_data()
        mat = np.zeros([len(X), len(C)])
        # loop over codebok
        for i in range(len(C)):
            mat[:, i] = (np.linalg.norm(X - C[i], axis=1))
        return sum(np.min(mat, axis=1)**2)

    def get_params(self):
        """return main parameters of dataset"""
        pars = {
            "description": self.get_description(),
            "n_samples": self.get_n(),
            "n_features": self.get_d(),
            "called_as": self._called_as,
        }
        if hasattr(self, "gamma"):
            pars["gamma"] = self.gamma
        if hasattr(self, "inner"):
            pars["inner"] = self.inner
        return pars

    def get_description(self):
        """return description of dataset"""
        return self.description

    def get_d(self):
        """return number of features (d)"""
        return self.get_data().shape[1]

    def get_n(self):
        """return number of samples"""
        return len(self.get_data())

    def initialize(self, k, init):
        if init == "random":
            return self.random_init(k=k)
        else:
            raise UnknownInitException("Unknown init method:", init)

    def random_init(self, k=None):
        """get codebook of size k at random from dataset"""
        if k is not None:
            self.k = k
        else:
            # keep current value of k
            pass
        if self.data.any():
            a = list(range(self.get_n()))
            np.random.shuffle(a)
            self.ibook = np.zeros(
                (self.k, max([self.d2, self.get_d()])))  # ibook definition
            for i in range(self.k):
                self.ibook[i] = self.data[a[i]]
            return self.ibook
        else:
            raise NoDataException("no data found")

    def shuffle(self):
        """ randomize data order
        """
        np.random.shuffle(self.data)

    def get_data(self):
        """return dataset"""
        return self.data.astype(DataSet._num_type)

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

    def plot(self,  # NOSONAR
             ax=None,
             X=None,
             C=None,
             title=None,  # title string, font size from rc
             subtitle=None,  # additional text shown *under* the figure
             dotsize=1,  # size of data points
             dotcolor="green",  # color of datapoints
             centersize=6,  # sice of centroids
             centermarker="o",  # marker symbol used for centroids
             centerrim=True,  # have white rime around centers
             ticks=False,  # have ticks
             fontsize=30,  # fontsize for the "subtitle" string
             return_fig=False,  # return figure
             showdim=0,  # (int) for high-dim. data show this col and next or
             # (2-tuple) the given 2 columns, e.g. (2,7)
             figsize=(6, 6),  # figsize if no ax is given
             voro=False,  # show voronoi diagram
             showaxis=True,  # show axis
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
        doshow = False
        if ax is None:
            # standalone figure
            doshow = show
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        else:
            # plot on axis object
            fig = None

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
        if isstuff(C):
            if len(C.shape) == 2 and C.shape[1] == 1:
                # blow up C to 2 dimensions
                C = make_2D(C, 0.0)
            # select 2 dimensions to display
            C = C[:, showdim]

        # remove ticks if not desired
        if not ticks:
            noticks(ax)

        # Voronoi diagram wrt centers (# of centers must be > 2)
        if voro and isstuff(C) and len(C) > 2:
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

        # data range
        minx = np.min(X[:, 0])
        maxx = np.max(X[:, 0])
        dx = maxx - minx
        miny = np.min(X[:, 1])
        maxy = np.max(X[:, 1])
        dy = maxy - miny
        dmax = max(dx, dy)
        if (dmax > 1) and not forceunitsquare or square:
            # do not do for data extending larger than 1 in any direction
            unitsquare = 0

        if unitsquare:
        # limit plot area to unit square plus small margin
            xlim = (0 - unitpad, 1 + unitpad)
            ylim = xlim
            eqlim(ax, xlim=xlim, ylim=ylim)
        elif square:
            xlim = (minx, minx + dmax)
            ylim = (miny, miny + dmax)
            eqlim(ax, xlim=xlim, ylim=ylim, X=X)
        else:  # no scaling here
            omnipad = unitpad
            if hasattr(D, "gamma"):
                omnipad += D.gamma / (D.inner - 1) / 2
            # make limits dependent on data
            xlim = (np.min(X[:, 0]) - omnipad * dx,
                    np.max(X[:, 0]) + omnipad * dx)
            if X.shape[1] == 1:
                ylim = (0, 0.5)
            else:
                ylim = (np.min(X[:, 1]) - omnipad * dy,
                        np.max(X[:, 1]) + omnipad * dy)
            eqlim(ax, xlim=xlim, ylim=ylim, X=X)

        # same scale for all dimensions
        ax.set_aspect(1.0)

        zz = -2

        # data
        scat(X, s=dotsize, c=dotcolor, ax=ax, zorder=zz + 4)

        # codebook with white rim
        if isstuff(C):
            for i in range(len(C)):
                if centerrim:
                    scat(C[i:i + 1], s=centersize * 1.4,
                            c="white", ax=ax, zorder=zz + 5)
                scat(C[i:i + 1], s=centersize * 1, c="#ff6666",
                        ax=ax, zorder=zz + 5, marker=centermarker)

        # title
        ax.set_title(title, fontsize=fontsize)

        # transparent overlay
        over_alpha = 0.5
        over_color = "red"
        if subtitle is not None:
            ax.text(
                0.5,
                -0.02,
                subtitle,
                horizontalalignment='center',
                verticalalignment='top',
                transform=ax.transAxes,
                fontsize=fontsize,
                fontweight="normal",
                alpha=over_alpha,
                color=over_color,
                zorder=12)

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

def isstuff(x):
    return (x is not None) and \
        not (isinstance(x, int) and x == 0) and \
        not (isinstance(x, str) and x == "") and \
        not (isinstance(x, bool) and not x)

# modify graph o have no axis ticks


def noticks(ax):
    ax.tick_params(length=0)
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])

# in a list of axes set all dims to that of the first axis
# or to the dimensions given in the parameters


def eqlim(axs, xlim=None, ylim=None, unitpad=None, X=None):
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
#
# plot
#


def scat(P, s=1, c="r", ax=None, zorder=0, marker="o"):
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


def scale_to_unit_square(X):
    minx = np.min(X[:, 0])
    maxx = np.max(X[:, 0])
    miny = np.min(X[:, 1])
    maxy = np.max(X[:, 1])
    dx = maxx - minx
    dy = maxy - miny
    X = X - [minx, miny]
    X = X / max(dx, dy)
    return X


def make_2D(X, y=0):
    if (len(X.shape) == 2 and X.shape[1] == 1) or len(X.shape) == 1:
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        x = np.zeros((len(X), 2)) + y
        x[:, :-1] = X
        X = x
    return X
