import time
import dataset
import sklearn.cluster  # KMeans class from scikit-lean
import aux
import bkmeans
import pandas as pd


def correct_numerics(imp):
    # correct negative deltas very close to 0
    if imp < 0 and imp > -0.000000001:
        imp = 0
    return imp

def run_models(D, k, dotext=True, km_args={}, bkm_args={}):  # NOSONAR
    """ run k-means++ and breathing k-means on given problem

    dotext: write out results for each single run
    """
    assert k > 0

    X = D.get_data()

    # k-means++ (scikit-learn)
    if dotext:
        print("k-means++:              ", end="")

    t0 = time.time()
    km = sklearn.cluster.KMeans(n_clusters=k, **km_args)
    km.fit(X)
    t_km = time.time() - t0

    if dotext:
        print(f" SSE={km.inertia_:.3f}                 t={t_km:6.2f}s ")

    # breathing k-means
    if dotext:
        print("breathing k-means:      ", end="")

    t0 = time.time()

    bkm = bkmeans.BKMeans(
        n_clusters=k,
        init=km.cluster_centers_,
        n_init=1,
        **bkm_args)
    bkm.fit(X)
    # add km++ time to bkm
    t_bkm = time.time() - t0 + t_km

    # SSE improvement
    if dotext:
        overhead = t_bkm / t_km - 1
        imp = 1 - bkm.inertia_ / km.inertia_
        perci = f"({imp:.2%} lower)"
        print(
            f" SSE={bkm.inertia_:.3f} {perci:16s}t={t_bkm:6.2f}s",
            end="")
        print(f" ({overhead:+6.2%})")

    return km, bkm, t_km, t_bkm


def handle_problem(D, k, dotext=True):
    assert isinstance(k, int) and k > 0
    km, bkm, t_km, t_bkm = run_models(D=D, k=k, dotext=dotext)
    return km, bkm

def solve_problem(D, k, doplot=None, dotext=None,
                  retval=False, repeat=1,
                  km_args={}, bkm_args={}, **kwargs):
    """stored data sets with known optimum for a particular k-value
    D: Dataset
    k: k
    doplot: show plot?
    dotext: show text output?
    retval: algorithm objects and figure object
    """
    if 'tightlayout' not in kwargs:
        kwargs['tightlayout'] = 1
    if 'pad' not in kwargs:
        kwargs['pad'] = 1
    if dotext is None:
        dotext = True
    if doplot is None:
        doplot = repeat == 1
    assert isinstance(k, int) and k > 0
    # data container
    stat = {
        "err_km": [],
        "err_bkm": [],
        "cpu_km": [],
        "cpu_bkm": []
    }
    for i in range(repeat):
        # one or several runs
        km, bkm, t_km, t_bkm = run_models(
            D=D, k=k, dotext=dotext, km_args=km_args, bkm_args=bkm_args)  # NOSONAR
        stat["err_km"].append(km.inertia_)
        stat["err_bkm"].append(bkm.inertia_)
        stat["cpu_km"].append(t_km)
        stat["cpu_bkm"].append(t_bkm)
        if doplot:
            fig = aux.do_fig(
                km=km,
                bkm=bkm,
                D=D,
                **kwargs)
        else:
            fig = None
    if(repeat > 1):
        # statistical summary
        df = pd.DataFrame(stat).mean()
        imp = 1 - df.loc["err_bkm"] / df.loc["err_km"]
        ovi = df.loc["cpu_bkm"] / df.loc["cpu_km"] - 1
        tabi = "\n".join([x[2:]
                          for x in df.to_frame().T.to_string().split("\n")])
        print(
            f"\nSummary of {repeat} runs:\n\n{tabi}\n\n{imp:.2%} lower SSE,  {ovi:7.2%} extra CPU time")
    if retval:
        return km, bkm, fig
