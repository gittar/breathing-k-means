import time
import sklearn.cluster  # KMeans class from scikit-lean
import aux
from IPython.display import display
from bkmeans import BKMeans
import pandas as pd

#
# run_models()
# handle_problems
# solve_problem()
#

def correct_numerics(imp):
    # correct negative deltas very close to 0
    if imp < 0 and imp > -0.000000001:
        #print("corrected numerics:",imp)
        imp = 0
    return imp

def xpdiff2(x,pos,neg,what=""):
    if x >= 0:
        return (f"({pos}{what})")
    else:
        return (f"({neg}{what})")

def xpdiff(x,pos,neg):
    if x >= 0:
        return (f"{x:.2%} {pos}")
    else:
        return (f"{-x:.2%} {neg}")

def run_models(D, k, 
    dotext=True, 
    km_args={}, 
    bkm_args={}):  # NOSONAR
    """ run k-means++ and breathing k-means on given problem

    dotext: write out results for each single run
    km_args: parameters for KMeans, e.g., {"tol":1E-6}
    bkm_args: parameters for BKMeans, e.g., {"m":10}
    make statistics in relation to this optimum
    """

    X = D.get_data()

    """for dataset X and given k-value run k-means++ and breathing k-means"""
    #
    # k-means++ (scikit-learn)
    #
    if dotext:
        print("k-means++:              ", end="")

    km = sklearn.cluster.KMeans(n_clusters=k, **km_args)
    # start taking time
    t0 = time.time()
    km.fit(X)
    # compute CPU time
    t_km = time.time() - t0

    if dotext:
        print(f" SSE={km.inertia_:.3e}                 t={t_km:6.2f} sec ")

    #
    # breathing k-means
    #
    if dotext:
        print("breathing k-means:      ", end="")

    bkm = BKMeans(n_clusters=k, **bkm_args)
    # start taking time
    t0 = time.time()
    bkm.fit(X)
    # compute CPU time
    t_bkm = time.time() - t0

    # SSE improvement
    if dotext:
        extra_cpu_frac = t_bkm / t_km - 1
        imp = 1 - bkm.inertia_ / km.inertia_
        if imp >= 0:
            perci = f"({imp:.2%} lower)"
        else:
            perci = f"({-imp:.2%} higher!)"
        print(
            f" SSE={bkm.inertia_:.3e} {perci:16s}t={t_bkm:6.2f} sec",
            end="")
        # show extra time
        if extra_cpu_frac >= 0:
            print(f" ({extra_cpu_frac:6.2%} slower)")
        else:
            print(f" ({-extra_cpu_frac:6.2%} faster)")

    return km, bkm, t_km, t_bkm

def fmt(x):
    return f"{x:.3e}"

def fmtsse(x):
    return f"{x:.2%} "+xpdiff2(x,"lower","higher"," SSE")

def fmtcpu(x):
    return f"{x:.2%} "+xpdiff2(x,"faster","slower","")

def prepare_frame(stat,D,k,repeat):
    """prepare statistics using pandas
    """
    df0=pd.DataFrame(stat)      
    xx={}
    xx["data set"]=[D.get_name()]
    xx["n"]=[D.get_n()]
    xx["d"]=[D.get_d()]
    xx["k"]=[k]
    xx["$\phi(\mbox{km}$++)"]=[df0.mean()["err_km"]]
    xx["$\phi(\mbox{bkm})$"]=[df0.mean()["err_bkm"]]
    delta_phi=1-df0.mean()["err_bkm"]/df0.mean()["err_km"]
    xx["$\Delta\phi$"]=[delta_phi]
    xx[r"$t_{\textrm{km++}}$"]=[df0.mean()["cpu_km"]]
    xx[r"$t_{\textrm{bkm}}$"]=[df0.mean()["cpu_bkm"]]
    delta_cpu=1-df0.mean()["cpu_bkm"]/df0.mean()["cpu_km"]
    xx["$\Delta t$"]=[delta_cpu]
    xx["runs"]=repeat
    df=pd.DataFrame(xx)
    df.reset_index(drop=True, inplace=True)
    df2=df.copy()
    df2.reset_index(drop=True, inplace=True)
    df2=df2.style.format({
        df2.columns[6]:fmtsse,
        df2.columns[9]:fmtcpu,
        df2.columns[7]:"{:.2f} sec",
        df2.columns[8]:"{:.2f} sec",
        df2.columns[4]:fmt,
        df2.columns[5]:fmt,
        }).hide_index()
    if delta_phi >= 0:
        # ok (better or equal)
        phi_col='#afa'
    else:
        # worse
        phi_col='#faa'
    cell_err = { 
        'selector': '.col6.row0',
        'props': [('background-color', phi_col)],
    }
    cell_header = {  
        'selector': 'th',
        'props': [('background-color', "#ddd"),('color','black')],
    }
    if delta_cpu > 0:
        cpu_col='#afa'
    else:
        cpu_col='#faa'
    cell_cpu = {  
        'selector': '.col9.row0',
        'props': [('background-color', cpu_col)]
    }
    df2.set_properties(**{'background-color': '#eee',
                           'color': 'black',
                           'border-color': 'white'})
    df2.set_table_styles([
        {'selector': 'th', 'props': [('border-left','1px solid white')]},
        {'selector': 'td', 'props': [('border-left','1px solid #aaa')]},
        {'selector': 'td:nth-child(1)', 'props': [('border-left','0px solid red')]},
        cell_err,cell_cpu,cell_header])
    return df, df2

def solve_problem(D, k=0, 
    doplot=None, 
    dotext=None,
    retval=False, 
    repeat=1,
    km_args={}, 
    bkm_args={}, 
    show_stat=1,
    **kwargs):
    """run km++ and breathing k-means on the given data set D
    D: Dataset
    k: k
    doplot: show plot?
    dotext: show text output?
    retval: return km, bkm and fig
    repeat: number of expriments to perform (this is not "n_init")
    km_args: parameters for KMeans, e.g., {"tol":1E-6}
    bkm_args: parameters for BKMeans, e.g., {"m":10}
    showstat: show statistics
    **kwargs: parameters for doplot()
    """
    if repeat==1:
        dotext=0
    if 'tightlayout' not in kwargs:
        kwargs['tightlayout'] = 1
    if 'pad' not in kwargs:
        kwargs['pad'] = 1
    if dotext is None:
        # ouput text per default
        dotext = True
    if doplot is None:
        # if only one run, plot per default
        doplot = repeat == 1
    assert isinstance(k, int), "k has type" + str(type(k))
    assert isinstance(k, int) and k >= 0
    # data container
    stat = {
        "err_km": [],
        "err_bkm": [],
        "cpu_km": [],
        "cpu_bkm": []
    }
    for i in range(repeat):
        #
        # one or several runs
        #
        km, bkm, t_km, t_bkm = run_models(
            D=D, k=k, dotext=dotext,
            km_args=km_args, bkm_args=bkm_args)  # NOSONAR
        stat["err_km"].append(km.inertia_)
        stat["err_bkm"].append(bkm.inertia_)
        stat["cpu_km"].append(t_km)
        stat["cpu_bkm"].append(t_bkm)

    if(repeat >= 1):
        #
        # statistical summary
        #
        df, df2=prepare_frame(stat,D,k,repeat)
    else:
        df2 = None
    if show_stat:
        display(df2)
    if doplot:
        fig = aux.do_fig(
            km=km,
            bkm=bkm,
            D=D,
            **kwargs)
    else:
        fig = None
    if retval:
        return km, bkm, fig, df2
