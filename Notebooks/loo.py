"""
Some information on infering p_loo and pareto k value
http://discourse.mc-stan.org/t/a-quick-note-what-i-infer-from-p-loo-and-pareto-k-values/3446
"""

from pymc3.model import modelcontext
from pymc3.diagnostics import effective_n
from pymc3 import stats as pmstat
from scipy.misc import logsumexp
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

def loo(trace, model=None, reff=None, progressbar=False):
    """Calculates leave-one-out (LOO) cross-validation for out of sample
    predictive model fit, following Vehtari et al. (2015). Cross-validation is
    computed using Pareto-smoothed importance sampling (PSIS).

    Parameters
    ----------
    trace : result of MCMC run
    model : PyMC Model
        Optional model. Default None, taken from context.
    reff : float
        relative MCMC efficiency, `effective_n / N` i.e. number of effective
        samples divided by the number of actual samples. Computed from trace by
        default.
    progressbar: bool
        Whether or not to display a progress bar in the command line. The
        bar shows the percentage of completion, the evaluation speed, and
        the estimated time to completion

    Returns
    -------
    df_loo: pandas.DataFrame 
        Estimation and standard error of `elpd_loo`, `p_loo`, and `looic`
    pointwise: dict
        point-wise value of `elpd_loo`, `p_loo`, `looic` and pareto shape `k`
    """
    model = modelcontext(model)

    if reff is None:
        if trace.nchains == 1:
            reff = 1.
        else:
            eff = effective_n(trace)
            eff_ave = pmstat.dict2pd(eff, 'eff').mean()
            samples = len(trace) * trace.nchains
            reff = eff_ave / samples

    log_py = pmstat._log_post_trace(trace, model, progressbar=progressbar)
    if log_py.size == 0:
        raise ValueError('The model does not contain observed values.')

    shape_str = ' by '.join(map(str, log_py.shape))
    print('Computed from ' + shape_str + ' log-likelihood matrix')

    lw, ks = pmstat._psislw(-log_py, reff)
    lw += log_py

    elpd_loo_i = logsumexp(lw, axis=0)
    elpd_loo = elpd_loo_i.sum()
    elpd_loo_se = (len(elpd_loo_i) * np.var(elpd_loo_i)) ** 0.5

    loo_lppd_i = - 2 * elpd_loo_i
    loo_lppd = loo_lppd_i.sum()
    loo_lppd_se = (len(loo_lppd_i) * np.var(loo_lppd_i)) ** 0.5

    lppd_i = logsumexp(log_py, axis=0, b=1. / log_py.shape[0])
    p_loo_i = lppd_i - elpd_loo_i
    p_loo = p_loo_i.sum()
    p_loo_se = (len(p_loo_i) * np.var(p_loo_i)) ** 0.5

    df_loo = (pd.DataFrame(dict(Estimate=[elpd_loo, p_loo, loo_lppd],
                                SE=[elpd_loo_se, p_loo_se, loo_lppd_se]))
                .rename(index={0: 'elpd_loo', 
                               1: 'p_loo', 
                               2: 'looic'}))
    pointwise = dict(elpd_loo=elpd_loo_i,
                        p_loo=p_loo_i,
                        looic=loo_lppd_i, 
                        ks=ks)
    return df_loo, pointwise

def ks_summary(ks):
    kcounts, _ = np.histogram(ks, bins=[-np.Inf, .5, .7, 1, np.Inf])
    kprop = kcounts/len(ks)*100
    df_k = (pd.DataFrame(dict(_=['(good)', '(ok)', '(bad)', '(very bad)'],
                              Count=kcounts,
                              Pct=kprop))
            .rename(index={0: '(-Inf, 0.5]', 
                           1: ' (0.5, 0.7]', 
                           2: '   (0.7, 1]',
                           3: '   (1, Inf)'}))
    
    if np.sum(kcounts[1:])==0:
        print("All Pareto k estimates are good (k < 0.5)")
    elif np.sum(kcounts[2:])==0:
        print("All Pareto k estimates are ok (k < 0.7)")
    else:
        print("Pareto k diagnostic values:")
        print(df_k)

    return df_k

def compare(list_pointwise):
    if len(list_pointwise) < 2:
        raise ValueError('`compare` requires at least two models.')
    else:
        pa = list_pointwise[0]
        filter_col = [key for key in pa.keys() if 'elpd' in key.lower()][0]
        Na = pa[filter_col].shape[0]
        comp = []
        indexs = []
        for i, pb in enumerate(list_pointwise[1:]):
            Nb = pb[filter_col].shape[0]
            if Na != Nb:
                raise ValueError("Models don't have the same number of data points. "
                                 "Found N_1 = {} and N_2 = {}.".format(Na, Nb))
            sqrtN = np.sqrt(Na)
            diff = np.asarray(pb[filter_col] - pa[filter_col])
            comp.append(dict(elpd_diff=np.sum(diff), se=sqrtN*np.std(diff)))
            indexs.append('m{}-m0'.format(i+1))
    return pd.DataFrame(comp, index=indexs)

def plot_khat(ks, ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 5))

    ax.hlines([0, .5, .7, 1], xmin=-1, xmax=len(ks)+1,
              alpha=.25, color='r')

    alphas = .5 + .5*(ks>.5)
    rgba_c = np.zeros((len(ks), 4))
    rgba_c[:, 2] = .8
    rgba_c[:, 3] = alphas
    ax.scatter(np.arange(len(ks)), ks, c=rgba_c, marker='+')

    ax.set_xlabel('Data point')
    ax.set_ylabel(r'Shape parameter $\kappa$')
    return ax
        