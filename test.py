import numpy as np
from hmmlearn import hmm
from mySSA import mySSA
import pandas as pd

"""
Existing implementations for some functions for comparision
"""


def log_likelihood_control(X, k, means, cov):
    A = X.reshape(-1, 1)
    remodel = hmm.GaussianHMM(n_components=k, init_params="")
    remodel.covars_ = cov
    remodel.means_ = means
    res = remodel._compute_log_likelihood(A)
    return res


def forward_control(loglikelihood, start, transition):
    n, k = loglikelihood.shape
    remodel = hmm.GaussianHMM(n_components=k, init_params="")
    remodel.startprob_ = start
    remodel.transmat_ = transition
    res = remodel._do_forward_pass(loglikelihood)
    return res


def backward_control(loglikelihood, transition):
    _, k = loglikelihood.shape
    remodel = hmm.GaussianHMM(n_components=k, init_params="")
    # startprops dont matter for backwards
    remodel.startprob_ = np.zeros(k)
    remodel.transmat_ = transition
    res = remodel._do_backward_pass(loglikelihood)
    return res


def diagonal_avg_control(Xi):
    ts = pd.DataFrame([1, 2, 3])
    ssa = mySSA(ts)
    comp = ssa.diagonal_averaging(Xi)
    return comp
