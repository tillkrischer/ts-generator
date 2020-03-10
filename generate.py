#!/usr/bin/env python3

import argparse
import numpy as np
import sklearn.cluster
import matplotlib.pyplot as plt


def parseCSV(filename):
    """
    read the specified file, 
    split along commas and cast to float
    return numpy array
    """
    with open(filename) as f:
        datafile = f.read()
        lines = datafile.split("\n")
        X = []
        for l in lines:
            items = l.split(",")
            fs = []
            for item in items:
                try:
                    num = float(item)
                    fs.append(num)
                except:
                    pass
            if len(fs) > 0:
                X.append(fs)
        Xnp = np.array(X)
        return Xnp


def logsumexp(ns):
    """
    log sum exp trick 
    to avoid underflow
    """
    max = np.max(ns)
    # negative inf case
    if np.isneginf(max):
        return float("-inf")
    ds = ns - max
    sumOfExp = np.exp(ds).sum()
    return max + np.log(sumOfExp)


def lognormalize_gamma(g):
    """
    normalize gamma along the rows
    since in log space we subtract instead of divide
    so that the rows sum up to 0 (instead of 1 in regular space)
    """
    l = g.shape[0]
    a = np.zeros(l)

    for i in range(l):
        a[i] = logsumexp(g[i])

    g_norm = g - a.reshape(-1, 1)
    return np.exp(g_norm)


def log_normal_pdf(x, mu, sigmasq):
    """
    compute log of normal distribution pdf
    """
    dsq = (x - mu)**2
    lp = - 0.5 * np.log(sigmasq * 2 * np.pi) - 0.5 * dsq / sigmasq
    return lp


def log_likelihood(X, k, means, cov):
    """
    compute log likelihood for every datapoint of every possible state
    """
    ll = np.zeros((len(X), k))
    for i in range(len(X)):
        for j in range(k):
            ll[i, j] = log_normal_pdf(X[i], means[j], cov[j])

    return ll


def forward(loglikelihood, start, transition):
    """
    perform forward pass to compute alpha
    """
    n, k = loglikelihood.shape
    # taking the log of 0 gives -inf, which is fine in this case
    with np.errstate(divide="ignore"):
        logstart = np.log(start)
        logtrans = np.log(transition)
    alpha = np.zeros((n, k))
    temp = np.zeros(k)

    for i in range(k):
        alpha[0, i] = logstart[i] + loglikelihood[0, i]

    for t in range(1, n):
        for j in range(k):
            for i in range(k):
                temp[i] = alpha[t-1, i] + logtrans[i, j]
            alpha[t, j] = logsumexp(temp) + loglikelihood[t, j]
    return alpha


def backward(loglikelihood, transition):
    """
    perform backward pass to compute beta
    """
    n, k = loglikelihood.shape
    # taking the log of 0 gives -inf, which is fine in this case
    with np.errstate(divide="ignore"):
        logtrans = np.log(transition)
    beta = np.zeros((n, k))
    temp = np.zeros(k)

    for i in range(k):
        beta[-1, i] = 0.0

    for t in range(n-2, -1, -1):
        for i in range(k):
            for j in range(k):
                temp[j] = logtrans[i, j] + loglikelihood[t+1, j] + beta[t+1, j]
            beta[t, i] = logsumexp(temp)
    return beta


def compute_trans(a, b, ll, transition):
    """
    computes log of xi
    """
    n, k = ll.shape
    with np.errstate(divide="ignore"):
        logtrans = np.log(transition)
    logxisum = np.full((k, k), float("-inf"))
    denom = logsumexp(a[-1])

    for t in range(n-1):
        for i in range(k):
            for j in range(k):
                logxi = a[t, i] + logtrans[i, j] + \
                    ll[t+1, j] + b[t+1, j] - denom
                logxisum[i, j] = logsumexp([logxisum[i, j], logxi])
    return logxisum


def normalize(a, axis=None):
    """
    normalize along axis
    making sure not to divide by zero
    """
    a_sum = a.sum(axis)
    if axis and a.ndim > 1:
        a_sum[a_sum == 0] = 1
        shape = list(a.shape)
        shape[axis] = 1
        a_sum.shape = shape

    return a / a_sum


def update_params(sumgamma, xgammasum, xgammasumsquared, start, trans):
    """
    M-step
    copute new parameters from computed expectations
    """
    norm_trans = normalize(trans, axis=1)
    norm_start = normalize(start)

    means = xgammasum / sumgamma
    # cov is based on newly computed means
    num = (means**2 * sumgamma - 2 * means *
           xgammasum + xgammasumsquared)

    # is prior necessary ?
    cov = (num + 0.01) / np.maximum(sumgamma, 1e-5)

    return norm_start, norm_trans, means, cov


def init(components, X):
    """
    initialize HMM parameters 
    """
    # join sequences if necessary
    if X.ndim == 2:
        X = np.concatenate(X)
    # initial means using kmeans
    kmeans = sklearn.cluster.KMeans(n_clusters=components)
    kmeans.fit(X.reshape(-1, 1))
    means = kmeans.cluster_centers_.reshape(-1)

    # initial covariance
    covar = np.cov(X)
    cov = np.tile(covar, components)
    # init start probability
    startprop = np.tile(1/components, components)
    # init transition matrix
    transmat = np.tile(1/components, components **
                       2).reshape(components, components)
    return means, cov, startprop, transmat


def fit(A, k, it, verbose):
    """
    fit a HMM to training data A
    returns learned parameters
    """
    means, cov, startprop, transmat = init(k, A)

    for i in range(it):
        e_sumgamma = np.zeros(k)
        e_xgammasum = np.zeros(k)
        e_xgammasumsquared = np.zeros(k)
        e_start = np.zeros(k)
        e_trans = np.zeros((k, k))
        prop = 0
        for X in A:

            n = len(X)
            loglikelihood = log_likelihood(X, k, means, cov)

            # forward pass
            logalpha = forward(loglikelihood, startprop, transmat)
            prop += logsumexp(logalpha[-1])

            # backward pass
            logbeta = backward(loglikelihood, transmat)

            loggamma = logalpha + logbeta
            gamma = lognormalize_gamma(loggamma)

            e_start += gamma[0]
            e_sumgamma += np.einsum("ij->j", gamma)
            e_xgammasum += np.dot(gamma.T, X)
            e_xgammasumsquared += np.dot(gamma.T, X**2)
            if n > 1:
                logtrans = compute_trans(
                    logalpha, logbeta, loglikelihood, transmat)
                e_trans += np.exp(logtrans)

        if verbose:
            print(f"it: {i}  logprop: {prop}")

        startprop, transmat, means, cov = update_params(
            e_sumgamma, e_xgammasum, e_xgammasumsquared, e_start, e_trans)

    return startprop, transmat, means, cov


def generate_hmm(startprop, transmat, means, cov, l):
    """
    generate time series from HMM parameters
    """
    k = startprop.size
    start = np.random.choice(range(k), p=startprop)
    mu = means[start]
    hiddenstates = [start]
    sigma = np.sqrt(cov[start])
    first_output = np.random.normal(mu, sigma)
    samples = [first_output]
    for _ in range(l-1):
        prevstate = hiddenstates[-1]
        nextstate = np.random.choice(range(k), p=transmat[prevstate])
        hiddenstates += [nextstate]
        mu = means[nextstate]
        sigma = np.sqrt(cov[nextstate])
        next_output = np.random.normal(mu, sigma)
        samples += [next_output]
    return samples


def embed(X, k):
    """
    embed time series X with a windows size of k
    """
    l = X.size
    C = np.zeros((k, l - k + 1))
    for i in range(k):
        for j in range(l - k + 1):
            C[i, j] = X[j+i]
    return C


def decompose(X):
    """
    decompose matrix X into eigenvectors
    and corresponding X components
    """
    rank = np.linalg.matrix_rank(X)
    Xsq = X @ X.T
    U, S, _ = np.linalg.svd(Xsq)
    s = np.sqrt(S)

    Xcomp = []
    eigenvectors = []

    for i in range(rank):
        egv = U[:, i]
        lamb = s[i]
        Q = X.T@(egv/lamb)
        P = lamb*egv
        Xcomp += [np.outer(P, (Q.T))]
        eigenvectors += [egv]
    return Xcomp, eigenvectors


def diagonal_avg(M):
    """
    avarage diagonals to turn matrix form of component
    back into time series
    """
    k, l = M.shape
    res = []
    for i in range(l + k - 1):
        count = 0
        s = 0
        for j in range(k):
            x = j
            y = i - j
            if x >= 0 and x < k and y >= 0 and y < l:
                count += 1
                s += M[x, y]
        res += [s/count]
    return res


def compute_R(eigenv, ncomponents):
    """
    compute recurrence matrix X from eigenvectors
    uses first ncomonents eigenvectors
    """
    vsq = 0
    R = np.zeros(eigenv[0].size-1)
    for i in range(ncomponents):
        eig = eigenv[i]
        pi = eig[-1]
        vsq += pi**2
        R += pi * eig[:-1]
    R = R / (1 - vsq)
    return R


def reconstruct(Xcomp, ncomponents):
    """
    combine first ncomponents components to 
    reconstruct the time series
    """
    Xfull = np.zeros(Xcomp[0].shape)
    for i in range(ncomponents):
        X = Xcomp[i]
        Xfull += X

    x_tilde = diagonal_avg(Xfull)
    return x_tilde


def forecast(x_tilde, R, ncomponents, steps):
    """
    forecast additional steps of the time series
    using the recurrence method and matrix R
    """
    n = len(x_tilde)
    d = R.size
    new_series = x_tilde[:]
    for i in range(steps):
        x = R.T @ new_series[i+n-d:i+n]
        new_series += [x]
    return new_series


def handle_hmm_learn(training, k, it, verbose, l):
    """
    learn parameters using training data
    then generate time series according to parameters
    """
    X = parseCSV(training)
    startprop, transmat, means, cov = fit(X, k, it, verbose)
    ts = generate_hmm(startprop, transmat, means, cov, l)
    return ts


def handle_hmm_param_simple(means, l):
    """
    pick default values for other parameters given means
    then generate time series according to parameters
    """
    # means are given
    meansnp = np.array(means)
    k = meansnp.size

    # even start prop
    startprop = np.tile(1/k, k)

    # create a balanced transition matrix, that favors staying in the current state over changing state
    trans = np.ones((k, k)) + k * np.eye(k)
    transmat = trans / sum(trans)

    # covariance is 1/5 of the difference between largest and smallest mean
    c = ((max(meansnp) - min(meansnp)) / 5)**2
    cov = np.tile(c, k)

    ts = generate_hmm(startprop, transmat, meansnp, cov, l)
    return ts


def handle_hmm_param(m, c, s, t, l):
    """
    generate time series according to HMM parameters
    """
    k = len(m)
    means = np.array(m)
    cov = np.array(c)
    startprop = np.array(s)
    transmat = np.array(t).reshape((k, k))

    ts = generate_hmm(startprop, transmat, means, cov, l)
    return ts


def handle_ssa(orig, window, components, l):
    """
    perform SSA on given time series
    then forecast up to required length 
    return combined reconstruction and forecast as
    new time series
    """
    X = parseCSV(orig).reshape(-1)
    n = X.size

    C = embed(X, window)
    Xcomp, eigenv = decompose(C)

    x_tilde = reconstruct(Xcomp, components)

    R = compute_R(eigenv, components)

    additional_samples = max(0, l-n)
    fc = forecast(x_tilde, R, components, additional_samples)
    return fc[:l]


def scale_time_series(ts, low, high):
    """
    scale result so that new lowest value is low
    and highest is high
    """
    tsmin = min(ts)
    tsdiff = max(ts) - tsmin
    newdiff = high - low
    newts = []
    for x in ts:
        scaled = low + ((x - tsmin) / tsdiff) * newdiff
        newts += [scaled]
    return newts


def format_output(ts, time_inc, output_file):
    """
    write generated time series in csv form either to file
    or stdout
    """
    output = "Time, Value"
    output += "\n"
    for i, v in enumerate(ts):
        if not i == 0:
            output += "\n"
        output += f"{i*time_inc}, {v}"
    if output_file:
        with open(output_file, "w") as f:
            f.write(output)
    else:
        print(output)


def display_timeseries(ts, time_inc):
    """
    display generated timeseries using
    matplotlib
    """
    n = len(ts)
    # x = range(0, n * time_inc, time_inc)
    x = [time_inc * x for x in range(n)]
    plt.plot(x, ts)
    plt.show()


def handle_args():
    """
    main function
    handle all cmd parameters
    """
    parser = argparse.ArgumentParser(description='Generate Time Series')
    # general arguments
    generalargs = parser.add_argument_group("general")
    generalargs.add_argument(
        '--method', choices=['hmm-learn', 'hmm-param-simple', 'hmm-param', 'ssa'], required=True, help="choose")
    generalargs.add_argument('--length', metavar='n', required=True,
                             type=int, help='number of samples')
    generalargs.add_argument('-o', '--output', metavar='output.csv',
                             help='output file, stdout if none')
    generalargs.add_argument('--time-increment', metavar='t', default=1,
                             type=float, help='time increments')
    generalargs.add_argument(
        '--scaling', type=float, metavar=("min", "max"), nargs=2, help='scale result by so that the given min and max are fullfilled')
    generalargs.add_argument('--display', action='store_true',
                             help='display using matplotlib')

    # hmm-learn related arguments
    hmmlearnargs = parser.add_argument_group("hmm-learn")
    hmmlearnargs.add_argument(
        '--hmm-training', metavar='data.csv', help='training data: one series per line')
    hmmlearnargs.add_argument('--hmm-components', metavar='k',
                              type=int, help='hmm #components')
    hmmlearnargs.add_argument('--hmm-iterations', metavar='N',
                              type=int, help='learning #iterations')
    hmmlearnargs.add_argument('--show-progress', action='store_true',
                              help='progress during learning')

    # hmm-param-simple related arguments
    hmmparamsimpleargs = parser.add_argument_group("hmm-param-simple")
    hmmparamsimpleargs.add_argument(
        '--hmm-simple-means', type=float, metavar="m", nargs="+", help='set of means')

    # hmm-param related arguments
    hmmparamargs = parser.add_argument_group("hmm-param")
    hmmparamargs.add_argument(
        '--hmm-means', type=float, metavar="m",  nargs="+", help='set of means len=k')
    hmmparamargs.add_argument(
        '--hmm-cov', type=float, metavar="c", nargs="+", help='set of covariances len=k')
    hmmparamargs.add_argument(
        '--hmm-start-prop', type=float, metavar="s", nargs="+", help='start probabilitis len=k')
    hmmparamargs.add_argument(
        '--hmm-trans-prop', type=float, metavar="t", nargs="+", help='transition probabilitis len=k**2')

    # ssa related arguments
    ssaargs = parser.add_argument_group("ssa")
    ssaargs.add_argument(
        '--ssa-original', metavar='data.csv', help='base time series')
    ssaargs.add_argument(
        '--ssa-window', type=int, metavar='k', help='should be a multiple of the seasonality')
    ssaargs.add_argument(
        '--ssa-components', type=int, metavar='c', help='# of components used in reconstruction')

    args = parser.parse_args()

    timeseries = []

    if args.method == "hmm-learn":
        if not args.hmm_training or not args.hmm_components or not args.hmm_iterations:
            print(
                "--hmm-training, --hmm-components, and --hmm-iterations required for hmm-learn")
            return
        timeseries = handle_hmm_learn(
            args.hmm_training, args.hmm_components, args.hmm_iterations, args.show_progress, args.length)

    if args.method == "hmm-param-simple":
        if not args.hmm_simple_means:
            print("--hmm-simple-means required for hmm-param-simple")
            return
        timeseries = handle_hmm_param_simple(
            args.hmm_simple_means, args.length)

    if args.method == "hmm-param":
        if not args.hmm_means or not args.hmm_cov \
                or not args.hmm_start_prop or not args.hmm_trans_prop:
            print(
                "--hmm-means, --hmm-cov, --hmm-start-prop, and --hmm-trans-prop required for hmm-param-simple")
            return
        if not len(args.hmm_means) == len(args.hmm_cov) \
                or not len(args.hmm_means) == len(args.hmm_start_prop) \
                or not len(args.hmm_means)**2 == len(args.hmm_trans_prop):
            print("dimensions don't match")
            return
        timeseries = handle_hmm_param(
            args.hmm_means, args.hmm_cov, args.hmm_start_prop, args.hmm_trans_prop, args.length)

    elif args.method == "ssa":
        if not args.ssa_original or not args.ssa_window or not args.ssa_components:
            print("--ssa-original, --ssa-window, and --ssa-components required for ssa")
            return
        timeseries = handle_ssa(
            args.ssa_original, args.ssa_window, args.ssa_components, args.length)

    if args.scaling:
        low, high = args.scaling
        timeseries = scale_time_series(timeseries, low, high)

    format_output(timeseries, args.time_increment, args.output)

    if args.display:
        display_timeseries(timeseries, args.time_increment)


handle_args()
