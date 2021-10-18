import timeit
import numpy as np
import collections
from . import ndft

def benchmark(title=None):
    r"""
    Returns the current value of the global timer, then restarts it.

    If a title is supplied, displays the current value as well. Note
    that the global timer ``TIMER`` is started on module load.

    Args:
        title (str): Optional timer title to display.

    Returns:
        float: The current value of the timer in seconds, to millisecond precision.
    """
    global TIMER
    t_diff = timeit.default_timer() - TIMER
    TIMER = timeit.default_timer()
    if title is not None:
        print(title + ": " + str(t_diff))
    return t_diff

def to_db(data):
    r"""
    Converts data from linear to logarithmic units.

    If data is supplied in Watts, output is in dBmW.

    Args:
        data (NumPy nD array): Data to convert.

    Returns:
        NumPy nD array: The converted data.
    """
    # converts data (in Watts) to dBmW
    return 30 + 10*np.log10(data)

def direct_acf(data,timestamps,maxlag,mincount=10,retcounts=False,removemean=False):
    r"""
    Estimates the (normalised) autocovariance of the time series :math:`F_j(t)`, sampled at the given times `t_i`.

    The autocovariance at all sufficiently common gaps of length :math:`0\leq\tau\leq m` is estimated. 
    Written by Keith Briggs.

    Args:
        data (2D NumPy complex array): The input data: index ``(i,j)`` is :math:`F_j(t_i)`
        timestamps (Numpy float array): The times :math:`t_i`; must be sorted in increasing order.
            Must be relatively low precision (e.g. integers) for meaningful results.
        maxlag (float): The maximal lag :math:`m` to measure the autocovariance at.
        mincount (int): The minimal number of gaps of a certain lag that must exist for
            the autocovariance at that lag to be estimated.
        retcounts (bool): If `True`, additionally returns the number of gaps at each lag.
        removemean (bool): If `False`, does not remove the mean of the data and so calculates
            autocorrelation instead.

    Returns:
        dict from float to complex: Dictionary mapping lags to autocovariances. If a key is missing,
        then either the lag does not appear often enough to get a reasonable estimate, or the lag is out of range.
    """
    # data can be a k-index array; the first index is the time axis
    # timestamps must be a monotonically increasing sequence
    # retcounts=True returns the number of gaps as well
    k={}; sx={}; sy={}; sxx={}; syy={}; sxy={}
    data_shape=data[0].shape
    for i,x in enumerate(data):
        assert x.shape==data_shape
        t0=timestamps[i]
        for j,y in enumerate(data[i+1:]):
            lag=timestamps[i+1+j]-t0
            assert lag>0
            if lag>maxlag: break
            if lag not in k:
                k[lag]=0
                sx[lag] =np.zeros(data_shape,dtype=complex)
                sy[lag] =np.zeros(data_shape,dtype=complex)
                sxx[lag]=np.zeros(data_shape,dtype=complex)
                syy[lag]=np.zeros(data_shape,dtype=complex)
                sxy[lag]=np.zeros(data_shape,dtype=complex)
            k[lag]+=1
            sx[lag]+=x
            sy[lag]+=y
            sxx[lag]+=x*np.conj(x)
            syy[lag]+=y*np.conj(y)
            sxy[lag]+=x*np.conj(y)
    ac=collections.OrderedDict()
    ac[0]=np.ones(data_shape,dtype=complex)
    lags=[lag for lag in k if k[lag]>=mincount]
    lags.sort()
    for lag in lags:
        n=k[lag]
        covar=n*sxy[lag]
        var_x=n*sxx[lag]
        var_y=n*syy[lag]
        if removemean:
            covar=covar-sx[lag]*np.conj(sy[lag])
            var_x=var_x-sx[lag]*np.conj(sx[lag])
            var_y=var_y-sy[lag]*np.conj(sy[lag])
        ac[lag]=covar/np.sqrt(var_x*var_y)

    if retcounts:
        return (ac, k)
    return ac

def plot_sampled_data(x, ys, ax, label="Unknown", db=False, symmetric=False, **kwargs):
    r"""
    Plots the mean of the 2D data ``ys`` against ``xs`` on axes ``ax``, with error bars of one standard deviation.

    Requires pyplot to be installed.

    Args:
        x (NumPy array): The x-axis (independent) data.
        y (2D NumPy array): The y-axis (dependent) data; the columns are the individual data series.
        ax (Pyplot axes object): The axes to plot on.
        label (str): The data label.
        db (bool): If `True`, converts the y-axis to logarithmic units.
        symmetric (True): If `True`, discards the second half of the data. Use if data is symmetric.
    Any keyword arguments are passed to the ``pyplot.plot`` and ``pyplot.fillbetween`` methods. Do not pass ``alpha``.
    """
    # remove first (negative) half
    if symmetric:
        end = len(x)//2
        x = x[:end]
        ys = ys[:,:end]

    mean = np.mean(ys, axis=0)

    # convert to decibels
    if db:
        mean = to_db(mean)
        ys = to_db(ys)
    
    stddev = np.std(ys, axis=0)
    ax.fill_between(x, mean-stddev, mean+stddev, alpha=.5, **kwargs)
    ax.plot(x, mean, label=label, **kwargs)

def gen_paths(N, max_depth):
    r"""
    Generates all (simple) paths of length :math:`k` or less through 
    the complete graph with :math:`N` vertices.

    Equivalenty, generates all combinations of :math:`\{0,\dots,N-1\}` of length less 
    than :math:`k` where all adjacent values are distinct. Used to find possible signal
    paths in a multipath model.

    Args:
        N (int): The number of vertices in the graph.
        max_depth (int): The maximum path length.

    Returns:
        generator of lists of integers: A generator for the paths, specified by the vertices visited in order.
    """
    return paths([], N, min(N, max_depth))

def paths(cur, N, rem_depth):
    r"""Recursive helper function for :meth:`gen_paths`."""
    if rem_depth == 0:
        yield cur
    else:
        last = None if len(cur) == 0 else cur[-1]
        for i in range(N):
            if i != last:
                # this is a bit cursed but it works
                for p in paths(cur + [i], N, rem_depth - 1):
                    yield p
                yield cur

def psd_with_freqs(data, **kwargs):
    r"""
    Wrapper for :meth:`ndft.calc_psd` that also returns the frequency axis.
    Returns:
        (NumPy real array, NumPy complex array): (frequencies, PSD)
    """
    max_freq = kwargs.get('max_freq', 1)

    psds = ndft.calc_psd(data, **kwargs)
    N = len(psds)
    freqs = unif_pts(N, max_freq)
    return (freqs, psds)

def acf_with_lags(threshold=0.1, **kwargs):
    r"""
    Wrapper for :meth:`ndft.calc_acf` that also returns the period axis.

    Args:
        threshold (float): Minimum lag strength needed for an ACF estimate
            to be returned at that lag. Must be in :math:`[0,1)`.
    Returns:
        (NumPy real array, NumPy complex array): (lags, ACF)
    """
    max_freq = kwargs.get('max_freq', 1)
    if not 'max_freq' in kwargs:
        max_freq = 1
    if kwargs.get('lag_strengths') is None:
        kwargs['lag_strengths'] = ndft.lag_strength(**kwargs)
    ls = kwargs['lag_strengths']
    inds = (ls > threshold)
    acf = ndft.calc_acf(**kwargs)
    if (L := len(inds)) < (M := len(acf)):
        inds = np.tile(inds, M//L + 1)[:M]
    lags = np.arange(len(inds))/max_freq
    return (lags[inds], acf[inds])

def unif_pts(N, max=1):
    r"""
    Generates :math:`N` uniformly-spaced points on :math:`[0,m)`.

    Used to generate frequency axis for PSD.

    Args:
        N (int): The value of :math:`N`.
        max (float): The value of :math:`m`. Must be positive.
    """
    return np.arange(N)/N*max

# start timing on import
TIMER = timeit.default_timer()