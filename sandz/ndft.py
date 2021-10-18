import numpy as np
import scipy.fft as fft
import sandz.periodic as periodic

IMPORT_STATUS = "none"
DEFAULT_NDFT = None
MAX_LAG = 50
try:
    import pynfft as nfft_c
except:
    print("Could not find nonessential package pynfft. Try running \"pip install pynfft\".")

try:
    import nfft as nfft_py
    IMPORT_STATUS = "py"
except:
    print("Could not find nonessential package nfft. Try running \"pip install nfft\".")

def ndft_mat(fs, ts):
    r"""
    Generates the NDFT matrix mapping data at times :math:`t_i` to frequencies :math:`f_j`.

    Args:
        ts (NumPy float array): The times :math:`t_i`.
        fs (NumPy float array): the frequencies :math:`f_j` .

    Returns:
        2D NumPy complex array: The NDFT matrix.
    """
    return np.exp(-2*np.pi*1j*fs[:, None]*ts)

def spec_win(times, **kwargs):
    r"""
    Calculates the spectral window function of :math:`t_i`. Arguments and returns are as in :meth:`calc_psd`.
    
    Equivalently, estimates the PSD of :math:`F\equiv1/T`, sampled at times :math:`t_i`, using the NDFT,
    where :math:`T` is the number of samples.
    """
    T = len(times)
    psd = calc_psd(np.ones(T), times=times, **kwargs)
    return psd / psd[0]

def lag_strength(**kwargs):
    r"""
    Calculates the lag strengths of :math:`t_i`. Arguments and returns are as in 
    :meth:`calc_acf`, except that `data`, `psds` and `norm_lag_str` cannot be given.
    
    Equivalently, estimates the autocovariance of :math:`F\equiv1`, sampled at times :math:`t_i`, 
    using the NDFT, where :math:`T` is the number of samples.
    """
    # remove illegal args
    kwargs.pop('data', None)
    kwargs.pop('psd', None)
    kwargs.pop('norm_lag_str', None)

    if 'times' in kwargs:
        T = len(kwargs['times'])
    elif 'mask' in kwargs:
        T = np.count_nonzero(kwargs['mask'])
    else: 
        raise Exception("No times or mask argument found: cannot calculate lag strengths.")
    return calc_acf(data=np.ones(T), norm_lag_str=False, **kwargs)

def exe_ndft(data, max_freq, times, os=10):
    r"""
    Calculates the exact NDFT of a signal :math:`F(t)` sampled at :math:`T` times :math:`t_i`.
    
    The output spectrum is sampled at :math:`T\cdot k` equispaced frequencies on :math:`[0,r)`,
    where :math:`k` is the oversampling factor.

    Args:
        data (NumPy complex array): The sample values :math:`F(t_i)`.
        max_freq (float): The maximum frequency :math:`r`; must be positive.
        times (NumPy real array): The times :math:`t_i`.
        os (int): The oversampling factor :math:`k`. Must be a positive integer.
    """
    N = len(times) * os
    return ndft_mat(np.arange(N)/N*max_freq, times) @ data

def exe_nfft_py(data, max_freq, times, os=10, **kwargs):
    r"""Calculates a fast approximation to the NDFT using the NFFT Python package. See :meth:`exe_ndft`."""
    N = len(times) * os
    ndft = nfft_py.nfft_adjoint(-times/N*max_freq, data, N, **kwargs)
    return np.concatenate([ndft[N//2:],ndft[:N//2]])

def exe_nfft_c(data, max_freq, times, os=10):
    r"""
    Calculates a fast approximation to the NDFT using the pyNFFT Python wrapper for the NFFT3 C library.
    See :meth:`exe_ndft`.

    Prone to memory errors.
    """
    N = len(times) * os
    plan = nfft_c.NFFT(N, len(times))
    plan.x = -times/N*max_freq
    plan.f = data
    plan.precompute()
    ndft = plan.adjoint()
    return np.concatenate([ndft[N//2:],ndft[:N//2]])

def exe_ndft_fft(data, max_freq, times):
    r"""
    Calculates an exact NDFT using a large zero-interpolated FFT.

    The time domain is rounded to steps of size :math:`1/f_N`, where :math:`f_N` is
    the value of ``max_freq``.
    
    Args:
        times (NumPy real array): The sample times :math:`t_i`, in increasing order.
    """
    temp=np.unique(np.rint(np.array(times)*max_freq).astype(int), return_index=True)
    times, inds = np.unique(np.rint(np.array(times)*max_freq).astype(int), return_index=True)
    times -= times[0]
    L = times[-1] + 1
    data = data[inds]

    gapped_data = np.zeros(L, dtype="complex")
    gapped_data[times] = data
    return fft.fft(gapped_data)

def exe_nfft_per(data, max_freq, mask):
    r"""
    Calculates an exact NDFT for periodic data.
    
    This is a wrapper for :meth:`periodic.nfft_per`. Only frequency range :math:`1` is supported as
    integer times are assumed.
    """
    if not np.isclose(max_freq, 1):
        raise ValueError("Periodic NFFT only supports frequency range 1")
    return periodic.nfft_per(data, mask)

def calc_psd(data, ndft_fun=None, max_freq=1, **kwargs):
    r"""
    Estimates the PSD of a signal :math:`F(t)` using the NDFT at equispaced points on :math:`[0,r)`.

    Args:
        data (NumPy complex array): The values :math:`F(t_i)` of the signal.
        ndft_fun (function): The function that performs the NDFT.
            Must take parameters ``(data, max_freq)``, and return an estimate for the NDFT of :math:`F` 
            at equispaced points on :math:`[0,r)`, in increasing order. Any additional keyword arguments
            are passed to this function. This library provides several implementations:
                * :meth:`exe_ndft`: Exact NDFT
                * :meth:`exe_nfft_py`: Approximation to the NDFT implemented in Python
                * :meth:`exe_nfft_c`: Approximation to the NDFT implemented in C
                * :meth:`exe_ndft_fft`: NDFT evaluated by embedding in a larger FFT, possibly rounding the sample times.
                * :meth:`exe_nfft_per`: Exact NDFT for periodic nonuniform sampling.
        max_freq (float): The maximum frequency :math:`r`; must be positive.

    Returns:
        NumPy real array: An estimate of the PSD of :math:`F(t)` at equispaced points on :math:`[0,r)`.
    """
    if ndft_fun is None:
        ndft_fun = DEFAULT_NDFT    
    spec = ndft_fun(data, max_freq, **kwargs)
    return np.square(np.abs(spec))/len(data)

def calc_acf(data=None, psd=None, lag_strengths=None, max_freq=1, max_lag=MAX_LAG, norm_lag_str=True, norm_var=True, **kwargs):
    r"""
    Estimates the normalised autocorrelation of a signal :math:`F(t)` using the NDFT.

    The autocorrelation is calculated for lags :math:`\tau_k=k\cdot f_{\text{max}}` in the range :math:`0\leq\tau_k<M`.
    Any additional keyword arguments (such as `times`) are passed to the NDFT estimator :func:`calc_psd`.

    Args:
        data (NumPy complex array): The values :math:`F(t_i)` of the signal.
        psd (NumPy real array): Optional pre-calculates PSDs (see :meth:`calc_psd`). If provided, ``data`` is ignored.
            Note that, if this option is used and ``lag_strengths`` is not specified, the varargs passed to the original
            :meth:`calc_psd` call should also be passed to this method, so that the right number of lag strengths can be
            calculated.
        lag_strengths (NumPy real array): Optional pre-calculated lag strengths (see :meth:`lag_strength`). Recommended for
            repeated calls with the same time sampling structure. If not enough lag strengths are provided, they will be
            assumed to be periodic.
        max_freq (float): The period :math:`f_{\text{max}}` of the spectrum of :math:`F` (i.e. twice the Nyquist frequency).
            Should normally have value :math:`1`.
        max_lag (int): The maximum lag :math:`M`.
        norm_lag_str (bool): If `False`, does not perform normalisation by lag strength.
        norm_var (bool): If `False`, does not perform normalisation by variance (value at lag 0).

    Returns:
        NumPy complex array: An estimate of the (normalised) autocorrelation of :math:`F(t)` for the first `M` integer 
        lags.
    """
    kwargs['max_freq'] = max_freq # setting for calc_psd
    num_lags = int(max_lag*max_freq)

    if psd is None:
        if not data is None:
            psd = calc_psd(data, **kwargs)
        else:
            raise Exception("No data or psds argument found: cannot calculate autocorrelation.")
    acf = fft.fft(psd)[:num_lags]

    # normalise
    if norm_var:
        acf /= acf[0]
    if norm_lag_str:
        if lag_strengths is None:
            lag_strengths = lag_strength(max_lag=max_lag, **kwargs)
        if L := len(lag_strengths) < num_lags:
            # extend in case of periodic lag strength
            lag_strengths = np.tile(lag_strengths, num_lags//L + 1)
        acf /= lag_strengths[:num_lags]
    return acf

def calc_deconvolved_psd(data, **kwargs):
    r"""
    Estimates the PSD of a signal :math:`F(t)` by deconvolving its spectral window function from the naive estimate from the NDFT.

    Note that this is ill-conditioned.

    Args:
        `max_lag` (int): Deconvolution is truncated after this many steps. This should be set to the greatest lag at which
            :func:`calc_acf` gives a good estimate for the autocovariance. 
    All other arguments and returns are as in :meth:`calc_psd`.
    """
    # remove illegal args
    kwargs.pop('psd', None)
    kwargs.pop('lag_strengths', None)
    kwargs.pop('norm_lag_str', None)
    kwargs.pop('norm_var', None)

    raw_acfs = calc_acf(data, norm_var=False, **kwargs)
    return np.abs(fft.ifft(raw_acfs, norm="forward"))/len(data)

# on module load
if IMPORT_STATUS == "py":
    DEFAULT_NDFT = exe_nfft_py
else:
    DEFAULT_NDFT = exe_ndft