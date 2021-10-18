import numpy as np
import numpy.linalg as la
import numpy.random as r
from .ndft import ndft_mat
from . import util

try:
    import sympy as s
except:
    print("Could not find nonessential package sympy. Try running \"pip install sympy\".")

def arma(phis, thetas, sigma_sq, N, seed=None):
    r"""
    Generates and returns the first N steps of a (complex) ARMA(p,q) process.

    See *Spectral Analysis of Univariate Time Series* (Percival, Walden) p.35.

    Args:
        phis (NumPy complex array): The AR parameters :math:`\phi_i` (length :math:`p`)
        thetas (NumPy complex array): The MA parameters :math:`\theta_i` (length :math:`q`).
        sigma_sq (float): Variance of the underlying Gaussian noise.
        N (int): Number of steps to generate.
        seed: Optional seed for the underlying noise.

    Returns:
        NumPy complex array: The ARMA time series.
    """
    p = len(phis) 
    q = len(thetas)

    if seed:
        r.seed(seed)

    # setup dot products
    thetas = -1 * np.flip(thetas)
    phis = np.flip(phis)

    offset = max(p,q) # number of intialisation steps
    M = N + offset
    E = (r.randn(M) + 1j*r.randn(M)) / np.sqrt(sigma_sq/2) # noise
    X = np.copy(E)

    # calculate ARMA process
    for m in range(offset, M):
        X[m] += E[m-q:m] @ thetas + X[m-p:m] @ phis

    return X[offset:] # remove setup noise

def acf_exact(phis, thetas, N):
    r"""
    Calculates the (theoretical) autocorrelation of an ARMA(p,q) process at integral lags from :math:`0` to :math:`N`, inclusive

    Only works for real-valued parameters :math:`\phi_i` and :math:`\theta_i`.

    Args:
        phis (NumPy float array): The AR parameters :math:`\phi_i` (length :math:`p`).
        thetas (NumPy float array): The MA parameters :math:`\theta_i` (length :math:`q`).
        N (int): Maximum lag :math:`N` to calcuate the ACF for.

    Returns:
        NumPy complex array: The autocorrelation function.
    """
    x = s.symbols("x")
    p = len(phis)
    q = len(thetas)
    assert N >= p

    # change to correct sign convention
    thetas = -1 * np.concatenate([[-1], thetas])

    # create symbolic polynomials
    util.benchmark()
    phi = s.Integer(1)
    for i, c in enumerate(phis):
        phi -= s.Rational(c)*(x**(i+1)) # flip coefficients of AC part

    # compute (truncated) series expansion
    series = (1/phi).series(x, x0=0, n=q+1).removeO()
    coeffs = s.Poly(series, x).all_coeffs() # note coefficients are in reverse order
    coeffs = np.copy(np.array([float(c) for c in coeffs]))
    coeffs = np.pad(coeffs, (q+1-len(coeffs),0)) # pad out extra zeros
    psis = [coeffs[-(i+1):] @ thetas[:i+1] for i in range(q+1)]

    # calculate first p entries
    phis_pad = np.pad(-1 * np.concatenate([[-1], phis]), [0, p]) # pad makes out of range entries zeros
    m = max(p+1,q+1)
    ics = np.array([thetas[i:] @ psis[:-i or None] for i in range(m)]) # uses cov[0]=1
    mat = np.array([[phis_pad[i] if j==0 else phis_pad[i+j]+phis_pad[i-j] for j in range(p+1)] for i in range(p+1)])
    covs = np.empty(N+1, dtype="complex")
    covs[0:p+1] = la.inv(mat) @ ics[0:p+1] # solve ICs
    phis_rev = np.flip(phis)

    # apply recurrence relation
    for i in range(p+1, N+1):
        covs[i] = covs[i-p:i] @ phis_rev + (ics[i] if i<m else 0)

    return covs/covs[0]

def psd_exact(phis, thetas, sigma_sq, freqs):
    r"""
    Calculates the (theoretical) PSD of an ARMA(p,q) process at the frequencies :math:`f_i`.

    See *Spectral Analysis of Univariate Time Series* (Percival, Walden) p.145.

    Args:
        phis (NumPy complex array): The AR parameters :math:`\phi_i` (length :math:`p`)
        thetas (NumPy complex array): The MA parameters :math:`\theta_i` (length :math:`q`).
        sigma_sq (float): Variance of the underlying Gaussian noise.
        freqs (NumPy float array): The frequencies :math:`f_i` to calculate the PSD at.

    Returns:
        NumPy float array: The PSD.
    """
    s_b = ndft_mat(freqs, range(1, len(phis)+1)) @ phis # +1 since we are zero-indexed
    s_t = ndft_mat(freqs, range(1, len(thetas)+1)) @ thetas
    return np.square(sigma_sq*np.abs(1-s_t)/np.abs(1-s_b))