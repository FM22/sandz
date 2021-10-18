import numpy as np

def dft_mat(n, sgn=-1.0):
    r"""
    Generates an :math:`n\times n` DFT matrix.

    Args:
        n (int): The matrix size :math:`n`.
        sgn (float): The sign convention to use; must be :math:`\pm1`.
    """
    v = np.arange(n)
    return np.exp(sgn*2j*np.pi*np.outer(v,v)/n)

def ndft_mat(pattern, p, sgn=-1.0):
    r"""
    Generates an :math:`p\times q` NDFT matrix consisting of columns
    :math:`k_1,\dots,k_q` of a :math:`p\times p` DFT matrix.

    Args:
        pattern (int array): The columns :math:`k_1,\dots,k_q` to use; must be integers in `[0,p)`.
        p (int): The dimension :math:`p`.
        sgn (float): The sign convention to use; must be :math:`\pm1`.
    """
    return np.exp(sgn*2j*np.pi*np.outer(np.arange(p),pattern)/p)

def fft(inp, N=None, threshold=3):
    r"""
    Mixed-radix FFT implementation.

    Uses the (mixed-radix) Cooley-Tukey algorithm; see *Computational frameworks for the FFT* (Van Loan) p.81.
    Calculates a factorisation recursively at runtime.

    Args:
        inp (NumPy complex array): The input vector.
        N (int): The length of ``inp``.
        threshold (int): The maximum size at which to compute the DFT directly instead of
            attempting to factorise.
    Returns:
        NumPy complex array: The FFT of ``inp``.
    """
    if N == None:
        N = len(inp)

    if N <= threshold:
        return dft_mat(N) @ inp
    
    # try to find a factor p>1    
    p = 3
    check_lim = int(np.sqrt(N)+1)
    while p < check_lim and N % p: p += 1
    if p == check_lim:
        # compute directly if prime
        return dft_mat(N) @ inp
    
    # found factorisation n=p*m
    m = N//p

    # Ω 🠔 diag(1,ω,...,ω^{m-1}), where ω^N=1
    Ω = np.exp(-2j*np.pi*np.arange(m)/N)

    # z 🠔 diag(I_m,Ω,...,Ω^{p-1}) @ (I_p⊗F_m) @ Π^T_{p,m} @ inp
    # p size-m FFTs
    z = np.empty(N,dtype=complex)
    for j in range(p):
        z[j*m:(j+1)*m] = Ω**j * fft(inp[j:N:p], m)
    
    # y 🠔 (F_p⊗I_m) @ z
    # m size-p FFTs
    y = np.empty(N, dtype=complex)
    for j in range(m):
        y[j:N:m] = fft(z[j:N:m], p)
    
    return y

def nfft_per_pattern(inp, pattern, p, dft_fun=fft, **kwargs):
    r"""
    Calculates the NDFT of a signal :math:`F(t)` sampled at integer times `t_i`, where the sampling is
    periodic with integer period :math:`p`.

    More precisely, let there be :math:`q` sample times :math:`t_1,\dots,t_q` in :math:`[0,p)`. Then :math:`t_{aq+b}=ap+t_b`.
    Note that the signal must be sampled over a whole number of periods, that is, the length of ``inp`` must be a multiple
    of :math:`q`.

    The algorithm is based off the mixed-radix Cooley-Tukey algorithm; compare :meth:`fft`.

    Args:
        inp (NumPy complex array): The values :math:`F(t_i)` of the signal.
            pattern (int array): The (integer) sample times :math:`t_1,\dots,t_q` in the range :math:`[0,p)`.
        p (int): The period :math:`p` of the sampling.
        dft_fun (function): A function that performs a DFT; must take parameters ``(data, N)``, where ``N`` 
            is the length of ``data``, and return the DFT of ``data``. Any keyword arguments are passed to
            this function.
    
    Returns:
        NumPy complex array: The NDFT of the vector :math:`F(t_i)`.
    """
    # given factorisation N=q*m
    q = len(pattern)
    m = len(inp)//q
    N = m*p

    # G 🠔 cols pattern of F_q
    small_ndft = ndft_mat(pattern, p)

    # tw 🠔 diag(1,w,...,w^{m-1}), where w^N=1
    tw = np.exp(-2j*np.pi*np.arange(m)/N)

    # vec 🠔 diag(tw^pattern) @ (I_q⊗F_m) @ Π^T_{q,m} @ inp
    # q size-m FFTs
    vec = np.empty(m*q, dtype=complex)
    for j in range(q):
        vec[j*m:(j+1)*m] = tw**pattern[j] * dft_fun(inp[j:N:q], m, **kwargs)

    # y 🠔 (G⊗I_m) @ vec
    # m size-p.q NDFTs
    out = np.empty(N, dtype=complex)
    for j in range(m):
        out[j:N:m] = small_ndft @ vec[j:N:m]
    return out

def nfft_per(data, mask, **kwargs):
    r"""
    Calculates the NDFT of a signal :math:`F(t)` sampled at integer times `t_i`, where the sampling is
    periodic with integer period :math:`p`, with structure given by ``mask``.

    More precisely, :math:`t\in(t_i)` iff ``mask[t%b]==True``.

    This is a wrapper for :meth:`nfft_per_pattern`, and any keyword arguments are passed to that function.
    The data is padded so that it contains a whole number of periods. 

    Args:
        data (NumPy complex array): The values :math:`F(t_i)` of the signal.
        mask (bool array): A length-:math:`p` array with value `True` whenever that time is a sample
            time in :math:`(t_i)`.
    
    Returns:
        NumPy complex array: The NDFT of the vector :math:`F(t_i)`.
    """
    n = len(data)
    p = len(mask)
    q = np.count_nonzero(mask)
    L = n + (-n % q)

    # pad out to multiple of q
    padded_data = np.zeros(L, dtype=complex)
    padded_data[:n] = data

    p = len(mask)
    return nfft_per_pattern(data, np.arange(p)[mask], p, **kwargs)