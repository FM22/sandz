import sandz.ndft as ndft
import sandz.util as util
import sandz.arma as arma
import numpy as np
import numpy.testing as npt
import sys

def test_00():
    np.random.seed(0) # fix randomness

    ts = np.array([0,.2,.4,.5])
    data = np.array([2,-2,-1j,1j])
    ans_acf = np.array([1,-.4,-.8-.4j,.8j])

    npt.assert_allclose(([0, .1, .2, .3], ans_acf), util.acf_with_lags(times=ts, data=data, ndft_fun=ndft.exe_ndft, max_lag=.4, max_freq=10, os=100))
    if 'nfft' in sys.modules: # python version
        npt.assert_allclose(ans_acf, ndft.calc_acf(times=ts, data=data, max_lag=.4, ndft_fun=ndft.exe_nfft_py, max_freq=10))
    if 'pynfft' in sys.modules: # c version
        npt.assert_allclose(ans_acf, ndft.calc_acf(times=ts, data=data, max_lag=.4, ndft_fun=ndft.exe_nfft_c, max_freq=10))
    npt.assert_allclose([1,.25,.5,.25], ndft.lag_strength(times=10*ts, max_lag=4, ndft_fun=ndft.exe_ndft))

    data_padded = np.array([2,0,-2,0,-1j,1j])
    ans_psd_embed = np.abs(np.fft.fft(data_padded))**2/4
    ans_spec_win = np.array([1,1/16,1/16,.25,1/16,1/16])
    mask = np.array([True, False, True, False, True, True])
    npt.assert_allclose(([0, 10/6, 20/6, 30/6, 40/6, 50/6], ans_psd_embed), util.psd_with_freqs(data, times=ts, ndft_fun=ndft.exe_ndft_fft, max_freq=10))
    npt.assert_allclose(ans_psd_embed, ndft.calc_psd(data, mask=mask, ndft_fun=ndft.exe_nfft_per))
    npt.assert_allclose(ans_spec_win, ndft.spec_win(times=10*ts, ndft_fun=ndft.exe_ndft_fft))

    sample_times = np.sort(np.random.choice(np.arange(1000), 500, replace=False))
    npt.assert_allclose(np.zeros(99), ndft.calc_deconvolved_psd(
        data=np.array([1]*500), times=sample_times, max_freq=1, max_lag=100, ndft_fun=ndft.exe_ndft)[1:],
        atol = 10E-10) # unsure on the normalisation at lag 0; this test could be made much better

    assert set((tuple(i) for i in util.gen_paths(3, 2))) == {(),(0,),(1,),(2,),(0,1),(1,0),(0,2),(2,0),(1,2),(2,1)}
    npt.assert_allclose(util.to_db(np.array([10,100,53])), np.array([40,50,47.242758696])) # calculator output

    # no tests for arma module yet

if __name__ == '__main__':
    test_00()