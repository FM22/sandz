def ex_0():
    import ndft
    import arma
    import util
    import numpy as np
    np.random.seed(0)

    length = 1000
    inds = np.sort(np.random.choice(
        np.arange(length), length//2, replace=False))
    times = np.arange(length)[inds]
    arma_params = (
        np.array([2.7607, -3.8106, 2.6535, -0.9238]), np.empty(0), 1)

    full_data = arma.arma(*arma_params, length, seed=0)
    data = full_data[inds]

    (freqs, psd) = util.psd_with_freqs(
        data, times=times, ndft_fun=ndft.exe_ndft_fft)
    (lags, acf) = util.acf_with_lags(
        psd=psd, times=times, ndft_fun=ndft.exe_ndft_fft)

    true_psd = arma.psd_exact(*arma_params, freqs)
    true_acf = arma.acf_exact(*arma_params[:-1], len(lags)-1)

    psd_pwr = util.to_db(psd)
    true_psd_pwr = util.to_db(true_psd)

    import matplotlib.pyplot as plt

    ax = plt.subplot()
    ax.plot(freqs, psd_pwr, label="Estimated PSD")
    ax.plot(freqs, true_psd_pwr, label="True PSD", alpha=.8)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("PSD (dB)")
    ax.legend()
    ax.set(xlim=[0,1])
    plt.figure()
    ax = plt.subplot()
    ax.plot(lags, np.real(acf), label="Estimated ACF")
    ax.plot(lags, true_acf, label="True ACF", alpha=.8)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Normalised\n autocorrelation")
    ax.legend()
    ax.set(ylim=[-1.2,1.2])
    ax.grid(color='gray', alpha=0.5, lw=0.5)
    plt.show()

def ex_1():
    import ndft
    import arma
    import util
    import numpy as np
    import matplotlib.pyplot as plt
    np.random.seed(0)

    iterations = 100
    gen_len = 1000
    arma_params = (np.array([2.7607, -3.8106, 2.6535, -0.9238]), np.array([.9j, 1.1j]), 1)

    dropout_pattern = [True, True, False, True, False, True, True, True, True, True]
    dropout_pattern *=  gen_len//len(dropout_pattern)
    full_times = np.arange(gen_len)
    times_vec = [
        full_times, full_times[dropout_pattern], full_times[dropout_pattern]]
    labels = ["No missing data", "Periodic gaps", "Periodic gaps (deconvolved)"]

    psd = [None]*len(times_vec)
    for i, times in enumerate(times_vec):
        psd[i] = [None]*iterations
        for j in range(iterations):
            data = arma.arma(
                *arma_params, gen_len,
                seed=np.random.randint(2**16))[times_vec[i]]
            if i == 2:
                psd[i][j] = ndft.calc_deconvolved_psd(data, times=times_vec[i])
            else:
                psd[i][j] = ndft.calc_psd(data, times=times_vec[i])

    ax = plt.subplot()
    for i in range(len(times_vec)):
        util.plot_sampled_data(
            util.unif_pts(
                len(psd[i][0])), np.array(psd[i]), ax,
                label=labels[i], db=True, symmetric=False)
    freqs = util.unif_pts(gen_len)
    ax.plot(
        freqs, util.to_db(arma.psd_exact(*arma_params, freqs)),
        label="True value", color='black')
    ax.set(xlim=[0,1], ylim=[0,100])
    ax.legend(loc='lower right')
    ax.set_xlabel("Frequency")
    ax.set_ylabel("PSD (dB)")
    plt.show()

def ex_2():
    import ndft
    import util
    import numpy as np
    import matplotlib.pyplot as plt
    np.random.seed(0)

    length = 33333
    mask = [True, True, False]
    times = np.arange(length)[mask * (length//len(mask))]
    data = np.exp(2j*np.pi*.3*times) +\
        np.exp(2j*np.pi*.4*times) +\
        np.exp(2j*np.pi*.8*times) +\
        np.random.normal((len(times)))*0.01

    acfs = [None]*4
    labels = [
        "Approximate NuFFT", "Embedded in FFT",
        "Periodic NFFT", "Direct estimation"]
    util.benchmark()
    acfs[0] = np.real(ndft.calc_acf(
        data=data, times=times, ndft_fun=ndft.exe_nfft_py))
    util.benchmark(labels[0])
    acfs[1] = np.real(ndft.calc_acf(
        data=data, times=times, ndft_fun=ndft.exe_ndft_fft))
    util.benchmark(labels[1])
    acfs[2] = np.real(ndft.calc_acf(
        data=data, mask=mask, ndft_fun=ndft.exe_nfft_per))
    util.benchmark(labels[2])
    acfs[3] = [np.real(v) for v in util.direct_acf(
        data, times, ndft.MAX_LAG-1).values()]
    util.benchmark(labels[3])

    for i, acf in enumerate(acfs):
        plt.plot(np.arange(ndft.MAX_LAG), acf, label=labels[i])
    plt.legend(loc='lower right')
    plt.show()

def ex_3():
    import ndft
    import util
    import numpy as np

    times = np.array([0, 0.1, 0.2, 0.3])
    data = np.array([1,-1,-1,1])

    freqs, psd = util.psd_with_freqs(
        data, times=times, ndft_fun=ndft.exe_ndft_fft,
        max_freq=10)
    lags, acf = util.acf_with_lags(
        psd=psd, times=times, ndft_fun=ndft.exe_ndft_fft,
        max_freq=10, max_lag=.4)

    print(freqs)
    print(psd)
    print(lags)
    print(acf)