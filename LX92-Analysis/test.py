import numpy as np


def intensity_influence_estimator(ps, kbar,
                                  nroi=65000, binmin=1e-5, binmax=None, nbin_edge=10):
    """
    The concept of this function is simple.
    There may be a dependence on the measured speckle contrast of the
    :param ps:
    :param kbar:
    :param nroi:
    :param binmin:
    :param binmax:
    :param nbin_edge:
    :return:
    """
    binmin = np.log10(binmin)
    if binmax is None:
        binmax = np.log10(kbar.max())
    else:
        binmax = np.log10(binmax)

    kbar_bin = np.logspace(binmin, binmax, nbin_edge)
    w = np.digitize(kbar, kbar_bin)
    ps_bin = np.zeros((nbin_edge - 1, 3))
    ps_bin_error = np.zeros((nbin_edge - 1, 3))

    kbar_bin = np.zeros((nbin_edge - 1))
    kbar_bin_error = np.zeros((nbin_edge - 1))
    p2_shot_noise_error = np.zeros((nbin_edge - 1))

    for i in range(1, nbin_edge):
        w0 = np.where(w == i)[0]
        nframe = w0.size
        if nframe < 100:
            ps_bin[i - 1] = np.nan
        else:
            ps_bin[i - 1] = np.mean(ps[w0], axis=0)
            ps_bin_error[i - 1] = np.std(ps[w0], axis=0) / np.sqrt(nframe)
        if ps_bin_error[i - 1, 2] == 0:
            ps_bin[i - 1] = np.nan
        kbar_bin[i - 1] = np.mean(kbar[w0], axis=0)
        kbar_bin_error[i - 1] = np.std(kbar[w0], axis=0)
        # shot noise error, not alpha implemented right now
        p2_shot_noise_error[i - 1] = np.sqrt(ps_bin[i - 1, 2] / nroi / nframe)

    beta = ps_bin[:, 2] * 2 / kbar_bin ** 2 - 1
    delta_beta = ps_bin_error[:, 2] * 2 / kbar_bin ** 2

    return kbar_bin, kbar_bin_error, ps_bin, ps_bin_error, p2_shot_noise_error, beta, delta_beta
