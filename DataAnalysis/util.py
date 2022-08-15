import numpy as np
import scipy as sp
from scipy import stats


##################################################
#   Convert 1D data to 2D image
##################################################


#################################################
#    Maximal likelihood anlaysis
#################################################
def NB_dist(k, M, kavg, I0):
    temp1 = sp.gammaln(k + M) - sp.gammaln(k + 1) - sp.gammaln(M)
    temp2 = -k * np.log(1 + M / kavg)
    temp3 = -M * np.log(1 + kavg / M)
    return I0 * np.exp(temp1 + temp2 + temp3)


def chisqs(p, kavg, M, nroi):
    N = np.size(kavg)
    k = np.reshape(np.arange(4), (4, 1))
    k = np.tile(k, (1, N))
    kavg = np.tile(kavg, (4, 1))
    return -2 * np.nansum((p * nroi * np.log(1 / p * NB_dist(k, M, kavg, 1.))))


def getContrast_bk(ps, nroi, Mmax):
    nn = (Mmax - 1) * 1000 + 1
    #     Ms = np.linspace(0.1,Mmax,nn)
    Ms = np.linspace(1, Mmax, nn)
    chi2 = np.zeros(Ms.size)
    for ii, M0 in enumerate(Ms):
        chi2[ii] = chisqs(p=ps[:4], kavg=ps[-1], M=M0, nroi=nroi)
    pos = np.argmin(chi2)
    M0 = Ms[pos]
    # curvature as error analysis
    dM = Ms[1] - Ms[0]
    delta_M = np.sqrt(dM ** 2 / (chi2[pos + 1] + chi2[pos - 1] - 2 * chi2[pos]))


################################################
#    Maximal likelihood analysis -- Adopted by Haoyuan
################################################
def get_nbinomial_dist(k, kMean, M):
    """
    Get the probability mass function of the negative binomial distribution function.

    :param k:
    :param kMean:
    :param M:
    :return:
    """
    return stats.nbinom.pmf(k=k, n=M, p=1. / (1. + kMean / M))


def chiSquare(p, kMean, M, nroi):
    """
    Get the chi square test value for different conditions.

    :param p:
    :param kMean:
    :param M:
    :param nroi:
    :return:
    """
    # Get the number of different kMean to fit
    kMeanNum = np.size(kMean)

    # Create holder of k to calculate the probability for different k and kbar
    k = np.reshape(np.arange(4), (4, 1))
    k = np.tile(k, (1, kMeanNum))

    # Duplicate kMean to match the size of k to calculate the probability distribution for fitting
    kMean = np.tile(kMean, (4, 1))

    # Calculate the chi square value.
    return -2 * np.nansum((p * nroi * np.log(p / stats.nbinom.pmf(k=k, n=M, p=1. / (1. + kMean / M)))))


def getContrast(probability, kMean, nroi, Mmin=1, Mmax=100., Mnum=1000):
    """
    Calculate the chi square value at each specified M
    Find the value where chi square result is minimal.
    The inverse of the corresponding M is the contrast.

    :param probability: Probabilities calculated from each pattern (number of pattern, 4).
                        probability[0] = (P0, P1, P2, P3)
    :param kMean: The kbar per pattern. Shape (number of pattern, )
    :param nroi: Number of pixels in the ROI
    :param Mmin:
    :param Mmax:
    :param Mnum:1000
    :return:
    """

    # The list of M to calculate the chi square test.
    Ms = np.linspace(Mmin, Mmax, num=Mnum)

    # Create a holder for chi square
    chi2 = np.zeros(Ms.size)

    # Loop through all M
    for idx in range(Mnum):
        chi2[idx] = chiSquare(p=probability[:4],
                              kMean=kMean,
                              M=Ms[idx],
                              nroi=nroi)

    # Get the M where the chi square is minimal
    pos = np.argmin(chi2)
    M0 = Ms[pos]

    # Estimate the uncertainty of the value.
    # curvature as error analysis
    dM = Ms[1] - Ms[0]
    delta_M = np.sqrt(dM ** 2 / (chi2[pos + 1] + chi2[pos - 1] - 2 * chi2[pos]))

    return M0, delta_M
