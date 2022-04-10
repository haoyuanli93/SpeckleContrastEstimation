import numpy as np
from scipy.special import erfc


#################################################
#    Get the pixel number within the specified Q
#################################################
def get_detector_q_map(theta0, n_pix, pix_size, det_dist, incident_wavevec_A):
    """

    :param theta0:
    :param n_pix:
    :param pix_size:
    :param det_dist:
    :param incident_wavevec_A: The length of the incident wave-vector measured in A^-1
    :return:
    """
    position_holder = np.zeros((n_pix, n_pix, 3))

    # Define the central position vector
    position_center = np.array([0, np.sin(2 * theta0), np.cos(2 * theta0)]) * det_dist

    # Define the two normal directions
    norm_dir1 = np.array([0, np.cos(2 * theta0), - np.sin(2 * theta0)]) * pix_size
    norm_dir2 = np.array([1, 0, 0]) * pix_size

    # Fill in the position holder
    position_holder[:, :, :] += position_center[np.newaxis, np.newaxis, :]
    position_holder[:, :, :] += np.outer(np.linspace(-n_pix / 2, n_pix / 2, num=n_pix), norm_dir1)[np.newaxis, :, :]
    position_holder[:, :, :] += np.outer(np.linspace(-n_pix / 2, n_pix / 2, num=n_pix), norm_dir2)[:, np.newaxis, :]

    # Get the q value
    direction_holder = position_holder / np.sqrt(np.sum(np.square(position_holder), axis=-1))[:, :, np.newaxis]

    q_vec_holder = direction_holder - np.array([0, 0, 1])[np.newaxis, np.newaxis, :]
    q_vec_holder *= incident_wavevec_A

    return np.sqrt(np.sum(np.square(q_vec_holder), axis=-1))


def get_pixel_num_in_Q(det_Q_map, Q_max, Q_min):
    """
    Get the
    :param det_Q_map:
    :param Q_max:
    :param Q_min:
    :return:
    """
    holder = np.ones_like(det_Q_map, dtype=np.bool)
    holder[det_Q_map < Q_max] = 0
    holder[det_Q_map > Q_min] = 0

    return np.sum(holder)


##########################################################################
#    Calculate the contrast
##########################################################################
def deltaRadial(tth, gam, L, W, xi, k0, dlol, dx, dy, f1, Num=200):
    W = W / np.cos(gam)
    Q = 2. * k0 * np.sin(tth / 2.)
    A = dlol * Q * 1.e4 * np.sqrt(1 - (Q / (2 * k0)) ** 2)
    B = -dlol * 1.e4 * Q ** 2 / k0
    A = A + B * np.tan(gam)
    x0 = np.linspace(0., L, Num)
    y0 = np.linspace(0., W, Num)
    x, y = np.meshgrid(x0, y0, sparse=True)
    t = (L - x) * (W - y) / ((L * W) ** 2) * np.exp(-((x / xi) ** 2))
    t *= (dx * x + dy * y) ** 2
    t *= f1(A * x + B * y) + f1(A * x - B * y)
    delta_r = 2 * np.sum(t * x[0, 1] * y[1, 0])
    return delta_r


def deltaAzimuthal(M, xi):
    x = M / xi
    t = x * np.sqrt(np.pi) / 2. * (1 - erfc(x)) + np.exp(-x ** 2) - 1.0
    t *= (xi / x) ** 2
    return t


def deltav(Q, pp, f1, omega=0, Num=200):
    i = np.size(Q)
    tth = 2 * np.arcsin(Q / 2 / pp['k0'])
    gam = []

    if omega == 0:
        gam = np.zeros_like(Q)
    if omega == 1:
        gam = np.pi / 2. - tth / 2.
    if omega == 2:
        gam = -tth / 2.

    z = np.zeros_like(Q)
    deltas = np.zeros((6, i))
    t = betaAzimuthal(pp['M'], pp['xi_v'])
    d0 = deltaAzimuthal(pp['M'], pp['xi_v'])
    scale = 1.e4
    for j in range(i):
        t0 = betaRadial(tth[j], gam[j], pp['L'], pp['W'], pp['xi_h'], pp['k0'], pp['dlol'],
                        f1, Num=Num)
        t1 = deltaRadial(tth[j], gam[j], pp['L'], pp['W'], pp['xi_h'], pp['k0'], pp['dlol'], 1.0, 0.0,
                         f1, Num=Num)
        t2 = deltaRadial(tth[j], gam[j], pp['L'], pp['W'], pp['xi_h'], pp['k0'], pp['dlol'], 0.0, 1.0,
                         f1, Num=Num)
        z[j] = np.sqrt(t * t1 + t * t2)
        deltas[0, j] = scale ** 2 * t1 / t0
        deltas[1, j] = scale ** 2 * t2 / t0
        deltas[2, j] = scale ** 2 * d0 / t
        deltas[3, j] = t0 * t
        deltas[4, j] = t
        deltas[5, j] = t0
    return deltas


def betaRadial(tth, gam, L, W, xi, k0, dlol, f1, Num=200):
    """
    Calculate radial beta in scattering plane by brute force.

    tth: (rad) Scattering angle
    gam: No sure
    L: (um) Beam size on the sample within the diffraction plane
    W: (um) Sample thickness
    xi: (um) coherence length in the
    k0: (angstroms^-1) is the length of the incident wave-vector: 2*pi / wavelength
    dlol: energy resolution.  FWHM_spectrum / energy_center
    f1: function that defines the shape of the pulse
    """

    W = W / np.cos(gam)
    Q = 2. * k0 * np.sin(tth / 2.)
    A = dlol * Q * 1.e4 * np.sqrt(1 - (Q / (2 * k0)) ** 2)
    B = -dlol * 1.e4 * Q ** 2 / k0
    A = A + B * np.tan(gam)
    x0 = np.linspace(0., L, Num)
    y0 = np.linspace(0., W, Num)
    x, y = np.meshgrid(x0, y0, sparse=True)
    t = 2.0 * (L - x) * (W - y) / ((L * W) ** 2) * np.exp(-((x / xi) ** 2))
    t *= f1(A * x + B * y) + f1(A * x - B * y)
    beta_radial = np.sum(t * x[0, 1] * y[1, 0])
    return beta_radial


def betaAzimuthal(M, xi):
    """
    Get the speckle contribution from the azimuthal direction

    M: (um) Beam size on the sample perpendicular to the diffraction plane
    xi_op: (um) coherence length perpendicular to the diffraction plane
    """
    x = M / xi
    t = (x * np.sqrt(np.pi) * (1 - erfc(x)) + np.exp(-x ** 2) - 1) / x ** 2
    return t


def get_contrast(params, f1):
    """
    Get the contrast estimation.

    :param params:
    :param f1:
    :return:
    radial FWHM sepckle size(1e-4 inverse angstrom),
    azimuthal FWHM speckle size(1e-4 inverse angstrom),
    detector pixel size(1e-4 inverse angstrom),
    beta with perfect detector,
    real beta
    """

    pixelsize = params['pixelsize']
    detectordis = params['detectordis']

    k0 = params['k0']

    kpix = k0 * pixelsize * 1.0e-6 / detectordis  # detector pixel size in Q space
    kdet = kpix / np.sqrt(6.0)  # convert detector size to the rms space

    norm = 1.0e+4
    q = params['Q']
    theta_detector = params['tthetadet']

    deltas = deltav(q, params, f1=f1)
    d1 = 1.0 / deltas[2, :]
    d2 = 1.0 / (np.cos(theta_detector) ** 2 * deltas[0, :] + np.sin(theta_detector) ** 2 * deltas[1, :])

    dd4 = deltas[3, :]  # perfect detector beta

    fd1 = np.sqrt(d1 / (kdet ** 2 + d1))  # fraction delta_zz
    fd2 = np.sqrt(d2 / (kdet ** 2 + d2))  # fraction delta_rr

    # measured beta.
    beta3 = fd1 * fd2 * dd4
    factor = 2.35  # from rms to FWHM

    return norm * np.sqrt(d1) * factor, norm * np.sqrt(d2) * factor, norm * kpix, dd4, beta3
