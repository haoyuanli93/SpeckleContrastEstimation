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
    holder = np.ones_like(det_Q_map, dtype=bool)
    holder[det_Q_map < Q_min] = False
    holder[det_Q_map > Q_max] = False

    return np.sum(holder)


#################################
#    Beam profile
#################################
# square of various time correlation functions
def florentz(t):
    return np.exp(-np.abs(t))


def ftwobounce(t):
    # correct for HWHM
    tt = np.abs(t / 0.643594 / 2)
    return np.exp(-2 * tt) * (1 + tt) ** 2


def fgaussian(t):
    # correct for HWHM
    tt = np.abs(t / 1.17741 / 2)
    return np.exp(-tt ** 2)


##########################################################################
#    Calculate the contrast
##########################################################################
def deltaRadial(Q, gam,
                beam_size_in_plane, sample_thickness,
                trans_coh_length, incident_wavevec, energy_res,
                dx, dy,
                beam_profile_fun,
                Num=200):
    """

    :param Q:
    :param gam:
    :param beam_size_in_plane:
    :param sample_thickness:
    :param trans_coh_length:
    :param incident_wavevec:
    :param energy_res:
    :param dx:
    :param dy:
    :param beam_profile_fun:
    :param Num:
    :return:
    """
    sample_thickness = sample_thickness / np.cos(gam)
    A = energy_res * Q * 1.e4 * np.sqrt(1 - (Q / (2 * incident_wavevec)) ** 2)
    B = -energy_res * 1.e4 * Q ** 2 / incident_wavevec
    A = A + B * np.tan(gam)
    x0 = np.linspace(0., beam_size_in_plane, Num)
    y0 = np.linspace(0., sample_thickness, Num)
    x, y = np.meshgrid(x0, y0)
    t = (beam_size_in_plane - x) * (sample_thickness - y) / ((beam_size_in_plane * sample_thickness) ** 2)
    t *= np.exp(-((x / trans_coh_length) ** 2))
    t *= (dx * x + dy * y) ** 2
    t *= beam_profile_fun(A * x + B * y) + beam_profile_fun(A * x - B * y)
    delta_r = 2 * np.sum(t * x[0, 1] * y[1, 0])
    return delta_r


def deltaAzimuthal(beam_size_out_plane, trans_coh_length):
    """

    :param beam_size_out_plane:
    :param trans_coh_length:
    :return:
    """
    x = beam_size_out_plane / trans_coh_length
    t = x * np.sqrt(np.pi) / 2. * (1 - erfc(x)) + np.exp(-x ** 2) - 1.0
    t *= (trans_coh_length / x) ** 2
    return t


def betaRadial(Q, gam, beam_size_in_plane, sample_thickness, trans_coh_length, k0, energy_res, beam_profile_fun,
               Num=200):
    """
    Calculate radial beta in scattering plane by brute force.

    tth: (rad) Scattering angle
    gam: No sure
    L: (um) Beam size on the sample within the diffraction plane
    W: (um) Sample thickness
    xi: (um) coherence length in the
    k0: (angstroms^-1) is the length of the incident wave-vector: 2*pi / wavelength
    energy_resolution: energy resolution.  FWHM_spectrum / energy_center
    f1: function that defines the shape of the pulse
    """

    sample_thickness = sample_thickness / np.cos(gam)
    A = energy_res * Q * 1.e4 * np.sqrt(1 - (Q / (2 * k0)) ** 2)
    B = -energy_res * 1.e4 * Q ** 2 / k0
    A = A + B * np.tan(gam)
    x0 = np.linspace(0., beam_size_in_plane, Num)
    y0 = np.linspace(0., sample_thickness, Num)
    x, y = np.meshgrid(x0, y0)
    t = 2.0 * (beam_size_in_plane - x) * (sample_thickness - y) / ((beam_size_in_plane * sample_thickness) ** 2)
    t *= np.exp(-((x / trans_coh_length) ** 2))
    t *= beam_profile_fun(A * x + B * y) + beam_profile_fun(A * x - B * y)
    beta_radial = np.sum(t * x[0, 1] * y[1, 0])
    return beta_radial


def betaAzimuthal(beam_size_out_plane, trans_coh_length):
    """
    :param beam_size_out_plane: (um) Beam size on the sample perpendicular to the diffraction plane
    :param trans_coh_length: (um) coherence length perpendicular to the diffraction plane
    :return:
    """
    x = beam_size_out_plane / trans_coh_length
    t = (x * np.sqrt(np.pi) * (1 - erfc(x)) + np.exp(-x ** 2) - 1) / x ** 2
    return t


def deltav(Q, params, beam_profile_fun, omega=0, Num=200):
    """

    :param Q:
    :param params:
    :param beam_profile_fun:
    :param omega:
    :param Num:
    :return:
    """
    tth = 2 * np.arcsin(Q / 2 / params['incident_wavevec'])

    if omega == 0:
        gam = 0.
    elif omega == 1:
        gam = np.pi / 2. - tth / 2.
    else:
        gam = -tth / 2.

    deltas = np.zeros(6)

    t = betaAzimuthal(params['beam_size_out_plane'], params['trans_coh_length_in_plane'])
    d0 = deltaAzimuthal(params['beam_size_out_plane'], params['trans_coh_length_in_plane'])
    scale = 1.e4

    t0 = betaRadial(Q=Q, gam=gam,
                    beam_size_in_plane=params['beam_size_in_plane'],
                    sample_thickness=params['sample_thickness'],
                    trans_coh_length=params['trans_coh_length_out_plane'],
                    k0=params['incident_wavevec'],
                    energy_res=params['energy_resolution'],
                    beam_profile_fun=beam_profile_fun,
                    Num=Num)
    t1 = deltaRadial(Q=Q, gam=gam,
                     beam_size_in_plane=params['beam_size_in_plane'],
                     sample_thickness=params['sample_thickness'],
                     trans_coh_length=params['trans_coh_length_out_plane'],
                     incident_wavevec=params['incident_wavevec'],
                     energy_res=params['energy_resolution'],
                     dx=1.0,
                     dy=0.0,
                     beam_profile_fun=beam_profile_fun,
                     Num=Num)
    t2 = deltaRadial(Q=Q,
                     gam=gam,
                     beam_size_in_plane=params['beam_size_in_plane'],
                     sample_thickness=params['sample_thickness'],
                     trans_coh_length=params['trans_coh_length_out_plane'],
                     incident_wavevec=params['incident_wavevec'],
                     energy_res=params['energy_resolution'],
                     dx=0.0,
                     dy=1.0,
                     beam_profile_fun=beam_profile_fun,
                     Num=Num)

    deltas[0] = scale ** 2 * t1 / t0
    deltas[1] = scale ** 2 * t2 / t0
    deltas[2] = scale ** 2 * d0 / t
    deltas[3] = t0 * t
    deltas[4] = t
    deltas[5] = t0
    return deltas


def get_contrast(params, beam_profile_fun=ftwobounce):
    """
    Get the contrast estimation.

    :param params:
    :param beam_profile_fun:
    :return:
    radial FWHM sepckle size(1e-4 inverse angstrom),
    azimuthal FWHM speckle size(1e-4 inverse angstrom),
    detector pixel size(1e-4 inverse angstrom),
    beta with perfect detector,
    real beta
    """

    pixelsize = params['pixel_size']
    detectordis = params['detector_distance']

    k0 = params['incident_wavevec']

    kpix = k0 * pixelsize * 1.0e-6 / detectordis  # detector pixel size in Q space
    kdet = kpix / np.sqrt(6.0)  # convert detector size to the rms space

    norm = 1.0e+4
    q = params['Q']
    theta_detector = 2 * np.arcsin(q / 2. / k0)

    deltas = deltav(Q=q, params=params, beam_profile_fun=beam_profile_fun)
    d1 = 1.0 / deltas[2]
    d2 = 1.0 / (np.cos(theta_detector) ** 2 * deltas[0] + np.sin(theta_detector) ** 2 * deltas[1])

    dd4 = deltas[3]  # perfect detector beta

    fd1 = np.sqrt(d1 / (kdet ** 2 + d1))  # fraction delta_zz
    fd2 = np.sqrt(d2 / (kdet ** 2 + d2))  # fraction delta_rr

    # measured beta.
    beta3 = fd1 * fd2 * dd4
    factor = 2.35  # from rms to FWHM

    return norm * np.sqrt(d1) * factor, norm * np.sqrt(d2) * factor, norm * kpix, dd4, beta3


