import numpy as np
from scipy import integrate

"""
All quantities used in this script are in the SI unit.
"""


##########################################
#    Basic functions
##########################################
def wavelength(energy_eV):
    """
    input  : x-ray energy in [eV]
    output : x-ray wavelength in [m]
    """
    h = 4.135667516e-15  # planck constant [eV*s]
    c = 299792458  # speed of light [m/s]
    lmbda = h * c / energy_eV
    return lmbda


##########################################
#    Visualization
##########################################
def custom_ticks(tick_values, tick_labels, value_range, axis, axis_handle):
    """
    helper function to plot axis ticks and tick labels
    """
    if len(tick_values) != len(tick_labels):
        print('WARNING: Please match tick_values and tick_labels!')

    tick_list = []
    label_list = []
    for t, tick in enumerate(tick_values):
        idx = np.argmin(np.abs(tick - value_range))
        tick_list.append(idx)
        label_list.append(tick_labels[t])

    if axis == 'x':
        axis_handle.set_xticks(tick_list)
        axis_handle.set_xticklabels(label_list)
    if axis == 'y':
        axis_handle.set_yticks(tick_list)
        axis_handle.set_yticklabels(label_list)

    return


##########################################
#    Get constrast
##########################################
def contrast_hruszkewycz(momentum_transfer,
                         xray_energy,
                         energy_resolution,
                         xray_focus,
                         sample_thickness,
                         detector_distance,
                         detector_pixelsize):
    """
    returns speckle contrast estimation according to:
    Hruszkewycz et al,,Physical review letters 109.18 (2012): 185502

    equation (2),(4) of the supplementary material are used.
    """

    # aliases following Hruszkewycz
    q = momentum_transfer
    e = xray_energy
    r = energy_resolution
    s = xray_focus
    t = sample_thickness
    l = detector_distance
    p = detector_pixelsize

    # scattering angle
    theta = np.arcsin(q * wavelength(e) / (4 * np.pi))

    # Mrad
    temp = (q * r) ** 2 * (s ** 2 * np.cos(theta) ** 2 + t ** 2 * np.sin(theta) ** 2)
    Mrad = np.sqrt(1 + temp / (4 * np.pi ** 2))
    # print('beta_rad: ',1./Mrad)

    # Mdet
    temp = (p ** 4 * s ** 2) * (s ** 2 * np.cos(2 * theta) ** 2 + t ** 2 * np.sin(2 * theta) ** 2)
    Mdet = np.sqrt(1 + temp / (wavelength(e) ** 4 * l ** 4 * Mrad ** 2))
    # print('beta_det: ',1./Mdet)

    return 1. / (Mrad * Mdet)


def contrast_moeller(momentum_transfer,
                     xray_energy,
                     energy_resolution,
                     xray_focus,
                     sample_thickness,
                     detector_distance,
                     detector_pixelsize):
    """
    returns speckle contrast estimation according to:
    Moeller et al, IUCrJ 6.5 (2019)

    equation (12),(13),(14) are used.
    """

    # aliases following Moeller et al
    q = momentum_transfer
    e = xray_energy
    r = energy_resolution  # i.e. DeltaLambda/Lambda
    a = xray_focus
    d = sample_thickness
    l = detector_distance
    p = detector_pixelsize

    # wavevector
    k = 2. * np.pi / wavelength(e)
    # scattering angle
    theta = np.arcsin(q * wavelength(e) / (4 * np.pi))

    # beta_res
    w = 2 * np.pi * p * a / (wavelength(e) * l)
    fun1 = lambda v: 2 / w ** 2 * (w - v) * (np.sin(v / 2) / (v / 2)) ** 2
    out1 = integrate.quad(fun1, 0, w, limit=500)
    beta_res = out1[0] ** 2
    # print('beta_res: ',beta_res)

    # beta_cl
    if a > 100e-6:
        print('WARNING: please check transverse coherence for xray_focus larger than 100um.')
    A = r * q * np.sqrt(1 - q ** 2 / (4 * k ** 2))
    B = (-1. / 2) * r * q ** 2 / k
    fun2 = lambda x, z: (a - x) * (d - z) * np.exp(-x ** 2 / a ** 2) * (
            np.exp(-2 * np.abs(A * x + B * z)) + np.exp(-2 * np.abs(A * x - B * z)))
    out2 = integrate.dblquad(fun2, 0, d, 0, a)
    beta_cl = out2[0] * 2 / (a * d) ** 2
    # print('beta_cl: ',beta_cl)

    # contrast
    return beta_res * beta_cl


#######################################################
#     Get sample scattering intensity
#######################################################

def iq_estimate(detector_distance,
                sample_thickness,
                pixel_size=50e-6,
                photon_per_pulse=3e8,
                diff_sigma=9.94,
                attenuation_length=2000e-6):
    """
    returns crude estimate of scattering intensity [photons/pixel/shot]

    The default values for the photon_per_pulse, diff_sigma,
    and attenuation_length are for CO2 at supercritical condition

    :param detector_distance:
    :param sample_thickness:
    :param pixel_size: The size of the pixel on the detector. For epix detector
                        this is 50 um.
    :param photon_per_pulse: incomming xray at the sample position [photons/pulse]
    :param diff_sigma: differential scattering cross section of supercritical CO2 in units [1/m]
    :param attenuation_length: attenuation length of co2 [m] at 8.3keV and 0.6g/cm**3
    :return:
    """

    # solid angle covered by a single detector pixel
    omega = (pixel_size / detector_distance) ** 2

    # effective sample thickness at 8.3keV [m]
    d_eff = attenuation_length * (1 - np.exp(-sample_thickness / attenuation_length))

    # scattering intensity [photons/pixel/shot]
    I_out = photon_per_pulse * diff_sigma * omega * d_eff

    return I_out


def get_water_diff_cross_section(q):
    pass
