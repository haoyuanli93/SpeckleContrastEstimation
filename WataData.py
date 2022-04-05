import numpy as np
from scipy import interpolate

##########################################
#   Water data
##########################################
"""
X-ray diffraction data on liquid water at 22.0 Celsius, 
measured at APS sector 11-ID-C by LB Skinner & CJ Benmore 
in 2011 (LB Skinner et al. J. Chem Phys. 2013)


q_array is the momentum transfer, in inverse angstrom

diff_array is the measured x-ray elastic differential cross-section,
 normalized per molecule (& after subtraction of Compton scattering)
"""


def get_water_differential_crosssection(q_angstrom_vals):
    """
    Get the water differential cross section at the specified q_vals

    :param q_angstrom_vals: numpy array. 4 pi sin(theta) / lambda
                    2 theta is the scattering angle
                    The uni
    :return:
    """
    diff_fun = interpolate.interp1d(x=q_array,
                                    y=diff_array)
    return diff_fun(q_angstrom_vals)


def get_water_attenuation_coefficient(photon_energy_in_eV, density):
    """
    Get the water differential cross section at the specified q_vals

    :param photon_energy_in_eV: photon energy in eV
    :param density: Density of water in kg/m^3
    :return:
    """
    mu_fun = interpolate.interp1d(x=energy_array_eV,
                                  y=mu_array)
    return mu_fun(photon_energy_in_eV) * density


q_array = np.array([
    0.000000, 0.025000, 0.050000, 0.075000, 0.100000, 0.125000, 0.150000, 0.175000, 0.200000, 0.225000, 0.250000,
    0.275000, 0.300000, 0.325000, 0.350000, 0.375000, 0.400000, 0.425000, 0.450000, 0.475000, 0.500000, 0.525000,
    0.550000, 0.575000, 0.600000, 0.625000, 0.650000, 0.675000, 0.700000, 0.725000, 0.750000, 0.775000, 0.800000,
    0.825000, 0.850000, 0.875000, 0.900000, 0.925000, 0.950000, 0.975000, 1.000000, 1.025000, 1.050000, 1.075000,
    1.100000, 1.125000, 1.150000, 1.175000, 1.200000, 1.225000, 1.250000, 1.275000, 1.300000, 1.325000, 1.350000,
    1.375000, 1.400000, 1.425000, 1.450000, 1.475000, 1.500000, 1.525000, 1.550000, 1.575000, 1.600000, 1.625000,
    1.650000, 1.675000, 1.700000, 1.725000, 1.750000, 1.775000, 1.800000, 1.825000, 1.850000, 1.875000, 1.900000,
    1.925000, 1.950000, 1.975000, 2.000000, 2.025000, 2.050000, 2.075000, 2.100000, 2.125000, 2.150000, 2.175000,
    2.200000, 2.225000, 2.250000, 2.275000, 2.300000, 2.325000, 2.350000, 2.375000, 2.400000, 2.425000, 2.450000,
    2.475000, 2.500000, 2.525000, 2.550000, 2.575000, 2.600000, 2.625000, 2.650000, 2.675000, 2.700000, 2.725000,
    2.750000, 2.775000, 2.800000, 2.825000, 2.850000, 2.875000, 2.900000, 2.925000, 2.950000, 2.975000, 3.000000,
    3.025000, 3.050000, 3.075000, 3.100000, 3.125000, 3.150000, 3.175000, 3.200000, 3.225000, 3.250000, 3.275000,
    3.300000, 3.325000, 3.350000, 3.375000, 3.400000, 3.425000, 3.450000, 3.475000, 3.500000, 3.525000, 3.550000,
    3.575000, 3.600000, 3.625000, 3.650000, 3.675000, 3.700000, 3.725000, 3.750000, 3.775000, 3.800000, 3.825000,
    3.850000, 3.875000, 3.900000, 3.925000, 3.950000, 3.975000, 4.000000, 4.025000, 4.050000, 4.075000, 4.100000,
    4.125000, 4.150000, 4.175000, 4.200000, 4.225000, 4.250000, 4.275000, 4.300000, 4.325000, 4.350000, 4.375000,
    4.400000, 4.425000, 4.450000, 4.475000, 4.500000, 4.525000, 4.550000, 4.575000, 4.600000, 4.625000, 4.650000,
    4.675000, 4.700000, 4.725000, 4.750000, 4.775000, 4.800000, 4.825000, 4.850000, 4.875000, 4.900000, 4.925000,
    4.950000, 4.975000, 5.000000, 5.025000, 5.050000, 5.075000, 5.100000, 5.125000, 5.150000, 5.175000, 5.200000,
    5.225000, 5.250000, 5.275000, 5.300000, 5.325000, 5.350000, 5.375000, 5.400000, 5.425000, 5.450000, 5.475000,
    5.500000, 5.525000, 5.550000, 5.575000, 5.600000, 5.625000, 5.650000, 5.675000, 5.700000, 5.725000, 5.750000,
    5.775000, 5.800000, 5.825000, 5.850000, 5.875000, 5.900000, 5.925000, 5.950000, 5.975000, 6.000000, 6.025000,
    6.050000, 6.075000, 6.100000, 6.125000, 6.150000, 6.175000, 6.200000, 6.225000, 6.250000, 6.275000, 6.300000,
    6.325000, 6.350000, 6.375000, 6.400000, 6.425000, 6.450000, 6.475000, 6.500000, 6.525000, 6.550000, 6.575000,
    6.600000, 6.625000, 6.650000, 6.675000, 6.700000, 6.725000, 6.750000, 6.775000, 6.800000, 6.825000, 6.850000,
    6.875000, 6.900000, 6.925000, 6.950000, 6.975000, 7.000000, 7.025000, 7.050000, 7.075000, 7.100000, 7.125000,
    7.150000, 7.175000, 7.200000, 7.225000, 7.250000, 7.275000, 7.300000, 7.325000, 7.350000, 7.375000, 7.400000,
    7.425000, 7.450000, 7.475000, 7.500000, 7.525000, 7.550000, 7.575000, 7.600000, 7.625000, 7.650000, 7.675000,
    7.700000, 7.725000, 7.750000, 7.775000, 7.800000, 7.825000, 7.850000, 7.875000, 7.900000, 7.925000, 7.950000,
    7.975000, 8.000000, 8.025000, 8.050000, 8.075000, 8.100000, 8.125000, 8.150000, 8.175000, 8.200000, 8.225000,
    8.250000, 8.275000, 8.300000, 8.325000, 8.350000, 8.375000, 8.400000, 8.425000, 8.450000, 8.475000, 8.500000,
    8.525000, 8.550000, 8.575000, 8.600000, 8.625000, 8.650000, 8.675000, 8.700000, 8.725000, 8.750000, 8.775000,
    8.800000, 8.825000, 8.850000, 8.875000, 8.900000, 8.925000, 8.950000, 8.975000, 9.000000, 9.025000, 9.050000,
    9.075000, 9.100000, 9.125000, 9.150000, 9.175000, 9.200000, 9.225000, 9.250000, 9.275000, 9.300000, 9.325000,
    9.350000, 9.375000, 9.400000, 9.425000, 9.450000, 9.475000, 9.500000, 9.525000, 9.550000, 9.575000, 9.600000,
    9.625000, 9.650000, 9.675000, 9.700000, 9.725000, 9.750000, 9.775000, 9.800000, 9.825000, 9.850000, 9.875000,
    9.900000, 9.925000, 9.950000, 9.975000, 10.000000, 10.025000, 10.050000, 10.075000, 10.100000, 10.125000, 10.150000,
    10.175000, 10.200000, 10.225000, 10.250000, 10.275000, 10.300000, 10.325000, 10.350000, 10.375000, 10.400000,
    10.425000, 10.450000, 10.475000, 10.500000, 10.525000, 10.550000, 10.575000, 10.600000, 10.625000, 10.650000,
    10.675000, 10.700000, 10.725000, 10.750000, 10.775000, 10.800000, 10.825000, 10.850000, 10.875000, 10.900000,
    10.925000, 10.950000, 10.975000, 11.000000, 11.025000, 11.050000, 11.075000, 11.100000, 11.125000, 11.150000,
    11.175000, 11.200000, 11.225000, 11.250000, 11.275000, 11.300000, 11.325000, 11.350000, 11.375000, 11.400000,
    11.425000, 11.450000, 11.475000, 11.500000, 11.525000, 11.550000, 11.575000, 11.600000, 11.625000, 11.650000,
    11.675000, 11.700000, 11.725000, 11.750000, 11.775000, 11.800000, 11.825000, 11.850000, 11.875000, 11.900000,
    11.925000, 11.950000, 11.975000, 12.000000, 12.025000, 12.050000, 12.075000, 12.100000, 12.125000, 12.150000,
    12.175000, 12.200000, 12.225000, 12.250000, 12.275000, 12.300000, 12.325000, 12.350000, 12.375000, 12.400000,
    12.425000, 12.450000, 12.475000, 12.500000, 12.525000, 12.550000, 12.575000, 12.600000, 12.625000, 12.650000,
    12.675000, 12.700000, 12.725000, 12.750000, 12.775000, 12.800000, 12.825000, 12.850000, 12.875000, 12.900000,
    12.925000, 12.950000, 12.975000, 13.000000, 13.025000, 13.050000, 13.075000, 13.100000, 13.125000, 13.150000,
    13.175000, 13.200000, 13.225000, 13.250000, 13.275000, 13.300000, 13.325000, 13.350000, 13.375000, 13.400000,
    13.425000, 13.450000, 13.475000, 13.500000, 13.525000, 13.550000, 13.575000, 13.600000, 13.625000, 13.650000,
    13.675000, 13.700000, 13.725000, 13.750000, 13.775000, 13.800000, 13.825000, 13.850000, 13.875000, 13.900000,
    13.925000, 13.950000, 13.975000, 14.000000, 14.025000, 14.050000, 14.075000, 14.100000, 14.125000, 14.150000,
    14.175000, 14.200000, 14.225000, 14.250000, 14.275000, 14.300000, 14.325000, 14.350000, 14.375000, 14.400000,
    14.425000, 14.450000, 14.475000, 14.500000, 14.525000, 14.550000, 14.575000, 14.600000, 14.625000, 14.650000,
    14.675000, 14.700000, 14.725000, 14.750000, 14.775000, 14.800000, 14.825000, 14.850000, 14.875000, 14.900000,
    14.925000, 14.950000, 14.975000, 15.000000, 15.025000, 15.050000, 15.075000, 15.100000, 15.125000, 15.150000,
    15.175000, 15.200000, 15.225000, 15.250000, 15.275000, 15.300000, 15.325000, 15.350000, 15.375000, 15.400000,
    15.425000, 15.450000, 15.475000, 15.500000, 15.525000, 15.550000, 15.575000, 15.600000, 15.625000, 15.650000,
    15.675000, 15.700000, 15.725000, 15.750000, 15.775000, 15.800000, 15.825000, 15.850000, 15.875000, 15.900000,
    15.925000, 15.950000, 15.975000, 16.000000, 16.025000, 16.050000, 16.075000, 16.100000, 16.125000, 16.150000,
    16.175000, 16.200000, 16.225000, 16.250000, 16.275000, 16.300000, 16.325000, 16.350000, 16.375000, 16.400000,
    16.425000, 16.450000, 16.475000, 16.500000, 16.525000, 16.550000, 16.575000, 16.600000, 16.625000, 16.650000,
    16.675000, 16.700000, 16.725000, 16.750000, 16.775000, 16.800000, 16.825000, 16.850000, 16.875000, 16.900000,
    16.925000, 16.950000, 16.975000, 17.000000, 17.025000, 17.050000, 17.075000, 17.100000, 17.125000, 17.150000,
    17.175000, 17.200000, 17.225000, 17.250000, 17.275000, 17.300000, 17.325000, 17.350000, 17.375000, 17.400000,
    17.425000, 17.450000, 17.475000, 17.500000, 17.525000, 17.550000, 17.575000, 17.600000, 17.625000, 17.650000,
    17.675000, 17.700000, 17.725000, 17.750000, 17.775000, 17.800000, 17.825000, 17.850000, 17.875000, 17.900000,
    17.925000, 17.950000, 17.975000, 18.000000, 18.025000, 18.050000, 18.075000, 18.100000, 18.125000, 18.150000,
    18.175000, 18.200000, 18.225000, 18.250000, 18.275000, 18.300000, 18.325000, 18.350000, 18.375000, 18.400000,
    18.425000, 18.450000, 18.475000, 18.500000, 18.525000, 18.550000, 18.575000, 18.600000, 18.625000, 18.650000,
    18.675000, 18.700000, 18.725000, 18.750000, 18.775000, 18.800000, 18.825000, 18.850000, 18.875000, 18.900000,
    18.925000, 18.950000, 18.975000, 19.000000, 19.025000, 19.050000, 19.075000, 19.100000, 19.125000, 19.150000,
    19.175000, 19.200000, 19.225000, 19.250000, 19.275000, 19.300000, 19.325000, 19.350000, 19.375000, 19.400000,
    19.425000, 19.450000, 19.475000, 19.500000, 19.525000, 19.550000, 19.575000, 19.600000, 19.625000, 19.650000,
    19.675000, 19.700000, 19.725000, 19.750000, 19.775000, 19.800000, 19.825000, 19.850000, 19.875000, 19.900000,
    19.925000, 19.950000, 19.975000, 20.000000, 20.025000, 20.050000, 20.075000, 20.100000, 20.125000, 20.150000,
    20.175000, 20.200000, 20.225000, 20.250000, 20.275000, 20.300000, 20.325000, 20.350000, 20.375000, 20.400000,
    20.425000, 20.450000, 20.475000, 20.500000, 20.525000, 20.550000, 20.575000, 20.600000, 20.625000, 20.650000,
    20.675000, 20.700000, 20.725000, 20.750000, 20.775000, 20.800000, 20.825000, 20.850000, 20.875000, 20.900000,
    20.925000, 20.950000, 20.975000, 21.000000, 21.025000, 21.050000, 21.075000, 21.100000, 21.125000, 21.150000,
    21.175000, 21.200000, 21.225000, 21.250000, 21.275000, 21.300000, 21.325000, 21.350000, 21.375000, 21.400000,
    21.425000, 21.450000, 21.475000, 21.500000, 21.525000, 21.550000, 21.575000, 21.600000, 21.625000, 21.650000,
    21.675000, 21.700000, 21.725000, 21.750000, 21.775000, 21.800000, 21.825000, 21.850000, 21.875000, 21.900000,
    21.925000, 21.950000, 21.975000, 22.000000, 22.025000, 22.050000, 22.075000, 22.100000, 22.125000, 22.150000,
    22.175000, 22.200000, 22.225000, 22.250000, 22.275000, 22.300000, 22.325000, 22.350000, 22.375000, 22.400000,
    22.425000, 22.450000, 22.475000, 22.500000, 22.525000, 22.550000, 22.575000, 22.600000, 22.625000, 22.650000,
    22.675000, 22.700000, 22.725000, 22.750000, 22.775000, 22.800000, 22.825000, 22.850000, 22.875000, 22.900000,
    22.925000, 22.950000, 22.975000, 23.000000, 23.025000, 23.050000, 23.075000, 23.100000, 23.125000, 23.150000,
    23.175000, 23.200000, 23.225000, 23.250000, 23.275000, 23.300000, 23.325000, 23.350000, 23.375000, 23.400000,
    23.425000, 23.450000, 23.475000, 23.500000, 23.525000, 23.550000, 23.575000, 23.600000, 23.625000, 23.650000,
    23.675000, 23.700000, 23.725000, 23.750000, 23.775000, 23.800000, 23.825000, 23.850000, 23.875000, 23.900000,
    23.925000, 23.950000, 23.975000, 24.000000, 24.025000, 24.050000, 24.075000, 24.100000, 24.125000, 24.150000,
    24.175000, 24.200000, 24.225000, 24.250000, 24.275000, 24.300000, 24.325000, 24.350000, 24.375000, 24.400000,
    24.425000, 24.450000, 24.475000, 24.500000, 24.525000, 24.550000, 24.575000, 24.600000, 24.625000, 24.650000,
    24.675000, 24.700000, 24.725000, 24.750000, 24.775000, 24.800000, 24.825000, 24.850000, 24.875000, 24.900000,
    24.925000, 24.950000, 24.975000, ])

diff_array = np.array(
    [6.430845, 6.430845, 6.430845, 6.430845, 6.430845, 6.430845, 6.430845, 6.430845, 6.430845, 6.430845,
     6.430845, 6.430845, 6.430845, 6.430845, 6.430845, 6.430845, 6.430845, 6.512938, 6.558730,
     6.618283, 6.691507, 6.747961, 6.805342, 6.864190, 6.924869, 6.984576, 7.034866, 7.094118,
     7.186708, 7.255293, 7.369529, 7.471169, 7.584982, 7.720884, 7.877707, 8.021136, 8.221140,
     8.385709, 8.620608, 8.854704, 9.082699, 9.373518, 9.685242, 10.018807, 10.408773, 10.774321,
     11.220205, 11.734989, 12.189855, 12.799583, 13.424802, 14.033961, 14.828550, 15.637616,
     16.528005, 17.553852, 18.509427, 19.672152, 20.980018, 22.225444, 23.788452, 25.386487,
     27.036841, 29.034846, 30.713986, 32.868410, 35.025720, 37.139066, 39.391172, 41.922414,
     43.817005, 46.484839, 48.575271, 50.825245, 52.689323, 54.331170, 55.775327, 57.051316,
     57.746737, 58.359737, 58.590118, 58.557636, 58.241851, 57.763937, 57.046041, 56.120137,
     55.249168, 54.148690, 53.041939, 52.022613, 50.833869, 49.712209, 48.679306, 47.571501,
     46.740688, 45.791963, 44.973048, 44.261147, 43.539629, 42.897730, 42.380771, 41.859178,
     41.480511, 41.117797, 40.806306, 40.576356, 40.346120, 40.156909, 39.998983, 39.845448,
     39.684667, 39.519355, 39.298034, 39.087556, 38.761152, 38.386377, 37.948934, 37.395540,
     36.724170, 36.009767, 35.067600, 34.230496, 33.151075, 32.012148, 30.896171, 29.615910,
     28.334665, 27.296310, 25.968489, 24.866814, 23.697363, 22.622744, 21.666648, 20.747973,
     19.788553, 19.089005, 18.250963, 17.614968, 16.975594, 16.406753, 15.936136, 15.494766,
     15.064401, 14.744281, 14.411503, 14.123595, 13.904062, 13.668552, 13.478125, 13.311514,
     13.171059, 13.059964, 12.958723, 12.872699, 12.806203, 12.746898, 12.700798, 12.663861,
     12.634985, 12.611581, 12.594536, 12.582567, 12.570351, 12.560885, 12.552072, 12.543427,
     12.538352, 12.529939, 12.518506, 12.506630, 12.485074, 12.464020, 12.440043, 12.412239,
     12.379191, 12.340591, 12.300900, 12.253566, 12.205000, 12.147196, 12.086557, 12.023165,
     11.949716, 11.870333, 11.787872, 11.695629, 11.609011, 11.509859, 11.400811, 11.303982,
     11.190329, 11.073118, 10.963393, 10.833159, 10.706273, 10.573835, 10.432644, 10.305310,
     10.161613, 10.011286, 9.874392, 9.714757, 9.564727, 9.418716, 9.254641, 9.118354, 8.956975,
     8.799988, 8.652302, 8.501997, 8.336357, 8.204657, 8.032491, 7.895680, 7.736404, 7.594724,
     7.455319, 7.309647, 7.161374, 7.044166, 6.900234, 6.776631, 6.653026, 6.528395, 6.419981,
     6.306042, 6.193446, 6.103131, 5.999002, 5.904168, 5.813852, 5.726285, 5.650688, 5.573461,
     5.493820, 5.433654, 5.364794, 5.303488, 5.251948, 5.199085, 5.151778, 5.108536, 5.067690,
     5.035134, 5.003769, 4.972889, 4.948301, 4.922753, 4.899693, 4.878327, 4.860623, 4.844982,
     4.830265, 4.817288, 4.807395, 4.799095, 4.790518, 4.782164, 4.773535, 4.766960, 4.761299,
     4.755625, 4.750504, 4.743854, 4.737513, 4.731198, 4.725617, 4.717564, 4.709165, 4.701209,
     4.691198, 4.680475, 4.669663, 4.658660, 4.646006, 4.634106, 4.617551, 4.600199, 4.585069,
     4.566431, 4.547004, 4.528057, 4.505861, 4.483778, 4.459231, 4.432629, 4.407584, 4.380553,
     4.350146, 4.325339, 4.292921, 4.262261, 4.231714, 4.199936, 4.166851, 4.133527, 4.096996,
     4.064985, 4.026295, 3.989434, 3.954996, 3.917034, 3.880308, 3.846516, 3.807437, 3.776216,
     3.738617, 3.704347, 3.669291, 3.633618, 3.595840, 3.562491, 3.526208, 3.497385, 3.463433,
     3.431269, 3.403474, 3.372887, 3.344222, 3.317633, 3.289664, 3.262596, 3.237217, 3.211414,
     3.190967, 3.167043, 3.145921, 3.127057, 3.106691, 3.089240, 3.072915, 3.056063, 3.041761,
     3.026392, 3.013442, 3.002508, 2.990773, 2.979575, 2.970284, 2.961423, 2.952315, 2.945167,
     2.937195, 2.930805, 2.925376, 2.919704, 2.915653, 2.912154, 2.906745, 2.903406, 2.901143,
     2.897586, 2.894781, 2.892854, 2.890040, 2.886656, 2.883233, 2.880299, 2.877606, 2.875250,
     2.872169, 2.869277, 2.866288, 2.862988, 2.858372, 2.853965, 2.850295, 2.845837, 2.839433,
     2.833702, 2.827940, 2.820992, 2.815630, 2.809845, 2.801203, 2.793537, 2.785685, 2.776469,
     2.767777, 2.757430, 2.747485, 2.737409, 2.725792, 2.716072, 2.704683, 2.693004, 2.682451,
     2.670464, 2.659157, 2.647366, 2.633316, 2.620727, 2.607567, 2.594014, 2.582082, 2.568770,
     2.557095, 2.544040, 2.531361, 2.518556, 2.504604, 2.490923, 2.478561, 2.465416, 2.452240,
     2.437887, 2.426132, 2.415287, 2.403771, 2.390690, 2.380115, 2.368199, 2.358315, 2.348351,
     2.337399, 2.327500, 2.318586, 2.308940, 2.299277, 2.290720, 2.282071, 2.274826, 2.267562,
     2.261725, 2.254402, 2.248008, 2.242165, 2.236151, 2.230429, 2.225072, 2.219465, 2.215304,
     2.210028, 2.204834, 2.202197, 2.198993, 2.196756, 2.193554, 2.190952, 2.187596, 2.184464,
     2.180817, 2.178339, 2.174533, 2.172445, 2.169681, 2.167355, 2.165013, 2.161863, 2.159989,
     2.157823, 2.154636, 2.152083, 2.149195, 2.146250, 2.144636, 2.142437, 2.140197, 2.136169,
     2.132821, 2.128920, 2.124833, 2.122008, 2.118706, 2.114665, 2.110851, 2.105799, 2.101573,
     2.095735, 2.089293, 2.083390, 2.078599, 2.074922, 2.069861, 2.063794, 2.057039, 2.051701,
     2.046261, 2.040684, 2.034634, 2.028759, 2.020961, 2.015151, 2.008529, 2.002462, 1.996577,
     1.989736, 1.983813, 1.977643, 1.970269, 1.963402, 1.955568, 1.947368, 1.941431, 1.934423,
     1.927761, 1.920621, 1.915044, 1.908474, 1.902262, 1.896118, 1.889914, 1.883588, 1.878931,
     1.871336, 1.865242, 1.859572, 1.854502, 1.849044, 1.844128, 1.837274, 1.832131, 1.827582,
     1.824265, 1.818962, 1.812715, 1.808704, 1.805336, 1.800757, 1.795580, 1.791883, 1.788042,
     1.784740, 1.781541, 1.779088, 1.776511, 1.772732, 1.769002, 1.766110, 1.762773, 1.759578,
     1.756953, 1.755220, 1.752113, 1.749117, 1.746154, 1.743429, 1.739016, 1.735889, 1.733771,
     1.732181, 1.729359, 1.726217, 1.722410, 1.720706, 1.717850, 1.715347, 1.713076, 1.710787,
     1.707858, 1.703906, 1.700042, 1.696489, 1.693548, 1.690346, 1.685749, 1.682567, 1.680217,
     1.677652, 1.674174, 1.670251, 1.665156, 1.661961, 1.657982, 1.653668, 1.650170, 1.646832,
     1.641765, 1.636681, 1.633621, 1.630475, 1.624414, 1.619790, 1.615485, 1.610736, 1.605895,
     1.602356, 1.598819, 1.593857, 1.589067, 1.585835, 1.581391, 1.576579, 1.572052, 1.567700,
     1.563635, 1.557598, 1.552454, 1.547546, 1.542587, 1.538272, 1.534231, 1.529142, 1.525452,
     1.521506, 1.516869, 1.512705, 1.508019, 1.503466, 1.499365, 1.495122, 1.490444, 1.486357,
     1.482150, 1.479013, 1.475760, 1.471414, 1.466999, 1.464414, 1.460478, 1.456780, 1.452788,
     1.448569, 1.444074, 1.441170, 1.437290, 1.434686, 1.431937, 1.428119, 1.424199, 1.421259,
     1.417218, 1.413375, 1.410759, 1.407357, 1.404388, 1.401468, 1.399042, 1.397032, 1.393609,
     1.390764, 1.386229, 1.382098, 1.377734, 1.374828, 1.372974, 1.370863, 1.367882, 1.365477,
     1.362126, 1.358589, 1.355920, 1.353060, 1.348985, 1.345744, 1.342519, 1.340099, 1.337150,
     1.333978, 1.331236, 1.327657, 1.323693, 1.320901, 1.317384, 1.313977, 1.308823, 1.306356,
     1.302777, 1.298856, 1.296234, 1.293068, 1.289935, 1.287209, 1.283237, 1.279284, 1.276181,
     1.272202, 1.267563, 1.264065, 1.260392, 1.257739, 1.254555, 1.249488, 1.245408, 1.241140,
     1.238343, 1.235105, 1.231948, 1.229279, 1.225930, 1.221135, 1.218371, 1.215060, 1.211512,
     1.208000, 1.204916, 1.201025, 1.198342, 1.194956, 1.191899, 1.188623, 1.184095, 1.180096,
     1.177021, 1.173391, 1.169793, 1.167240, 1.163279, 1.160928, 1.158051, 1.154862, 1.151415,
     1.148007, 1.144442, 1.141056, 1.138405, 1.135172, 1.131362, 1.128658, 1.126203, 1.123530,
     1.120614, 1.117385, 1.114360, 1.110829, 1.107733, 1.104846, 1.102505, 1.099815, 1.098074,
     1.094724, 1.091280, 1.088526, 1.085175, 1.082842, 1.079995, 1.078113, 1.075396, 1.071858,
     1.068253, 1.065357, 1.062267, 1.059210, 1.055924, 1.053564, 1.051033, 1.049316, 1.045701,
     1.043165, 1.040269, 1.037200, 1.034639, 1.032077, 1.029183, 1.025900, 1.023176, 1.020123,
     1.017467, 1.014441, 1.011526, 1.007948, 1.004642, 1.001059, 0.998694, 0.996941, 0.994604,
     0.991948, 0.988997, 0.985647, 0.981828, 0.978941, 0.975886, 0.973342, 0.970200, 0.967662,
     0.965952, 0.963169, 0.959811, 0.956389, 0.953213, 0.949698, 0.946632, 0.943755, 0.940956,
     0.938102, 0.934320, 0.932841, 0.930322, 0.926985, 0.922853, 0.920216, 0.917235, 0.914982,
     0.912324, 0.909871, 0.907718, 0.904706, 0.901742, 0.897863, 0.895660, 0.892699, 0.889668,
     0.886971, 0.884407, 0.882065, 0.879503, 0.876375, 0.874619, 0.872768, 0.869359, 0.865943,
     0.863821, 0.861533, 0.859906, 0.857051, 0.853083, 0.849975, 0.847705, 0.845824, 0.844121,
     0.841124, 0.837943, 0.834680, 0.833088, 0.831144, 0.829375, 0.827185, 0.823606, 0.821060,
     0.818832, 0.815629, 0.812965, 0.810566, 0.809128, 0.806544, 0.803973, 0.800968, 0.798549,
     0.796751, 0.794830, 0.792188, 0.789738, 0.787079, 0.784717, 0.782293, 0.780306, 0.777995,
     0.775317, 0.772536, 0.770926, 0.769466, 0.767731, 0.765562, 0.763116, 0.760436, 0.757533,
     0.755779, 0.753899, 0.751389, 0.748357, 0.746592, 0.744833, 0.743052, 0.740796, 0.738103,
     0.735227, 0.733006, 0.730098, 0.728031, 0.726672, 0.724533, 0.722763, 0.720228, 0.718228,
     0.714796, 0.711990, 0.709641, 0.708469, 0.706828, 0.704823, 0.702258, 0.699609, 0.697854,
     0.695744, 0.693021, 0.690340, 0.687920, 0.686369, 0.685049, 0.684501, 0.682444, 0.679410,
     0.675886, 0.673654, 0.671876, 0.668928, 0.665649, 0.663588, 0.661334, 0.659510, 0.657376,
     0.655269, 0.653340, 0.651122, 0.649710, 0.648678, 0.646505, 0.643362, 0.640792, 0.639072,
     0.637294, 0.635286, 0.633179, 0.631928, 0.630570, 0.628751, 0.626891, 0.624264, 0.622209,
     0.620291, 0.618722, 0.616996, 0.615135, 0.611952, 0.610032, 0.607627, 0.605612, 0.603983,
     0.602685, 0.600668, 0.598608, 0.595597, 0.593011, 0.591910, 0.590792, 0.588336, 0.586069,
     0.584846, 0.583269, 0.581518, 0.579541, 0.579070, 0.577103, 0.574424, 0.571932, 0.569806,
     0.567590, 0.565612, 0.564134, 0.562445, 0.560844, 0.558679, 0.557002, 0.555180, 0.553456,
     0.552001, 0.550392, 0.548593, 0.546023, 0.543457, 0.542255, 0.541513, 0.538928, 0.536781,
     0.535829, 0.534113, 0.532195, 0.530560, 0.528439, 0.527144, 0.525988, 0.523683, 0.522525,
     0.520966, 0.518883, 0.516411, 0.514720, 0.513690, 0.511354, 0.509600, 0.508425, 0.506524,
     0.505256, 0.504369, 0.501946, 0.500960, 0.499259, 0.496844, 0.495156, 0.494846, 0.493596,
     0.491182, 0.487864, 0.486554, 0.484742, 0.482504, 0.481306, 0.479986, 0.479395, 0.478598,
     0.477078, 0.474625, 0.473057, 0.471968, 0.470666, 0.468150, 0.466484, 0.465359, 0.464180,
     0.462473, ])

##########################################################
#   Data for absorption
##########################################################
"""
Data are taken from 
https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/water.html
"""
energy_array_eV = np.array([
    1.00000E-03,
    1.50000E-03,
    2.00000E-03,
    3.00000E-03,
    4.00000E-03,
    5.00000E-03,
    6.00000E-03,
    8.00000E-03,
    1.00000E-02,
    1.50000E-02,
    2.00000E-02,
    3.00000E-02,
    4.00000E-02,
    5.00000E-02,
    6.00000E-02,
    8.00000E-02,
    1.00000E-01,
    1.50000E-01,
    2.00000E-01,
    3.00000E-01,
    4.00000E-01,
    5.00000E-01,
    6.00000E-01,
    8.00000E-01,
    1.00000E+00,
    1.25000E+00,
    1.50000E+00,
    2.00000E+00,
    3.00000E+00,
    4.00000E+00,
    5.00000E+00,
    6.00000E+00,
    8.00000E+00,
    1.00000E+01,
    1.50000E+01,
    2.00000E+01, ]) * 1e6

# The unit is in m^2 / kg
mu_array = np.array([
    4.078E+03,
    1.376E+03,
    6.173E+02,
    1.929E+02,
    8.278E+01,
    4.258E+01,
    2.464E+01,
    1.037E+01,
    5.329E+00,
    1.673E+00,
    8.096E-01,
    3.756E-01,
    2.683E-01,
    2.269E-01,
    2.059E-01,
    1.837E-01,
    1.707E-01,
    1.505E-01,
    1.370E-01,
    1.186E-01,
    1.061E-01,
    9.687E-02,
    8.956E-02,
    7.865E-02,
    7.072E-02,
    6.323E-02,
    5.754E-02,
    4.942E-02,
    3.969E-02,
    3.403E-02,
    3.031E-02,
    2.770E-02,
    2.429E-02,
    2.219E-02,
    1.941E-02,
    1.813E-02,
]) / 10.