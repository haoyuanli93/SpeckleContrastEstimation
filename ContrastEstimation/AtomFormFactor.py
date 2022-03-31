"""
The purpose of this module is to calculate the
atomic form factor at a specific Q value.

I follow the definition and data shown in webpage:

"""

import numpy as np
from ContrastEstimation.IUCrAtomDataBase import atom_dict

atom_list = list(atom_dict.keys())


def get_available_atom_list():
    """
    Return a list of all atoms available for the calcluation of the atomic form factor.

    :return:
    """
    return atom_list


def get_atomic_formfactor_Gaussian_fitting_parameters(atom_name):
    """
    Get the parameters used in the Gaussian fitting of the atomic form factor.

    :param atom_name:
    :return:
    """
    if atom_name in atom_list:
        return atom_dict[atom_name]
    else:
        print("Warning! this program does not have the data for atom: {}".format(atom_name))
        print("Return a numpy array of np.zeros(9, dtype=np.float64) as a place holder.")

        return np.zeros(9, dtype=np.float64)


def get_atomic_formfactor(atom_name, q_detector_in_A):
    """
    Get the atomic form factor of the specificed atom at the specified detector q.
    Notice that, the detector q is defined in the following way:

    Assume that the incident wave-vector is K
    |K| = 2 pi / wavelength

    Assume that the diffracted wave-vector is K'

    Then the wave-vector change
    Q = K' - K

    Then
    q_detector = |Q|

    The calculation is done according to the formula and data provided in website
    http://lampx.tugraz.at/~hadley/ss1/crystaldiffraction/atomicformfactors/formfactors.php

    :param atom_name:
    :param q_detector_in_A:
    :return:
    """
    atomic_formfactor = 0

    # Get the Gaussian fitting parameters
    gaussian_parameters = get_atomic_formfactor_Gaussian_fitting_parameters(atom_name=atom_name)

    # Perform the fitting calculation
    for x in range(4):
        atomic_formfactor += gaussian_parameters[2 * x] * np.exp(- gaussian_parameters[2 * x + 1]
                                                                 * (q_detector_in_A / 4. / np.pi) ** 2)
    atomic_formfactor += gaussian_parameters[-1]

    return atomic_formfactor
