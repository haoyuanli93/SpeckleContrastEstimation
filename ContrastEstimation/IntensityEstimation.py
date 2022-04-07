"""
This module aims to calculate the x-ray scattering structure factor
for a given sample under given experiment condition.

1. If we do not have a MD result, then we calculate the structure factor
    based on a gas model, e.i., I assume that particles are distributed uniformly
    across the space.
2. If we have the MD result, then I calculate the structure factor directly
    from the MD result

Notice that, the scattering intensity is defined as

TODO: Finish this introduction

"""
import numpy as np
from scipy import interpolate

from ContrastEstimation import AtomFormFactor
from ContrastEstimation.AtomAttenuationParams import atom_attenuation_param
from ContrastEstimation.AtomMassParams import atom_mass_list

# Define constants
N_A = 6.02214086 * 1e23  # Avogadro's Constant
re0 = 2.8179403227 * 1e-15  # classical electron radius


def get_molecular_formfactor_for_uniform_sample(molecule_structure, q_detector_in_A):
    """
    According to the theory presented in the document,
    the averaged molecular form factor over rotations is

    # Wrong Equation
    mff = sqrt( sum_n (|f_n(q)|^2) )

    where n loops through all atoms in this molecule.

    :param molecule_structure:  A list of the shape
                                [["atom 1", np.array([x1, y1, z1]),],
                                 ["atom 2", np.array([x2, y2, z2]),],
                                 ["atom 3", np.array([x2, y2, z2]),],
                                 ... ]

    :param q_detector_in_A:
    :return:
    """
    # molecular form factor
    mff = 0.

    # Get the list of all atom kinds
    atom_num = len(molecule_structure)

    if atom_num == 1:
        return AtomFormFactor.get_atomic_formfactor(atom_name=molecule_structure[0][0],
                                                    q_detector_in_A=q_detector_in_A)
    else:
        for atom_idx1 in range(atom_num):
            # Get the atomic form factor
            aff1 = AtomFormFactor.get_atomic_formfactor(atom_name=molecule_structure[atom_idx1][0],
                                                        q_detector_in_A=q_detector_in_A)

            # Add the formfactor to the holder
            mff += aff1 ** 2

        for atom_idx2 in range(1, atom_num):
            aff1 = AtomFormFactor.get_atomic_formfactor(atom_name=molecule_structure[atom_idx2][0],
                                                        q_detector_in_A=q_detector_in_A)

            for atom_idx3 in range(atom_idx2 - 1):
                aff2 = AtomFormFactor.get_atomic_formfactor(atom_name=molecule_structure[atom_idx3][0],
                                                            q_detector_in_A=q_detector_in_A)

                phase = q_detector_in_A * np.linalg.norm(molecule_structure[atom_idx2][1] -
                                                         molecule_structure[atom_idx3][1])

                mff += 2 * aff1 * aff2 * np.sin(phase) / phase

        return np.sqrt(mff)


def get_mass_attenuation_coefficient(molecule_structure, energy_keV):
    """
    Get the mass attenuation coefficient of the specified molecule.

    :param energy_keV: The incident photon energy measured in eV.
    :param molecule_structure:
    :return:
    """
    atom_num = len(molecule_structure)

    # Get the molecular mass
    molecular_mass = 0.
    for atom_idx in range(atom_num):
        atom_type = molecule_structure[atom_idx][0]
        molecular_mass += atom_mass_list[atom_type][1]

    molecule_mass_attenuation_coefficient = 0.
    for atom_idx in range(atom_num):
        atom_type = molecule_structure[atom_idx][0]

        # Get the list of the mass attenuation coefficient:
        atom_data = atom_attenuation_param[atom_type]

        energy_list = np.log10(atom_data[:, 0])
        mu_rho_list = atom_data[:, 1]

        # Get the interpolation function
        mu_rho_fun = interpolate.interp1d(energy_list, mu_rho_list, kind='cubic')

        # Get the interpolated value
        mu_rho_fit = mu_rho_fun(np.log10(energy_keV * 1e-3))

        # update the molecule_mass_attenuation_coefficient
        molecule_mass_attenuation_coefficient += atom_mass_list[atom_type][1] / molecular_mass * mu_rho_fit

    return molecule_mass_attenuation_coefficient


def get_attenuation_coefficient(molecule_structure_list, photon_energy_keV, density_list):
    """
    Get the attenuation coefficient

    :param molecule_structure_list:
    :param photon_energy_keV:
    :param density_list: The density of each kind of molecules in this compound. The unit is g / cm^3
    :return:
    """

    # Get the total attenuation coefficient
    total_attenuation_coefficient = 0.
    for idx in range(len(molecule_structure_list)):
        mu_rho = get_mass_attenuation_coefficient(molecule_structure_list[idx], photon_energy_keV)
        total_attenuation_coefficient += mu_rho * density_list[idx]

    return total_attenuation_coefficient


def get_attenuation_length_cm(molecule_structure_list, photon_energy_keV, partial_density_list):
    """

    :param molecule_structure_list:
    :param photon_energy_keV:
    :param partial_density_list:  The density of each kind of molecules in this compound. The unit is g / cm^3
    :return:
    """
    return 1. / get_attenuation_coefficient(molecule_structure_list, photon_energy_keV, partial_density_list)


def get_differential_crosssection_for_uniform_sample(molecule_structure, molecular_molar_density,
                                                     q_detector_in_A):
    """
    The differential cross section obtained in this function
    has been normalized by the sample thickness.
    Therefore, it has a unit of m^-1

    :param molecule_structure:
    :param molecular_molar_density:  The unit is in mol / L
    :param q_detector_in_A:
    :return:
    """
    # Get the molecular form factor
    mff = get_molecular_formfactor_for_uniform_sample(molecule_structure=molecule_structure,
                                                      q_detector_in_A=q_detector_in_A)

    # get the differential cross section
    # The additional factor 1000 convert the molar density from mol/L to mol/m^3.

    return molecular_molar_density * 1000. * N_A * (mff * re0) ** 2


def get_scatter_intensity_with_a_unifrom_sample(differential_list,
                                                density_list,
                                                sample_thickness,
                                                q_detector,
                                                photon_energy_keV):
    pass



