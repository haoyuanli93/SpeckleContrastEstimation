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


def convert_density_mole_L_to_g_cm3(molecule_structure, density_mole_L):
    """

    :param molecule_structure:
    :param density_mole_L:
    :return:
    """
    molecule_mass = get_molecule_molar_mass_in_g(molecule_structure=molecule_structure)

    return density_mole_L * molecule_mass / 1000.


def convert_density_g_cm3_to_mole_L(molecule_structure, density_g_cm3):
    """

    :param molecule_structure:
    :param density_g_cm3:
    :return:
    """
    molecule_mass = get_molecule_molar_mass_in_g(molecule_structure=molecule_structure)
    return density_g_cm3 / molecule_mass * 1000.


def get_molecule_molar_mass_in_g(molecule_structure):
    """
    Get the molecule molar mass.
    i.e. the mass of a mole of the molecule measured in gram
    :param molecule_structure:
    :return:
    """
    molar_mass = 0.

    atom_num = len(molecule_structure)
    for atom_idx in range(atom_num):
        molar_mass += atom_mass_list[molecule_structure[atom_idx][0]][1]

    return molar_mass


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

        for atom_idx2 in range(atom_num):
            for atom_idx3 in range(atom_num):
                aff1 = AtomFormFactor.get_atomic_formfactor(atom_name=molecule_structure[atom_idx2][0],
                                                            q_detector_in_A=q_detector_in_A)
                aff2 = AtomFormFactor.get_atomic_formfactor(atom_name=molecule_structure[atom_idx3][0],
                                                            q_detector_in_A=q_detector_in_A)
                if atom_idx3 == atom_idx2:
                    continue
                else:
                    phase = q_detector_in_A * np.linalg.norm(molecule_structure[atom_idx2][1] -
                                                             molecule_structure[atom_idx3][1])

                    mff += aff1 * aff2 * np.sinc(phase / np.pi)

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
    molecular_mass = get_molecule_molar_mass_in_g(molecule_structure=molecule_structure)

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


def get_attenuation_coefficient(molecule_structure, photon_energy_keV, density):
    """
    Get the attenuation coefficient

    :param molecule_structure:
    :param photon_energy_keV:
    :param density: The density of each kind of molecules in this compound. The unit is g / cm^3
    :return:
    """

    # Get the total attenuation coefficient
    mu_rho = get_mass_attenuation_coefficient(molecule_structure, photon_energy_keV)
    total_attenuation_coefficient = mu_rho * density

    return total_attenuation_coefficient


def get_attenuation_length_cm(molecule_structure, photon_energy_keV, partial_density):
    """

    :param molecule_structure:
    :param photon_energy_keV:
    :param partial_density:  The density of each kind of molecules in this compound. The unit is g / cm^3
    :return:
    """
    return 1. / get_attenuation_coefficient(molecule_structure, photon_energy_keV, partial_density)


def get_differential_crosssection_for_uniform_sample(molecule_structure,
                                                     density_g_cm3,
                                                     q_detector_in_A):
    """
    The differential cross section obtained in this function
    has been normalized by the sample thickness.
    Therefore, it has a unit of m^-1

    :param molecule_structure:
    :param density_g_cm3:  The unit is in mol / L
    :param q_detector_in_A:
    :return:
    """
    # Get the molecular form factor
    mff = get_molecular_formfactor_for_uniform_sample(molecule_structure=molecule_structure,
                                                      q_detector_in_A=q_detector_in_A)

    # Convert the density in g/cm^3 to mole/m^3
    molecule_mass = get_molecule_molar_mass_in_g(molecule_structure=molecule_structure)
    density_mole_m3 = density_g_cm3 * (100 ** 3) / molecule_mass

    return N_A * density_mole_m3 * (mff * re0) ** 2


def get_differential_crosssection_with_structure_factor(molecule_structure,
                                                        density_g_cm3,
                                                        q_detector_in_A,
                                                        structure_factor):
    """
    The differential cross section obtained in this function
    has been normalized by the sample thickness.
    Therefore, it has a unit of m^-1

    :param molecule_structure:
    :param density_g_cm3:  The unit is in mol / L
    :param q_detector_in_A:
    :param structure_factor:
    :return:
    """
    differential_crosssection = get_differential_crosssection_for_uniform_sample(molecule_structure,
                                                                                 density_g_cm3,
                                                                                 q_detector_in_A)

    return differential_crosssection * structure_factor


def get_scatter_intensity_with_differetial_crosssection(diff_cross_list,
                                                        atten_length,
                                                        sample_thickness,
                                                        pixel_size,
                                                        detector_distance,
                                                        incident_photon_count):
    """
    Get the scattering intensity for a given differential cross section.

    :param diff_cross_list:
    :param atten_length:
    :param sample_thickness:
    :param pixel_size:
    :param detector_distance:
    :param incident_photon_count:
    :return:
    """
    # Effective sample thickness
    d_eff = atten_length * (1 - np.exp(-sample_thickness / atten_length))

    # Solid angle spanned by the pixel
    d_omega = (pixel_size / detector_distance) ** 2

    return incident_photon_count * np.sum(diff_cross_list) * d_omega * d_eff


def get_scatter_intensity_with_a_unifrom_sample(molecule_structure_list,
                                                density_g_cm3_list,
                                                sample_thickness,
                                                pixel_size,
                                                detector_distance,
                                                incident_photon_count,
                                                q_detector,
                                                photon_energy_keV):
    """
    Calculate the scattering intensity for a specific
    detector distance and sample thickness.

    :param molecule_structure_list:
    :param density_g_cm3_list:
    :param sample_thickness:
    :param pixel_size:
    :param detector_distance:
    :param incident_photon_count:
    :param q_detector:
    :param photon_energy_keV:
    :return:
    """
    # Diffraction cross section
    diff_list = [get_differential_crosssection_for_uniform_sample(
        molecule_structure=molecule_structure_list[x],
        density_g_cm3=density_g_cm3_list[x],
        q_detector_in_A=q_detector) for x in range(len(molecule_structure_list))]

    attenuation_coef_list = [get_attenuation_coefficient(molecule_structure=molecule_structure_list[x],
                                                         photon_energy_keV=photon_energy_keV,
                                                         density=density_g_cm3_list[x]) for x in
                             range(len(molecule_structure_list))]
    attenuation_length = 1. / np.sum(attenuation_coef_list)  # unit cm

    # Convert to m
    attenuation_length /= 100.

    # Effective sample thickness
    d_eff = attenuation_length * (1 - np.exp(-sample_thickness / attenuation_length))

    # Solid angle spanned by the pixel
    d_omega = (pixel_size / detector_distance) ** 2

    return incident_photon_count * np.sum(diff_list) * d_omega * d_eff


def get_scatter_intensity_with_a_unifrom_sample_batch(molecule_structure_list,
                                                      density_g_cm3_list,
                                                      sample_thickness_list,
                                                      pixel_size,
                                                      detector_distance_list,
                                                      incident_photon_count,
                                                      q_detector,
                                                      photon_energy_keV):
    """
    Get the scattering intensity for a series of
    detector distance and sample thickness


    :param molecule_structure_list:
    :param density_g_cm3_list:
    :param sample_thickness_list:
    :param pixel_size:
    :param detector_distance_list:
    :param incident_photon_count:
    :param q_detector:
    :param photon_energy_keV:
    :return:
    """
    # Diffraction cross section
    diff_list = [get_differential_crosssection_for_uniform_sample(
        molecule_structure=molecule_structure_list[x],
        density_g_cm3=density_g_cm3_list[x],
        q_detector_in_A=q_detector) for x in range(len(molecule_structure_list))]

    attenuation_coef_list = [get_attenuation_coefficient(molecule_structure=molecule_structure_list[x],
                                                         photon_energy_keV=photon_energy_keV,
                                                         density=density_g_cm3_list[x]) for x in
                             range(len(molecule_structure_list))]
    attenuation_length = 1. / np.sum(attenuation_coef_list)  # unit cm

    # Convert to m
    attenuation_length /= 100.

    # Effective sample thickness
    d_eff = attenuation_length * (1 - np.exp(-sample_thickness_list / attenuation_length))

    # Solid angle spanned by the pixel
    d_omega = (pixel_size / detector_distance_list) ** 2

    return incident_photon_count * np.sum(diff_list) * np.outer(d_eff, d_omega)
