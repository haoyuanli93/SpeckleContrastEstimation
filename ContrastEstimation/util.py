"""
This module contains some auxiliary functions.
"""
import numpy as np
from ContrastEstimation.AtomMassParams import atom_mass_list

atom_types = list(atom_mass_list.keys())

h = 4.135667516e-15  # planck constant [eV*s]
c = 299792458  # speed of light [m/s]
N_A = 6.02214086 * 1e23  # Avogadro's Constant
re0 = 2.8179403227 * 1e-15  # classical electron radius


###############################
#   Simple conversion
###############################
def kev_to_wavelength_A(energy_keV):
    """

    :param energy_keV:
    :return:
    """

    lmbda = h * c / (energy_keV * 1000) * 1e10
    return lmbda


def kev_to_wavevec_A(energy_keV):
    """

    :param energy_keV:
    :return:
    """
    lmbda = h * c / (energy_keV * 1000) * 1e10
    wavevec = 2 * np.pi / lmbda
    return wavevec


################################
#   MOlecule mass conversion
################################
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


########################################
#   Automatically calculate the experiment
#   condition to facilitate the processing
########################################
def auto_update_expParams(expParams):
    """

    :param expParams:
    :return:
    """

    # Get central wave-vector
    expParams.update({'incident_wavevec': kev_to_wavevec_A(energy_keV=expParams["photon_energy_keV"])})

    # Get the energy resolution
    expParams.update({'energy_resolution': expParams["bandwidth_keV"] / expParams["photon_energy_keV"]})

    return expParams


########################################
#     IO
########################################
def get_molecule_from_pdb(pdb_file_name):
    """
    Get the molecule structure from the pdb file.

    :param pdb_file_name:
    :return:
    """

    with open(pdb_file_name, 'r') as pdb_file:

        # Define holders
        atoms_list = []  # dict to save atom positions and chain id
        atom_count = 0

        # Read the file
        lines = pdb_file.readlines()

        # Get line number
        line_num = len(lines)

        # Loop through the lines to parse each line
        for line_idx in range(line_num):

            # Get the line
            line = lines[line_idx]

            # Check if this line is about atoms
            if line[0:4] == 'ATOM' or line[0:6] == 'HETATM':
                # Count the atom number
                atom_count += 1

                # Get the atom info
                tmp = [str(line[11:14].strip()).capitalize(), float(line[30:38].strip()), float(line[38:46].strip()),
                       float(line[46:54].strip())]

                # Format the data
                atoms_list.append([tmp[0], np.array(tmp[1:], dtype=np.float64)])

    return atoms_list


def show_formated_molecule_structure(molecule_structure):
    """
    Print the molecule structure that can be copy-paste to the
    molecule zoo directly as source code.


    :param molecule_structure:
    :return:
    """
    atom_num = len(molecule_structure)

    print('[')
    for atom_idx in range(atom_num):
        print("[ \'{}\', np.array([{},{},{}]),],".format(molecule_structure[atom_idx][0],
                                                         molecule_structure[atom_idx][1][0],
                                                         molecule_structure[atom_idx][1][1],
                                                         molecule_structure[atom_idx][1][2],
                                                         ))

    print('],')
