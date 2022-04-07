"""
This module contains some auxiliary functions.
"""
import numpy as np
from ContrastEstimation.AtomMassParams import atom_mass_list

atom_types = list(atom_mass_list.keys())


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
