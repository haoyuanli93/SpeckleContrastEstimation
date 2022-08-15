import numpy as np
from numba import jit
import math
from ContrastEstimation.AtomFormFactor import get_atomic_formfactor
from ContrastEstimation.util import re0


def load_atom_info(file_name):
    """
    Parse the lammps output file
    :param file_name:
    :return:
    """
    with open(file_name, 'r') as data_file:
        #################################
        # Get the atom number
        #################################
        for idx in range(4):
            my_line = data_file.readline()
        atom_num = int(my_line)

        print("There are {:.2e} atoms in this file.".format(atom_num))

        ################################
        # Get the box size
        ################################
        _ = data_file.readline()

        my_line = data_file.readline().split()
        xlo = float(my_line[0])
        xhi = float(my_line[1])

        my_line = data_file.readline().split()
        ylo = float(my_line[0])
        yhi = float(my_line[1])

        my_line = data_file.readline().split()
        zlo = float(my_line[0])
        zhi = float(my_line[1])

        ################################
        # Get the atom type and position
        ################################
        # Create holders for different info
        type_holder = np.zeros(atom_num, dtype=np.int64)
        position_holder = np.zeros((atom_num, 3), dtype=np.float64)

        # Skip the line of description
        _ = data_file.readline().split()

        # Loop through the remaining lines
        for atom_idx in range(atom_num):
            info = data_file.readline().split()
            type_holder[atom_idx] = int(info[1])
            position_holder[atom_idx, :] = np.array([float(info[2]), float(info[3]), float(info[4])],
                                                    dtype=np.float64)

    return atom_num, np.array([[xlo, xhi], [ylo, yhi], [zlo, zhi]]), type_holder, position_holder


def categorize_atoms(atom_types, position_holder):
    """

    :param atom_types:
    :param position_holder:
    :return:
    """

    # Sort the atom_types array based on the type
    sorted_idx = np.argsort(atom_types)

    # Get the sorted array
    atom_type_sorted = atom_types[sorted_idx]
    position_sorted = position_holder[sorted_idx]

    # Get the index of the start and end of each kind of atoms
    atom_type_list, atom_type_start_idx, atom_type_count = np.unique(atom_type_sorted,
                                                                     return_index=True,
                                                                     return_counts=True)

    return (atom_type_list, atom_type_start_idx, atom_type_count,
            np.ascontiguousarray(atom_type_sorted), np.ascontiguousarray(position_sorted))


def get_q_vector_list_in_range(box_size_xyz_A, q_low_A, q_high_A):
    """
    Get the q vector list in the range between q_low_A and q_high_A

    :param box_size_xyz_A:
    :param q_low_A:
    :param q_high_A:
    :return:
    """

    q_min_x = np.pi * 2 / box_size_xyz_A[0]
    q_min_y = np.pi * 2 / box_size_xyz_A[1]
    q_min_z = np.pi * 2 / box_size_xyz_A[2]

    # Get the number of q to calculate
    q_num_x = int(q_high_A / q_min_x) + 1
    q_num_y = int(q_high_A / q_min_y) + 1
    q_num_z = int(q_high_A / q_min_z) + 1

    # Define a Q grid
    q_grid = np.zeros((2 * q_num_x + 1,
                       2 * q_num_y + 1,
                       2 * q_num_z + 1,
                       3), dtype=np.float64)

    q_grid[:, :, :, 0] = q_min_x * np.arange(start=-q_num_x, stop=q_num_x + 1, step=1)[:, np.newaxis, np.newaxis]
    q_grid[:, :, :, 1] = q_min_y * np.arange(start=-q_num_y, stop=q_num_y + 1, step=1)[np.newaxis, :, np.newaxis]
    q_grid[:, :, :, 2] = q_min_z * np.arange(start=-q_num_z, stop=q_num_z + 1, step=1)[np.newaxis, np.newaxis, :]

    q_length = np.linalg.norm(q_grid, axis=-1)

    # Reshape the Q grid
    q_num_tot = (2 * q_num_x + 1) * (2 * q_num_y + 1) * (2 * q_num_z + 1)
    q_grid = np.reshape(q_grid, newshape=(q_num_tot, 3))
    q_length = np.reshape(q_length, newshape=q_num_tot)

    # Get the q_list with in the range
    return np.ascontiguousarray(q_grid[(q_length < q_high_A) & (q_length > q_low_A)])


###############################################################
#     Calculate the diffraction intensity with GPU
#
#     I borrow this piece of code from my previous implementation of the pysingfel
###############################################################
@jit(parallel=True)
def _get_MD_formfactor_at_Q_list(cos_holder, sin_holder,
                                 form_factor_list, q_list, atom_position,
                                 split_idx, atom_type_num, q_num, atom_num):
    for atom_iter in range(atom_num):

        # Determine the atom type
        atom_type = 0
        for atom_type_idx in range(atom_type_num):
            atom_type += int(bool(atom_iter > split_idx[atom_type_idx]))

        # Calculate the Q
        for q_idx in range(q_num):
            form_factor = form_factor_list[atom_type, q_idx]

            phase = (q_list[q_idx, 0] * atom_position[atom_iter, 0] +
                     q_list[q_idx, 1] * atom_position[atom_iter, 1] +
                     q_list[q_idx, 2] * atom_position[atom_iter, 2])

            cos_holder[q_idx] += form_factor * math.cos(phase)
            sin_holder[q_idx] += form_factor * math.sin(phase)


def get_MD_formfactor_at_Q_list(q_list_A, atom_position_array, atom_type_array, atom_type_name_list):
    """

    :param q_list_A:
    :param atom_position_array:
    :param atom_type_array:
    :param atom_type_name_list:
    :return:
    """

    # convert the reciprocal space into a 1d series.
    q_len_array = np.linalg.norm(q_list_A, axis=-1)
    q_num = q_list_A.shape[0]

    # Organize the atom info
    (atom_type_unique,
     atom_type_start_point,
     atom_type_count,
     atom_type_sorted,
     atom_position_sorted) = categorize_atoms(atom_types=atom_type_array, position_holder=atom_position_array)

    # Get the form factor of each atom at each reciprocal point
    form_factor_list = np.zeros((len(atom_type_name_list), q_num), dtype=np.float64)
    for atom_type_idx in range(len(atom_type_name_list)):
        for q_idx in range(q_num):
            form_factor_list[atom_type_idx, q_idx] = get_atomic_formfactor(atom_name=atom_type_name_list[atom_type_idx],
                                                                           q_detector_in_A=q_len_array[q_idx])

    # create
    cos_holder = np.zeros(q_num, dtype=np.float64)
    sin_holder = np.zeros(q_num, dtype=np.float64)

    # Calculate the pattern
    _get_MD_formfactor_at_Q_list(cos_holder,
                                 sin_holder,
                                 form_factor_list,
                                 np.ascontiguousarray(q_list_A),
                                 atom_position_sorted,
                                 atom_type_start_point,
                                 len(atom_type_name_list),
                                 q_num,
                                 atom_type_array.shape[0])

    return cos_holder + 1.j * sin_holder


@jit(parallel=True)
def _get_MD_formfactor_at_Q_list_parallel_at_Q(cos_holder, sin_holder,
                                               form_factor_list, q_list, atom_position,
                                               split_idx, atom_type_num):
    # Loop through all types of atoms
    for atom_type_idx in range(atom_type_num):
        form_factor = form_factor_list[atom_type_idx, q_idx]

        # Loop through each of the atom in this type
        for atom_idx in range(split_idx[atom_type_idx], split_idx[atom_type_idx + 1]):
            phase = (q_list[:, 0] * atom_position[atom_idx, 0] +
                     q_list[:, 1] * atom_position[atom_idx, 1] +
                     q_list[:, 2] * atom_position[atom_idx, 2])

            cos_holder[:] += form_factor * math.cos(phase)
            sin_holder[:] += form_factor * math.sin(phase)


def get_MD_formfactor_at_Q_list_parallel_at_Q(q_list_A, atom_position_array, atom_type_array, atom_type_name_list):
    """

    :param q_list_A:
    :param atom_position_array:
    :param atom_type_array:
    :param atom_type_name_list:
    :return:
    """

    # convert the reciprocal space into a 1d series.
    q_len_array = np.linalg.norm(q_list_A, axis=-1)
    q_num = q_list_A.shape[0]
    atom_num = atom_type_array.shape[0]

    # Organize the atom info
    (atom_type_unique,
     atom_type_start_point,
     atom_type_count,
     atom_type_sorted,
     atom_position_sorted) = categorize_atoms(atom_types=atom_type_array, position_holder=atom_position_array)
    atom_type_num = len(atom_type_name_list)

    # Construct the split idx
    split_idx = np.zeros(atom_type_start_point.shape[0] + 1, dtype=np.int64)
    split_idx[:atom_type_start_point.shape[0]] = atom_type_start_point[:]
    split_idx[-1] = atom_type_array.shape[0]

    # Get the form factor of each atom at each reciprocal point
    form_factor_list = np.zeros((atom_type_num, q_num), dtype=np.float64)
    for atom_type_idx in range(atom_type_num):
        for q_idx in range(q_num):
            form_factor_list[atom_type_idx, q_idx] = get_atomic_formfactor(atom_name=atom_type_name_list[atom_type_idx],
                                                                           q_detector_in_A=q_len_array[q_idx])

    # create
    cos_holder = np.zeros(q_num, dtype=np.float64)
    sin_holder = np.zeros(q_num, dtype=np.float64)

    # Calculate the pattern
    _get_MD_formfactor_at_Q_list_parallel_at_Q(cos_holder,
                                               sin_holder,
                                               form_factor_list,
                                               q_list_A,
                                               atom_position_sorted,
                                               split_idx,
                                               atom_type_num,)

    return cos_holder + 1.j * sin_holder


def get_diffracted_flux_with_MD_formfactor(in_flux, dOmega, q_in, d_eff_m, box_size_A,
                                           q_list_A, formfactorMD):
    """

    :param in_flux:
    :param dOmega:
    :param q_in:
    :param d_eff_m:
    :param box_size_A:
    :param q_list_A:
    :param formfactorMD:
    :return:
    """

    # convert the reciprocal space into a 1d series.
    q_len_array = np.linalg.norm(q_list_A, axis=-1)

    # Get the polarization factor
    polarization_factor = np.square(1 - 0.5 * np.square(q_len_array / 2. / q_in))

    # Get the intensity
    intensity = np.multiply(np.square(np.abs(formfactorMD)), polarization_factor)
    intensity *= (in_flux * dOmega * (re0 ** 2))
    intensity *= (d_eff_m * 1e10 / (box_size_A[2, 1] - box_size_A[2, 0]))

    return intensity
