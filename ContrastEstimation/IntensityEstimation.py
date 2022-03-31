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
from ContrastEstimation import AtomFormFactor


def get_molecular_formfactor_for_uniform_sample(molecular_constitution, q_detector_in_A):
    """
    According to the theory presented in the document,
    the averaged molecular form factor over rotations is

    mff = sqrt( sum_n (|f_n(q)|^2) )

    where n loops through all atoms in this molecule.

    :param molecular_constitution:  A dictionary showing the number of each kind of atoms in a single molecule
    :param q_detector_in_A:
    :return:
    """
    # molecular form factor
    mff = 0.

    # Get the list of all atom kinds
    atom_types = list(molecular_constitution.keys())

    for atom in atom_types:
        # Get the atomic form factor
        aff = AtomFormFactor.get_atomic_formfactor(atom_name=atom, q_detector_in_A=q_detector_in_A)

        # Add the formfactor to the holder
        mff += molecular_constitution[atom] * aff ** 2

    return np.sqrt(mff)


def get_differential_crosssection_for_uniform_sample(molecular_structure, q_detector):
    # Loop through the atomic position

    pass


def get_scatter_intensity_with_a_unifrom_sample(differential, density, sample_thickness, q_detector, q_incident):
    pass
