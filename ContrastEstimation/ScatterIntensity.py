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


def get_differential_crosssection_for_uniform_sample(molecular_structure, q_detector):
    # Loop through the
    pass


def get_scatter_intensity_with_a_unifrom_sample(differential, density, sample_thickness, q_detector, q_incident):
    pass
