import numpy as np

"""   a1   b1   a2   b2   a3   b3   a4   b4   c """
carbon = [2.31, 20.8439, 1.02, 10.2075, 1.5886, 0.5687, 0.865, 51.6512, 0.2156]
oxygen = [3.0485, 13.2771, 2.2868, 5.7011, 1.5463, 0.3239, 0.867, 32.9089, 0.2508]
xeon = [20.2933, 3.9282, 19.0298, 0.344, 8.9767, 26.4659, 1.99, 64.2658, 3.7118]


def get_formfactor(molecular_position, q_detector):
    """

    :param molecule_dict:
    :param wavelength:
    :return:
    """

    molecular_form_factor = 0.

    # Get a random orientation
    for rot_idx in range(10000):


    pass


def get_atomic_formfactor(atom_type, q_detector):

    holder = 0
    for x in range(4):
        holder += atom_type[2 * x] * np.exp(- atom_type[2 * x + 1] * (q_detector / 4. / np.pi) ** 2)

    holder += atom_type[-1]

    return holder



       [[0.122   0.548   0.312],
      [-1.194   0.580   0.312],
      [-2.510   0.611   0.312],]