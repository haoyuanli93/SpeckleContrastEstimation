import numpy as np
import h5py as h5


def assemble_image(imgs):
    """
    return the assembled image of size [2080,2080]
    with the 4 epix
    input: list of 4 epix map
    output: map of size (2080,2080) assembled
    """
    shape = [704, 768]
    edge = [170, 140]
    frame = np.zeros([2080, 2080])
    # epix1
    frame[edge[0]:shape[0] + edge[0], edge[1]:shape[1] + edge[1]] = np.rot90(imgs[0], 2)
    # epix2
    frame[edge[0]:shape[0] + edge[0], -edge[1] - shape[1]:-edge[1]] = np.rot90(imgs[1], 2)
    # epix3
    frame[-edge[0] - shape[0]:-edge[0], edge[1]:shape[1] + edge[1]] = imgs[2]
    # epix4
    frame[-edge[0] - shape[0]:-edge[0], -edge[1] - shape[1]:-edge[1]] = imgs[3]
    return frame


def disassemble_image(image):
    shape = [704, 768]
    edge = [170, 140]

    img1 = image[shape[0] + edge[0]: edge[0]: -1, shape[1] + edge[1]: edge[1]: -1]
    img2 = image[shape[0] + edge[0]: edge[0]: -1, -edge[1]: -edge[1] - shape[1]: -1]
    img3 = image[-edge[0] - shape[0]: -edge[0], edge[1]: shape[1] + edge[1]]
    img4 = image[-edge[0] - shape[0]: -edge[0], -edge[1] - shape[1]: -edge[1]]
    return (img1, img2, img3, img4)


def reconstruct_img(photons_i, photons_j, shape):
    """
    Get the 2D photon count image with the location of each photon with the droplet algorithm
    """
    nx, ny = shape
    phot_img, _, _ = np.histogram2d(photons_j + 0.5, photons_i + 0.5, bins=[np.arange(nx + 1), np.arange(ny + 1)])
    return phot_img


def get_ePix_sum(run_num, exp_name):
    smalldata_path = '/sdf/data/lcls/ds/xpp/{}/hdf5/smalldata/'.format(exp_name)

    with h5.File(smalldata_path + '{}_Run{:04d}.h5'.format(exp_name, run_num)) as dataFile:
        epix1 = np.array(dataFile['Sums/epix_alc1_calib'])
        epix2 = np.array(dataFile['Sums/epix_alc2_calib'])
        epix3 = np.array(dataFile['Sums/epix_alc3_calib'])
        epix4 = np.array(dataFile['Sums/epix_alc4_calib'])

    return (epix1, epix2, epix3, epix4)


def get_pixel_position(det_centers=(1912.1455488502193, 812.9391386934965),
                       det_dist=1.59e6):  # All the lengths are in um

    # Assign pixel positions
    pixel_position_holder = np.zeros((2080, 2080, 3))
    pixel_position_holder[:, :, 2] = det_dist
    pixel_position_holder[:, :, 0] = (np.arange(2080) - det_centers[1])[np.newaxis, :] * 50.0
    pixel_position_holder[:, :, 1] = (np.arange(2080) - det_centers[0])[:, np.newaxis] * 50.0

    return pixel_position_holder


def get_pixel_Q(wavelength, pixel_position):
    # Get angle with respect to incident beam
    pixel_direction = pixel_position / (np.linalg.norm(pixel_position, axis=-1))[:, :, np.newaxis]

    # Get the k_out direction
    Q_direction = np.copy(pixel_direction)
    Q_direction[:, :, 2] -= 1

    # Get the Q
    q_map = np.pi * 2 / wavelength * Q_direction
    q_map = np.linalg.norm(q_map, axis=-1)
    return q_map
