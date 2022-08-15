<<<<<<< HEAD
def plot_contrast(data, run, epix, q, beam, weights=None):
    '''
    data is in {'kbar', 'beta'}
    beam is in {'vcc','cc','both'}
    #* (beta < 1) * (beta > -1) * (p1 != 0) * (p2 != 0)
    '''

    prob_name = 'run{:04d}_epx{}_q{}.npz'.format(run, epx, q)
    path = '/cds/home/t/tanduc/Masks_Tan-Duc/may2k22/contrast_files/'
    npz = np.load(path + prob_name)
    p1 = npz['p1']
    kbar = npz['kbar']
    beta = npz['beta']

    file_path = '/cds/data/drpsrcf/XPP/xppc00120/scratch/hdf5/smalldata/'
    fn = 'run{:04d}_epx{}.h5'.format(run, epix)

    with h5py.File(file_path + fn, 'r') as f:
        vcc = np.array(f['vcc'])
        cc = np.array(f['cc'])
        i3 = np.array(f['i3'])

    mean_i3 = np.mean(i3)
    std_i3 = np.std(i3)
    n_beta = len(beta)
    n_vcc = len(vcc)

    vcc = vcc[:n_beta]
    cc = cc[:n_beta]
    i3 = i3[:n_beta]
    both = ((vcc > 4) * (cc > 4)).astype('int')
    x = np.arange(1, len(cc) + 1, 1)

    if beam == 'vcc':

        events = (vcc > 4) * (both == 0) * (i3 >= mean_i3 - std_i3) * (i3 <= mean_i3 + std_i3) * (p1 != 0)
        if data == 'kbar':

            new_data = np.array([kbar[i] for i in range(len(events)) if events[i]])
            avg, err = cum_avg(new_data)
            print('kbar: %.7f' % avg[-1])
        elif data == 'beta':
            new_data = np.array([beta[i] for i in range(len(events)) if events[i]])
            avg, err = cum_avg(new_data)
            print('contrast: %.4f' % avg[-1])
        else:
            print('choose a correct data type')

    elif beam == 'cc':
        events = (cc > 4) * (both == 0) * (i3 >= mean_i3 - std_i3) * (i3 <= mean_i3 + std_i3) * (p1 != 0)

        if data == 'kbar':
            new_data = np.array([kbar[i] for i in range(len(events)) if events[i]])
            avg, err = cum_avg(new_data)
            print('kbar: %.7f' % avg[-1])
        elif data == 'beta':
            new_data = np.array([beta[i] for i in range(len(events)) if events[i]])
            avg, err = cum_avg(new_data)
            print('contrast: %.4f' % avg[-1])
        else:
            print('choose a correct data type')

    elif beam == 'both':
        events = (both == 1)  # * (i3 >= mean_i3 - std_i3) * (i3 <= mean_i3 + std_i3) * (p1 != 0)

        if data == 'kbar':
            new_data = np.array([kbar[i] for i in range(len(kbar)) if events[i]])
            avg, err = cum_avg(new_data)
            print('kbar: %.7f' % avg[-1])
        elif data == 'beta':
            new_data = np.array([beta[i] for i in range(len(beta)) if events[i]])
            avg, err = cum_avg(new_data)
            print('contrast: %.4f' % avg[-1])
        else:
            print('choose a correct data type')

    else:
        print('choose a correct beam configuration')

    print(len(events[events]))
    fig = plt.figure(figsize=(9, 9))
    plt.plot(new_data, '.', color='navy', alpha=0.5)
    plt.plot(avg, color='black')
    plt.fill_between(np.arange(len(avg)), avg - err, avg + err, color='crimson', alpha=0.3)
    plt.xlabel('event #')
    plt.ylabel(data)
    if data == 'beta':
        plt.ylim(-1, 3)
        plt.title('Contrast for run{:04d} and epix{}'.format(run, epix))
    elif data == 'kbar':
        plt.title('Kbar for run{:04d} and epix{}'.format(run, epix))
    plt.show()

    plt.clf()
    return None
=======
import numpy as np

dist_min = 0.1
box_size = 50.

density = 5.1 / 69.72  # mol/ml
NA = 6.0221408 * 0.1

molecule_number = int((box_size ** 3) * density * NA)
print("There are {} molecules".format(molecule_number))


def get_molecule_positions(box_size_A, molecule_num, random_number_seed):
    """
    Create a random array representing
    the positions of water molecules in size a box

    :param box_size_A:
    :param molecule_num:
    :param random_number_seed: Force one to select a random number seed so that one won't forget
                                to use a different seed for a different simulation
    :return:
    """
    molecule_num = int(molecule_num)
    box_size_A = float(box_size_A)
    np.random.seed(random_number_seed)

    # First divides the whole space into several cubes according to the molecule number
    axis_part_num = int(np.cbrt(float(molecule_num)) + 1)

    # Create a numpy array to represent this partition
    grid_coordinate = np.zeros((axis_part_num, axis_part_num, axis_part_num, 3), dtype=np.float64)
    grid_coordinate[:, :, :, 0] = np.arange(axis_part_num)[:, np.newaxis, np.newaxis]
    grid_coordinate[:, :, :, 1] = np.arange(axis_part_num)[np.newaxis, :, np.newaxis]
    grid_coordinate[:, :, :, 2] = np.arange(axis_part_num)[np.newaxis, np.newaxis, :]

    # Convert the 3D coordinate to 1D to randomly choose from it
    grid_coordinate = np.reshape(a=grid_coordinate, newshape=(axis_part_num ** 3, 3))

    # Shuffle the array and choose from it
    np.random.shuffle(grid_coordinate)

    # Choose the first several samples as the initial position of the molecules
    grid_coordinate = grid_coordinate[:molecule_num, :]

    # Convert the grid coordinate to the molecule positions in A
    grid_coordinate *= (box_size_A / float(axis_part_num))

    # Move the center to 0
    grid_coordinate -= (box_size_A / 2.)

    # Purturb the water molecules
    max_move = (box_size_A / float(axis_part_num) - dist_min)
    # grid_coordinate += (np.random.rand(molecule_num, 3) - 0.5) * max_move
    grid_coordinate += np.random.rand(molecule_num, 3) * max_move

    return grid_coordinate


# Get the coordinate of the molecules
mol_coordinate = get_molecule_positions(box_size_A=box_size, molecule_num=molecule_number, random_number_seed=193)

with open("./system.lt", 'w') as data_file:
    data_file.write("Position data for Silicon-Carbon system  \n")
    data_file.write("{}  atoms \n".format(molecule_number))
    data_file.write("1  atom types  \n")
    data_file.write("\n")
    data_file.write("\n")

    data_file.write("{} {} xlo xhi \n".format(-box_size / 2., box_size / 2.))
    data_file.write("{} {} ylo yhi \n".format(-box_size / 2., box_size / 2.))
    data_file.write("{} {} zlo zhi \n".format(-box_size / 2., box_size / 2.))
    data_file.write("\n")
    data_file.write("\n")

    data_file.write("Atoms")
    data_file.write("\n")
    data_file.write("\n")

    for mol_idx in range(molecule_number):
        data_file.write("{}	 1  {}  {}  {}\n".format(mol_idx,
                                                        mol_coordinate[mol_idx, 0],
                                                        mol_coordinate[mol_idx, 1],
                                                        mol_coordinate[mol_idx, 2],
                                                        ))
>>>>>>> 05650882da8e47fdf5716a9b219d0bc74493bce4
