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
