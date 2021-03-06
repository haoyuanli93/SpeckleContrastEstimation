atom_mass_list = {"H": [int(1), 1.00797, "Hydrogen", ],
                  "He": [int(2), 4.00260, "Helium", ],
                  "Li": [int(3), 6.941, "Lithium", ],
                  "Be": [int(4), 9.01218, "Beryllium", ],
                  "B": [int(5), 10.81, "Boron", ],
                  "C": [int(6), 12.011, "Carbon", ],
                  "N": [int(7), 14.0067, "Nitrogen", ],
                  "O": [int(8), 15.9994, "Oxygen", ],
                  "F": [int(9), 18.998403, "Fluorine", ],
                  "Ne": [int(10), 20.179, "Neon", ],
                  "Na": [int(11), 22.98977, "Sodium", ],
                  "Mg": [int(12), 24.305, "Magnesium", ],
                  "Al": [int(13), 26.98154, "Aluminum", ],
                  "Si": [int(14), 28.0855, "Silicon", ],
                  "P": [int(15), 30.97376, "Phosphorus", ],
                  "S": [int(16), 32.06, "Sulfur", ],
                  "Cl": [int(17), 35.453, "Chlorine", ],
                  "Ar": [int(18), 39.948, "Argon", ],
                  "K": [int(19), 39.0983, "Potassium", ],
                  "Ca": [int(20), 40.08, "Calcium", ],
                  "Sc": [int(21), 44.9559, "Scandium", ],
                  "Ti": [int(22), 47.90, "Titanium", ],
                  "V": [int(23), 50.9415, "Vanadium", ],
                  "Cr": [int(24), 51.996, "Chromium", ],
                  "Mn": [int(25), 54.9380, "Manganese", ],
                  "Fe": [int(26), 55.847, "Iron", ],
                  "Co": [int(27), 58.9332, "Cobalt", ],
                  "Ni": [int(28), 58.70, "Nickel", ],
                  "Cu": [int(29), 63.546, "Copper", ],
                  "Zn": [int(30), 65.38, "Zinc", ],
                  "Ga": [int(31), 69.72, "Gallium", ],
                  "Ge": [int(32), 72.59, "Germanium", ],
                  "As": [int(33), 74.9216, "Arsenic", ],
                  "Se": [int(34), 78.96, "Selenium", ],
                  "Br": [int(35), 79.904, "Bromine", ],
                  "Kr": [int(36), 83.80, "Krypton", ],
                  "Rb": [int(37), 85.4678, "Rubidium", ],
                  "Sr": [int(38), 87.62, "Strontium", ],
                  "Y": [int(39), 88.9059, "Yttrium", ],
                  "Zr": [int(40), 91.22, "Zirconium", ],
                  "Nb": [int(41), 92.9064, "Niobium", ],
                  "Mo": [int(42), 95.94, "Molybdenum", ],
                  "Tc": [int(43), 98.0, "Technetium", ],
                  "Ru": [int(44), 101.07, "Ruthenium", ],
                  "Rh": [int(45), 102.9055, "Rhodium", ],
                  "Pd": [int(46), 106.4, "Palladium", ],
                  "Ag": [int(47), 107.868, "Silver", ],
                  "Cd": [int(48), 112.41, "Cadmium", ],
                  "In": [int(49), 114.82, "Indium", ],
                  "Sn": [int(50), 118.69, "Tin", ],
                  "Sb": [int(51), 121.75, "Antimony", ],
                  "Te": [int(52), 127.60, "Tellurium", ],
                  "I": [int(53), 126.9045, "Iodine", ],
                  "Xe": [int(54), 131.30, "Xenon", ],
                  "Cs": [int(55), 132.9054, "Cesium", ],
                  "Ba": [int(56), 137.33, "Barium", ],
                  "La": [int(57), 138.9055, "Lanthanum", ],
                  "Ce": [int(58), 140.12, "Cerium", ],
                  "Pr": [int(59), 140.9077, "Praseodymium", ],
                  "Nd": [int(60), 144.24, "Neodymium", ],
                  "Pm": [int(61), 145.0, "Promethium", ],
                  "Sm": [int(62), 150.4, "Samarium", ],
                  "Eu": [int(63), 151.96, "Europium", ],
                  "Gd": [int(64), 157.25, "Gadolinium", ],
                  "Tb": [int(65), 158.9254, "Terbium", ],
                  "Dy": [int(66), 162.50, "Dysprosium", ],
                  "Ho": [int(67), 164.9304, "Holmium", ],
                  "Er": [int(68), 167.26, "Erbium", ],
                  "Tm": [int(69), 168.9342, "Thulium", ],
                  "Yb": [int(70), 173.04, "Ytterbium", ],
                  "Lu": [int(71), 174.967, "Lutetium", ],
                  "Hf": [int(72), 178.49, "Hafnium", ],
                  "Ta": [int(73), 180.9479, "Tantalum", ],
                  "W": [int(74), 183.85, "Tungsten", ],
                  "Re": [int(75), 186.207, "Rhenium", ],
                  "Os": [int(76), 190.2, "Osmium", ],
                  "Ir": [int(77), 192.22, "Iridium", ],
                  "Pt": [int(78), 195.09, "Platinum", ],
                  "Au": [int(79), 196.9665, "Gold", ],
                  "Hg": [int(80), 200.59, "Mercury", ],
                  "Tl": [int(81), 204.37, "Thallium", ],
                  "Pb": [int(82), 207.2, "Lead", ],
                  "Bi": [int(83), 208.9804, "Bismuth", ],
                  "Po": [int(84), 209.0, "Polonium", ],
                  "At": [int(85), 210.0, "Astatine", ],
                  "Rn": [int(86), 222.0, "Radon", ],
                  "Fr": [int(87), 223.0, "Francium", ],
                  "Ra": [int(88), 226.0254, "Radium", ],
                  "Ac": [int(89), 227.0278, "Actinium", ],
                  "Th": [int(90), 232.0381, "Thorium", ],
                  "Pa": [int(91), 231.0359, "Protactinium", ],
                  "U": [int(92), 238.029, "Uranium", ],
                  }
