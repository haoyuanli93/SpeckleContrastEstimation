import numpy as np
import matplotlib.pyplot as plt


def show_diode(ipm2_val, diode_g1, diode1, diode2, diode3, diode4, diode5, diode6, diode_sample, mask):
    fig, axis = plt.subplots(ncols=3, nrows=3)
    fig.set_figheight(10)
    fig.set_figwidth(10)

    # ipm2 vs g1 diode
    axis[0, 0].scatter(np.concatenate(ipm2_val)[np.concatenate(mask)],
                       np.concatenate(diode_g1)[np.concatenate(mask)],
                       marker='.',
                       c='b')

    axis[0, 0].set_xlabel("ipm2")
    axis[0, 0].set_ylabel("dg1")

    # g1 diode vs d1
    axis[0, 1].scatter(np.concatenate(diode_g1)[np.concatenate(mask)],
                       np.concatenate(diode1)[np.concatenate(mask)],
                       marker='.',
                       c='b')

    axis[0, 1].set_xlabel("dg1")
    axis[0, 1].set_ylabel("d1")

    # d1 vs d2
    axis[0, 2].scatter(np.concatenate(diode1)[np.concatenate(mask)],
                       np.concatenate(diode2)[np.concatenate(mask)],
                       marker='.',
                       c='b')

    axis[0, 2].set_xlabel("d1")
    axis[0, 2].set_ylabel("d2")

    # d2 vs d3
    axis[1, 0].scatter(np.concatenate(diode2)[np.concatenate(mask)],
                       np.concatenate(diode3)[np.concatenate(mask)],
                       marker='.',
                       c='b')

    axis[1, 0].set_xlabel("d2")
    axis[1, 0].set_ylabel("d3")

    # d3 vs d4
    axis[1, 1].scatter(np.concatenate(diode3)[np.concatenate(mask)],
                       np.concatenate(diode4)[np.concatenate(mask)],
                       marker='.',
                       c='b')

    axis[1, 1].set_xlabel("d3")
    axis[1, 1].set_ylabel("d4")

    # d4 vs d5
    axis[1, 2].scatter(np.concatenate(diode4)[np.concatenate(mask)],
                       np.concatenate(diode5)[np.concatenate(mask)],
                       marker='.',
                       c='b')

    axis[1, 2].set_xlabel("d4")
    axis[1, 2].set_ylabel("d5")

    # d5 vs d6
    axis[2, 0].scatter(np.concatenate(diode5)[np.concatenate(mask)],
                       np.concatenate(diode6)[np.concatenate(mask)],
                       marker='.',
                       c='b')

    axis[2, 0].set_xlabel("d5")
    axis[2, 0].set_ylabel("d6")

    # d1 vs d6
    axis[2, 1].scatter(np.concatenate(diode1)[np.concatenate(mask)],
                       np.concatenate(diode6)[np.concatenate(mask)],
                       marker='.',
                       c='b')

    axis[2, 1].set_xlabel("d1")
    axis[2, 1].set_ylabel("d6")

    # d6 vs d sample
    axis[2, 2].scatter(np.concatenate(diode6)[np.concatenate(mask)],
                       np.concatenate(diode_sample)[np.concatenate(mask)],
                       marker='.',
                       c='b')

    axis[2, 2].set_xlabel("d6")
    axis[2, 2].set_ylabel("d_sample")

    plt.tight_layout()
    plt.show()
