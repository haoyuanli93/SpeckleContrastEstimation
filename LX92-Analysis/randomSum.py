"""
This script is created to facilitate the analysis of the
speckle contrast with the random summation process.

Target
1. Read through XTC file and small-data file at the same time to reduce time
    a. Being able to handle multiple runs at the same time
    b. Being able to distinguish between CC and VCC
    c. Being able to skip the first 1 seconds of data after the change of the CC, VCC shutter.
2. Try to realize MPI data access through psana-mpi
3. Being able to produce the output into a hdf5 file

Notice:
Each epix detector is handed separately.
"""

import numpy as np
import h5py as h5
import psana
from mpi4py import MPI
import argparse
import os

# -----------------------------------------------
#   Parse the exp name, det name, run number
# -----------------------------------------------
parser = argparse.ArgumentParser(description='Please specify the experiment name,'
                                             ' detector name,'
                                             ' run numbers to process')
parser.add_argument("--expName",
                    type=str, required=True,
                    help="Name of the experiment. e.g. xpplx9221")
parser.add_argument("--detName",
                    type=str, required=True,
                    help="Name of the detector. e.g. epix_alc1, epix_alc2, epix_alc3."
                         " Notice that this algorithm only handle a single detector at each time.")
parser.add_argument("--smdPath",
                    type=str, required=True,
                    help="The folder containing h5 files for the smd output")
parser.add_argument("--runNumList",
                    nargs='+', type=int, required=True,
                    help="The run number to process")

args = parser.parse_args()

# Load the argument
expName = args.expName
detName = args.detName
runNumList = args.runNumList
smdPath = args.smdPath

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ----------------------------------------
#  The leader node analyze all the files
#  and verify the data quality and distribute the works
# ----------------------------------------

print("Information verification:")
print("experiment name:", expName)
print("detector name:", detName)
print("smd path:", smdPath)
print("runs to analyze:", runNumList)

if rank == 0:
    # ----------------------------------------------------------
    #   Loop through all the smd file to check if they exist
    # ----------------------------------------------------------
    print("Checking the status of each smd file")
    for run in runNumList:
        # Construct the location of the smd file
        smdFileName = "{}/{}_Run{:0>4d}.h5".format(smdPath, expName, run)
        # Check if the file exist:
        if not os.path.isfile(smdFileName):
            print("Does not find the smd output for run {} ".format(run) +
                  "at location {}".format(smdFileName))
        else:
            # Check the size of the file
            fileSize = float(os.path.getsize(smdFileName))
            print("smd File size for run {} is {:.2f} GB".format(run, fileSize * 9.3132257461548E-10))
        # Ideally, I also want to check if the hdf5 file is corrupted or not at this stage.
        # In this end, I give up since I feel it does not worth it to implement the code which add mental burden

    # ----------------------------------------------------------
    #  Load output from the smd file for pulse intensity, CC/VCC flag, delay time, l3e and other quantities
    # that maybe used as a flag.
    # ----------------------------------------------------------
