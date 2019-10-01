#!/usr/bin/env python

"""
extract floating point time series from music files

mpirun -np <#> --use-hwthread-cpus python get_raw.py <dataset name>
mpirun -np  8  --use-hwthread-cpus python get_raw.py train

Returns:
    raw_train.npy

dir tree:
    ./_train/
    ./_validation/
    ./_test/
            prog/
            nonprog/
"""

import os
import sys
import itertools

import numpy as np
import librosa

from mpi4py import MPI
from time import time

LOC = f'./_{sys.argv[1]}'           # train validation test
genres = 'prog nonprog'.split()

t0 = time()


def get_raw(filename, genre):
    file_path = f'{LOC}/{genre}/{filename}'
    y, sr = librosa.load(file_path, sr=22050, mono=True)
    return (y, genre, filename)


# MPI Processes
# initialize the world
comm = MPI.COMM_WORLD
print("Rank %d from %d running in total..." % (comm.rank, comm.size))

data = []
for genre in genres:
    # get file names scatter, wait
    if comm.rank == 0:
        files = os.listdir(f'{LOC}/{genre}')
        file_arr = np.array_split(files, comm.size)
    else:
        file_arr = None

    file_arr = comm.scatter(file_arr, root=0)
    comm.Barrier()

    # work on the genre, wait
    for filename in file_arr:
        index = file_arr.tolist().index(filename) + 1
        to_append = get_raw(filename, genre)
        data.append(to_append)
        print(f'{genre} song {index}/{len(file_arr)} processed by rank {comm.rank}')

    comm.Barrier()


# gather data from all ranks to 0
data = comm.gather(data, root=0)  # list of data lists
if comm.rank == 0:
    print(f'Writing the raw_{sys.argv[1]} file ...')

    # in each rank: Xr = [(np.array, str, str)], data is list of list: [ Xr ]
    # out_data is now metged ranks into list of tuples [(np.array, str, str)]
    out_data = list(itertools.chain.from_iterable(data))

    # after loading data = np.array('data.npy')
    # stack or concat data[:,0] >> arrays
    np.save(f'raw_{sys.argv[1]}', out_data)

    dt = time() - t0
    print(f'Program finished processing {sys.argv[1]} in {dt/60} minutes')
