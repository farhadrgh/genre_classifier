#!/usr/bin/env python

"""
Extract features from music files

mpirun -np <#> --use-hwthread-cpus python extract_feature.py <dataset name>
mpirun -np  8  --use-hwthread-cpus python extract_feature.py train

Dir tree:
    ./_train/
    ./_validation/
    ./_test/
            prog/
            nonprog/

"""

import os, sys
import csv

import numpy as np
import pandas as pd
import librosa

from mpi4py import MPI
from time import time


LOC = f'./_{sys.argv[1]}' # train validation
out_file = f'_{sys.argv[1]}.csv'
genres = 'prog nonprog'.split()

t0 = time()

def get_features(file_path, genre):

    y, sr = librosa.load(file_path, mono=True)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rmse(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    
    to_append = f'{np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'

    for e in mfcc:
        to_append += f' {np.mean(e)}'
    to_append += f' {genre}'

    return to_append


# MPI initialize the world
comm = MPI.COMM_WORLD
print("Rank %d from %d running in total..." % (comm.rank, comm.size))

data = []
for g in genres:
    # get file names scatter, wait
    if comm.rank == 0: 
        files = os.listdir(f'{LOC}/{g}')
        file_arr = np.array_split(files, comm.size)
    else:
        file_arr = None
    
    file_arr = comm.scatter(file_arr, root=0)
    comm.Barrier()
    
    # work on the genre, wait
    for filename in file_arr:
        #index = np.where(file_arr==filename)[0][0] + 1
        index = file_arr.tolist().index(filename) + 1

        to_append = get_features(f'{LOC}/{g}/{filename}', f'{g}')
        to_append += f' {filename.replace(" ", "")}'
        data.append(to_append)

        print(f'{g} song {index}/{len(file_arr)} processed by rank {comm.rank}')

    comm.Barrier()


# gather data from all ranks to 0
data = comm.gather(data, root=0) # list of data lists

if comm.rank == 0:

    dt = time() - t0
    print(f'Program finished processing {sys.argv[1]} in {dt} seconds')

    header = 'chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label name'
    header = header.split()
    
    file = open(out_file, 'w', newline='')
    with file:
        writer = csv.writer(file)
        writer.writerow(header)
        for rank in range(comm.size):
            for to_append in data[rank]:
                writer.writerow(to_append.split())

