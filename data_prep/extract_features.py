#!/usr/bin/env python

"""
Extract features from mp3

mpirun -np <#> --use-hwthread-cpus python extract_features.py <dataset name>
mpirun -np  8  --use-hwthread-cpus python extract_features.py train
depending on OS python -u to force writout

Returns:

    _train.npy if GRID == True   for LSTM, Conv1D
    _train.csv if GRID == False  for DNN, RandomForest

Dir tree:
    ./_train/
    ./_validation/
    ./_test/
            prog/
            nonprog/
"""

import os
import sys
import csv
import itertools

import numpy as np
import librosa

from mpi4py import MPI
from time import time

GRID = True
LOC = f'./_{sys.argv[1]}'           # train validation test
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


def get_part_features(file_path, genre):

    sr = 22050         # sample rate: #samples/sec
    ts_length = 256   # time series length of each datapoint
    hop_length = 512   # number of samples in each frame of ts
    # part_len = 10 * sr # #samples in 10s portions
    offset = 10      # skip first 10 sec of song

    # to fixed imbalanced dataset
    if genre == 'prog':
        part_len = 8 * sr
    elif genre == 'nonprog':
        part_len = 12 * sr

    y, sr = librosa.load(file_path, sr=sr, offset=offset, mono=True)
    tot_length = len(y) - offset * sr  # samples (except last 10s)
    n_parts = (tot_length // part_len)  # 1 sample every part_len

    if n_parts == 0:  # songs less than part_len
        n_parts = 1

    data = np.zeros((n_parts, ts_length, 43), dtype=np.float64)
    target = np.zeros((n_parts, 1))

    duration = ts_length * hop_length  # samples per part ~ 6s

    # 5x dataset
    # starts = np.random.randint(low=0,
    #                          high = tot_length,
    #                          size = 5 * n_parts)

    for n in range(n_parts):

        start = n * part_len
        end = start + duration
        part = y[start:end]

        mfcc = librosa.feature.mfcc(y=part, sr=sr, hop_length=hop_length, n_mfcc=20)
        spectral_cent = librosa.feature.spectral_centroid(y=part, sr=sr, hop_length=hop_length)
        chroma_stft = librosa.feature.chroma_stft(y=part, sr=sr, hop_length=hop_length)
        spectral_cont = librosa.feature.spectral_contrast(y=part, sr=sr, hop_length=hop_length)

        rmse = librosa.feature.rmse(y=part, hop_length=hop_length)
        rolloff = librosa.feature.spectral_rolloff(y=part, sr=sr, hop_length=hop_length)
        zcr = librosa.feature.zero_crossing_rate(y=part, hop_length=hop_length)

        if genre == 'prog':
            target[n] = 1
        else:
            target[n] = 0

        data[n, :, 0:20] = mfcc.T[0:ts_length, :]
        data[n, :, 20:21] = spectral_cent.T[0:ts_length, :]
        data[n, :, 21:33] = chroma_stft.T[0:ts_length, :]
        data[n, :, 33:40] = spectral_cont.T[0:ts_length, :]
        data[n, :, 40:41] = rmse.T[0:ts_length, :]
        data[n, :, 41:42] = rolloff.T[0:ts_length, :]
        data[n, :, 42:43] = zcr.T[0:ts_length, :]

    return (data, target)


# MPI processes
# initialize the world
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
        index = file_arr.tolist().index(filename) + 1

        if not GRID:
            to_append = get_features(f'{LOC}/{g}/{filename}', f'{g}')
            to_append += f' {filename.replace(" ", "")}'

        elif GRID:
            # to_append (np.array, int)
            try:
                array, target = get_part_features(f'{LOC}/{g}/{filename}', f'{g}')
                to_append = [array, target, f' {filename.replace(" ", "")}']
            except:
                print(f'Error: Something is wrong with {filename}')
                to_append = []

        data.append(to_append)

        print(f'{g} song {index}/{len(file_arr)} processed by rank {comm.rank}')

    comm.Barrier()


# gather data from all ranks to 0
data = comm.gather(data, root=0)  # list of data lists
if comm.rank == 0:
    print(f'Writing the _{sys.argv[1]} file ...')

    if GRID:
        # in each rank: Xr = [(np.array, int)], data is [ Xr ]
        # out_data is now merged ranks into list of tuples [()]
        out_data = list(itertools.chain.from_iterable(data))

        # after loading data = np.array('data.npy')
        # stack or concat data[:,0] >> arrays
        np.save(f'_{sys.argv[1]}', out_data)

    elif not GRID:

        header = 'chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
        for i in range(1, 21):
            header += f' mfcc{i}'
        header += ' label name'
        header = header.split()

        file = open(f'_{sys.argv[1]}.csv', 'w', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(header)
            for rank in range(comm.size):
                for to_append in data[rank]:
                    writer.writerow(to_append.split())

    dt = time() - t0
    print(f'Program finished processing {sys.argv[1]} in {dt/60} minutes')
