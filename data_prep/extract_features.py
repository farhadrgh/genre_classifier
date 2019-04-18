#!/usr/bin/env python

"""
Extract features from music files for the following networks:
    LSTM + attention transformer
    DNN

mpirun -np <#> --use-hwthread-cpus python extract_feature.py <dataset name>
mpirun -np  8  --use-hwthread-cpus python extract_feature.py train

Returns:

    _train.npy if LSTM == True
    _train.csv if LSTM == False

Dir tree:
    ./_train/
    ./_validation/
    ./_test/
            prog/
            nonprog/

"""

import os, sys
import csv
import itertools

import numpy as np
import pandas as pd
import librosa

from mpi4py import MPI
from time import time

LSTM = True
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


def get_features_LSTM(file_path, genre):

    ts_length = 128
    data = np.zeros((ts_length, 33), dtype=np.float64)

    y, sr = librosa.load(file_path, mono=True)
    ten_sec = 10*sr
    y = y[ten_sec:-ten_sec]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=512, n_mfcc=13)
    spectral_cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
    spectral_cont = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=512)

    if genre == 'prog': target = 1
    else: target = 0

    data[:, 0:13] = mfcc.T[0:ts_length, :]
    data[:, 13:14] = spectral_cent.T[0:ts_length, :]
    data[:, 14:26] = chroma_stft.T[0:ts_length, :]
    data[:, 26:33] = spectral_cont.T[0:ts_length, :]

    return (data, target)


def get_part_features_LSTM(file_path, genre):

    ts_length = 256
    hop_length = 512
    sr = 22050
    skip_length = 30

    tot_length = librosa.get_duration(filename=file_path, sr=sr) # get duration in sec 
    n_samples = int(tot_length // 60)  # one sample every 60 second

    if n_samples == 0: # songs less than 60
        n_samples = 1

    data = np.zeros((n_samples, ts_length, 33), dtype=np.float64)
    target = np.zeros((n_samples, 1))

    duration = ts_length * hop_length/sr + 0.5 # seconds
    
    for n in range(n_samples):
        
        offset = skip_length + (n * 60)
        y, sr = librosa.load(file_path, sr=sr, offset=offset , duration=duration , mono=True)
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
        spectral_cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
        spectral_cont = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)

        if genre == 'prog': target[n] = 1
        else: target[n] = 0

        data[n, :, 0:13] = mfcc.T[0:ts_length, :]
        data[n, :, 13:14] = spectral_cent.T[0:ts_length, :]
        data[n, :, 14:26] = chroma_stft.T[0:ts_length, :]
        data[n, :, 26:33] = spectral_cont.T[0:ts_length, :]
    
    return (data, target)


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
        index = file_arr.tolist().index(filename) + 1

        if LSTM == False:
            to_append = get_features(f'{LOC}/{g}/{filename}', f'{g}')
            to_append += f' {filename.replace(" ", "")}'

        elif LSTM == True:
            # to_append (np.array, int)
            try:
                array, target = get_part_features_LSTM(f'{LOC}/{g}/{filename}', f'{g}')
                to_append = [array, target, f' {filename.replace(" ", "")}']
            except:
                print(f'Error: Something is wrong with {filename}')
                to_append = []

        data.append(to_append)
        
        print(f'{g} song {index}/{len(file_arr)} processed by rank {comm.rank}')

    comm.Barrier()


# gather data from all ranks to 0
data = comm.gather(data, root=0) # list of data lists
if comm.rank == 0:

    dt = time() - t0
    print(f'Program finished processing {sys.argv[1]} in {dt/60} minutes')

    header = 'chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label name'
    header = header.split()

    if LSTM == True:
        # in each rank: X = [(np.array, int)], data is [ Xi ]
        out_data = list(itertools.chain.from_iterable(data))
        # out_data is now metged ranks into [()i]

        np.save(f'_{sys.argv[1]}', out_data)
        # after loading data = np.array('data.npy')
        # stack or concat data[:,0] >> arrays

    elif LSTM == False:
        file = open(out_file, 'w', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(header)
            for rank in range(comm.size):
                for to_append in data[rank]:
                    writer.writerow(to_append.split())
