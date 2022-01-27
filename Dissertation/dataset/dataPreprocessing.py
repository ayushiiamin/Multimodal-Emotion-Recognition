""" REFERENCES """

""" 
[1] neelaryan, “CMU_MOSI_explore,” Kaggle.com, 2020.
Available at: https://www.kaggle.com/neelaryan/cmu-mosi-explore/notebook 



"""


import sys
assert sys.version_info >= (3, 5)

import sklearn
assert sklearn.__version__ >= "0.20"

import numpy as np
import os
import tarfile
import urllib
import pandas as pd
import urllib.request

from shutil import copyfile
import glob
import pickle

import h5py
import cv2
from scipy.io import wavfile
from tqdm import tqdm

from joblib import Parallel, delayed
import multiprocessing


with open('mosei_senti_data.pkl', 'rb') as fp:    #[1]
    data = pickle.load(fp)

# for i in range(16265):
#     print(data['train']['vision'][i][-1])
#     cur = data['train']['labels'][i]
#     if len(cur) > 1:
#         print(data['train']['vision'][i].shape)
#         print(data['train']['audio'][i].shape)
#         print(data['train']['text'][i].shape)
#         print(data['train']['labels'][i])
#         print(data['train']['id'][i])

def cmumosei_round(a):                 #[2]
    if a < -2:
        res = -3
    if -2 <= a and a < -1:
        res = -2
    if -1 <= a and a < 0:
        res = -1
    if 0 <= a and a <= 0:
            res = 0
    if 0 < a and a <= 1:
            res = 1
    if 1 < a and a <= 2:
            res = 2
    if a > 2:
            res = 3
    return res

labelset = sorted(set(list(np.squeeze(data['train']['labels']))))
print(labelset)
labelset = [cmumosei_round(label) for label in labelset]
print(labelset)



inputData = data
outputDataPath = "/home/ayushiamin/Dissertation/dataset/preprocessed_data"

