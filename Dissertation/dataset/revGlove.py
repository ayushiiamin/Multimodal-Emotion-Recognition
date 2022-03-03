from operator import index
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

import mmsdk
from mmsdk import mmdatasdk as md
import math
from sklearn.metrics.pairwise import cosine_similarity



dataMosei_train_text = h5py.File("data/text_train_emb.h5", "r")

text_trainKeys = []

for key in dataMosei_train_text.keys():
    text_trainKeys.append(key)

print(text_trainKeys)

print(" ")
print("-----------Create train array-----------")

textTrainArr = np.array(dataMosei_train_text.get("d1"))
print(textTrainArr.shape)
print(textTrainArr[45].shape)
print(textTrainArr[67][17].shape)

print(" ")
print("-----------Importing raw words + their glove embeddings from SDK-----------")

raw_words = 'CMU_MOSEI_TimestampedWords'
raw_words_embedded = 'CMU_MOSEI_TimestampedWordVectors'

raw_features = [raw_words]
raw_features_glove = [raw_words_embedded]

word_dict = {word: os.path.join('data/', word) + '.csd' for word in raw_features}
word_dataset = md.mmdataset(word_dict)

word_dict_glove = {emb: os.path.join('data/', emb) + '.csd' for emb in raw_features_glove}
glove_dataset = md.mmdataset(word_dict_glove)

print(" ")
print("RAW WORDS PRINTS")

print(list(word_dataset.keys()))
print("=" * 80)
print(list(word_dataset[raw_words].keys())[:10])
print("=" * 80)

print(len(list(word_dataset[raw_words].keys())))
print("=" * 80)

some_id = list(word_dataset[raw_words].keys())[4]         ########################################
print(some_id)
print("=" * 80)

print(list(word_dataset[raw_words][some_id].keys()))
print("=" * 80)
print(list(word_dataset[raw_words][some_id]['intervals'].shape))
print("=" * 80)
print(list(word_dataset[raw_words][some_id]['features'].shape))
print("=" * 80)
print(list(word_dataset[raw_words][some_id]['features'][4]))
print("=" * 80)

print(" ")
print("GLOVE EMBEDDINGS PRINTS")

print(list(glove_dataset.keys()))
print("=" * 80)
print(list(glove_dataset[raw_words_embedded].keys())[:10])
print("=" * 80)

print(len(list(glove_dataset[raw_words_embedded].keys())))
print("=" * 80)

some_id = list(glove_dataset[raw_words_embedded].keys())[4]         ########################################
print(some_id)
print("=" * 80)

print(list(glove_dataset[raw_words_embedded][some_id].keys()))
print("=" * 80)
print(list(glove_dataset[raw_words_embedded][some_id]['intervals'].shape))
print("=" * 80)
print(list(glove_dataset[raw_words_embedded][some_id]['features'].shape))
print("=" * 80)

print(" ")
print("-----------Creating the cosine similarity function-----------")

def cosine_similarity(v1,v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    if(math.sqrt(sumxx*sumyy) != 0.0):
        return sumxy/math.sqrt(sumxx*sumyy)

print("FUNCTION CREATED")

print(" ")
print("-----------Reverting the embeddings-----------")

for k in range(len(list(word_dataset[raw_words].keys()))):
    vid_id = list(word_dataset[raw_words].keys())[k]
    print(k)
    for j in range(len( ((glove_dataset[raw_words_embedded][vid_id]['features'])) )):
      if(cosine_similarity( (textTrainArr[0][5]), ((glove_dataset[raw_words_embedded][vid_id]['features'][j])) ) == 1.0):
        print("MATCH FOUND")
        print( (word_dataset[raw_words][vid_id]['features'][j]) )
        print("=" * 80)

# for j in range(len( ((glove_dataset[raw_words_embedded][list(glove_dataset[raw_words_embedded].keys())[4]]['features'])) )):
#     if(cosine_similarity( (textTrainArr[0][15]), ((glove_dataset[raw_words_embedded][list(glove_dataset[raw_words_embedded].keys())[4]]['features'][j])) ) == 1.0):
#         print("MATCH FOUND")
#         print("j value - ", j)
#         print( (word_dataset[raw_words][list(glove_dataset[raw_words_embedded].keys())[4]]['features'][j]) )
#         print("=" * 80)

print(" ")
# print(some_id)
print("END OF LOOP")