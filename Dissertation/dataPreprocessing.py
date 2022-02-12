""" REFERENCES """

"""
[1] ytrewq, “How to unpack pkl file?,” Stack Overflow, 2014. 
Available at: https://stackoverflow.com/questions/24906126/how-to-unpack-pkl-file. 

[2] “Python dictionary len() Method,” Tutorialspoint.com, 2021.
Available at: https://www.tutorialspoint.com/python/dictionary_len.htm 
"""


#Importing the required libraries
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



#Unpacking the pkl file so that we can access the data for pre-processing
#and view the file contents
with open("mosei_senti_data.pkl", "rb") as fle:          #[1]
    dataMosei = pickle.load(fle)


#To understand the raw dataset, we must print out certain details
#such as its type and shape
print(type(dataMosei))
print(len(dataMosei))        #[2]

print(" ")

#Since the raw dataset is of type dictionary with length 3,
#we need to now print out the 3 keys that are present in the dataset and their type
print(dataMosei.keys())
print(type(dataMosei['train']))
print(type(dataMosei['valid']))
print(type(dataMosei['test']))

print(" ")

#The dataset has around 3 keys, and each of them are dictionaries
#we need to now print out the keys and check their type
print(dataMosei['train'].keys())
print(dataMosei['valid'].keys())
print(dataMosei['test'].keys())

print(type(dataMosei['train']['vision']))
print(type(dataMosei['train']['audio']))
print(type(dataMosei['train']['text']))
print(type(dataMosei['train']['labels']))
print(type(dataMosei['train']['id']))

print(" ")

#Let us print out the shape of the numpy arrays
print(dataMosei["train"]["labels"].shape)
print(dataMosei["train"]["vision"].shape)
print(dataMosei["train"]["audio"].shape)
print(dataMosei["train"]["text"].shape)
print(dataMosei["train"]["audio"].shape)
print(dataMosei["train"]["id"].shape)

print(" ")

#Printing out a sample of the labels
print(dataMosei['train']['labels'][56])

print((dataMosei['train']['labels'][56][0][0]))
print(type(dataMosei['train']['labels'][56][0][0]))

print(" ")

#Since the label values are floats, we need to map them to
#integers, so that it will be easier when detecting/classifying the 
#emotions

def assignIntLabel(idVal):
    if idVal < -2:
        intID = -3
    elif idVal >= -2 and idVal < -1:
        intID = -2
    elif idVal >= -1 and idVal < 0:
        intID = -1
    elif idVal >= 0 and idVal <= 0:
        intID = 0
    elif idVal > 0 and idVal <= 1:
        intID = 1
    elif idVal > 1 and idVal <= 2:
        intID = 2
    elif idVal > 2:
        intID = 3
    return intID


print(" ")
#print(set(np.squeeze(dataMosei['train']['labels'])))

print((dataMosei["train"]["id"][2]))

df = pd.DataFrame({'ID': dataMosei["train"]["id"][:, 0], 'start': dataMosei["train"]["id"][:, 1], 'end': dataMosei["train"]["id"][:, 2]})

print(df)

print((dataMosei["train"]["labels"][2]))

#np.where(dataMosei["train"]["vision"] == "--qXJuDtHPw_5")