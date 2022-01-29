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


#Unpack the .pkl (pickle) file
