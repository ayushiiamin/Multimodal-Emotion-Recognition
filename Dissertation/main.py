""" The main class """

""" Importing the required libraries """

#%%
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

import matplotlib as mpl
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

"""x = torch.rand(5, 3)
print(x)"""


""" Importing the dataset """

from mmsdk import mmdatasdk

import numpy
def myavg(intervals,features):
        return numpy.average(features,axis=0)

#cmumosei_highlevel = mmdatasdk.mmdataset(mmdatasdk.cmu_mosei.highlevel,'cmumosei/')
##cmumosei_highlevel.align('glove_vectors',collapse_functions=[myavg])
#cmumosei_highlevel.add_computational_sequences(mmdatasdk.cmu_mosei.labels,'cmumosei/')
#cmumosei_highlevel.align('Opinion Segment Labels')
