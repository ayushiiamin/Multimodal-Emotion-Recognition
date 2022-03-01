""" REFERENCES """

"""
[1] G. Slick, “hdf5 file to pandas dataframe,” Stack Overflow, 2016. 
Available at: https://stackoverflow.com/questions/40472912/hdf5-file-to-pandas-dataframe.

[2] “How to rename columns in Pandas DataFrame - GeeksforGeeks,” GeeksforGeeks, 2018.
Available at: https://www.geeksforgeeks.org/how-to-rename-columns-in-pandas-dataframe/.

[3] A2Zadeh, “specific feature size of MOSEI dataset? · Issue #132 · A2Zadeh/CMU-MultimodalSDK,” GitHub, 2019. 
Available at: https://github.com/A2Zadeh/CMU-MultimodalSDK/issues/132.

[4] Stanko, “Update row values where certain condition is met in pandas,” Stack Overflow, 2016. 
Available at: https://stackoverflow.com/questions/36909977/update-row-values-where-certain-condition-is-met-in-pandas/36910033.

[5] “Index of /raw_datasets/processed_data/cmu-mosei,” Cmu.edu, 2019. 
Available at: http://immortal.multicomp.cs.cmu.edu/raw_datasets/processed_data/cmu-mosei/readme.MD.

[6] Yari, “Max and Min value for each colum of one Dataframe,” Stack Overflow, 2015. 
Available at: https://stackoverflow.com/questions/29276301/max-and-min-value-for-each-colum-of-one-dataframe/54999422.

[7] Mitkp, “Set value for particular cell in pandas DataFrame using index,” Stack Overflow, 2012. 
Available at: https://stackoverflow.com/questions/13842088/set-value-for-particular-cell-in-pandas-dataframe-using-index.

[8] convman, “Multimodal/modify_labels.ipynb at master · convman/Multimodal,” GitHub, 2022.
Available at: https://github.com/convman/Multimodal/blob/master/data/modify_labels.ipynb.

[9] “Selecting Rows and Columns Based on Conditions in Python Pandas DataFrame - KeyToDataScience,” KeyToDataScience, 2020.
Available at: https://keytodatascience.com/selecting-rows-conditions-pandas-dataframe/.

[10] markov zain, “Find the column name which has the maximum value for each row,” Stack Overflow, 2015. 
Available at: https://stackoverflow.com/questions/29919306/find-the-column-name-which-has-the-maximum-value-for-each-row#:~:text=To%20create%20the%20new%20column,idxmax()%20(or%20equivalently%20df.

[11] user2290820, “Python: Loops for simultaneous operation, Two or possibly more?,” Stack Overflow, 2013. 
Available at: https://stackoverflow.com/questions/16552508/python-loops-for-simultaneous-operation-two-or-possibly-more.
"""




#Importing the required libraries
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




#Reading the h5 emotion train file and storing it in a 
#variable called dataMosei_train
#The code also prints it type
dataMosei_train_ey = h5py.File("data/ey_train.h5", "r")              #[3]        

print(type(dataMosei_train_ey))

#Since h5 files act like a dictionary, hence we need to access the keys
#Creating an empty list to store the keys in a list
ey_trainKeys = []

#This for loop will iterate through the object keys of the h5 file,
#and store it in the empty list created above for this
#This is done so that it will be easier to interpret which keys are present in the h5 file
for key in dataMosei_train_ey.keys():
    ey_trainKeys.append(key)

#print(dataMosei_train.keys())
print(ey_trainKeys)


#Since we want to store the data in a CSV file, we first covert it to a DataFrame
#Initially, we convert the h5 file to a NumPy array, as it will be easier to build the pandas DataFrame
df_eyTrain = pd.DataFrame(np.array(dataMosei_train_ey.get("d1")))          #[1]

print(df_eyTrain.head())

#Printing the column values of the newly created DataFrame
print(df_eyTrain.columns)


#As per [5], the emotions in the ey_train, ey_test, and ey_valid h5 files are in the order -
#["Anger", "Disgust", "Fear", "Happy", "Sad", "Surprise"]
#Hence, we need to rename the columns as per the above provided order
df_eyTrain.columns = ["Anger", "Disgust", "Fear", "Happy", "Sad", "Surprise"]     #[2]

#Printing the updated column values
print(df_eyTrain.columns)

print(df_eyTrain.head())


#Print out the min and max of each column
print(df_eyTrain.agg([min, max]))            #[6]


#The minimum value an emotion can have in the dataset is 0.0
#and the maximum value it can have is 3.0
#0.0 value indicates that the particular emotion is not present
#3.0 value indicates that there is a high presence of that particular emotion

#From the output of (df_eyTrain.agg([min, max])), we can see that the emotion
#"Happy", has a maximum value of 3.333333, however the maximum value any emotion
#can have is 3.0. Hence, we need to change the maximum value of "Happy" from 3.333333
#to 3.0

#This below line of code, checks which values of "Happy" are greater than 3,
#if the boolean condition comes true, then change the value to 3
df_eyTrain.loc[df_eyTrain["Happy"] > 3, "Happy"] = 3           #[4]


#Print out the min and max of each column again
print(df_eyTrain.agg([min, max]))

#Printing out the number of rows and columns in the DataFrame
print(df_eyTrain.shape)


#As mentioned previously, the maximum value an emotion can have is 3.0 (indicates a high presence of that emotion) 
#and the minimum value the emotion can have is 0.0 (indicates the emotion is not present)

#The DataFrame has 6 columns of the order – “Anger”, “Disgust”, “Fear”, “Happy”, “Sad”, “Surprise”
#Under each column, a value between [0,3] is written, indicating the presence level of the emotion

#To ensure the ML model detects the right emotion, a list called “emoListTrain” is initialized to store the 
#column names (emotions) which have the maximum value for each row. The highest value in a row will indicate 
#a strong presence of that emotion.

#This below line of code uses the idxmax() function to find the column name with the maximum value
#The result (column name) is then stored in a list called emoListTrain
# emoListTrain = list(df_eyTrain.idxmax(axis=1))     #[10]


# for row, emo in zip(range(15290), emoListTrain):       #[11]
#     df_eyTrain.loc[row, emo] = 1           #[7]
#     df_eyTrain.loc[row, df_eyTrain.columns != emo] = 0       #[8]  #[9]

# df_eyTrain.to_csv('./csv_files/eTrain.csv', index = False)

print("ey_train.h5 converted to CSV")





print(" ")
print("TEST FILE NOW")

dataMosei_test_ey = h5py.File("data/ey_test.h5", "r")

ey_testKeys = []

for key in dataMosei_test_ey.keys():
    ey_testKeys.append(key)

print(ey_testKeys)

df_eyTest = pd.DataFrame(np.array(dataMosei_test_ey.get("d1")))

print(df_eyTest.head())

df_eyTest.columns = ["Anger", "Disgust", "Fear", "Happy", "Sad", "Surprise"]

print(df_eyTest.head())

print(df_eyTest.agg([min, max]))

df_eyTest.loc[df_eyTest["Happy"] > 3, "Happy"] = 3

print(df_eyTest.agg([min, max]))

print(df_eyTest.shape)

emoListTest = list(df_eyTest.idxmax(axis=1))

# for row, emo in zip(range(4832), emoListTest):       #[11]
#     df_eyTest.loc[row, emo] = 1           #[7]
#     df_eyTest.loc[row, df_eyTest.columns != emo] = 0

# df_eyTest.to_csv('./csv_files/eTest.csv', index = False)

print("ey_test.h5 converted to CSV")


print(" ")
print("VALID FILE NOW")

dataMosei_valid_ey = h5py.File("data/ey_valid.h5", "r")

ey_validKeys = []

for key in dataMosei_valid_ey.keys():
    ey_validKeys.append(key)

print(ey_validKeys)

df_eyValid = pd.DataFrame(np.array(dataMosei_valid_ey.get("d1")))

print(df_eyValid.head())

df_eyValid.columns = ["Anger", "Disgust", "Fear", "Happy", "Sad", "Surprise"]

print(df_eyValid.head())

print(df_eyValid.agg([min, max]))

print(df_eyValid.shape)

emoListValid = list(df_eyValid.idxmax(axis=1))

# for row, emo in zip(range(2291), emoListValid):
#     df_eyValid.loc[row, emo] = 1
#     df_eyValid.loc[row, df_eyValid.columns != emo] = 0

# df_eyValid.to_csv("./csv_files/eValid.csv", index = False)

print("ey_valid.h5 converted to CSV")

print(" ")
print(" ")

print("Audio/text/video now")

print(" ")

print("Text now")
print("-----------Text train-----------")

dataMosei_train_text = h5py.File("data/text_train_emb.h5", "r")

text_trainKeys = []

for key in dataMosei_train_text.keys():
    text_trainKeys.append(key)

print(text_trainKeys)

textTrainArr = np.array(dataMosei_train_text.get("d1"))
print(textTrainArr.shape)

# print(textTrainArr[4])
print(" ")

newTextTrainArr = textTrainArr.reshape(15290*20,300)

print(newTextTrainArr[0])

# np.set_printoptions(formatter={'float_kind':'{:f}'.format})

# list4 = textTrainArr.tolist()

# print("--------------------------------------")
# print(list4)

# print(list(textTrainArr[4]))
# print(textTrainArr[0])


import mmsdk
from mmsdk import mmdatasdk as md
from sklearn.metrics.pairwise import cosine_similarity

# visual_field = 'CMU_MOSI_VisualFacet_4.1'
# acoustic_field = 'CMU_MOSI_COVAREP'

# text_field = 'CMU_MOSEI_TimestampedWordVectors'
text_field = 'CMU_MOSEI_TimestampedWords'

text_field_GLOVE = 'CMU_MOSEI_TimestampedWordVectors'



features = [
    text_field
]

glove_features = [text_field_GLOVE]
# recipe = {feat: os.path.join(DATA_PATH, feat) + '.csd' for feat in features}
# dataset = md.mmdataset(recipe)

recipe = {feat: os.path.join('data/', feat) + '.csd' for feat in features}
dataset = md.mmdataset(recipe)

recipeGlove = {glo: os.path.join('data/', glo) + '.csd' for glo in glove_features}
datasetGlo = md.mmdataset(recipeGlove)
# text_field = 'CMU_MOSI_ModifiedTimestampedWords'

print(" ")

print(list(dataset.keys()))
print("=" * 80)

print(list(dataset[text_field].keys())[:10])
print("=" * 80)

some_id = list(dataset[text_field].keys())[15]
print(list(dataset[text_field][some_id].keys()))
print("=" * 80)

print(list(dataset[text_field][some_id]['intervals'].shape))
print("=" * 80)

print(list(dataset[text_field][some_id]['features'].shape))
print("=" * 80)

# print(list(textTrainArr[4]))
# print(" ")
print(list(dataset[text_field][some_id]['features'][4]))


print(" ")

print(list(datasetGlo.keys()))
print("=" * 80)

print(list(datasetGlo[text_field_GLOVE].keys())[:10])
print("=" * 80)

some_id = list(datasetGlo[text_field_GLOVE].keys())[15]
print(list(datasetGlo[text_field_GLOVE][some_id].keys()))
print("=" * 80)

print(list(datasetGlo[text_field_GLOVE][some_id]['intervals'].shape))
print("=" * 80)

print(list(datasetGlo[text_field_GLOVE][some_id]['features'].shape))
print("=" * 80)

# print(list(textTrainArr[4]))
# print(" ")
print((datasetGlo[text_field_GLOVE][some_id]['features'][9]))
print("=" * 80)

print((dataset[text_field][some_id]['features'][9]))
print("=" * 80)

# print(list(datasetGlo[text_field_GLOVE][some_id]['features'][5]))


print(" ")
print(" ")
import math
def cosine_similarity(v1,v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)

v1,v2 = (newTextTrainArr[28]), ((datasetGlo[text_field_GLOVE][some_id]['features'][202]))
print(cosine_similarity(v1,v2))
# cosIne = cosine_similarity( (list(dataset[text_field_GLOVE][some_id]['features'][3])), (list(datasetGlo[text_field_GLOVE][some_id]['features'][4])) )
# print(cosIne)

print(" ")
print(len(newTextTrainArr))
print(len(datasetGlo[text_field_GLOVE][some_id]['features'][1]))

print("LETS SPLIT ARRAY")

print(" ")




for j in range(len( ((datasetGlo[text_field_GLOVE][some_id]['features'])) )):
    if(cosine_similarity( (newTextTrainArr[752]), ((datasetGlo[text_field_GLOVE][some_id]['features'][j])) ) == 1.0):
        print("FOUND IT!!")
        print("j value - ", j)
        print( (dataset[text_field][some_id]['features'][j]) )
        # print( (dataset[text_field][some_id]['intervals'][j]) )
        # print( (datasetGlo[text_field_GLOVE][some_id]['intervals'][j]) )
        print("=" * 80)
        # break
    # else:
    #     print("BAD LUCK")

print("REACHED END OF LOOP")


print(" ")


# print(textTrainArr[0])




# label_test = 'CMU_MOSEI_Labels'


# label_features = [label_test]

# recipeLabel = {lab: os.path.join('data/', lab) + '.csd' for lab in label_features}
# datasetLab = md.mmdataset(recipeLabel)

# print(" ")

# print(list(datasetLab.keys()))
# print("=" * 80)

# print(list(datasetLab[label_test].keys())[:10])
# print("=" * 80)

# some_id = list(datasetLab[label_test].keys())[15]
# print(list(datasetLab[label_test][some_id].keys()))
# print("=" * 80)

# print(list(datasetLab[label_test][some_id]['intervals'].shape))
# print("=" * 80)

# print(list(datasetLab[label_test][some_id]['features'].shape))
# print("=" * 80)

# print(list(datasetLab[label_test][some_id]['intervals'][0]))
