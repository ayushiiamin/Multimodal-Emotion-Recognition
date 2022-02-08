""" REFERENCES """

""" 
[1] neelaryan, “CMU_MOSI_explore,” Kaggle.com, 2020.
Available at: https://www.kaggle.com/neelaryan/cmu-mosi-explore/notebook 



"""




import sys
assert sys.version_info >= (3, 5)

import pickle
import pandas as pd
import os
import numpy as np


with open('mosei_senti_data.pkl', 'rb') as fp:   #[1]
    data = pickle.load(fp)

print(data.keys())
print(data['train'].keys())
#print(data['train'].shape)
print(data['train']['vision'][56])
print(data['train']['audio'][0].shape)
print(data['train']['text'][0].shape)
print(data['train']['labels'][2])
print(data['train']['id'][0])


for i in range(16265):
    print(data['train']['vision'][i][-1])
    cur = data['train']['labels'][i]
    if len(cur) > 1:
        print(data['train']['vision'][i].shape)
        print(data['train']['audio'][i].shape)
        print(data['train']['text'][i].shape)
        print(data['train']['labels'][i])
        print(data['train']['id'][i])

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

#print(data['train']['labels'][56])

print(" ")
print(" ")
print(" ")

print(data.keys())
print(data['test'].keys())
#print(data['train'].shape)
#print(data['test']['vision'][56])
print(data['test']['audio'][0].shape)
print(data['test']['text'][0].shape)
print(data['test']['labels'][2])
print(data['test']['id'][0])


print(type(data))


print(type(data['train']['vision']))
print(type(data['valid']['labels']))
print(data.keys())
print(data['test'].keys())


print(data['test']['id'][5])
print(data['test']['labels'][6])