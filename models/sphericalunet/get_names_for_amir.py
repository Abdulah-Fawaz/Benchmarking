#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 11:08:18 2021

@author: fa19
"""

import numpy as np

root = '/home/fa19/Documents/Benchmarking/data/'
data='birth_age/'

data_root = root + data

train_root = data_root + 'train.npy'

a = np.load(train_root, allow_pickle=True)

l = np.char.split(a[:,0].astype(str), '_')


filename = 'birth_age_train.txt'
with open(filename, 'w') as f:
    for item in l:
        f.write("%s\n" % item)