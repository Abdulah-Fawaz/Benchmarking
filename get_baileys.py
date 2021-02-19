#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 16:49:59 2021

@author: fa19
"""


import numpy as np
import pandas as pd

f = pd.read_csv('/home/fa19/Downloads/DHCPNDH1_DATA_2021-02-17_1252.csv')
#f.columns = ['participationid', 'redcap_event_name', 'imdscore', 'imdquintile', 'age_at_assess_m',
#       'age_at_assess_d', 'corr_age_at_assess_m', 'corr_age_at_assess_d',
#       'cog_comp', 'lang_comp_20', 'mot_comp_20']
df = f[['participationid', 'cog_comp']].dropna()

df2 = df[df['cog_comp']!=90.0]

df3 = df2[df2['cog_comp']!=95.0]

list_of_subjects = os.listdir('/data/rsn/merged')
list_of_subjects = [x for x in list_of_subjects if 'L' in x]
subjects = [a.split('_')[0] for a in list_of_subjects]

subjects = [a.split('-')[1] for a in subjects]

sessions = [a.split('_')[1] for a in list_of_subjects]
sessions = [a.split('-')[1] for a in sessions]

ba = np.load('/home/fa19/Documents/Benchmarking/data/birth_age_confounded/full.npy', allow_pickle=True)

new_ba = np.empty_like(ba, shape = [ba.shape[0],4])
count=0
collection = []
for i in range(len(subjects)):

    subject = subjects[i]

    cog_score = df3[df3['participationid']==subject]['cog_comp']
    print(cog_score)
    if len(cog_score) == 0:
        print(i, 'error')
    else:
        count+=1
        a = ['sub-'+subjects[i]+ '_ses-'+sessions[i]]
        a.append(int(cog_score.item()))
        collection.append(a)
    
    
C = np.array(collection, dtype=object)



D = C[:,[0,-1]]
D[:,-1] = (D[:,-1]<100).astype(int)



C_pre = C[C[:,-1] < 100]

C_post = C[C[:,-1] >= 100]

import random

np.random.shuffle(C_pre)

np.random.shuffle(C_post)

split_pre = int(len(C_pre)//10)
split_post = int(len(C_post)//10)
train_pre, val_pre, test_pre = C_pre[split_pre * 2: , :], C_pre[split_pre:split_pre * 2, :], C_pre[:split_pre, :] 

train_post, val_post, test_post = C_post[split_post:, :], C_post[split_post:split_post * 2, :], C_post[:split_post, :] 



train_arr = np.concatenate([train_pre, train_post])


val_arr = np.concatenate([val_pre, val_post])

test_arr = np.concatenate([test_pre, test_post])

train_arr[:,-1] = (train_arr[:,-1] >= 100).astype(int)

val_arr[:,-1] = (val_arr[:,-1] >= 100).astype(int)

test_arr[:,-1] = (test_arr[:, -1] >= 100).astype(int)
train_arr = train_arr[:,[0,-1]]

val_arr = val_arr[:,[0,-1]]
test_arr = test_arr[:,[0,-1]]



np.save('full.npy', D)

np.save('train.npy', train_arr)
np.save('test.npy', test_arr)
np.save('validation.npy', val_arr)






