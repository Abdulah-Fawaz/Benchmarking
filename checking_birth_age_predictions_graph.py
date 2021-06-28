#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 14:35:18 2021

@author: fa19
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_differences(link):
    f=plt.figure(1)
    predictions, birth_ages = np.load(link)
    loss = np.abs(predictions-birth_ages)
    
    F = np.load('/home/fa19/Documents/Benchmarking/data/birth_age_confounded/test.npy', allow_pickle=True)
    
    scan_ages = F[:,1]
    
    scan_ages = np.append(scan_ages, scan_ages)
    diffs = scan_ages - birth_ages
        
    
    
    plt.scatter(diffs,loss)
    plt.xlabel('difference between ages')
    plt.ylabel('L1 Loss')
    f.show()
    return
plt.close()

plot_differences('/home/fa19/Documents/Benchmarking/results/sphericalunet/birth_age_confounded/et6236/U_preds_labels.npy')

plot_differences('/home/fa19/Documents/Benchmarking/results/monet/birth_age_confounded/sv0024/U_preds_labels.npy')

plot_differences('/home/fa19/Documents/s2cnn_small_results_2/birth_age_confounded/di7100/U_preds_labels.npy')

plot_differences('/home/fa19/Documents/Benchmarking/results/gconvnet_nopool/birth_age_confounded/ql4807/U_preds_labels.npy')