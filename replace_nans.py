#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 19:05:05 2021

@author: fa19
"""

import numpy as np

import nibabel as nb


import os

root = '/data/rsn/warped/'

means_bayley = torch.Tensor([0.03561912, 0.1779468,  1.02368241, 1.30365072, 1.42005161,  1.80373678, 1.0485854,  1.44855442,  0.74604417])


for filename in os.listdir(root):
    
    file = root + filename
    
    loaded_file = nb.load(file)
    
    
    
    for i in range(len( loaded_file.darrays)):
        
        modality = loaded_file.darrays[i]
        
        numpy_loaded = modality.data
        
        
        numpy_loaded[np.isnan(numpy_loaded)] = means_bayley[i].item()
        
        loaded_file.darrays[i].data = numpy_loaded
    
    
    nb.save(loaded_file, file)