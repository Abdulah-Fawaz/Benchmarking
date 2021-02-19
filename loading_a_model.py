#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 14:38:28 2021

@author: fa19
"""

import torch
import os
os.chdir('models/presnet')
m = torch.load('/home/fa19/Documents/results2/presnet/birth_age_confounded/au4015/end_model')


