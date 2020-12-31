#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 22:58:21 2020

@author: fa19
"""


class model_1():
    def __init__(self, whatever, whatever2):
        super(model_1, self).__init__()
    
    def forward(self,x):
        return x + self.whatever + self.whatever2
    

class model_2():
    def __init__(self, whatever, whatever2):
        super(model_1, self).__init__()
    
    def backwards(self,x):
        return x + self.whatever- self.whatever2
    
