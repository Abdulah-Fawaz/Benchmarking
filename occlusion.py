#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 10:55:30 2021

@author: fa19
"""

from scipy.interpolate import griddata
import os
import params
from params import gen_id
import sys
import numpy as np
from os.path import abspath, dirname
import torch
import torch.nn as nn
#sys.path.append(dirname(abspath(__file__)))
from utils import pick_criterion, import_from, load_testing

from data_utils.utils import load_model, make_fig



import json
from json.decoder import JSONDecodeError

from data_utils.MyDataLoader import My_dHCP_Data, My_dHCP_Data_Graph

def get_device(args):
    device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')
    print('device is ', device)
    torch.cuda.set_device(args.device)
    return device



import networkx as nx





def knbrs(G, start, k):
    nbrs = set([start])
    for l in range(k):
        nbrs = set((nbr for n in nbrs for nbr in G[n]))
    return list(nbrs)

def main():

    args = params.parse()
    device = get_device(args)
    
    model_dir = '/home/fa19/Documents/Benchmarking/results/sphericalunet/birth_age_confounded/et6236/best_model'
   
    
    model_name = args.model
    
    dsarr = 'birth_age_confounded'
    
    location_of_model = 'models/' + model_name
    
    print(model_name)
    print(dsarr)
    task=args.task
    print(task)
    resdir = '/home/fa19/Desktop/'

    means = torch.Tensor([1.1267, 0.0345, 1.0176, 0.0556])
    stds = torch.Tensor([0.3522, 0.1906, 0.3844, 4.0476]) 
    
    chosen_model = load_model(args)
    
    model = torch.load(model_dir).to(device)
    model.eval()
    
    T = np.load('/home/fa19/Documents/Benchmarking/data/'+dsarr+'/large_birth_age_confounded.npy', allow_pickle = True)
    
    T = T.reshape([1,len(T)])

    test_ds = My_dHCP_Data(T, projected = False, 
                              rotations = False, 
                              parity_choice = 'both', 
                              number_of_warps = 0,
                              normalisation = 'std',
                              warped_files_directory='/home/fa19/Documents/dHCP_Data_merged/Warped',
                              unwarped_files_directory='/home/fa19/Documents/dHCP_Data_merged/merged')
    
        
        
    edges_ico_6 = np.load('data/ico_6_edges.npy', allow_pickle = True)
    
    G = nx.Graph()
    
    G.add_edges_from(edges_ico_6)
    
    loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=1)
#    model = torch.load('/home/fa19/Documents/Benchmarking/results/sphericalunet/bayley/ho2860/best_model').cuda()
    

    occlusion_results = np.zeros(40962)
    model.eval()

    for i, batch in enumerate(loader):
        original_image = batch['image']
    
    
        if model_name == 'sphericalunet':
            
            original_image = original_image.permute(2,1,0)

    original_image = original_image.to(device)
    
    original_target = batch['label'].item()
    
    
    size_of_hole = 10
    
    import copy
    
    for i in range(40962):
        
        mask = knbrs(G, i, size_of_hole)
        
        new_image = copy.deepcopy(original_image)
        for m in range(4):
            new_image[mask,m] = -1 * means[m]
            new_image[mask,m] = new_image[mask,m] / stds[m]
        
        

        new_image = new_image.to(device)
#    test_labels = test_labels.unsqueeze(1)
        if task == 'regression':
            output = model(new_image).item()
            
            
            
        elif task == 'regression_confounded':
            
            metadata = batch['metadata'].to(device)            
            #print(metadata.shape)
    
            output = model(new_image, metadata).item()
        
    
        occlusion_results[i] = output - original_target        
        if i%100 == 0:
            print('Completed ', i)
    np.save(resdir+'/K=10_occlusion_large_mean_birth_age.npy', occlusion_results)

    
    
    
if __name__ == '__main__':
    main()