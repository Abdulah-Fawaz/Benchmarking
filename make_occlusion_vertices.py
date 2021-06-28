#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 19:26:15 2021

@author: fa19
"""

import numpy as np

size_ = 'small_modalitychebnet_nopool'
size_ = 'large_brth_agechebnet_nopool'

raw_k_numpy ='/home/fa19/Documents/Benchmarking/results/chebnet_nopool/scan_age/oq2934/K=10_occlusion_small_scan_agechebnet_nopool.npy'

raw_occlusion_results = np.load(raw_k_numpy)
raw_k_file_dir = raw_k_numpy.split('.npy')[0]


if len(raw_occlusion_results.shape) == 1:
    raw_occlusion_results = np.reshape(raw_occlusion_results, [raw_occlusion_results.shape[0],1])
    
    
import nibabel as nb
random_image = nb.load('/home/fa19/Documents/dHCP_Data_merged/merged/CC00589XX21_184000_L.shape.gii')

random_image.darrays.append(random_image.darrays[0])
for i in range(raw_occlusion_results.shape[1]):
    random_image.darrays[i].data = raw_occlusion_results[:,i]

nb.save(random_image, raw_k_file_dir + '_raw.shape.gii')

icosahedron_faces =nb.load('/home/fa19/Downloads/icosahedrons/ico-6.surf.gii')
icosahedron_faces = icosahedron_faces.darrays[1].data

edge_2 = []
for row in icosahedron_faces:
    row.sort()
    edge_2.append([row[0],row[1]])
    edge_2.append([row[1],row[2]])
    edge_2.append([row[0],row[2]])

import networkx as nx

G = nx.Graph()

#G.add_edges_from(edges.T.astype(list))

G.add_edges_from(edge_2)



def knbrs(G, start, k):
    nbrs = set([start])
    for l in range(k):
        nbrs = set((nbr for n in nbrs for nbr in G[n]))
    return list(nbrs)




vertices_list = []

for k in range(40962):
    vertices_list.append([])
    



for i in range(40962):
    mask = knbrs(G,i,10)
    for r in mask:
        vertices_list[r].append(i)
        
import pickle

with open('data/vertices_occlusion_list.pkl', 'wb') as handle:
    pickle.dump(vertices_list, handle)




#for i in range(len(solution)):
#    solution[i] = np.mean(solution[i])
#
#
#    solution = np.array(solution)
#
#    random_image.darrays[m].data = solution.astype(float)
#nb.save(random_image, raw_k_file_dir + '_processed.shape.gii')


#nb.save(random_image, 'original_image_' + str(size_) + '.shape.gii')

