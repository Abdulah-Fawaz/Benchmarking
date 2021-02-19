#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 01:56:38 2021

@author: fa19
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 15:11:43 2020

@author: fa19
"""

import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np


import torch_geometric
import torch_scatter
import torch_geometric.nn as gnn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import degree

import numpy as np




hex_6 = torch.LongTensor(np.load('data/hexagons_6.npy'))
hex_5 = torch.LongTensor(np.load('data/hexagons_5.npy'))
hex_4 = torch.LongTensor(np.load('data/hexagons_4.npy'))
hex_3 = torch.LongTensor(np.load('data/hexagons_3.npy'))
hex_2 = torch.LongTensor(np.load('data/hexagons_2.npy'))
hex_1 = torch.LongTensor(np.load('data/hexagons_1.npy'))



reverse_hex_6 = np.load('data/reverse_hex_6.npy')
reverse_hex_5 = np.load('data/reverse_hex_5.npy')
reverse_hex_4 = np.load('data/reverse_hex_4.npy')
reverse_hex_3 = np.load('data/reverse_hex_3.npy')
reverse_hex_2 = np.load('data/reverse_hex_2.npy')
reverse_hex_1 = np.load('data/reverse_hex_1.npy')



edge_index_6 = torch.LongTensor(np.load('data/edge_index_6.npy'))
edge_index_5 = torch.LongTensor(np.load('data/edge_index_5.npy'))
edge_index_4 = torch.LongTensor(np.load('data/edge_index_4.npy'))
edge_index_3 = torch.LongTensor(np.load('data/edge_index_3.npy'))
edge_index_2 = torch.LongTensor(np.load('data/edge_index_2.npy'))
edge_index_1 = torch.LongTensor(np.load('data/edge_index_1.npy'))

polar_coords = torch.Tensor(np.load('data/ico_6_polar.npy'))

cartesian_cords = torch.Tensor(np.load('data/ico_6_cartesian.npy'))



pseudo_6 = torch.Tensor(np.load('data/relative_coords_6.npy'))
pseudo_5 = torch.Tensor(np.load('data/relative_coords_5.npy'))
pseudo_4 = torch.Tensor(np.load('data/relative_coords_4.npy'))
pseudo_3 = torch.Tensor(np.load('data/relative_coords_3.npy'))




#############################################################################################
       
hexes = [hex_6, hex_5, hex_4, hex_3, hex_2, hex_1]
edges_list = [edge_index_6, edge_index_5, edge_index_4, edge_index_3, edge_index_2, edge_index_1]




def chebconv(inchans, outchans, K = 3):
    return gnn.ChebConv(inchans, outchans, K)

def gcnconv(inchans, outchans):
    return gnn.GCNConv(inchans, outchans)

def gmmconv(inchans, outchans, kernel_size=3):
    return gnn.GMMConv(inchans, outchans, dim=3, kernel_size=kernel_size)


class GraphResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, inchannels, outchannels, conv_style, activation_function, edges_level, downsample=None, device = 'cuda'):
        super(GraphResidualBlock, self).__init__()
        self.conv_style = conv_style
        self.conv1 = conv_style(inchannels, outchannels)
        self.device = device
        #self.bn1 = nn.BatchNorm2d(planes) NO BN FOR NOW
        #self.relu = nn.ReLU(inplace=True)
        
        self.activation_function = activation_function

        self.conv2 = conv_style(outchannels, outchannels) # second one is outchannels becuse exapansion is 1
        
        #self.bn2 = nn.BatchNorm2d(planes)
        
        self.downsample = downsample
        self.edges_level = edges_level
        self.edges = edges_list[self.edges_level].to('cuda')
    def forward(self, x):
        residual = x
        e = self.edges
        out = self.conv1(x, self.edges)

        out = self.activation_function(out)

        out = self.conv2(out, self.edges)
        
        if self.downsample is not None:
            out = hex_pooling(self.edges_level, self.device)(out)

            residual = self.downsample(x, e)
            


        out += residual
        out = self.activation_function(out)

        return out
    

def transform(data):
    row, col = data.edge_index
    deg = degree(col, data.num_nodes)
    data.edge_attr = torch.stack(
        [1 / torch.sqrt(deg[row]), 1 / torch.sqrt(deg[col])], dim=-1)
    return data

def stack_coords(arr, times=1):
    return torch.cat(times*[arr])



class monet_regression(nn.Module):
    def __init__(self, num_features, conv_style=gmmconv,activation_function=nn.ReLU(), in_channels = 4, device='cuda'):
        super(monet_regression, self).__init__()
        self.conv_style = conv_style
        self.device = device
        self.in_channels = 4
        self.conv1 = conv_style(self.in_channels, num_features[0])
        self.conv2 = conv_style(num_features[0], num_features[1])
        self.conv3 = conv_style(num_features[1], num_features[2])
        self.conv4 = conv_style(num_features[2], num_features[3])
        self.pool1 = hex_pooling(0, self.device)
        self.pool2 = hex_pooling(1, self.device)
        self.pool3 = hex_pooling(2, self.device)
        self.pool4 = hex_pooling(3, self.device)
        
        
        self.activation_function = activation_function
        
        self.fc = nn.Linear(num_features[3] * 2, num_features[3])
        self.fc2 = nn.Linear(num_features[3], 1)

        

        #print "block.expansion=",block.expansion
#        self.fc = nn.Linear(512 * block.expansion, num_classes)

#        for m in self.modules():
#            if isinstance(m, nn.Conv2d):
#                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                m.weight.data.normal_(0, math.sqrt(2. / n))
#            elif isinstance(m, nn.BatchNorm2d):
#                m.weight.data.fill_(1)
#                m.bias.data.zero_()


    def forward(self, data):

        x = data.x

        e = data.edge_index


        batch=data.batch

        x = self.conv1(x,e, pseudo_6.to(self.device))
        x = self.activation_function(x)
        x = self.pool1(x)


        x = self.conv2(x,edge_index_5.to(self.device), pseudo_5.to(self.device))
        x = self.activation_function(x)
        x = self.pool2(x)
        
        
        x = self.conv3(x,edge_index_4.to(self.device), pseudo_4.to(self.device))
        x = self.activation_function(x)
        x = self.pool3(x)
        
        
        x = self.conv4(x,edge_index_3.to(self.device), pseudo_3.to(self.device))
        x = self.activation_function(x)
        x = self.pool4(x)
        
        
        
        x_max = gnn.global_max_pool(x, batch[:162].to(self.device))
        x_mean = gnn.global_mean_pool(x, batch[:162].to(self.device))
        
        x_c = torch.cat([x_max, x_mean], dim = 1)
        


        x_out = self.fc(x_c)

        x_out = self.activation_function(x_out)
        x_out = self.fc2(x_out)
        
        return x_out.squeeze(1)

class gconvnet_regression_2(nn.Module):

    def __init__(self, num_features,  block=GraphResidualBlock, layers=[2,2,2,2], conv_style=gcnconv,activation_function=nn.ReLU(), in_channels = 4, device='cuda'):
        self.inchans = num_features[0]
        super(gconvnet_regression_2, self).__init__()
        self.conv_style = conv_style
        self.device = device
        self.in_channels = 4
        self.conv1 = conv_style(self.in_channels, num_features[0])
        self.pool1 = hex_pooling(0, self.device)
        self.activation_function = activation_function
        

        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, num_features[0], layers[0], 1,1, device)
        self.layer2 = self._make_layer(block,  num_features[1], layers[1],  1,1, device)
        self.layer3 = self._make_layer(block,  num_features[2], layers[2], 2,2, device)
        self.layer4 = self._make_layer(block,  num_features[3], layers[3], 3,3, device)
        self.fc = nn.Linear(num_features[3]*162, 1)

        self.dropout = nn.Dropout(p=0.5,inplace=True)

        #print "block.expansion=",block.expansion
#        self.fc = nn.Linear(512 * block.expansion, num_classes)

#        for m in self.modules():
#            if isinstance(m, nn.Conv2d):
#                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                m.weight.data.normal_(0, math.sqrt(2. / n))
#            elif isinstance(m, nn.BatchNorm2d):
#                m.weight.data.fill_(1)
#                m.bias.data.zero_()

    def _make_layer(self, block, chans, blocks, ico_level, edge_level, device):
        downsample = None

        if self.inchans != chans * block.expansion:
            downsample = do_downsample(self.inchans, chans, self.conv_style, ico_level,device)
        else:
            downsample= None
        layers = []
        
        layers.append(block(self.inchans, chans,  self.conv_style,self.activation_function, 
                            edge_level, downsample, device))        
        self.inchans = chans * block.expansion
        
        for i in range(1, blocks):
            layers.append(block(self.inchans, chans, self.conv_style, self.activation_function, 
                                edge_level+1, None, device))
        

        return nn.Sequential(*layers)

    def forward(self, data):
        x = data.x
        e = data.edge_index

        x = self.conv1(x,e)

        x = self.activation_function(x)
        x = self.pool1(x)

        x = self.layer1(x)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

#        x = self.avgpool(x)
#        #print "avepool: ",x.data.shape

        x = x.flatten()
#        #print "view: ",x.data.shape        
        x = self.dropout(x)

        x = self.fc(x)

        return x       
    
    
class monet_regression_confounded(nn.Module):
    def __init__(self, num_features, conv_style=gmmconv,activation_function=nn.ReLU(), in_channels = 4, device='cuda'):
        super(monet_regression_confounded, self).__init__()
        self.conv_style = conv_style
        self.device = device
        self.in_channels = 4
        self.conv1 = conv_style(self.in_channels, num_features[0])
        self.conv2 = conv_style(num_features[0], num_features[1])
        self.conv3 = conv_style(num_features[1], num_features[2])
        self.conv4 = conv_style(num_features[2], num_features[3])
        self.pool1 = hex_pooling(0, self.device)
        self.pool2 = hex_pooling(1, self.device)
        self.pool3 = hex_pooling(2, self.device)
        self.pool4 = hex_pooling(3, self.device)
        self.convm = nn.Conv1d(1,4, kernel_size=1)
        
        
        self.activation_function = activation_function
        
        self.fc = nn.Linear((num_features[3] * 2 )+ 4, num_features[3])
        self.fc2 = nn.Linear(num_features[3], 1)
        self.dropout = nn.Dropout(0.5)
        

        #print "block.expansion=",block.expansion
#        self.fc = nn.Linear(512 * block.expansion, num_classes)

#        for m in self.modules():
#            if isinstance(m, nn.Conv2d):
#                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                m.weight.data.normal_(0, math.sqrt(2. / n))
#            elif isinstance(m, nn.BatchNorm2d):
#                m.weight.data.fill_(1)
#                m.bias.data.zero_()


    def forward(self, data):

        x = data.x

        e = data.edge_index

        m = data.metadata.to(self.device)
        batch=data.batch

        x = self.conv1(x,e, pseudo_6.to(self.device))
        x = self.activation_function(x)
        x = self.pool1(x)


        x = self.conv2(x,edge_index_5.to(self.device), pseudo_5.to(self.device))
        x = self.activation_function(x)
        x = self.pool2(x)
        
        
        x = self.conv3(x,edge_index_4.to(self.device), pseudo_4.to(self.device))
        x = self.activation_function(x)
        x = self.pool3(x)
        
        
        x = self.conv4(x,edge_index_3.to(self.device), pseudo_3.to(self.device))
        x = self.activation_function(x)
        x = self.pool4(x)
        
        
        
        x_max = gnn.global_max_pool(x, batch[:162].to(self.device))
        x_mean = gnn.global_mean_pool(x, batch[:162].to(self.device))
        
        x_c = torch.cat([x_max, x_mean], dim = 1)
        
        m = self.convm(m.unsqueeze(1))
        m = nn.ReLU()(m)
        m = m.reshape(m.shape[0],-1)

        x_c = torch.cat([x_c, m], dim=1)

        x_out = self.fc(x_c)
        x_out = self.dropout(x_out)
        x_out = self.activation_function(x_out)
        x_out = self.fc2(x_out)
        
        return x_out.squeeze(1)
    
    
        
class hex_pooling(nn.Module):
    def __init__(self, ico_level, device):
        super(hex_pooling, self).__init__()
        self.hex = hexes[ico_level].to(device)
    
    def forward(self, x):
        x = x.reshape(len(self.hex), -1)[self.hex]
        L = int((len(x)+6)/4)
        x = torch.max(x, dim = 1)[0][: L]
        
        return x
    
    
class do_downsample(nn.Module):

    def __init__(self, inchans, outchans, conv_style, ico_level, device):
        super(do_downsample, self).__init__()

        self.conv_style = conv_style
        self.inchans = inchans
        self.outchans = outchans
        self.ico_level = ico_level
        self.device = device
        self.conv1 = conv_style(self.inchans, self.outchans)
        self.pooling_method = hex_pooling(self.ico_level, self.device)
        
        
    def forward(self,x,e):

        
        x = self.conv1(x,e)
        
        x = self.pooling_method(x)
        
    