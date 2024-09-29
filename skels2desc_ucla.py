import json
import re

import numpy as np
import scipy.io as sio
import os
import sys
import time
from matplotlib import pyplot as plt
from pylab import figure
import cv2
import math
from PIL import Image
from natsort import natsorted
import glob
import torch.nn as nn
import scipy.sparse as sp
from einops import rearrange, repeat
import torch
from torch.utils.data import Dataset, DataLoader
from arguments import args


def angle_3D(p1, p2, p3):

    AB = math.sqrt((p2[0]-p1[0])**2+(p2[1]-p1[1])**2+(p2[2]-p1[2])**2)
    AC = math.sqrt((p3[0]-p1[0])**2+(p3[1]-p1[1])**2+(p3[2]-p1[2])**2)
    BC = math.sqrt((p3[0]-p2[0])**2+(p3[1]-p2[1])**2+(p3[2]-p2[2])**2)

    if ((AB == 0) or (AC == 0)):

        ang=0

    else:

        ang= math.acos(round((AB**2+AC**2-BC**2)/(2*AB*AC), 2))
      
    return ang




def Distance3D (x,y,z,x0,y0,z0):
    dist  = math.sqrt((x-x0)**2 +(y-y0)**2 + (z-z0)**2)
    return dist

def Distance3Dxyz (x,y,z,x0,y0,z0):

    distx  = math.sqrt((x-x0)**2)
    disty  = math.sqrt((y-y0)**2)
    distz  = math.sqrt((z-z0)**2)
    return distx, disty, distz



def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def adj_mx_from_edges(num_pts, sparse=False):


    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [0, 16],
                        [16, 17], [17, 18], [18, 19], [0, 12],
                        [12, 13], [13, 14], [14, 15], [2, 8],
                        [8, 9], [9, 10], [10, 11], [2, 4],[4,5],[5,6],
                        [6, 7]], dtype=torch.long)

   

    
    edges = np.array(edges, dtype=np.int32)
    data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
    adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)

    # build symmetric adjacency matrix
    adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
    adj_mx = normalize(adj_mx + sp.eye(adj_mx.shape[0]))


    adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)

    return adj_mx


def graph(x):

    in_channels=3

    out_channels=3

    adj=adj_mx_from_edges(20)

    kernel = adj.size(0)
   

    x=torch.unsqueeze(x,0)
    x=torch.unsqueeze(x,0)

    x=x.transpose(1,3)
    x=x.transpose(2,3)

    #x = nn.Conv2d(in_channels, out_channels , kernel_size=(1, 1))(x)

    x=torch.squeeze(x)

    x = torch.matmul(x, adj)

    return torch.flatten(x)



def data_UCLA_skel(mod, ds_path):
    dataset = 'UTD_MHAD'

    folder_path = ds_path

    if mod == 'test':
        data_dict = [{"file_name": "a05_s04_e02_v03", "length": 21, "label": 5},
                          {"file_name": "a12_s09_e04_v03", "length": 26, "label": 10},
                          {"file_name": "a03_s03_e04_v03", "length": 35, "label": 3},
                          {"file_name": "a08_s02_e01_v03", "length": 101, "label": 7},
                          {"file_name": "a03_s05_e03_v03", "length": 26, "label": 3},
                          {"file_name": "a12_s10_e01_v03", "length": 21, "label": 10},
                          {"file_name": "a01_s07_e03_v03", "length": 31, "label": 1},
                          {"file_name": "a03_s08_e02_v03", "length": 21, "label": 3},
                          {"file_name": "a11_s10_e03_v03", "length": 51, "label": 9},
                          {"file_name": "a11_s03_e00_v03", "length": 46, "label": 9},
                          {"file_name": "a03_s02_e00_v03", "length": 32, "label": 3},
                          {"file_name": "a11_s01_e04_v03", "length": 16, "label": 9},
                          {"file_name": "a09_s08_e04_v03", "length": 63, "label": 8},
                          {"file_name": "a09_s06_e01_v03", "length": 41, "label": 8},
                          {"file_name": "a09_s07_e01_v03", "length": 51, "label": 8},
                          {"file_name": "a02_s08_e01_v03", "length": 21, "label": 2},
                          {"file_name": "a01_s04_e01_v03", "length": 23, "label": 1},
                          {"file_name": "a02_s02_e02_v03", "length": 31, "label": 2},
                          {"file_name": "a02_s07_e05_v03", "length": 31, "label": 2},
                          {"file_name": "a06_s02_e00_v03", "length": 16, "label": 6},
                          {"file_name": "a03_s02_e02_v03", "length": 22, "label": 3},
                          {"file_name": "a11_s09_e04_v03", "length": 22, "label": 9},
                          {"file_name": "a09_s03_e04_v03", "length": 61, "label": 8},
                          {"file_name": "a04_s01_e02_v03", "length": 23, "label": 4},
                          {"file_name": "a12_s01_e01_v03", "length": 17, "label": 10},
                          {"file_name": "a02_s07_e03_v03", "length": 9, "label": 2},
                          {"file_name": "a05_s08_e04_v03", "length": 19, "label": 5},
                          {"file_name": "a02_s07_e02_v03", "length": 31, "label": 2},
                          {"file_name": "a04_s07_e02_v03", "length": 16, "label": 4},
                          {"file_name": "a01_s08_e03_v03", "length": 27, "label": 1},
                          {"file_name": "a08_s03_e01_v03", "length": 68, "label": 7},
                          {"file_name": "a04_s08_e03_v03", "length": 21, "label": 4},
                          {"file_name": "a03_s10_e00_v03", "length": 17, "label": 3},
                          {"file_name": "a04_s03_e03_v03", "length": 21, "label": 4},
                          {"file_name": "a06_s06_e02_v03", "length": 21, "label": 6},
                          {"file_name": "a09_s03_e00_v03", "length": 81, "label": 8},
                          {"file_name": "a09_s03_e03_v03", "length": 46, "label": 8},
                          {"file_name": "a04_s02_e02_v03", "length": 21, "label": 4},
                          {"file_name": "a08_s01_e02_v03", "length": 78, "label": 7},
                          {"file_name": "a04_s04_e00_v03", "length": 11, "label": 4},
                          {"file_name": "a03_s02_e03_v03", "length": 39, "label": 3},
                          {"file_name": "a05_s04_e00_v03", "length": 21, "label": 5},
                          {"file_name": "a05_s07_e03_v03", "length": 36, "label": 5},
                          {"file_name": "a06_s10_e00_v03", "length": 31, "label": 6},
                          {"file_name": "a11_s07_e00_v03", "length": 31, "label": 9},
                          {"file_name": "a03_s01_e01_v03", "length": 24, "label": 3},
                          {"file_name": "a04_s06_e01_v03", "length": 16, "label": 4},
                          {"file_name": "a08_s02_e04_v03", "length": 96, "label": 7},
                          {"file_name": "a09_s08_e03_v03", "length": 46, "label": 8},
                          {"file_name": "a05_s07_e00_v03", "length": 36, "label": 5},
                          {"file_name": "a05_s02_e02_v03", "length": 21, "label": 5},
                          {"file_name": "a04_s06_e04_v03", "length": 21, "label": 4},
                          {"file_name": "a05_s09_e03_v03", "length": 21, "label": 5},
                          {"file_name": "a03_s06_e02_v03", "length": 15, "label": 3},
                          {"file_name": "a01_s01_e00_v03", "length": 27, "label": 1},
                          {"file_name": "a06_s06_e03_v03", "length": 11, "label": 6},
                          {"file_name": "a06_s10_e02_v03", "length": 25, "label": 6},
                          {"file_name": "a02_s07_e04_v03", "length": 36, "label": 2},
                          {"file_name": "a09_s06_e00_v03", "length": 80, "label": 8},
                          {"file_name": "a04_s07_e04_v03", "length": 16, "label": 4},
                          {"file_name": "a05_s02_e01_v03", "length": 19, "label": 5},
                          {"file_name": "a01_s06_e04_v03", "length": 17, "label": 1},
                          {"file_name": "a04_s08_e01_v03", "length": 17, "label": 4},
                          {"file_name": "a01_s09_e00_v03", "length": 31, "label": 1},
                          {"file_name": "a08_s03_e03_v03", "length": 67, "label": 7},
                          {"file_name": "a12_s03_e00_v03", "length": 21, "label": 10},
                          {"file_name": "a11_s02_e03_v03", "length": 29, "label": 9},
                          {"file_name": "a12_s07_e02_v03", "length": 13, "label": 10},
                          {"file_name": "a05_s06_e01_v03", "length": 16, "label": 5},
                          {"file_name": "a06_s02_e04_v03", "length": 16, "label": 6},
                          {"file_name": "a06_s04_e00_v03", "length": 16, "label": 6},
                          {"file_name": "a05_s09_e01_v03", "length": 26, "label": 5},
                          {"file_name": "a11_s10_e04_v03", "length": 24, "label": 9},
                          {"file_name": "a03_s01_e00_v03", "length": 33, "label": 3},
                          {"file_name": "a11_s02_e01_v03", "length": 14, "label": 9},
                          {"file_name": "a04_s02_e00_v03", "length": 31, "label": 4},
                          {"file_name": "a11_s01_e01_v03", "length": 14, "label": 9},
                          {"file_name": "a02_s06_e03_v03", "length": 21, "label": 2},
                          {"file_name": "a12_s10_e03_v03", "length": 16, "label": 10},
                          {"file_name": "a01_s06_e00_v03", "length": 21, "label": 1},
                          {"file_name": "a05_s07_e01_v03", "length": 41, "label": 5},
                          {"file_name": "a01_s09_e01_v03", "length": 26, "label": 1},
                          {"file_name": "a02_s06_e00_v03", "length": 18, "label": 2},
                          {"file_name": "a11_s09_e00_v03", "length": 26, "label": 9},
                          {"file_name": "a03_s03_e01_v03", "length": 47, "label": 3},
                          {"file_name": "a03_s08_e00_v03", "length": 22, "label": 3},
                          {"file_name": "a06_s04_e01_v03", "length": 21, "label": 6},
                          {"file_name": "a02_s05_e01_v03", "length": 34, "label": 2},
                          {"file_name": "a03_s04_e04_v03", "length": 29, "label": 3},
                          {"file_name": "a01_s09_e02_v03", "length": 26, "label": 1},
                          {"file_name": "a08_s03_e04_v03", "length": 46, "label": 7},
                          {"file_name": "a01_s10_e00_v03", "length": 6, "label": 1},
                          {"file_name": "a01_s02_e02_v03", "length": 26, "label": 1},
                          {"file_name": "a09_s03_e01_v03", "length": 36, "label": 8},
                          {"file_name": "a05_s06_e00_v03", "length": 26, "label": 5},
                          {"file_name": "a05_s01_e02_v03", "length": 22, "label": 5},
                          {"file_name": "a02_s02_e04_v03", "length": 28, "label": 2},
                          {"file_name": "a06_s07_e03_v03", "length": 26, "label": 6},
                          {"file_name": "a04_s02_e04_v03", "length": 16, "label": 4},
                          {"file_name": "a02_s07_e01_v03", "length": 31, "label": 2},
                          {"file_name": "a03_s07_e03_v03", "length": 11, "label": 3},
                          {"file_name": "a12_s08_e01_v03", "length": 16, "label": 10},
                          {"file_name": "a05_s01_e03_v03", "length": 19, "label": 5},
                          {"file_name": "a02_s09_e02_v03", "length": 43, "label": 2},
                          {"file_name": "a05_s08_e03_v03", "length": 26, "label": 5},
                          {"file_name": "a04_s06_e00_v03", "length": 16, "label": 4},
                          {"file_name": "a09_s01_e02_v03", "length": 41, "label": 8},
                          {"file_name": "a12_s09_e00_v03", "length": 24, "label": 10},
                          {"file_name": "a04_s09_e02_v03", "length": 26, "label": 4},
                          {"file_name": "a03_s03_e03_v03", "length": 43, "label": 3},
                          {"file_name": "a08_s07_e03_v03", "length": 63, "label": 7},
                          {"file_name": "a08_s09_e02_v03", "length": 134, "label": 7},
                          {"file_name": "a08_s09_e00_v03", "length": 91, "label": 7},
                          {"file_name": "a06_s06_e04_v03", "length": 11, "label": 6},
                          {"file_name": "a01_s07_e04_v03", "length": 26, "label": 1},
                          {"file_name": "a05_s04_e01_v03", "length": 24, "label": 5},
                          {"file_name": "a04_s07_e00_v03", "length": 21, "label": 4},
                          {"file_name": "a05_s08_e01_v03", "length": 21, "label": 5},
                          {"file_name": "a11_s06_e03_v03", "length": 16, "label": 9},
                          {"file_name": "a01_s04_e03_v03", "length": 21, "label": 1},
                          {"file_name": "a11_s06_e04_v03", "length": 12, "label": 9},
                          {"file_name": "a12_s07_e03_v03", "length": 21, "label": 10},
                          {"file_name": "a06_s07_e05_v03", "length": 21, "label": 6},
                          {"file_name": "a01_s02_e04_v03", "length": 23, "label": 1},
                          {"file_name": "a03_s01_e03_v03", "length": 36, "label": 3},
                          {"file_name": "a12_s02_e02_v03", "length": 21, "label": 10},
                          {"file_name": "a03_s06_e01_v03", "length": 17, "label": 3},
                          {"file_name": "a05_s02_e03_v03", "length": 21, "label": 5},
                          {"file_name": "a03_s02_e04_v03", "length": 23, "label": 3},
                          {"file_name": "a08_s02_e03_v03", "length": 103, "label": 7},
                          {"file_name": "a08_s03_e02_v03", "length": 66, "label": 7},
                          {"file_name": "a09_s01_e01_v03", "length": 40, "label": 8},
                          {"file_name": "a02_s01_e01_v03", "length": 30, "label": 2},
                          {"file_name": "a08_s06_e00_v03", "length": 96, "label": 7},
                          {"file_name": "a12_s08_e02_v03", "length": 16, "label": 10},
                          {"file_name": "a02_s08_e00_v03", "length": 26, "label": 2},
                          {"file_name": "a01_s08_e02_v03", "length": 36, "label": 1},
                          {"file_name": "a09_s04_e01_v03", "length": 36, "label": 8},
                          {"file_name": "a04_s01_e04_v03", "length": 16, "label": 4},
                          {"file_name": "a08_s10_e03_v03", "length": 68, "label": 7},
                          {"file_name": "a02_s05_e00_v03", "length": 28, "label": 2},
                          {"file_name": "a06_s04_e03_v03", "length": 16, "label": 6},
                          {"file_name": "a06_s09_e03_v03", "length": 21, "label": 6},
                          {"file_name": "a05_s03_e02_v03", "length": 21, "label": 5},
                          {"file_name": "a06_s03_e04_v03", "length": 16, "label": 6},
                          {"file_name": "a06_s01_e03_v03", "length": 21, "label": 6},
                          {"file_name": "a11_s03_e01_v03", "length": 21, "label": 9},
                          {"file_name": "a09_s02_e01_v03", "length": 31, "label": 8},
                          {"file_name": "a02_s02_e00_v03", "length": 42, "label": 2},
                          {"file_name": "a01_s01_e03_v03", "length": 25, "label": 1},
                          {"file_name": "a08_s06_e02_v03", "length": 93, "label": 7},
                          {"file_name": "a12_s01_e03_v03", "length": 18, "label": 10},
                          {"file_name": "a09_s09_e01_v03", "length": 56, "label": 8},
                          {"file_name": "a04_s10_e03_v03", "length": 16, "label": 4},
                          {"file_name": "a06_s09_e04_v03", "length": 16, "label": 6},
                          {"file_name": "a02_s04_e01_v03", "length": 31, "label": 2},
                          {"file_name": "a12_s10_e04_v03", "length": 21, "label": 10},
                          {"file_name": "a06_s03_e01_v03", "length": 26, "label": 6},
                          {"file_name": "a02_s03_e04_v03", "length": 62, "label": 2},
                          {"file_name": "a11_s09_e02_v03", "length": 26, "label": 9},
                          {"file_name": "a08_s08_e02_v03", "length": 51, "label": 7},
                          {"file_name": "a03_s02_e01_v03", "length": 36, "label": 3},
                          {"file_name": "a12_s02_e00_v03", "length": 19, "label": 10},
                          {"file_name": "a12_s08_e03_v03", "length": 14, "label": 10},
                          {"file_name": "a02_s09_e03_v03", "length": 31, "label": 2},
                          {"file_name": "a09_s02_e02_v03", "length": 33, "label": 8},
                          {"file_name": "a05_s09_e04_v03", "length": 21, "label": 5},
                          {"file_name": "a01_s04_e00_v03", "length": 21, "label": 1},
                          {"file_name": "a08_s04_e03_v03", "length": 68, "label": 7},
                          {"file_name": "a12_s09_e03_v03", "length": 17, "label": 10},
                          {"file_name": "a02_s04_e03_v03", "length": 31, "label": 2},
                          {"file_name": "a04_s03_e04_v03", "length": 21, "label": 4},
                          {"file_name": "a12_s06_e01_v03", "length": 11, "label": 10},
                          {"file_name": "a11_s04_e03_v03", "length": 36, "label": 9},
                          {"file_name": "a05_s03_e00_v03", "length": 20, "label": 5},
                          {"file_name": "a12_s07_e00_v03", "length": 11, "label": 10},
                          {"file_name": "a06_s03_e02_v03", "length": 21, "label": 6},
                          {"file_name": "a03_s03_e05_v03", "length": 33, "label": 3},
                          {"file_name": "a11_s08_e01_v03", "length": 26, "label": 9},
                          {"file_name": "a06_s10_e01_v03", "length": 21, "label": 6},
                          {"file_name": "a04_s03_e02_v03", "length": 11, "label": 4},
                          {"file_name": "a02_s03_e03_v03", "length": 56, "label": 2},
                          {"file_name": "a09_s10_e04_v03", "length": 51, "label": 8},
                          {"file_name": "a04_s08_e04_v03", "length": 21, "label": 4},
                          {"file_name": "a11_s08_e00_v03", "length": 35, "label": 9},
                          {"file_name": "a02_s01_e00_v03", "length": 39, "label": 2},
                          {"file_name": "a04_s02_e03_v03", "length": 19, "label": 4},
                          {"file_name": "a04_s02_e01_v03", "length": 36, "label": 4},
                          {"file_name": "a06_s08_e00_v03", "length": 21, "label": 6},
                          {"file_name": "a08_s08_e01_v03", "length": 52, "label": 7},
                          {"file_name": "a02_s03_e01_v03", "length": 45, "label": 2},
                          {"file_name": "a11_s02_e02_v03", "length": 29, "label": 9},
                          {"file_name": "a09_s07_e02_v03", "length": 38, "label": 8},
                          {"file_name": "a02_s05_e03_v03", "length": 21, "label": 2},
                          {"file_name": "a01_s07_e02_v03", "length": 31, "label": 1},
                          {"file_name": "a03_s05_e00_v03", "length": 20, "label": 3},
                          {"file_name": "a09_s03_e02_v03", "length": 38, "label": 8},
                          {"file_name": "a01_s03_e07_v03", "length": 28, "label": 1},
                          {"file_name": "a09_s04_e04_v03", "length": 56, "label": 8},
                          {"file_name": "a11_s10_e00_v03", "length": 16, "label": 9},
                          {"file_name": "a04_s04_e01_v03", "length": 13, "label": 4},
                          {"file_name": "a02_s08_e02_v03", "length": 21, "label": 2},
                          {"file_name": "a04_s01_e07_v03", "length": 16, "label": 4},
                          {"file_name": "a11_s06_e00_v03", "length": 26, "label": 9},
                          {"file_name": "a05_s02_e00_v03", "length": 27, "label": 5},
                          {"file_name": "a02_s02_e03_v03", "length": 29, "label": 2},
                          {"file_name": "a05_s06_e02_v03", "length": 16, "label": 5},
                          {"file_name": "a08_s01_e03_v03", "length": 76, "label": 7},
                          {"file_name": "a08_s09_e01_v03", "length": 91, "label": 7},
                          {"file_name": "a02_s08_e04_v03", "length": 36, "label": 2},
                          {"file_name": "a01_s02_e03_v03", "length": 29, "label": 1},
                          {"file_name": "a11_s08_e05_v03", "length": 28, "label": 9},
                          {"file_name": "a03_s09_e02_v03", "length": 26, "label": 3},
                          {"file_name": "a04_s08_e00_v03", "length": 17, "label": 4},
                          {"file_name": "a12_s03_e04_v03", "length": 16, "label": 10},
                          {"file_name": "a08_s04_e01_v03", "length": 56, "label": 7},
                          {"file_name": "a12_s04_e03_v03", "length": 11, "label": 10},
                          {"file_name": "a04_s09_e03_v03", "length": 31, "label": 4},
                          {"file_name": "a05_s06_e03_v03", "length": 26, "label": 5},
                          {"file_name": "a09_s06_e02_v03", "length": 56, "label": 8},
                          {"file_name": "a06_s08_e05_v03", "length": 21, "label": 6},
                          {"file_name": "a12_s02_e03_v03", "length": 21, "label": 10},
                          {"file_name": "a11_s03_e03_v03", "length": 36, "label": 9},
                          {"file_name": "a11_s07_e04_v03", "length": 23, "label": 9},
                          {"file_name": "a04_s01_e00_v03", "length": 31, "label": 4},
                          {"file_name": "a03_s08_e03_v03", "length": 14, "label": 3},
                          {"file_name": "a04_s10_e00_v03", "length": 12, "label": 4},
                          {"file_name": "a08_s03_e00_v03", "length": 86, "label": 7},
                          {"file_name": "a02_s08_e03_v03", "length": 21, "label": 2},
                          {"file_name": "a01_s09_e03_v03", "length": 26, "label": 1},
                          {"file_name": "a01_s01_e04_v03", "length": 28, "label": 1},
                          {"file_name": "a01_s07_e00_v03", "length": 28, "label": 1},
                          {"file_name": "a02_s03_e00_v03", "length": 46, "label": 2},
                          {"file_name": "a01_s02_e00_v03", "length": 21, "label": 1},
                          {"file_name": "a03_s09_e04_v03", "length": 21, "label": 3},
                          {"file_name": "a01_s06_e02_v03", "length": 26, "label": 1},
                          {"file_name": "a03_s07_e02_v03", "length": 17, "label": 3},
                          {"file_name": "a03_s05_e04_v03", "length": 39, "label": 3},
                          {"file_name": "a08_s07_e01_v03", "length": 126, "label": 7},
                          {"file_name": "a04_s07_e03_v03", "length": 26, "label": 4},
                          {"file_name": "a08_s04_e04_v03", "length": 56, "label": 7},
                          {"file_name": "a08_s08_e00_v03", "length": 68, "label": 7},
                          {"file_name": "a02_s09_e00_v03", "length": 37, "label": 2},
                          {"file_name": "a06_s03_e00_v03", "length": 16, "label": 6},
                          {"file_name": "a09_s09_e04_v03", "length": 68, "label": 8},
                          {"file_name": "a05_s04_e04_v03", "length": 21, "label": 5},
                          {"file_name": "a09_s04_e03_v03", "length": 31, "label": 8},
                          {"file_name": "a01_s09_e04_v03", "length": 28, "label": 1},
                          {"file_name": "a05_s10_e00_v03", "length": 33, "label": 5},
                          {"file_name": "a09_s08_e02_v03", "length": 49, "label": 8},
                          {"file_name": "a11_s07_e01_v03", "length": 20, "label": 9},
                          {"file_name": "a06_s01_e00_v03", "length": 21, "label": 6},
                          {"file_name": "a12_s08_e04_v03", "length": 14, "label": 10},
                          {"file_name": "a08_s09_e04_v03", "length": 75, "label": 7},
                          {"file_name": "a12_s10_e02_v03", "length": 21, "label": 10},
                          {"file_name": "a04_s01_e01_v03", "length": 33, "label": 4},
                          {"file_name": "a01_s08_e01_v03", "length": 21, "label": 1},
                          {"file_name": "a09_s07_e00_v03", "length": 41, "label": 8},
                          {"file_name": "a04_s09_e00_v03", "length": 21, "label": 4},
                          {"file_name": "a08_s02_e02_v03", "length": 111, "label": 7},
                          {"file_name": "a09_s09_e02_v03", "length": 81, "label": 8},
                          {"file_name": "a09_s02_e03_v03", "length": 31, "label": 8},
                          {"file_name": "a11_s09_e01_v03", "length": 16, "label": 9},
                          {"file_name": "a03_s10_e01_v03", "length": 11, "label": 3},
                          {"file_name": "a11_s03_e02_v03", "length": 21, "label": 9},
                          {"file_name": "a11_s08_e04_v03", "length": 19, "label": 9},
                          {"file_name": "a06_s08_e02_v03", "length": 11, "label": 6},
                          {"file_name": "a11_s04_e04_v03", "length": 21, "label": 9},
                          {"file_name": "a12_s01_e00_v03", "length": 18, "label": 10},
                          {"file_name": "a02_s06_e04_v03", "length": 21, "label": 2},
                          {"file_name": "a06_s07_e01_v03", "length": 16, "label": 6},
                          {"file_name": "a05_s10_e03_v03", "length": 26, "label": 5},
                          {"file_name": "a03_s06_e00_v03", "length": 23, "label": 3},
                          {"file_name": "a12_s02_e01_v03", "length": 21, "label": 10},
                          {"file_name": "a08_s10_e02_v03", "length": 76, "label": 7},
                          {"file_name": "a08_s02_e00_v03", "length": 86, "label": 7},
                          {"file_name": "a06_s10_e03_v03", "length": 21, "label": 6},
                          {"file_name": "a11_s04_e02_v03", "length": 21, "label": 9},
                          {"file_name": "a08_s09_e03_v03", "length": 121, "label": 7},
                          {"file_name": "a12_s06_e04_v03", "length": 16, "label": 10},
                          {"file_name": "a01_s07_e01_v03", "length": 26, "label": 1},
                          {"file_name": "a05_s02_e04_v03", "length": 26, "label": 5},
                          {"file_name": "a09_s08_e00_v03", "length": 52, "label": 8},
                          {"file_name": "a02_s04_e04_v03", "length": 33, "label": 2},
                          {"file_name": "a06_s07_e00_v03", "length": 8, "label": 6},
                          {"file_name": "a04_s09_e01_v03", "length": 34, "label": 4},
                          {"file_name": "a09_s01_e00_v03", "length": 41, "label": 8},
                          {"file_name": "a08_s10_e01_v03", "length": 111, "label": 7},
                          {"file_name": "a11_s10_e02_v03", "length": 61, "label": 9},
                          {"file_name": "a09_s10_e02_v03", "length": 49, "label": 8},
                          {"file_name": "a03_s07_e04_v03", "length": 11, "label": 3},
                          {"file_name": "a05_s08_e00_v03", "length": 26, "label": 5},
                          {"file_name": "a11_s09_e03_v03", "length": 15, "label": 9},
                          {"file_name": "a12_s04_e04_v03", "length": 14, "label": 10},
                          {"file_name": "a04_s01_e03_v03", "length": 16, "label": 4},
                          {"file_name": "a04_s10_e02_v03", "length": 16, "label": 4},
                          {"file_name": "a06_s10_e04_v03", "length": 16, "label": 6},
                          {"file_name": "a01_s08_e00_v03", "length": 21, "label": 1},
                          {"file_name": "a03_s10_e02_v03", "length": 28, "label": 3},
                          {"file_name": "a03_s07_e01_v03", "length": 11, "label": 3},
                          {"file_name": "a05_s04_e03_v03", "length": 21, "label": 5},
                          {"file_name": "a01_s01_e02_v03", "length": 25, "label": 1},
                          {"file_name": "a05_s10_e04_v03", "length": 19, "label": 5},
                          {"file_name": "a06_s08_e03_v03", "length": 21, "label": 6},
                          {"file_name": "a02_s04_e02_v03", "length": 33, "label": 2},
                          {"file_name": "a12_s01_e04_v03", "length": 15, "label": 10},
                          {"file_name": "a05_s07_e05_v03", "length": 18, "label": 5},
                          {"file_name": "a02_s01_e02_v03", "length": 28, "label": 2},
                          {"file_name": "a12_s10_e00_v03", "length": 21, "label": 10},
                          {"file_name": "a11_s02_e00_v03", "length": 31, "label": 9},
                          {"file_name": "a02_s09_e01_v03", "length": 40, "label": 2},
                          {"file_name": "a02_s04_e00_v03", "length": 46, "label": 2},
                          {"file_name": "a12_s01_e02_v03", "length": 14, "label": 10},
                          {"file_name": "a01_s03_e06_v03", "length": 31, "label": 1},
                          {"file_name": "a03_s01_e04_v03", "length": 36, "label": 3},
                          {"file_name": "a01_s03_e04_v03", "length": 34, "label": 1},
                          {"file_name": "a01_s06_e03_v03", "length": 21, "label": 1},
                          {"file_name": "a02_s06_e01_v03", "length": 16, "label": 2},
                          {"file_name": "a12_s07_e04_v03", "length": 21, "label": 10},
                          {"file_name": "a08_s10_e04_v03", "length": 86, "label": 7},
                          {"file_name": "a02_s03_e02_v03", "length": 58, "label": 2},
                          {"file_name": "a05_s06_e04_v03", "length": 18, "label": 5},
                          {"file_name": "a05_s10_e01_v03", "length": 26, "label": 5},
                          {"file_name": "a09_s10_e01_v03", "length": 55, "label": 8},
                          {"file_name": "a08_s08_e04_v03", "length": 61, "label": 7},
                          {"file_name": "a06_s01_e02_v03", "length": 21, "label": 6},
                          {"file_name": "a01_s01_e01_v03", "length": 21, "label": 1},
                          {"file_name": "a06_s08_e04_v03", "length": 17, "label": 6},
                          {"file_name": "a09_s06_e03_v03", "length": 56, "label": 8},
                          {"file_name": "a06_s09_e01_v03", "length": 21, "label": 6},
                          {"file_name": "a08_s06_e01_v03", "length": 134, "label": 7},
                          {"file_name": "a02_s01_e04_v03", "length": 38, "label": 2},
                          {"file_name": "a11_s01_e00_v03", "length": 14, "label": 9},
                          {"file_name": "a03_s03_e00_v03", "length": 41, "label": 3},
                          {"file_name": "a01_s04_e04_v03", "length": 21, "label": 1},
                          {"file_name": "a06_s01_e04_v03", "length": 16, "label": 6},
                          {"file_name": "a01_s10_e01_v03", "length": 24, "label": 1},
                          {"file_name": "a03_s09_e00_v03", "length": 26, "label": 3},
                          {"file_name": "a08_s10_e00_v03", "length": 71, "label": 7},
                          {"file_name": "a05_s10_e02_v03", "length": 34, "label": 5},
                          {"file_name": "a04_s10_e01_v03", "length": 16, "label": 4},
                          {"file_name": "a05_s03_e04_v03", "length": 14, "label": 5},
                          {"file_name": "a05_s07_e02_v03", "length": 26, "label": 5},
                          {"file_name": "a12_s02_e04_v03", "length": 16, "label": 10},
                          {"file_name": "a06_s02_e03_v03", "length": 17, "label": 6},
                          {"file_name": "a09_s01_e03_v03", "length": 41, "label": 8},
                          {"file_name": "a08_s04_e00_v03", "length": 49, "label": 7},
                          {"file_name": "a02_s10_e01_v03", "length": 32, "label": 2},
                          {"file_name": "a11_s04_e01_v03", "length": 21, "label": 9},
                          {"file_name": "a03_s05_e01_v03", "length": 39, "label": 3},
                          {"file_name": "a06_s07_e04_v03", "length": 21, "label": 6},
                          {"file_name": "a09_s09_e03_v03", "length": 56, "label": 8},
                          {"file_name": "a02_s06_e02_v03", "length": 21, "label": 2},
                          {"file_name": "a05_s01_e04_v03", "length": 21, "label": 5},
                          {"file_name": "a11_s03_e04_v03", "length": 26, "label": 9},
                          {"file_name": "a04_s08_e02_v03", "length": 21, "label": 4},
                          {"file_name": "a04_s09_e04_v03", "length": 21, "label": 4},
                          {"file_name": "a08_s07_e00_v03", "length": 51, "label": 7},
                          {"file_name": "a04_s01_e05_v03", "length": 16, "label": 4},
                          {"file_name": "a12_s07_e01_v03", "length": 16, "label": 10},
                          {"file_name": "a02_s01_e03_v03", "length": 40, "label": 2},
                          {"file_name": "a09_s04_e00_v03", "length": 35, "label": 8},
                          {"file_name": "a09_s01_e04_v03", "length": 37, "label": 8},
                          {"file_name": "a12_s08_e00_v03", "length": 16, "label": 10},
                          {"file_name": "a04_s06_e03_v03", "length": 16, "label": 4},
                          {"file_name": "a11_s06_e01_v03", "length": 21, "label": 9},
                          {"file_name": "a01_s10_e02_v03", "length": 26, "label": 1},
                          {"file_name": "a02_s10_e04_v03", "length": 29, "label": 2},
                          {"file_name": "a04_s07_e01_v03", "length": 21, "label": 4},
                          {"file_name": "a03_s04_e01_v03", "length": 39, "label": 3},
                          {"file_name": "a03_s01_e02_v03", "length": 31, "label": 3},
                          {"file_name": "a06_s09_e02_v03", "length": 26, "label": 6},
                          {"file_name": "a03_s07_e00_v03", "length": 21, "label": 3},
                          {"file_name": "a06_s04_e02_v03", "length": 21, "label": 6},
                          {"file_name": "a12_s04_e01_v03", "length": 16, "label": 10},
                          {"file_name": "a04_s06_e02_v03", "length": 21, "label": 4},
                          {"file_name": "a04_s04_e04_v03", "length": 21, "label": 4},
                          {"file_name": "a09_s04_e02_v03", "length": 37, "label": 8},
                          {"file_name": "a02_s02_e01_v03", "length": 26, "label": 2},
                          {"file_name": "a06_s09_e00_v03", "length": 21, "label": 6},
                          {"file_name": "a05_s09_e00_v03", "length": 28, "label": 5},
                          {"file_name": "a05_s03_e01_v03", "length": 17, "label": 5},
                          {"file_name": "a02_s05_e04_v03", "length": 29, "label": 2},
                          {"file_name": "a01_s06_e01_v03", "length": 21, "label": 1},
                          {"file_name": "a12_s04_e02_v03", "length": 13, "label": 10},
                          {"file_name": "a03_s05_e02_v03", "length": 36, "label": 3},
                          {"file_name": "a01_s03_e02_v03", "length": 37, "label": 1},
                          {"file_name": "a05_s08_e05_v03", "length": 21, "label": 5},
                          {"file_name": "a01_s03_e00_v03", "length": 29, "label": 1},
                          {"file_name": "a08_s06_e03_v03", "length": 120, "label": 7},
                          {"file_name": "a05_s09_e02_v03", "length": 26, "label": 5},
                          {"file_name": "a01_s02_e01_v03", "length": 27, "label": 1},
                          {"file_name": "a01_s03_e01_v03", "length": 33, "label": 1},
                          {"file_name": "a04_s03_e01_v03", "length": 16, "label": 4},
                          {"file_name": "a06_s06_e00_v03", "length": 21, "label": 6},
                          {"file_name": "a12_s06_e02_v03", "length": 18, "label": 10},
                          {"file_name": "a11_s03_e05_v03", "length": 26, "label": 9},
                          {"file_name": "a04_s10_e04_v03", "length": 16, "label": 4},
                          {"file_name": "a12_s03_e01_v03", "length": 11, "label": 10},
                          {"file_name": "a08_s04_e02_v03", "length": 67, "label": 7},
                          {"file_name": "a06_s04_e04_v03", "length": 13, "label": 6},
                          {"file_name": "a12_s06_e03_v03", "length": 17, "label": 10},
                          {"file_name": "a08_s01_e04_v03", "length": 71, "label": 7},
                          {"file_name": "a04_s03_e00_v03", "length": 14, "label": 4},
                          {"file_name": "a08_s01_e00_v03", "length": 51, "label": 7},
                          {"file_name": "a01_s03_e03_v03", "length": 41, "label": 1},
                          {"file_name": "a04_s01_e08_v03", "length": 16, "label": 4},
                          {"file_name": "a01_s04_e02_v03", "length": 26, "label": 1},
                          {"file_name": "a01_s10_e04_v03", "length": 26, "label": 1},
                          {"file_name": "a09_s02_e00_v03", "length": 41, "label": 8},
                          {"file_name": "a06_s07_e02_v03", "length": 16, "label": 6},
                          {"file_name": "a08_s07_e02_v03", "length": 46, "label": 7},
                          {"file_name": "a11_s10_e01_v03", "length": 36, "label": 9},
                          {"file_name": "a02_s07_e00_v03", "length": 31, "label": 2},
                          {"file_name": "a06_s08_e01_v03", "length": 16, "label": 6},
                          {"file_name": "a01_s10_e03_v03", "length": 31, "label": 1},
                          {"file_name": "a11_s02_e04_v03", "length": 35, "label": 9},
                          {"file_name": "a02_s09_e04_v03", "length": 1, "label": 2},
                          {"file_name": "a12_s03_e03_v03", "length": 21, "label": 10},
                          {"file_name": "a05_s01_e01_v03", "length": 21, "label": 5},
                          {"file_name": "a05_s08_e02_v03", "length": 16, "label": 5},
                          {"file_name": "a12_s09_e02_v03", "length": 23, "label": 10},
                          {"file_name": "a09_s08_e01_v03", "length": 48, "label": 8},
                          {"file_name": "a01_s08_e04_v03", "length": 23, "label": 1},
                          {"file_name": "a09_s09_e00_v03", "length": 56, "label": 8},
                          {"file_name": "a03_s10_e03_v03", "length": 13, "label": 3},
                          {"file_name": "a09_s02_e04_v03", "length": 36, "label": 8},
                          {"file_name": "a08_s01_e01_v03", "length": 61, "label": 7},
                          {"file_name": "a09_s10_e00_v03", "length": 54, "label": 8},
                          {"file_name": "a12_s09_e01_v03", "length": 18, "label": 10},
                          {"file_name": "a05_s01_e00_v03", "length": 20, "label": 5},
                          {"file_name": "a06_s02_e01_v03", "length": 16, "label": 6},
                          {"file_name": "a08_s08_e03_v03", "length": 62, "label": 7},
                          {"file_name": "a04_s04_e03_v03", "length": 21, "label": 4},
                          {"file_name": "a02_s10_e02_v03", "length": 31, "label": 2},
                          {"file_name": "a01_s03_e05_v03", "length": 31, "label": 1},
                          {"file_name": "a06_s03_e03_v03", "length": 19, "label": 6},
                          {"file_name": "a05_s07_e04_v03", "length": 21, "label": 5},
                          {"file_name": "a02_s10_e00_v03", "length": 38, "label": 2},
                          {"file_name": "a12_s04_e00_v03", "length": 16, "label": 10},
                          {"file_name": "a03_s04_e02_v03", "length": 27, "label": 3},
                          {"file_name": "a06_s02_e02_v03", "length": 21, "label": 6},
                          {"file_name": "a03_s04_e03_v03", "length": 31, "label": 3},
                          {"file_name": "a11_s08_e03_v03", "length": 12, "label": 9},
                          {"file_name": "a09_s07_e03_v03", "length": 44, "label": 8},
                          {"file_name": "a05_s03_e03_v03", "length": 14, "label": 5},
                          {"file_name": "a09_s10_e03_v03", "length": 54, "label": 8},
                          {"file_name": "a11_s06_e02_v03", "length": 18, "label": 9},
                          {"file_name": "a04_s04_e02_v03", "length": 11, "label": 4},
                          {"file_name": "a11_s08_e02_v03", "length": 21, "label": 9},
                          {"file_name": "a11_s07_e03_v03", "length": 21, "label": 9},
                          {"file_name": "a04_s01_e06_v03", "length": 19, "label": 4},
                          {"file_name": "a06_s01_e01_v03", "length": 21, "label": 6},
                          {"file_name": "a12_s06_e00_v03", "length": 11, "label": 10},
                          {"file_name": "a12_s03_e02_v03", "length": 18, "label": 10},
                          {"file_name": "a03_s04_e00_v03", "length": 26, "label": 3},
                          {"file_name": "a11_s01_e03_v03", "length": 18, "label": 9},
                          {"file_name": "a03_s08_e01_v03", "length": 21, "label": 3},
                          {"file_name": "a11_s04_e00_v03", "length": 31, "label": 9},
                          {"file_name": "a02_s05_e02_v03", "length": 26, "label": 2},
                          {"file_name": "a06_s06_e01_v03", "length": 19, "label": 6},
                          {"file_name": "a03_s03_e02_v03", "length": 32, "label": 3},
                          {"file_name": "a11_s07_e02_v03", "length": 16, "label": 9},
                          {"file_name": "a11_s01_e02_v03", "length": 15, "label": 9}]
        outlist_skel = []
        def load_data():
            # data: N C V T M
            data = []
            for da in data_dict:
                file_name = da['file_name']
                outlist_skel.append(da["label"])
                with open(folder_path + '/' + file_name + '.json', 'r') as f:
                    json_file = json.load(f)
                skeletons = json_file['skeletons']
                value = np.array(skeletons)
                data.append(value)
            return data

        inplist_skel = load_data()



    else:
        data_dict = [{"file_name": "a05_s04_e02_v01", "length": 26, "label": 5},
                          {"file_name": "a01_s05_e04_v01", "length": 46, "label": 1},
                          {"file_name": "a03_s03_e04_v01", "length": 42, "label": 3},
                          {"file_name": "a08_s02_e01_v01", "length": 106, "label": 7},
                          {"file_name": "a03_s05_e03_v01", "length": 31, "label": 3},
                          {"file_name": "a06_s05_e01_v01", "length": 20, "label": 6},
                          {"file_name": "a12_s10_e01_v01", "length": 37, "label": 10},
                          {"file_name": "a01_s07_e03_v01", "length": 39, "label": 1},
                          {"file_name": "a03_s08_e02_v01", "length": 61, "label": 3},
                          {"file_name": "a11_s10_e03_v01", "length": 49, "label": 9},
                          {"file_name": "a11_s03_e00_v01", "length": 41, "label": 9},
                          {"file_name": "a03_s02_e00_v01", "length": 31, "label": 3},
                          {"file_name": "a11_s01_e04_v01", "length": 21, "label": 9},
                          {"file_name": "a04_s05_e04_v01", "length": 49, "label": 4},
                          {"file_name": "a09_s08_e04_v01", "length": 76, "label": 8},
                          {"file_name": "a09_s06_e01_v01", "length": 41, "label": 8},
                          {"file_name": "a09_s07_e01_v01", "length": 77, "label": 8},
                          {"file_name": "a02_s08_e01_v01", "length": 21, "label": 2},
                          {"file_name": "a01_s04_e01_v01", "length": 41, "label": 1},
                          {"file_name": "a02_s02_e02_v01", "length": 53, "label": 2},
                          {"file_name": "a02_s07_e05_v01", "length": 39, "label": 2},
                          {"file_name": "a06_s02_e00_v01", "length": 34, "label": 6},
                          {"file_name": "a03_s02_e02_v01", "length": 26, "label": 3},
                          {"file_name": "a09_s03_e04_v01", "length": 75, "label": 8},
                          {"file_name": "a04_s01_e02_v01", "length": 44, "label": 4},
                          {"file_name": "a12_s01_e01_v01", "length": 45, "label": 10},
                          {"file_name": "a02_s07_e03_v01", "length": 53, "label": 2},
                          {"file_name": "a05_s08_e04_v01", "length": 19, "label": 5},
                          {"file_name": "a02_s07_e02_v01", "length": 35, "label": 2},
                          {"file_name": "a04_s07_e02_v01", "length": 78, "label": 4},
                          {"file_name": "a01_s08_e03_v01", "length": 64, "label": 1},
                          {"file_name": "a08_s03_e01_v01", "length": 86, "label": 7},
                          {"file_name": "a04_s08_e03_v01", "length": 79, "label": 4},
                          {"file_name": "a03_s10_e00_v01", "length": 52, "label": 3},
                          {"file_name": "a04_s03_e03_v01", "length": 76, "label": 4},
                          {"file_name": "a11_s05_e02_v01", "length": 20, "label": 9},
                          {"file_name": "a06_s06_e02_v01", "length": 21, "label": 6},
                          {"file_name": "a01_s08_e06_v01", "length": 27, "label": 1},
                          {"file_name": "a03_s09_e03_v01", "length": 29, "label": 3},
                          {"file_name": "a09_s03_e00_v01", "length": 105, "label": 8},
                          {"file_name": "a09_s03_e03_v01", "length": 49, "label": 8},
                          {"file_name": "a04_s02_e02_v01", "length": 120, "label": 4},
                          {"file_name": "a08_s01_e02_v01", "length": 84, "label": 7},
                          {"file_name": "a04_s04_e00_v01", "length": 30, "label": 4},
                          {"file_name": "a03_s02_e03_v01", "length": 50, "label": 3},
                          {"file_name": "a05_s04_e00_v01", "length": 49, "label": 5},
                          {"file_name": "a05_s07_e03_v01", "length": 34, "label": 5},
                          {"file_name": "a02_s10_e05_v01", "length": 51, "label": 2},
                          {"file_name": "a06_s10_e00_v01", "length": 35, "label": 6},
                          {"file_name": "a11_s07_e00_v01", "length": 26, "label": 9},
                          {"file_name": "a03_s01_e01_v01", "length": 131, "label": 3},
                          {"file_name": "a04_s06_e01_v01", "length": 35, "label": 4},
                          {"file_name": "a08_s02_e04_v01", "length": 106, "label": 7},
                          {"file_name": "a09_s08_e03_v01", "length": 85, "label": 8},
                          {"file_name": "a05_s02_e02_v01", "length": 19, "label": 5},
                          {"file_name": "a04_s06_e04_v01", "length": 23, "label": 4},
                          {"file_name": "a05_s09_e03_v01", "length": 38, "label": 5},
                          {"file_name": "a03_s06_e02_v01", "length": 23, "label": 3},
                          {"file_name": "a01_s01_e00_v01", "length": 44, "label": 1},
                          {"file_name": "a06_s06_e03_v01", "length": 28, "label": 6},
                          {"file_name": "a06_s10_e02_v01", "length": 35, "label": 6},
                          {"file_name": "a02_s07_e04_v01", "length": 45, "label": 2},
                          {"file_name": "a09_s06_e00_v01", "length": 80, "label": 8},
                          {"file_name": "a04_s07_e04_v01", "length": 89, "label": 4},
                          {"file_name": "a04_s05_e09_v01", "length": 38, "label": 4},
                          {"file_name": "a05_s02_e01_v01", "length": 17, "label": 5},
                          {"file_name": "a01_s06_e04_v01", "length": 24, "label": 1},
                          {"file_name": "a04_s08_e01_v01", "length": 77, "label": 4},
                          {"file_name": "a01_s09_e00_v01", "length": 37, "label": 1},
                          {"file_name": "a08_s03_e03_v01", "length": 157, "label": 7},
                          {"file_name": "a12_s03_e00_v01", "length": 31, "label": 10},
                          {"file_name": "a11_s02_e03_v01", "length": 29, "label": 9},
                          {"file_name": "a12_s07_e02_v01", "length": 25, "label": 10},
                          {"file_name": "a11_s05_e01_v01", "length": 53, "label": 9},
                          {"file_name": "a05_s06_e01_v01", "length": 18, "label": 5},
                          {"file_name": "a03_s08_e06_v01", "length": 25, "label": 3},
                          {"file_name": "a06_s02_e04_v01", "length": 32, "label": 6},
                          {"file_name": "a06_s04_e00_v01", "length": 49, "label": 6},
                          {"file_name": "a05_s09_e01_v01", "length": 33, "label": 5},
                          {"file_name": "a11_s05_e03_v01", "length": 41, "label": 9},
                          {"file_name": "a11_s10_e04_v01", "length": 55, "label": 9},
                          {"file_name": "a03_s01_e00_v01", "length": 106, "label": 3},
                          {"file_name": "a03_s08_e04_v01", "length": 31, "label": 3},
                          {"file_name": "a11_s02_e01_v01", "length": 8, "label": 9},
                          {"file_name": "a04_s02_e00_v01", "length": 25, "label": 4},
                          {"file_name": "a11_s01_e01_v01", "length": 31, "label": 9},
                          {"file_name": "a02_s06_e03_v01", "length": 35, "label": 2},
                          {"file_name": "a12_s10_e03_v01", "length": 40, "label": 10},
                          {"file_name": "a01_s05_e02_v01", "length": 51, "label": 1},
                          {"file_name": "a01_s06_e00_v01", "length": 36, "label": 1},
                          {"file_name": "a05_s07_e01_v01", "length": 35, "label": 5},
                          {"file_name": "a01_s09_e01_v01", "length": 38, "label": 1},
                          {"file_name": "a02_s06_e00_v01", "length": 32, "label": 2},
                          {"file_name": "a11_s09_e00_v01", "length": 12, "label": 9},
                          {"file_name": "a03_s03_e01_v01", "length": 33, "label": 3},
                          {"file_name": "a03_s08_e00_v01", "length": 42, "label": 3},
                          {"file_name": "a06_s04_e01_v01", "length": 31, "label": 6},
                          {"file_name": "a02_s05_e01_v01", "length": 58, "label": 2},
                          {"file_name": "a03_s04_e04_v01", "length": 36, "label": 3},
                          {"file_name": "a01_s09_e02_v01", "length": 32, "label": 1},
                          {"file_name": "a08_s03_e04_v01", "length": 51, "label": 7},
                          {"file_name": "a01_s10_e00_v01", "length": 36, "label": 1},
                          {"file_name": "a01_s02_e02_v01", "length": 33, "label": 1},
                          {"file_name": "a09_s03_e01_v01", "length": 61, "label": 8},
                          {"file_name": "a05_s06_e00_v01", "length": 39, "label": 5},
                          {"file_name": "a05_s01_e02_v01", "length": 26, "label": 5},
                          {"file_name": "a03_s06_e04_v01", "length": 24, "label": 3},
                          {"file_name": "a02_s02_e04_v01", "length": 36, "label": 2},
                          {"file_name": "a06_s07_e03_v01", "length": 32, "label": 6},
                          {"file_name": "a04_s02_e04_v01", "length": 28, "label": 4},
                          {"file_name": "a04_s05_e02_v01", "length": 75, "label": 4},
                          {"file_name": "a02_s07_e01_v01", "length": 38, "label": 2},
                          {"file_name": "a03_s07_e03_v01", "length": 62, "label": 3},
                          {"file_name": "a12_s08_e01_v01", "length": 32, "label": 10},
                          {"file_name": "a05_s01_e03_v01", "length": 31, "label": 5},
                          {"file_name": "a02_s09_e02_v01", "length": 60, "label": 2},
                          {"file_name": "a05_s08_e03_v01", "length": 29, "label": 5},
                          {"file_name": "a04_s06_e00_v01", "length": 67, "label": 4},
                          {"file_name": "a09_s01_e02_v01", "length": 130, "label": 8},
                          {"file_name": "a04_s09_e02_v01", "length": 68, "label": 4},
                          {"file_name": "a03_s03_e03_v01", "length": 36, "label": 3},
                          {"file_name": "a08_s07_e03_v01", "length": 86, "label": 7},
                          {"file_name": "a08_s09_e02_v01", "length": 148, "label": 7},
                          {"file_name": "a08_s09_e00_v01", "length": 120, "label": 7},
                          {"file_name": "a06_s06_e04_v01", "length": 30, "label": 6},
                          {"file_name": "a01_s07_e04_v01", "length": 34, "label": 1},
                          {"file_name": "a04_s05_e08_v01", "length": 28, "label": 4},
                          {"file_name": "a08_s05_e04_v01", "length": 80, "label": 7},
                          {"file_name": "a05_s04_e01_v01", "length": 30, "label": 5},
                          {"file_name": "a04_s07_e00_v01", "length": 117, "label": 4},
                          {"file_name": "a05_s08_e01_v01", "length": 35, "label": 5},
                          {"file_name": "a11_s06_e03_v01", "length": 22, "label": 9},
                          {"file_name": "a01_s04_e03_v01", "length": 58, "label": 1},
                          {"file_name": "a12_s07_e03_v01", "length": 21, "label": 10},
                          {"file_name": "a01_s02_e04_v01", "length": 19, "label": 1},
                          {"file_name": "a04_s04_e05_v01", "length": 34, "label": 4},
                          {"file_name": "a03_s01_e03_v01", "length": 113, "label": 3},
                          {"file_name": "a12_s02_e02_v01", "length": 63, "label": 10},
                          {"file_name": "a05_s02_e03_v01", "length": 26, "label": 5},
                          {"file_name": "a03_s02_e04_v01", "length": 33, "label": 3},
                          {"file_name": "a08_s02_e03_v01", "length": 100, "label": 7},
                          {"file_name": "a08_s03_e02_v01", "length": 83, "label": 7},
                          {"file_name": "a09_s01_e01_v01", "length": 106, "label": 8},
                          {"file_name": "a02_s01_e01_v01", "length": 59, "label": 2},
                          {"file_name": "a08_s06_e00_v01", "length": 103, "label": 7},
                          {"file_name": "a04_s04_e09_v01", "length": 36, "label": 4},
                          {"file_name": "a12_s08_e02_v01", "length": 29, "label": 10},
                          {"file_name": "a02_s08_e00_v01", "length": 27, "label": 2},
                          {"file_name": "a01_s08_e02_v01", "length": 201, "label": 1},
                          {"file_name": "a09_s04_e01_v01", "length": 74, "label": 8},
                          {"file_name": "a04_s01_e04_v01", "length": 120, "label": 4},
                          {"file_name": "a04_s05_e03_v01", "length": 31, "label": 4},
                          {"file_name": "a08_s10_e03_v01", "length": 70, "label": 7},
                          {"file_name": "a02_s05_e00_v01", "length": 26, "label": 2},
                          {"file_name": "a06_s04_e03_v01", "length": 19, "label": 6},
                          {"file_name": "a06_s09_e03_v01", "length": 44, "label": 6},
                          {"file_name": "a05_s03_e02_v01", "length": 40, "label": 5},
                          {"file_name": "a06_s03_e04_v01", "length": 30, "label": 6},
                          {"file_name": "a06_s01_e03_v01", "length": 26, "label": 6},
                          {"file_name": "a11_s03_e01_v01", "length": 31, "label": 9},
                          {"file_name": "a09_s02_e01_v01", "length": 67, "label": 8},
                          {"file_name": "a02_s02_e00_v01", "length": 57, "label": 2},
                          {"file_name": "a01_s01_e03_v01", "length": 51, "label": 1},
                          {"file_name": "a08_s06_e02_v01", "length": 90, "label": 7},
                          {"file_name": "a12_s01_e03_v01", "length": 57, "label": 10},
                          {"file_name": "a06_s05_e04_v01", "length": 15, "label": 6},
                          {"file_name": "a09_s09_e01_v01", "length": 179, "label": 8},
                          {"file_name": "a04_s10_e03_v01", "length": 20, "label": 4},
                          {"file_name": "a06_s09_e04_v01", "length": 35, "label": 6},
                          {"file_name": "a02_s04_e01_v01", "length": 55, "label": 2},
                          {"file_name": "a12_s10_e04_v01", "length": 57, "label": 10},
                          {"file_name": "a04_s03_e05_v01", "length": 44, "label": 4},
                          {"file_name": "a06_s03_e01_v01", "length": 31, "label": 6},
                          {"file_name": "a02_s03_e04_v01", "length": 51, "label": 2},
                          {"file_name": "a11_s09_e02_v01", "length": 42, "label": 9},
                          {"file_name": "a08_s08_e02_v01", "length": 61, "label": 7},
                          {"file_name": "a03_s02_e01_v01", "length": 28, "label": 3},
                          {"file_name": "a12_s02_e00_v01", "length": 38, "label": 10},
                          {"file_name": "a12_s08_e03_v01", "length": 26, "label": 10},
                          {"file_name": "a02_s09_e03_v01", "length": 45, "label": 2},
                          {"file_name": "a09_s02_e02_v01", "length": 54, "label": 8},
                          {"file_name": "a05_s09_e04_v01", "length": 39, "label": 5},
                          {"file_name": "a04_s04_e06_v01", "length": 28, "label": 4},
                          {"file_name": "a01_s04_e00_v01", "length": 21, "label": 1},
                          {"file_name": "a08_s04_e03_v01", "length": 125, "label": 7},
                          {"file_name": "a08_s05_e01_v01", "length": 135, "label": 7},
                          {"file_name": "a02_s04_e03_v01", "length": 28, "label": 2},
                          {"file_name": "a04_s03_e04_v01", "length": 51, "label": 4},
                          {"file_name": "a12_s06_e01_v01", "length": 21, "label": 10},
                          {"file_name": "a11_s04_e03_v01", "length": 51, "label": 9},
                          {"file_name": "a05_s03_e00_v01", "length": 46, "label": 5},
                          {"file_name": "a12_s07_e00_v01", "length": 34, "label": 10},
                          {"file_name": "a06_s03_e02_v01", "length": 70, "label": 6},
                          {"file_name": "a03_s03_e05_v01", "length": 30, "label": 3},
                          {"file_name": "a11_s08_e01_v01", "length": 19, "label": 9},
                          {"file_name": "a05_s05_e04_v01", "length": 26, "label": 5},
                          {"file_name": "a06_s10_e01_v01", "length": 30, "label": 6},
                          {"file_name": "a04_s03_e02_v01", "length": 97, "label": 4},
                          {"file_name": "a02_s03_e03_v01", "length": 56, "label": 2},
                          {"file_name": "a09_s10_e04_v01", "length": 66, "label": 8},
                          {"file_name": "a04_s08_e04_v01", "length": 71, "label": 4},
                          {"file_name": "a11_s08_e00_v01", "length": 14, "label": 9},
                          {"file_name": "a02_s01_e00_v01", "length": 55, "label": 2},
                          {"file_name": "a04_s02_e03_v01", "length": 51, "label": 4},
                          {"file_name": "a04_s02_e01_v01", "length": 48, "label": 4},
                          {"file_name": "a06_s08_e00_v01", "length": 15, "label": 6},
                          {"file_name": "a08_s08_e01_v01", "length": 90, "label": 7},
                          {"file_name": "a02_s03_e01_v01", "length": 50, "label": 2},
                          {"file_name": "a11_s02_e02_v01", "length": 35, "label": 9},
                          {"file_name": "a09_s07_e02_v01", "length": 44, "label": 8},
                          {"file_name": "a02_s05_e03_v01", "length": 41, "label": 2},
                          {"file_name": "a01_s07_e02_v01", "length": 35, "label": 1},
                          {"file_name": "a06_s05_e03_v01", "length": 18, "label": 6},
                          {"file_name": "a12_s05_e03_v01", "length": 40, "label": 10},
                          {"file_name": "a03_s05_e00_v01", "length": 63, "label": 3},
                          {"file_name": "a09_s03_e02_v01", "length": 48, "label": 8},
                          {"file_name": "a09_s04_e04_v01", "length": 139, "label": 8},
                          {"file_name": "a11_s10_e00_v01", "length": 50, "label": 9},
                          {"file_name": "a04_s04_e01_v01", "length": 26, "label": 4},
                          {"file_name": "a01_s08_e05_v01", "length": 63, "label": 1},
                          {"file_name": "a02_s08_e02_v01", "length": 35, "label": 2},
                          {"file_name": "a01_s05_e00_v01", "length": 83, "label": 1},
                          {"file_name": "a11_s06_e00_v01", "length": 33, "label": 9},
                          {"file_name": "a05_s02_e00_v01", "length": 40, "label": 5},
                          {"file_name": "a02_s02_e03_v01", "length": 41, "label": 2},
                          {"file_name": "a09_s05_e02_v01", "length": 61, "label": 8},
                          {"file_name": "a05_s06_e02_v01", "length": 19, "label": 5},
                          {"file_name": "a08_s01_e03_v01", "length": 130, "label": 7},
                          {"file_name": "a08_s09_e01_v01", "length": 153, "label": 7},
                          {"file_name": "a02_s08_e04_v01", "length": 51, "label": 2},
                          {"file_name": "a06_s05_e02_v01", "length": 21, "label": 6},
                          {"file_name": "a01_s02_e03_v01", "length": 31, "label": 1},
                          {"file_name": "a11_s08_e05_v01", "length": 34, "label": 9},
                          {"file_name": "a03_s09_e02_v01", "length": 19, "label": 3},
                          {"file_name": "a04_s08_e00_v01", "length": 86, "label": 4},
                          {"file_name": "a03_s09_e01_v01", "length": 6, "label": 3},
                          {"file_name": "a08_s04_e01_v01", "length": 109, "label": 7},
                          {"file_name": "a12_s04_e03_v01", "length": 41, "label": 10},
                          {"file_name": "a04_s09_e03_v01", "length": 43, "label": 4},
                          {"file_name": "a12_s05_e00_v01", "length": 32, "label": 10},
                          {"file_name": "a11_s05_e04_v01", "length": 41, "label": 9},
                          {"file_name": "a05_s06_e03_v01", "length": 19, "label": 5},
                          {"file_name": "a09_s06_e02_v01", "length": 31, "label": 8},
                          {"file_name": "a06_s08_e05_v01", "length": 19, "label": 6},
                          {"file_name": "a03_s06_e03_v01", "length": 25, "label": 3},
                          {"file_name": "a12_s02_e03_v01", "length": 77, "label": 10},
                          {"file_name": "a11_s03_e03_v01", "length": 36, "label": 9},
                          {"file_name": "a04_s01_e00_v01", "length": 141, "label": 4},
                          {"file_name": "a04_s04_e08_v01", "length": 36, "label": 4},
                          {"file_name": "a03_s08_e03_v01", "length": 31, "label": 3},
                          {"file_name": "a02_s10_e03_v01", "length": 71, "label": 2},
                          {"file_name": "a04_s10_e00_v01", "length": 12, "label": 4},
                          {"file_name": "a08_s03_e00_v01", "length": 84, "label": 7},
                          {"file_name": "a02_s08_e03_v01", "length": 56, "label": 2},
                          {"file_name": "a01_s09_e03_v01", "length": 35, "label": 1},
                          {"file_name": "a01_s01_e04_v01", "length": 46, "label": 1},
                          {"file_name": "a01_s07_e00_v01", "length": 35, "label": 1},
                          {"file_name": "a02_s03_e00_v01", "length": 86, "label": 2},
                          {"file_name": "a01_s02_e00_v01", "length": 25, "label": 1},
                          {"file_name": "a03_s09_e04_v01", "length": 38, "label": 3},
                          {"file_name": "a01_s06_e02_v01", "length": 28, "label": 1},
                          {"file_name": "a03_s07_e02_v01", "length": 8, "label": 3},
                          {"file_name": "a04_s05_e05_v01", "length": 56, "label": 4},
                          {"file_name": "a08_s07_e01_v01", "length": 155, "label": 7},
                          {"file_name": "a04_s07_e03_v01", "length": 109, "label": 4},
                          {"file_name": "a08_s04_e04_v01", "length": 146, "label": 7},
                          {"file_name": "a08_s08_e00_v01", "length": 56, "label": 7},
                          {"file_name": "a02_s09_e00_v01", "length": 55, "label": 2},
                          {"file_name": "a06_s03_e00_v01", "length": 35, "label": 6},
                          {"file_name": "a04_s05_e07_v01", "length": 39, "label": 4},
                          {"file_name": "a09_s09_e04_v01", "length": 81, "label": 8},
                          {"file_name": "a05_s04_e04_v01", "length": 27, "label": 5},
                          {"file_name": "a09_s04_e03_v01", "length": 63, "label": 8},
                          {"file_name": "a01_s09_e04_v01", "length": 25, "label": 1},
                          {"file_name": "a05_s10_e00_v01", "length": 59, "label": 5},
                          {"file_name": "a09_s08_e02_v01", "length": 100, "label": 8},
                          {"file_name": "a11_s07_e01_v01", "length": 10, "label": 9},
                          {"file_name": "a06_s01_e00_v01", "length": 32, "label": 6},
                          {"file_name": "a12_s08_e04_v01", "length": 26, "label": 10},
                          {"file_name": "a08_s09_e04_v01", "length": 88, "label": 7},
                          {"file_name": "a12_s10_e02_v01", "length": 66, "label": 10},
                          {"file_name": "a04_s01_e01_v01", "length": 84, "label": 4},
                          {"file_name": "a01_s08_e01_v01", "length": 19, "label": 1},
                          {"file_name": "a09_s07_e00_v01", "length": 63, "label": 8},
                          {"file_name": "a04_s09_e00_v01", "length": 112, "label": 4},
                          {"file_name": "a08_s02_e02_v01", "length": 163, "label": 7},
                          {"file_name": "a09_s09_e02_v01", "length": 192, "label": 8},
                          {"file_name": "a09_s02_e03_v01", "length": 66, "label": 8},
                          {"file_name": "a11_s09_e01_v01", "length": 26, "label": 9},
                          {"file_name": "a03_s10_e01_v01", "length": 31, "label": 3},
                          {"file_name": "a11_s03_e02_v01", "length": 21, "label": 9},
                          {"file_name": "a11_s08_e04_v01", "length": 65, "label": 9},
                          {"file_name": "a06_s08_e02_v01", "length": 20, "label": 6},
                          {"file_name": "a11_s04_e04_v01", "length": 51, "label": 9},
                          {"file_name": "a12_s01_e00_v01", "length": 62, "label": 10},
                          {"file_name": "a02_s06_e04_v01", "length": 25, "label": 2},
                          {"file_name": "a06_s07_e01_v01", "length": 29, "label": 6},
                          {"file_name": "a05_s10_e03_v01", "length": 46, "label": 5},
                          {"file_name": "a09_s05_e04_v01", "length": 60, "label": 8},
                          {"file_name": "a03_s06_e00_v01", "length": 28, "label": 3},
                          {"file_name": "a12_s02_e01_v01", "length": 45, "label": 10},
                          {"file_name": "a08_s10_e02_v01", "length": 102, "label": 7},
                          {"file_name": "a08_s02_e00_v01", "length": 116, "label": 7},
                          {"file_name": "a06_s10_e03_v01", "length": 37, "label": 6},
                          {"file_name": "a11_s04_e02_v01", "length": 37, "label": 9},
                          {"file_name": "a08_s09_e03_v01", "length": 125, "label": 7},
                          {"file_name": "a12_s06_e04_v01", "length": 18, "label": 10},
                          {"file_name": "a01_s07_e01_v01", "length": 31, "label": 1},
                          {"file_name": "a05_s02_e04_v01", "length": 21, "label": 5},
                          {"file_name": "a09_s08_e00_v01", "length": 71, "label": 8},
                          {"file_name": "a02_s04_e04_v01", "length": 44, "label": 2},
                          {"file_name": "a06_s07_e00_v01", "length": 20, "label": 6},
                          {"file_name": "a04_s09_e01_v01", "length": 79, "label": 4},
                          {"file_name": "a09_s01_e00_v01", "length": 97, "label": 8},
                          {"file_name": "a08_s10_e01_v01", "length": 100, "label": 7},
                          {"file_name": "a11_s10_e02_v01", "length": 22, "label": 9},
                          {"file_name": "a09_s10_e02_v01", "length": 40, "label": 8},
                          {"file_name": "a03_s07_e04_v01", "length": 28, "label": 3},
                          {"file_name": "a05_s08_e00_v01", "length": 31, "label": 5},
                          {"file_name": "a05_s05_e03_v01", "length": 21, "label": 5},
                          {"file_name": "a11_s09_e03_v01", "length": 19, "label": 9},
                          {"file_name": "a12_s04_e04_v01", "length": 37, "label": 10},
                          {"file_name": "a04_s01_e03_v01", "length": 84, "label": 4},
                          {"file_name": "a04_s10_e02_v01", "length": 35, "label": 4},
                          {"file_name": "a06_s10_e04_v01", "length": 42, "label": 6},
                          {"file_name": "a01_s08_e00_v01", "length": 42, "label": 1},
                          {"file_name": "a03_s10_e02_v01", "length": 60, "label": 3},
                          {"file_name": "a03_s07_e01_v01", "length": 18, "label": 3},
                          {"file_name": "a05_s04_e03_v01", "length": 27, "label": 5},
                          {"file_name": "a01_s01_e02_v01", "length": 64, "label": 1},
                          {"file_name": "a05_s10_e04_v01", "length": 29, "label": 5},
                          {"file_name": "a06_s08_e03_v01", "length": 24, "label": 6},
                          {"file_name": "a02_s04_e02_v01", "length": 29, "label": 2},
                          {"file_name": "a12_s01_e04_v01", "length": 61, "label": 10},
                          {"file_name": "a02_s01_e02_v01", "length": 69, "label": 2},
                          {"file_name": "a12_s10_e00_v01", "length": 31, "label": 10},
                          {"file_name": "a11_s02_e00_v01", "length": 25, "label": 9},
                          {"file_name": "a02_s09_e01_v01", "length": 38, "label": 2},
                          {"file_name": "a12_s06_e05_v01", "length": 43, "label": 10},
                          {"file_name": "a02_s04_e00_v01", "length": 51, "label": 2},
                          {"file_name": "a12_s01_e02_v01", "length": 58, "label": 10},
                          {"file_name": "a04_s02_e05_v01", "length": 57, "label": 4},
                          {"file_name": "a03_s01_e04_v01", "length": 69, "label": 3},
                          {"file_name": "a01_s03_e04_v01", "length": 54, "label": 1},
                          {"file_name": "a01_s06_e03_v01", "length": 21, "label": 1},
                          {"file_name": "a02_s06_e01_v01", "length": 25, "label": 2},
                          {"file_name": "a12_s07_e04_v01", "length": 19, "label": 10},
                          {"file_name": "a08_s10_e04_v01", "length": 123, "label": 7},
                          {"file_name": "a02_s03_e02_v01", "length": 50, "label": 2},
                          {"file_name": "a09_s05_e06_v01", "length": 57, "label": 8},
                          {"file_name": "a05_s10_e01_v01", "length": 36, "label": 5},
                          {"file_name": "a09_s10_e01_v01", "length": 65, "label": 8},
                          {"file_name": "a08_s08_e04_v01", "length": 92, "label": 7},
                          {"file_name": "a06_s01_e02_v01", "length": 30, "label": 6},
                          {"file_name": "a01_s01_e01_v01", "length": 47, "label": 1},
                          {"file_name": "a06_s08_e04_v01", "length": 17, "label": 6},
                          {"file_name": "a09_s06_e03_v01", "length": 44, "label": 8},
                          {"file_name": "a06_s09_e01_v01", "length": 69, "label": 6},
                          {"file_name": "a08_s06_e01_v01", "length": 152, "label": 7},
                          {"file_name": "a02_s01_e04_v01", "length": 31, "label": 2},
                          {"file_name": "a11_s01_e00_v01", "length": 51, "label": 9},
                          {"file_name": "a05_s05_e02_v01", "length": 21, "label": 5},
                          {"file_name": "a03_s03_e00_v01", "length": 37, "label": 3},
                          {"file_name": "a01_s04_e04_v01", "length": 31, "label": 1},
                          {"file_name": "a06_s01_e04_v01", "length": 30, "label": 6},
                          {"file_name": "a09_s05_e05_v01", "length": 88, "label": 8},
                          {"file_name": "a01_s10_e01_v01", "length": 33, "label": 1},
                          {"file_name": "a03_s09_e00_v01", "length": 22, "label": 3},
                          {"file_name": "a08_s10_e00_v01", "length": 91, "label": 7},
                          {"file_name": "a05_s10_e02_v01", "length": 28, "label": 5},
                          {"file_name": "a03_s08_e05_v01", "length": 51, "label": 3},
                          {"file_name": "a04_s10_e01_v01", "length": 30, "label": 4},
                          {"file_name": "a05_s03_e04_v01", "length": 20, "label": 5},
                          {"file_name": "a05_s07_e02_v01", "length": 21, "label": 5},
                          {"file_name": "a12_s02_e04_v01", "length": 53, "label": 10},
                          {"file_name": "a06_s02_e03_v01", "length": 21, "label": 6},
                          {"file_name": "a09_s01_e03_v01", "length": 100, "label": 8},
                          {"file_name": "a08_s04_e00_v01", "length": 99, "label": 7},
                          {"file_name": "a02_s10_e01_v01", "length": 81, "label": 2},
                          {"file_name": "a11_s04_e01_v01", "length": 26, "label": 9},
                          {"file_name": "a03_s05_e01_v01", "length": 56, "label": 3},
                          {"file_name": "a06_s07_e04_v01", "length": 38, "label": 6},
                          {"file_name": "a09_s09_e03_v01", "length": 150, "label": 8},
                          {"file_name": "a02_s06_e02_v01", "length": 25, "label": 2},
                          {"file_name": "a05_s01_e04_v01", "length": 26, "label": 5},
                          {"file_name": "a11_s03_e04_v01", "length": 26, "label": 9},
                          {"file_name": "a04_s08_e02_v01", "length": 97, "label": 4},
                          {"file_name": "a04_s09_e04_v01", "length": 54, "label": 4},
                          {"file_name": "a08_s07_e00_v01", "length": 72, "label": 7},
                          {"file_name": "a04_s01_e05_v01", "length": 50, "label": 4},
                          {"file_name": "a12_s07_e01_v01", "length": 32, "label": 10},
                          {"file_name": "a02_s01_e03_v01", "length": 76, "label": 2},
                          {"file_name": "a11_s10_e05_v01", "length": 21, "label": 9},
                          {"file_name": "a09_s04_e00_v01", "length": 99, "label": 8},
                          {"file_name": "a09_s05_e01_v01", "length": 60, "label": 8},
                          {"file_name": "a09_s01_e04_v01", "length": 50, "label": 8},
                          {"file_name": "a12_s08_e00_v01", "length": 44, "label": 10},
                          {"file_name": "a04_s06_e03_v01", "length": 161, "label": 4},
                          {"file_name": "a05_s05_e00_v01", "length": 65, "label": 5},
                          {"file_name": "a11_s06_e01_v01", "length": 18, "label": 9},
                          {"file_name": "a01_s10_e02_v01", "length": 50, "label": 1},
                          {"file_name": "a04_s05_e01_v01", "length": 40, "label": 4},
                          {"file_name": "a02_s10_e04_v01", "length": 36, "label": 2},
                          {"file_name": "a02_s06_e05_v01", "length": 27, "label": 2},
                          {"file_name": "a11_s05_e00_v01", "length": 32, "label": 9},
                          {"file_name": "a04_s05_e06_v01", "length": 31, "label": 4},
                          {"file_name": "a04_s07_e01_v01", "length": 97, "label": 4},
                          {"file_name": "a03_s04_e01_v01", "length": 39, "label": 3},
                          {"file_name": "a03_s01_e02_v01", "length": 99, "label": 3},
                          {"file_name": "a06_s09_e02_v01", "length": 50, "label": 6},
                          {"file_name": "a03_s07_e00_v01", "length": 22, "label": 3},
                          {"file_name": "a08_s05_e05_v01", "length": 54, "label": 7},
                          {"file_name": "a06_s04_e02_v01", "length": 25, "label": 6},
                          {"file_name": "a12_s04_e01_v01", "length": 31, "label": 10},
                          {"file_name": "a09_s05_e00_v01", "length": 86, "label": 8},
                          {"file_name": "a04_s06_e02_v01", "length": 120, "label": 4},
                          {"file_name": "a04_s04_e04_v01", "length": 38, "label": 4},
                          {"file_name": "a09_s04_e02_v01", "length": 73, "label": 8},
                          {"file_name": "a02_s02_e01_v01", "length": 35, "label": 2},
                          {"file_name": "a06_s09_e00_v01", "length": 82, "label": 6},
                          {"file_name": "a05_s09_e00_v01", "length": 20, "label": 5},
                          {"file_name": "a05_s03_e01_v01", "length": 54, "label": 5},
                          {"file_name": "a02_s05_e04_v01", "length": 31, "label": 2},
                          {"file_name": "a01_s06_e01_v01", "length": 35, "label": 1},
                          {"file_name": "a01_s04_e05_v01", "length": 20, "label": 1},
                          {"file_name": "a12_s04_e02_v01", "length": 41, "label": 10},
                          {"file_name": "a03_s05_e02_v01", "length": 85, "label": 3},
                          {"file_name": "a03_s10_e04_v01", "length": 165, "label": 3},
                          {"file_name": "a01_s03_e02_v01", "length": 51, "label": 1},
                          {"file_name": "a05_s08_e05_v01", "length": 31, "label": 5},
                          {"file_name": "a01_s03_e00_v01", "length": 25, "label": 1},
                          {"file_name": "a08_s06_e03_v01", "length": 175, "label": 7},
                          {"file_name": "a04_s04_e07_v01", "length": 37, "label": 4},
                          {"file_name": "a05_s09_e02_v01", "length": 22, "label": 5},
                          {"file_name": "a01_s02_e01_v01", "length": 32, "label": 1},
                          {"file_name": "a01_s03_e01_v01", "length": 53, "label": 1},
                          {"file_name": "a04_s03_e01_v01", "length": 33, "label": 4},
                          {"file_name": "a06_s06_e00_v01", "length": 27, "label": 6},
                          {"file_name": "a12_s06_e02_v01", "length": 22, "label": 10},
                          {"file_name": "a04_s10_e04_v01", "length": 21, "label": 4},
                          {"file_name": "a12_s03_e01_v01", "length": 54, "label": 10},
                          {"file_name": "a08_s04_e02_v01", "length": 124, "label": 7},
                          {"file_name": "a06_s04_e04_v01", "length": 29, "label": 6},
                          {"file_name": "a12_s06_e03_v01", "length": 26, "label": 10},
                          {"file_name": "a08_s01_e04_v01", "length": 141, "label": 7},
                          {"file_name": "a04_s03_e00_v01", "length": 33, "label": 4},
                          {"file_name": "a12_s05_e02_v01", "length": 45, "label": 10},
                          {"file_name": "a08_s01_e00_v01", "length": 111, "label": 7},
                          {"file_name": "a01_s03_e03_v01", "length": 41, "label": 1},
                          {"file_name": "a01_s04_e02_v01", "length": 44, "label": 1},
                          {"file_name": "a06_s05_e00_v01", "length": 30, "label": 6},
                          {"file_name": "a01_s10_e04_v01", "length": 70, "label": 1},
                          {"file_name": "a08_s05_e00_v01", "length": 110, "label": 7},
                          {"file_name": "a09_s02_e00_v01", "length": 40, "label": 8},
                          {"file_name": "a12_s04_e05_v01", "length": 42, "label": 10},
                          {"file_name": "a06_s07_e02_v01", "length": 41, "label": 6},
                          {"file_name": "a08_s07_e02_v01", "length": 95, "label": 7},
                          {"file_name": "a11_s10_e01_v01", "length": 38, "label": 9},
                          {"file_name": "a02_s07_e00_v01", "length": 33, "label": 2},
                          {"file_name": "a06_s08_e01_v01", "length": 17, "label": 6},
                          {"file_name": "a01_s10_e03_v01", "length": 32, "label": 1},
                          {"file_name": "a11_s02_e04_v01", "length": 38, "label": 9},
                          {"file_name": "a12_s03_e03_v01", "length": 31, "label": 10},
                          {"file_name": "a05_s01_e01_v01", "length": 21, "label": 5},
                          {"file_name": "a05_s08_e02_v01", "length": 13, "label": 5},
                          {"file_name": "a09_s08_e01_v01", "length": 84, "label": 8},
                          {"file_name": "a01_s08_e04_v01", "length": 34, "label": 1},
                          {"file_name": "a09_s09_e00_v01", "length": 128, "label": 8},
                          {"file_name": "a03_s10_e03_v01", "length": 43, "label": 3},
                          {"file_name": "a09_s05_e03_v01", "length": 96, "label": 8},
                          {"file_name": "a09_s02_e04_v01", "length": 84, "label": 8},
                          {"file_name": "a08_s01_e01_v01", "length": 81, "label": 7},
                          {"file_name": "a09_s10_e00_v01", "length": 76, "label": 8},
                          {"file_name": "a04_s04_e10_v01", "length": 22, "label": 4},
                          {"file_name": "a05_s01_e00_v01", "length": 24, "label": 5},
                          {"file_name": "a06_s02_e01_v01", "length": 38, "label": 6},
                          {"file_name": "a08_s08_e03_v01", "length": 82, "label": 7},
                          {"file_name": "a04_s04_e03_v01", "length": 31, "label": 4},
                          {"file_name": "a12_s05_e04_v01", "length": 41, "label": 10},
                          {"file_name": "a05_s10_e05_v01", "length": 48, "label": 5},
                          {"file_name": "a02_s10_e02_v01", "length": 49, "label": 2},
                          {"file_name": "a06_s03_e03_v01", "length": 40, "label": 6},
                          {"file_name": "a05_s07_e04_v01", "length": 20, "label": 5},
                          {"file_name": "a02_s10_e00_v01", "length": 50, "label": 2},
                          {"file_name": "a08_s05_e03_v01", "length": 90, "label": 7},
                          {"file_name": "a12_s04_e00_v01", "length": 65, "label": 10},
                          {"file_name": "a03_s04_e02_v01", "length": 46, "label": 3},
                          {"file_name": "a06_s02_e02_v01", "length": 30, "label": 6},
                          {"file_name": "a03_s04_e03_v01", "length": 47, "label": 3},
                          {"file_name": "a11_s08_e03_v01", "length": 46, "label": 9},
                          {"file_name": "a09_s07_e03_v01", "length": 47, "label": 8},
                          {"file_name": "a05_s03_e03_v01", "length": 26, "label": 5},
                          {"file_name": "a09_s10_e03_v01", "length": 58, "label": 8},
                          {"file_name": "a01_s05_e03_v01", "length": 51, "label": 1},
                          {"file_name": "a11_s06_e02_v01", "length": 21, "label": 9},
                          {"file_name": "a05_s05_e01_v01", "length": 31, "label": 5},
                          {"file_name": "a01_s05_e01_v01", "length": 54, "label": 1},
                          {"file_name": "a04_s04_e02_v01", "length": 46, "label": 4},
                          {"file_name": "a11_s08_e02_v01", "length": 32, "label": 9},
                          {"file_name": "a11_s07_e03_v01", "length": 13, "label": 9},
                          {"file_name": "a06_s01_e01_v01", "length": 26, "label": 6},
                          {"file_name": "a06_s10_e05_v01", "length": 20, "label": 6},
                          {"file_name": "a12_s06_e00_v01", "length": 23, "label": 10},
                          {"file_name": "a12_s03_e02_v01", "length": 26, "label": 10},
                          {"file_name": "a08_s05_e02_v01", "length": 73, "label": 7},
                          {"file_name": "a03_s04_e00_v01", "length": 36, "label": 3},
                          {"file_name": "a11_s01_e03_v01", "length": 45, "label": 9},
                          {"file_name": "a03_s08_e01_v01", "length": 55, "label": 3},
                          {"file_name": "a11_s04_e00_v01", "length": 27, "label": 9},
                          {"file_name": "a04_s05_e00_v01", "length": 83, "label": 4},
                          {"file_name": "a12_s05_e01_v01", "length": 30, "label": 10},
                          {"file_name": "a02_s05_e02_v01", "length": 30, "label": 2},
                          {"file_name": "a06_s06_e01_v01", "length": 20, "label": 6},
                          {"file_name": "a03_s03_e02_v01", "length": 62, "label": 3},
                          {"file_name": "a11_s07_e02_v01", "length": 38, "label": 9},
                          {"file_name": "a11_s01_e02_v01", "length": 26, "label": 9},
                          {"file_name": "a05_s04_e02_v02", "length": 46, "label": 5},
                          {"file_name": "a12_s09_e04_v02", "length": 16, "label": 10},
                          {"file_name": "a03_s03_e04_v02", "length": 35, "label": 3},
                          {"file_name": "a08_s02_e01_v02", "length": 145, "label": 7},
                          {"file_name": "a03_s05_e03_v02", "length": 26, "label": 3},
                          {"file_name": "a06_s05_e01_v02", "length": 21, "label": 6},
                          {"file_name": "a12_s10_e01_v02", "length": 21, "label": 10},
                          {"file_name": "a01_s07_e03_v02", "length": 26, "label": 1},
                          {"file_name": "a03_s08_e02_v02", "length": 21, "label": 3},
                          {"file_name": "a11_s10_e03_v02", "length": 21, "label": 9},
                          {"file_name": "a04_s06_e05_v02", "length": 24, "label": 4},
                          {"file_name": "a11_s03_e00_v02", "length": 40, "label": 9},
                          {"file_name": "a03_s02_e00_v02", "length": 32, "label": 3},
                          {"file_name": "a11_s01_e04_v02", "length": 21, "label": 9},
                          {"file_name": "a04_s05_e04_v02", "length": 30, "label": 4},
                          {"file_name": "a09_s08_e04_v02", "length": 48, "label": 8},
                          {"file_name": "a09_s06_e01_v02", "length": 33, "label": 8},
                          {"file_name": "a09_s07_e01_v02", "length": 36, "label": 8},
                          {"file_name": "a02_s08_e01_v02", "length": 21, "label": 2},
                          {"file_name": "a01_s04_e01_v02", "length": 41, "label": 1},
                          {"file_name": "a02_s02_e02_v02", "length": 31, "label": 2},
                          {"file_name": "a02_s07_e05_v02", "length": 31, "label": 2},
                          {"file_name": "a06_s02_e00_v02", "length": 25, "label": 6},
                          {"file_name": "a03_s02_e02_v02", "length": 22, "label": 3},
                          {"file_name": "a11_s09_e04_v02", "length": 21, "label": 9},
                          {"file_name": "a09_s03_e04_v02", "length": 61, "label": 8},
                          {"file_name": "a04_s01_e02_v02", "length": 37, "label": 4},
                          {"file_name": "a12_s01_e01_v02", "length": 47, "label": 10},
                          {"file_name": "a02_s07_e03_v02", "length": 9, "label": 2},
                          {"file_name": "a05_s08_e04_v02", "length": 21, "label": 5},
                          {"file_name": "a02_s07_e02_v02", "length": 31, "label": 2},
                          {"file_name": "a04_s07_e02_v02", "length": 18, "label": 4},
                          {"file_name": "a01_s08_e03_v02", "length": 31, "label": 1},
                          {"file_name": "a08_s03_e01_v02", "length": 81, "label": 7},
                          {"file_name": "a04_s08_e03_v02", "length": 16, "label": 4},
                          {"file_name": "a03_s10_e00_v02", "length": 17, "label": 3},
                          {"file_name": "a04_s03_e03_v02", "length": 44, "label": 4},
                          {"file_name": "a11_s05_e02_v02", "length": 29, "label": 9},
                          {"file_name": "a06_s06_e02_v02", "length": 18, "label": 6},
                          {"file_name": "a09_s03_e00_v02", "length": 88, "label": 8},
                          {"file_name": "a09_s03_e03_v02", "length": 58, "label": 8},
                          {"file_name": "a04_s02_e02_v02", "length": 104, "label": 4},
                          {"file_name": "a08_s01_e02_v02", "length": 83, "label": 7},
                          {"file_name": "a04_s04_e00_v02", "length": 46, "label": 4},
                          {"file_name": "a03_s02_e03_v02", "length": 39, "label": 3},
                          {"file_name": "a05_s04_e00_v02", "length": 19, "label": 5},
                          {"file_name": "a05_s07_e03_v02", "length": 16, "label": 5},
                          {"file_name": "a06_s10_e00_v02", "length": 26, "label": 6},
                          {"file_name": "a11_s07_e00_v02", "length": 26, "label": 9},
                          {"file_name": "a03_s01_e01_v02", "length": 24, "label": 3},
                          {"file_name": "a04_s06_e01_v02", "length": 16, "label": 4},
                          {"file_name": "a08_s02_e04_v02", "length": 102, "label": 7},
                          {"file_name": "a09_s08_e03_v02", "length": 41, "label": 8},
                          {"file_name": "a05_s07_e00_v02", "length": 16, "label": 5},
                          {"file_name": "a05_s02_e02_v02", "length": 27, "label": 5},
                          {"file_name": "a04_s06_e04_v02", "length": 21, "label": 4},
                          {"file_name": "a05_s09_e03_v02", "length": 21, "label": 5},
                          {"file_name": "a03_s06_e02_v02", "length": 15, "label": 3},
                          {"file_name": "a01_s01_e00_v02", "length": 30, "label": 1},
                          {"file_name": "a06_s06_e03_v02", "length": 13, "label": 6},
                          {"file_name": "a06_s10_e02_v02", "length": 21, "label": 6},
                          {"file_name": "a02_s07_e04_v02", "length": 36, "label": 2},
                          {"file_name": "a09_s06_e00_v02", "length": 68, "label": 8},
                          {"file_name": "a04_s07_e04_v02", "length": 21, "label": 4},
                          {"file_name": "a05_s02_e01_v02", "length": 36, "label": 5},
                          {"file_name": "a01_s06_e04_v02", "length": 17, "label": 1},
                          {"file_name": "a04_s08_e01_v02", "length": 21, "label": 4},
                          {"file_name": "a01_s09_e00_v02", "length": 31, "label": 1},
                          {"file_name": "a08_s03_e03_v02", "length": 71, "label": 7},
                          {"file_name": "a12_s03_e00_v02", "length": 41, "label": 10},
                          {"file_name": "a11_s02_e03_v02", "length": 26, "label": 9},
                          {"file_name": "a12_s07_e02_v02", "length": 11, "label": 10},
                          {"file_name": "a11_s05_e01_v02", "length": 35, "label": 9},
                          {"file_name": "a05_s06_e01_v02", "length": 14, "label": 5},
                          {"file_name": "a06_s02_e04_v02", "length": 14, "label": 6},
                          {"file_name": "a06_s04_e00_v02", "length": 18, "label": 6},
                          {"file_name": "a05_s09_e01_v02", "length": 31, "label": 5},
                          {"file_name": "a11_s05_e03_v02", "length": 34, "label": 9},
                          {"file_name": "a03_s01_e00_v02", "length": 33, "label": 3},
                          {"file_name": "a11_s02_e01_v02", "length": 32, "label": 9},
                          {"file_name": "a04_s02_e00_v02", "length": 57, "label": 4},
                          {"file_name": "a11_s01_e01_v02", "length": 26, "label": 9},
                          {"file_name": "a02_s06_e03_v02", "length": 21, "label": 2},
                          {"file_name": "a12_s10_e03_v02", "length": 21, "label": 10},
                          {"file_name": "a01_s05_e02_v02", "length": 19, "label": 1},
                          {"file_name": "a01_s06_e00_v02", "length": 21, "label": 1},
                          {"file_name": "a05_s07_e01_v02", "length": 21, "label": 5},
                          {"file_name": "a01_s09_e01_v02", "length": 26, "label": 1},
                          {"file_name": "a02_s06_e00_v02", "length": 18, "label": 2},
                          {"file_name": "a11_s09_e00_v02", "length": 11, "label": 9},
                          {"file_name": "a03_s03_e01_v02", "length": 47, "label": 3},
                          {"file_name": "a03_s08_e00_v02", "length": 22, "label": 3},
                          {"file_name": "a06_s04_e01_v02", "length": 21, "label": 6},
                          {"file_name": "a02_s05_e01_v02", "length": 34, "label": 2},
                          {"file_name": "a03_s04_e04_v02", "length": 29, "label": 3},
                          {"file_name": "a01_s09_e02_v02", "length": 22, "label": 1},
                          {"file_name": "a08_s03_e04_v02", "length": 59, "label": 7},
                          {"file_name": "a01_s10_e00_v02", "length": 28, "label": 1},
                          {"file_name": "a01_s02_e02_v02", "length": 23, "label": 1},
                          {"file_name": "a09_s03_e01_v02", "length": 42, "label": 8},
                          {"file_name": "a05_s06_e00_v02", "length": 23, "label": 5},
                          {"file_name": "a05_s01_e02_v02", "length": 31, "label": 5},
                          {"file_name": "a02_s02_e04_v02", "length": 28, "label": 2},
                          {"file_name": "a06_s07_e03_v02", "length": 21, "label": 6},
                          {"file_name": "a04_s02_e04_v02", "length": 23, "label": 4},
                          {"file_name": "a04_s05_e02_v02", "length": 29, "label": 4},
                          {"file_name": "a02_s07_e01_v02", "length": 31, "label": 2},
                          {"file_name": "a04_s02_e06_v02", "length": 28, "label": 4},
                          {"file_name": "a03_s07_e03_v02", "length": 11, "label": 3},
                          {"file_name": "a12_s08_e01_v02", "length": 14, "label": 10},
                          {"file_name": "a05_s01_e03_v02", "length": 31, "label": 5},
                          {"file_name": "a02_s09_e02_v02", "length": 43, "label": 2},
                          {"file_name": "a05_s08_e03_v02", "length": 26, "label": 5},
                          {"file_name": "a04_s06_e00_v02", "length": 18, "label": 4},
                          {"file_name": "a09_s01_e02_v02", "length": 67, "label": 8},
                          {"file_name": "a12_s09_e00_v02", "length": 21, "label": 10},
                          {"file_name": "a04_s09_e02_v02", "length": 16, "label": 4},
                          {"file_name": "a03_s03_e03_v02", "length": 43, "label": 3},
                          {"file_name": "a08_s07_e03_v02", "length": 54, "label": 7},
                          {"file_name": "a08_s09_e02_v02", "length": 76, "label": 7},
                          {"file_name": "a08_s09_e00_v02", "length": 71, "label": 7},
                          {"file_name": "a06_s06_e04_v02", "length": 16, "label": 6},
                          {"file_name": "a01_s07_e04_v02", "length": 21, "label": 1},
                          {"file_name": "a08_s05_e04_v02", "length": 45, "label": 7},
                          {"file_name": "a05_s04_e01_v02", "length": 26, "label": 5},
                          {"file_name": "a04_s07_e00_v02", "length": 23, "label": 4},
                          {"file_name": "a05_s08_e01_v02", "length": 21, "label": 5},
                          {"file_name": "a11_s06_e03_v02", "length": 17, "label": 9},
                          {"file_name": "a01_s04_e03_v02", "length": 34, "label": 1},
                          {"file_name": "a11_s06_e04_v02", "length": 8, "label": 9},
                          {"file_name": "a12_s07_e03_v02", "length": 16, "label": 10},
                          {"file_name": "a01_s02_e04_v02", "length": 21, "label": 1},
                          {"file_name": "a04_s04_e05_v02", "length": 132, "label": 4},
                          {"file_name": "a03_s01_e03_v02", "length": 36, "label": 3},
                          {"file_name": "a12_s02_e02_v02", "length": 38, "label": 10},
                          {"file_name": "a03_s06_e01_v02", "length": 17, "label": 3},
                          {"file_name": "a05_s02_e03_v02", "length": 28, "label": 5},
                          {"file_name": "a03_s02_e04_v02", "length": 23, "label": 3},
                          {"file_name": "a08_s02_e03_v02", "length": 113, "label": 7},
                          {"file_name": "a08_s03_e02_v02", "length": 67, "label": 7},
                          {"file_name": "a09_s01_e01_v02", "length": 55, "label": 8},
                          {"file_name": "a02_s01_e01_v02", "length": 30, "label": 2},
                          {"file_name": "a08_s06_e00_v02", "length": 86, "label": 7},
                          {"file_name": "a12_s08_e02_v02", "length": 16, "label": 10},
                          {"file_name": "a02_s08_e00_v02", "length": 26, "label": 2},
                          {"file_name": "a01_s08_e02_v02", "length": 33, "label": 1},
                          {"file_name": "a09_s04_e01_v02", "length": 74, "label": 8},
                          {"file_name": "a04_s01_e04_v02", "length": 26, "label": 4},
                          {"file_name": "a04_s05_e03_v02", "length": 31, "label": 4},
                          {"file_name": "a08_s10_e03_v02", "length": 61, "label": 7},
                          {"file_name": "a02_s05_e00_v02", "length": 28, "label": 2},
                          {"file_name": "a06_s04_e03_v02", "length": 24, "label": 6},
                          {"file_name": "a06_s09_e03_v02", "length": 21, "label": 6},
                          {"file_name": "a05_s03_e02_v02", "length": 21, "label": 5},
                          {"file_name": "a06_s03_e04_v02", "length": 12, "label": 6},
                          {"file_name": "a06_s01_e03_v02", "length": 16, "label": 6},
                          {"file_name": "a11_s03_e01_v02", "length": 23, "label": 9},
                          {"file_name": "a09_s02_e01_v02", "length": 33, "label": 8},
                          {"file_name": "a02_s02_e00_v02", "length": 42, "label": 2},
                          {"file_name": "a01_s01_e03_v02", "length": 39, "label": 1},
                          {"file_name": "a08_s06_e02_v02", "length": 83, "label": 7},
                          {"file_name": "a12_s01_e03_v02", "length": 41, "label": 10},
                          {"file_name": "a06_s05_e04_v02", "length": 16, "label": 6},
                          {"file_name": "a01_s04_e06_v02", "length": 24, "label": 1},
                          {"file_name": "a09_s09_e01_v02", "length": 41, "label": 8},
                          {"file_name": "a04_s10_e03_v02", "length": 16, "label": 4},
                          {"file_name": "a06_s09_e04_v02", "length": 16, "label": 6},
                          {"file_name": "a02_s04_e01_v02", "length": 31, "label": 2},
                          {"file_name": "a12_s10_e04_v02", "length": 14, "label": 10},
                          {"file_name": "a04_s03_e05_v02", "length": 42, "label": 4},
                          {"file_name": "a06_s03_e01_v02", "length": 25, "label": 6},
                          {"file_name": "a02_s03_e04_v02", "length": 62, "label": 2},
                          {"file_name": "a11_s09_e02_v02", "length": 25, "label": 9},
                          {"file_name": "a08_s08_e02_v02", "length": 53, "label": 7},
                          {"file_name": "a03_s02_e01_v02", "length": 36, "label": 3},
                          {"file_name": "a12_s02_e00_v02", "length": 50, "label": 10},
                          {"file_name": "a12_s08_e03_v02", "length": 13, "label": 10},
                          {"file_name": "a02_s09_e03_v02", "length": 31, "label": 2},
                          {"file_name": "a09_s02_e02_v02", "length": 46, "label": 8},
                          {"file_name": "a05_s09_e04_v02", "length": 21, "label": 5},
                          {"file_name": "a01_s04_e00_v02", "length": 26, "label": 1},
                          {"file_name": "a08_s04_e03_v02", "length": 121, "label": 7},
                          {"file_name": "a08_s05_e01_v02", "length": 59, "label": 7},
                          {"file_name": "a12_s09_e03_v02", "length": 16, "label": 10},
                          {"file_name": "a02_s04_e03_v02", "length": 31, "label": 2},
                          {"file_name": "a04_s03_e04_v02", "length": 49, "label": 4},
                          {"file_name": "a12_s06_e01_v02", "length": 16, "label": 10},
                          {"file_name": "a11_s04_e03_v02", "length": 32, "label": 9},
                          {"file_name": "a05_s03_e00_v02", "length": 22, "label": 5},
                          {"file_name": "a12_s07_e00_v02", "length": 18, "label": 10},
                          {"file_name": "a06_s03_e02_v02", "length": 16, "label": 6},
                          {"file_name": "a03_s03_e05_v02", "length": 33, "label": 3},
                          {"file_name": "a11_s08_e01_v02", "length": 22, "label": 9},
                          {"file_name": "a05_s05_e04_v02", "length": 17, "label": 5},
                          {"file_name": "a06_s10_e01_v02", "length": 17, "label": 6},
                          {"file_name": "a04_s03_e02_v02", "length": 108, "label": 4},
                          {"file_name": "a02_s03_e03_v02", "length": 56, "label": 2},
                          {"file_name": "a09_s10_e04_v02", "length": 36, "label": 8},
                          {"file_name": "a04_s08_e04_v02", "length": 36, "label": 4},
                          {"file_name": "a11_s08_e00_v02", "length": 35, "label": 9},
                          {"file_name": "a02_s01_e00_v02", "length": 39, "label": 2},
                          {"file_name": "a04_s02_e03_v02", "length": 45, "label": 4},
                          {"file_name": "a04_s02_e01_v02", "length": 113, "label": 4},
                          {"file_name": "a06_s08_e00_v02", "length": 19, "label": 6},
                          {"file_name": "a08_s08_e01_v02", "length": 49, "label": 7},
                          {"file_name": "a02_s03_e01_v02", "length": 45, "label": 2},
                          {"file_name": "a11_s02_e02_v02", "length": 33, "label": 9},
                          {"file_name": "a09_s07_e02_v02", "length": 29, "label": 8},
                          {"file_name": "a02_s05_e03_v02", "length": 21, "label": 2},
                          {"file_name": "a01_s07_e02_v02", "length": 23, "label": 1},
                          {"file_name": "a06_s05_e03_v02", "length": 15, "label": 6},
                          {"file_name": "a12_s05_e03_v02", "length": 33, "label": 10},
                          {"file_name": "a03_s05_e00_v02", "length": 20, "label": 3},
                          {"file_name": "a09_s03_e02_v02", "length": 58, "label": 8},
                          {"file_name": "a09_s04_e04_v02", "length": 138, "label": 8},
                          {"file_name": "a11_s10_e00_v02", "length": 21, "label": 9},
                          {"file_name": "a04_s04_e01_v02", "length": 35, "label": 4},
                          {"file_name": "a02_s08_e02_v02", "length": 21, "label": 2},
                          {"file_name": "a01_s05_e00_v02", "length": 27, "label": 1},
                          {"file_name": "a04_s01_e07_v02", "length": 34, "label": 4},
                          {"file_name": "a11_s06_e00_v02", "length": 27, "label": 9},
                          {"file_name": "a05_s02_e00_v02", "length": 36, "label": 5},
                          {"file_name": "a02_s02_e03_v02", "length": 29, "label": 2},
                          {"file_name": "a09_s05_e02_v02", "length": 51, "label": 8},
                          {"file_name": "a05_s06_e02_v02", "length": 16, "label": 5},
                          {"file_name": "a08_s01_e03_v02", "length": 80, "label": 7},
                          {"file_name": "a08_s09_e01_v02", "length": 62, "label": 7},
                          {"file_name": "a02_s08_e04_v02", "length": 36, "label": 2},
                          {"file_name": "a06_s05_e02_v02", "length": 21, "label": 6},
                          {"file_name": "a01_s02_e03_v02", "length": 24, "label": 1},
                          {"file_name": "a03_s09_e02_v02", "length": 26, "label": 3},
                          {"file_name": "a04_s08_e00_v02", "length": 31, "label": 4},
                          {"file_name": "a12_s03_e04_v02", "length": 46, "label": 10},
                          {"file_name": "a08_s04_e01_v02", "length": 126, "label": 7},
                          {"file_name": "a12_s04_e03_v02", "length": 35, "label": 10},
                          {"file_name": "a04_s09_e03_v02", "length": 26, "label": 4},
                          {"file_name": "a12_s05_e00_v02", "length": 31, "label": 10},
                          {"file_name": "a11_s05_e04_v02", "length": 25, "label": 9},
                          {"file_name": "a05_s06_e03_v02", "length": 30, "label": 5},
                          {"file_name": "a09_s06_e02_v02", "length": 39, "label": 8},
                          {"file_name": "a12_s02_e03_v02", "length": 27, "label": 10},
                          {"file_name": "a11_s03_e03_v02", "length": 21, "label": 9},
                          {"file_name": "a11_s07_e04_v02", "length": 17, "label": 9},
                          {"file_name": "a04_s01_e00_v02", "length": 43, "label": 4},
                          {"file_name": "a03_s08_e03_v02", "length": 14, "label": 3},
                          {"file_name": "a04_s10_e00_v02", "length": 21, "label": 4},
                          {"file_name": "a08_s03_e00_v02", "length": 116, "label": 7},
                          {"file_name": "a02_s08_e03_v02", "length": 21, "label": 2},
                          {"file_name": "a01_s09_e03_v02", "length": 24, "label": 1},
                          {"file_name": "a01_s01_e04_v02", "length": 29, "label": 1},
                          {"file_name": "a01_s07_e00_v02", "length": 21, "label": 1},
                          {"file_name": "a02_s03_e00_v02", "length": 46, "label": 2},
                          {"file_name": "a01_s02_e00_v02", "length": 23, "label": 1},
                          {"file_name": "a03_s09_e04_v02", "length": 21, "label": 3},
                          {"file_name": "a01_s06_e02_v02", "length": 21, "label": 1},
                          {"file_name": "a03_s07_e02_v02", "length": 17, "label": 3},
                          {"file_name": "a03_s05_e04_v02", "length": 39, "label": 3},
                          {"file_name": "a08_s07_e01_v02", "length": 104, "label": 7},
                          {"file_name": "a04_s07_e03_v02", "length": 21, "label": 4},
                          {"file_name": "a08_s04_e04_v02", "length": 124, "label": 7},
                          {"file_name": "a08_s08_e00_v02", "length": 58, "label": 7},
                          {"file_name": "a02_s09_e00_v02", "length": 37, "label": 2},
                          {"file_name": "a06_s03_e00_v02", "length": 24, "label": 6},
                          {"file_name": "a09_s09_e04_v02", "length": 36, "label": 8},
                          {"file_name": "a05_s04_e04_v02", "length": 21, "label": 5},
                          {"file_name": "a09_s04_e03_v02", "length": 61, "label": 8},
                          {"file_name": "a01_s09_e04_v02", "length": 28, "label": 1},
                          {"file_name": "a05_s10_e00_v02", "length": 26, "label": 5},
                          {"file_name": "a09_s08_e02_v02", "length": 36, "label": 8},
                          {"file_name": "a11_s07_e01_v02", "length": 15, "label": 9},
                          {"file_name": "a06_s01_e00_v02", "length": 21, "label": 6},
                          {"file_name": "a12_s08_e04_v02", "length": 14, "label": 10},
                          {"file_name": "a08_s09_e04_v02", "length": 56, "label": 7},
                          {"file_name": "a12_s10_e02_v02", "length": 16, "label": 10},
                          {"file_name": "a04_s01_e01_v02", "length": 83, "label": 4},
                          {"file_name": "a01_s08_e01_v02", "length": 26, "label": 1},
                          {"file_name": "a09_s07_e00_v02", "length": 31, "label": 8},
                          {"file_name": "a04_s09_e00_v02", "length": 26, "label": 4},
                          {"file_name": "a08_s02_e02_v02", "length": 134, "label": 7},
                          {"file_name": "a09_s09_e02_v02", "length": 57, "label": 8},
                          {"file_name": "a09_s02_e03_v02", "length": 46, "label": 8},
                          {"file_name": "a11_s09_e01_v02", "length": 14, "label": 9},
                          {"file_name": "a03_s10_e01_v02", "length": 11, "label": 3},
                          {"file_name": "a11_s03_e02_v02", "length": 36, "label": 9},
                          {"file_name": "a11_s08_e04_v02", "length": 16, "label": 9},
                          {"file_name": "a06_s08_e02_v02", "length": 16, "label": 6},
                          {"file_name": "a12_s01_e00_v02", "length": 21, "label": 10},
                          {"file_name": "a02_s06_e04_v02", "length": 21, "label": 2},
                          {"file_name": "a06_s07_e01_v02", "length": 21, "label": 6},
                          {"file_name": "a05_s10_e03_v02", "length": 21, "label": 5},
                          {"file_name": "a09_s05_e04_v02", "length": 66, "label": 8},
                          {"file_name": "a03_s06_e00_v02", "length": 23, "label": 3},
                          {"file_name": "a12_s02_e01_v02", "length": 40, "label": 10},
                          {"file_name": "a08_s10_e02_v02", "length": 56, "label": 7},
                          {"file_name": "a08_s02_e00_v02", "length": 111, "label": 7},
                          {"file_name": "a06_s10_e03_v02", "length": 21, "label": 6},
                          {"file_name": "a11_s04_e02_v02", "length": 33, "label": 9},
                          {"file_name": "a08_s09_e03_v02", "length": 66, "label": 7},
                          {"file_name": "a12_s06_e04_v02", "length": 11, "label": 10},
                          {"file_name": "a01_s07_e01_v02", "length": 27, "label": 1},
                          {"file_name": "a05_s02_e04_v02", "length": 22, "label": 5},
                          {"file_name": "a09_s08_e00_v02", "length": 41, "label": 8},
                          {"file_name": "a02_s04_e04_v02", "length": 33, "label": 2},
                          {"file_name": "a06_s07_e00_v02", "length": 15, "label": 6},
                          {"file_name": "a04_s09_e01_v02", "length": 21, "label": 4},
                          {"file_name": "a09_s01_e00_v02", "length": 42, "label": 8},
                          {"file_name": "a08_s10_e01_v02", "length": 91, "label": 7},
                          {"file_name": "a11_s10_e02_v02", "length": 56, "label": 9},
                          {"file_name": "a09_s10_e02_v02", "length": 41, "label": 8},
                          {"file_name": "a03_s07_e04_v02", "length": 11, "label": 3},
                          {"file_name": "a05_s08_e00_v02", "length": 26, "label": 5},
                          {"file_name": "a05_s05_e03_v02", "length": 25, "label": 5},
                          {"file_name": "a11_s09_e03_v02", "length": 11, "label": 9},
                          {"file_name": "a12_s04_e04_v02", "length": 36, "label": 10},
                          {"file_name": "a04_s01_e03_v02", "length": 30, "label": 4},
                          {"file_name": "a04_s10_e02_v02", "length": 21, "label": 4},
                          {"file_name": "a06_s10_e04_v02", "length": 21, "label": 6},
                          {"file_name": "a01_s08_e00_v02", "length": 21, "label": 1},
                          {"file_name": "a03_s10_e02_v02", "length": 28, "label": 3},
                          {"file_name": "a03_s07_e01_v02", "length": 11, "label": 3},
                          {"file_name": "a05_s04_e03_v02", "length": 22, "label": 5},
                          {"file_name": "a01_s01_e02_v02", "length": 31, "label": 1},
                          {"file_name": "a05_s10_e04_v02", "length": 21, "label": 5},
                          {"file_name": "a06_s08_e03_v02", "length": 21, "label": 6},
                          {"file_name": "a02_s04_e02_v02", "length": 33, "label": 2},
                          {"file_name": "a04_s01_e09_v02", "length": 33, "label": 4},
                          {"file_name": "a12_s01_e04_v02", "length": 37, "label": 10},
                          {"file_name": "a02_s01_e02_v02", "length": 28, "label": 2},
                          {"file_name": "a12_s10_e00_v02", "length": 21, "label": 10},
                          {"file_name": "a11_s02_e00_v02", "length": 40, "label": 9},
                          {"file_name": "a02_s09_e01_v02", "length": 40, "label": 2},
                          {"file_name": "a02_s04_e00_v02", "length": 46, "label": 2},
                          {"file_name": "a12_s01_e02_v02", "length": 27, "label": 10},
                          {"file_name": "a04_s02_e05_v02", "length": 61, "label": 4},
                          {"file_name": "a03_s01_e04_v02", "length": 36, "label": 3},
                          {"file_name": "a01_s03_e04_v02", "length": 46, "label": 1},
                          {"file_name": "a02_s06_e01_v02", "length": 16, "label": 2},
                          {"file_name": "a12_s07_e04_v02", "length": 11, "label": 10},
                          {"file_name": "a12_s03_e05_v02", "length": 33, "label": 10},
                          {"file_name": "a08_s10_e04_v02", "length": 66, "label": 7},
                          {"file_name": "a02_s03_e02_v02", "length": 58, "label": 2},
                          {"file_name": "a05_s06_e04_v02", "length": 21, "label": 5},
                          {"file_name": "a05_s10_e01_v02", "length": 21, "label": 5},
                          {"file_name": "a09_s10_e01_v02", "length": 49, "label": 8},
                          {"file_name": "a08_s08_e04_v02", "length": 61, "label": 7},
                          {"file_name": "a06_s01_e02_v02", "length": 11, "label": 6},
                          {"file_name": "a01_s01_e01_v02", "length": 28, "label": 1},
                          {"file_name": "a06_s08_e04_v02", "length": 21, "label": 6},
                          {"file_name": "a09_s06_e03_v02", "length": 47, "label": 8},
                          {"file_name": "a06_s09_e01_v02", "length": 16, "label": 6},
                          {"file_name": "a08_s06_e01_v02", "length": 116, "label": 7},
                          {"file_name": "a02_s01_e04_v02", "length": 38, "label": 2},
                          {"file_name": "a11_s01_e00_v02", "length": 31, "label": 9},
                          {"file_name": "a05_s05_e02_v02", "length": 17, "label": 5},
                          {"file_name": "a03_s03_e00_v02", "length": 41, "label": 3},
                          {"file_name": "a01_s04_e04_v02", "length": 34, "label": 1},
                          {"file_name": "a06_s01_e04_v02", "length": 21, "label": 6},
                          {"file_name": "a09_s05_e05_v02", "length": 48, "label": 8},
                          {"file_name": "a01_s10_e01_v02", "length": 21, "label": 1},
                          {"file_name": "a03_s09_e00_v02", "length": 26, "label": 3},
                          {"file_name": "a08_s10_e00_v02", "length": 67, "label": 7},
                          {"file_name": "a05_s10_e02_v02", "length": 21, "label": 5},
                          {"file_name": "a04_s10_e01_v02", "length": 23, "label": 4},
                          {"file_name": "a05_s03_e04_v02", "length": 26, "label": 5},
                          {"file_name": "a05_s07_e02_v02", "length": 36, "label": 5},
                          {"file_name": "a12_s02_e04_v02", "length": 37, "label": 10},
                          {"file_name": "a04_s02_e07_v02", "length": 47, "label": 4},
                          {"file_name": "a06_s02_e03_v02", "length": 13, "label": 6},
                          {"file_name": "a09_s01_e03_v02", "length": 56, "label": 8},
                          {"file_name": "a08_s04_e00_v02", "length": 86, "label": 7},
                          {"file_name": "a02_s10_e01_v02", "length": 32, "label": 2},
                          {"file_name": "a11_s04_e01_v02", "length": 15, "label": 9},
                          {"file_name": "a03_s05_e01_v02", "length": 39, "label": 3},
                          {"file_name": "a06_s07_e04_v02", "length": 19, "label": 6},
                          {"file_name": "a09_s09_e03_v02", "length": 51, "label": 8},
                          {"file_name": "a02_s06_e02_v02", "length": 21, "label": 2},
                          {"file_name": "a05_s01_e04_v02", "length": 21, "label": 5},
                          {"file_name": "a11_s03_e04_v02", "length": 12, "label": 9},
                          {"file_name": "a04_s08_e02_v02", "length": 21, "label": 4},
                          {"file_name": "a04_s09_e04_v02", "length": 36, "label": 4},
                          {"file_name": "a08_s07_e00_v02", "length": 53, "label": 7},
                          {"file_name": "a04_s01_e05_v02", "length": 37, "label": 4},
                          {"file_name": "a12_s07_e01_v02", "length": 14, "label": 10},
                          {"file_name": "a02_s01_e03_v02", "length": 40, "label": 2},
                          {"file_name": "a09_s04_e00_v02", "length": 84, "label": 8},
                          {"file_name": "a09_s05_e01_v02", "length": 65, "label": 8},
                          {"file_name": "a09_s01_e04_v02", "length": 65, "label": 8},
                          {"file_name": "a12_s08_e00_v02", "length": 13, "label": 10},
                          {"file_name": "a04_s06_e03_v02", "length": 12, "label": 4},
                          {"file_name": "a05_s05_e00_v02", "length": 41, "label": 5},
                          {"file_name": "a11_s06_e01_v02", "length": 17, "label": 9},
                          {"file_name": "a01_s10_e02_v02", "length": 26, "label": 1},
                          {"file_name": "a04_s05_e01_v02", "length": 26, "label": 4},
                          {"file_name": "a08_s05_e06_v02", "length": 24, "label": 7},
                          {"file_name": "a02_s10_e04_v02", "length": 29, "label": 2},
                          {"file_name": "a11_s05_e00_v02", "length": 27, "label": 9},
                          {"file_name": "a04_s07_e01_v02", "length": 21, "label": 4},
                          {"file_name": "a03_s04_e01_v02", "length": 39, "label": 3},
                          {"file_name": "a03_s01_e02_v02", "length": 31, "label": 3},
                          {"file_name": "a06_s09_e02_v02", "length": 16, "label": 6},
                          {"file_name": "a03_s07_e00_v02", "length": 21, "label": 3},
                          {"file_name": "a11_s05_e05_v02", "length": 29, "label": 9},
                          {"file_name": "a08_s05_e05_v02", "length": 44, "label": 7},
                          {"file_name": "a06_s04_e02_v02", "length": 41, "label": 6},
                          {"file_name": "a12_s04_e01_v02", "length": 36, "label": 10},
                          {"file_name": "a09_s05_e00_v02", "length": 70, "label": 8},
                          {"file_name": "a04_s06_e02_v02", "length": 16, "label": 4},
                          {"file_name": "a04_s04_e04_v02", "length": 53, "label": 4},
                          {"file_name": "a09_s04_e02_v02", "length": 61, "label": 8},
                          {"file_name": "a02_s02_e01_v02", "length": 26, "label": 2},
                          {"file_name": "a06_s09_e00_v02", "length": 16, "label": 6},
                          {"file_name": "a05_s09_e00_v02", "length": 21, "label": 5},
                          {"file_name": "a05_s03_e01_v02", "length": 28, "label": 5},
                          {"file_name": "a02_s05_e04_v02", "length": 29, "label": 2},
                          {"file_name": "a01_s06_e01_v02", "length": 24, "label": 1},
                          {"file_name": "a01_s04_e05_v02", "length": 29, "label": 1},
                          {"file_name": "a12_s04_e02_v02", "length": 23, "label": 10},
                          {"file_name": "a03_s05_e02_v02", "length": 36, "label": 3},
                          {"file_name": "a01_s03_e02_v02", "length": 61, "label": 1},
                          {"file_name": "a05_s04_e05_v02", "length": 21, "label": 5},
                          {"file_name": "a01_s03_e00_v02", "length": 26, "label": 1},
                          {"file_name": "a08_s06_e03_v02", "length": 103, "label": 7},
                          {"file_name": "a05_s09_e02_v02", "length": 21, "label": 5},
                          {"file_name": "a01_s02_e01_v02", "length": 21, "label": 1},
                          {"file_name": "a01_s03_e01_v02", "length": 42, "label": 1},
                          {"file_name": "a04_s03_e01_v02", "length": 29, "label": 4},
                          {"file_name": "a06_s06_e00_v02", "length": 16, "label": 6},
                          {"file_name": "a12_s06_e02_v02", "length": 26, "label": 10},
                          {"file_name": "a12_s03_e01_v02", "length": 44, "label": 10},
                          {"file_name": "a08_s04_e02_v02", "length": 116, "label": 7},
                          {"file_name": "a06_s04_e04_v02", "length": 20, "label": 6},
                          {"file_name": "a12_s06_e03_v02", "length": 14, "label": 10},
                          {"file_name": "a08_s01_e04_v02", "length": 81, "label": 7},
                          {"file_name": "a04_s03_e00_v02", "length": 28, "label": 4},
                          {"file_name": "a12_s05_e02_v02", "length": 28, "label": 10},
                          {"file_name": "a08_s01_e00_v02", "length": 146, "label": 7},
                          {"file_name": "a01_s03_e03_v02", "length": 53, "label": 1},
                          {"file_name": "a04_s01_e08_v02", "length": 83, "label": 4},
                          {"file_name": "a01_s04_e02_v02", "length": 26, "label": 1},
                          {"file_name": "a06_s05_e00_v02", "length": 30, "label": 6},
                          {"file_name": "a01_s10_e04_v02", "length": 21, "label": 1},
                          {"file_name": "a08_s05_e00_v02", "length": 61, "label": 7},
                          {"file_name": "a09_s02_e00_v02", "length": 32, "label": 8},
                          {"file_name": "a12_s04_e05_v02", "length": 29, "label": 10},
                          {"file_name": "a06_s07_e02_v02", "length": 21, "label": 6},
                          {"file_name": "a08_s07_e02_v02", "length": 40, "label": 7},
                          {"file_name": "a11_s10_e01_v02", "length": 31, "label": 9},
                          {"file_name": "a02_s07_e00_v02", "length": 31, "label": 2},
                          {"file_name": "a06_s08_e01_v02", "length": 16, "label": 6},
                          {"file_name": "a01_s10_e03_v02", "length": 25, "label": 1},
                          {"file_name": "a11_s02_e04_v02", "length": 35, "label": 9},
                          {"file_name": "a02_s09_e04_v02", "length": 1, "label": 2},
                          {"file_name": "a12_s03_e03_v02", "length": 39, "label": 10},
                          {"file_name": "a05_s01_e01_v02", "length": 24, "label": 5},
                          {"file_name": "a05_s08_e02_v02", "length": 16, "label": 5},
                          {"file_name": "a12_s09_e02_v02", "length": 21, "label": 10},
                          {"file_name": "a09_s08_e01_v02", "length": 40, "label": 8},
                          {"file_name": "a01_s08_e04_v02", "length": 21, "label": 1},
                          {"file_name": "a09_s09_e00_v02", "length": 51, "label": 8},
                          {"file_name": "a03_s10_e03_v02", "length": 13, "label": 3},
                          {"file_name": "a09_s05_e03_v02", "length": 46, "label": 8},
                          {"file_name": "a09_s02_e04_v02", "length": 49, "label": 8},
                          {"file_name": "a08_s01_e01_v02", "length": 91, "label": 7},
                          {"file_name": "a09_s10_e00_v02", "length": 41, "label": 8},
                          {"file_name": "a12_s09_e01_v02", "length": 16, "label": 10},
                          {"file_name": "a05_s01_e00_v02", "length": 26, "label": 5},
                          {"file_name": "a06_s02_e01_v02", "length": 13, "label": 6},
                          {"file_name": "a08_s08_e03_v02", "length": 56, "label": 7},
                          {"file_name": "a04_s04_e03_v02", "length": 61, "label": 4},
                          {"file_name": "a12_s05_e04_v02", "length": 36, "label": 10},
                          {"file_name": "a02_s10_e02_v02", "length": 31, "label": 2},
                          {"file_name": "a06_s03_e03_v02", "length": 16, "label": 6},
                          {"file_name": "a05_s07_e04_v02", "length": 21, "label": 5},
                          {"file_name": "a02_s10_e00_v02", "length": 38, "label": 2},
                          {"file_name": "a08_s05_e03_v02", "length": 46, "label": 7},
                          {"file_name": "a12_s04_e00_v02", "length": 46, "label": 10},
                          {"file_name": "a03_s04_e02_v02", "length": 27, "label": 3},
                          {"file_name": "a06_s02_e02_v02", "length": 11, "label": 6},
                          {"file_name": "a03_s04_e03_v02", "length": 31, "label": 3},
                          {"file_name": "a11_s08_e03_v02", "length": 21, "label": 9},
                          {"file_name": "a09_s07_e03_v02", "length": 35, "label": 8},
                          {"file_name": "a05_s03_e03_v02", "length": 26, "label": 5},
                          {"file_name": "a09_s10_e03_v02", "length": 31, "label": 8},
                          {"file_name": "a11_s06_e02_v02", "length": 16, "label": 9},
                          {"file_name": "a05_s05_e01_v02", "length": 23, "label": 5},
                          {"file_name": "a01_s05_e01_v02", "length": 35, "label": 1},
                          {"file_name": "a04_s04_e02_v02", "length": 34, "label": 4},
                          {"file_name": "a11_s08_e02_v02", "length": 17, "label": 9},
                          {"file_name": "a11_s07_e03_v02", "length": 21, "label": 9},
                          {"file_name": "a04_s01_e06_v02", "length": 31, "label": 4},
                          {"file_name": "a06_s01_e01_v02", "length": 21, "label": 6},
                          {"file_name": "a12_s03_e02_v02", "length": 39, "label": 10},
                          {"file_name": "a08_s05_e02_v02", "length": 51, "label": 7},
                          {"file_name": "a03_s04_e00_v02", "length": 26, "label": 3},
                          {"file_name": "a11_s01_e03_v02", "length": 31, "label": 9},
                          {"file_name": "a03_s08_e01_v02", "length": 21, "label": 3},
                          {"file_name": "a11_s04_e00_v02", "length": 32, "label": 9},
                          {"file_name": "a04_s05_e00_v02", "length": 36, "label": 4},
                          {"file_name": "a12_s05_e01_v02", "length": 31, "label": 10},
                          {"file_name": "a02_s05_e02_v02", "length": 26, "label": 2},
                          {"file_name": "a06_s06_e01_v02", "length": 16, "label": 6},
                          {"file_name": "a03_s03_e02_v02", "length": 32, "label": 3},
                          {"file_name": "a11_s07_e02_v02", "length": 21, "label": 9},
                          {"file_name": "a11_s01_e02_v02", "length": 21, "label": 9}]
        outlist_skel = []

        def load_data():
            # data: N C V T M
            data = []
            for da in data_dict:
                file_name = da['file_name']
                outlist_skel.append(da["label"])
                with open(folder_path + '/' + file_name + '.json', 'r') as f:
                    json_file = json.load(f)
                skeletons = json_file['skeletons']
                value = np.array(skeletons)
                data.append(value)
            return data

        inplist_skel = load_data()
    return inplist_skel, outlist_skel



def create_descriptors(inpseq_stack, labels_stack, path_gcn, path_desc, path_disj, path_disf, path_ang, img_size):



    #print(inpseq_stack[1].shape)
    #(226,20,3)
    #print(labels_stack[0])
    #time.sleep(100)





    '''
    max_all= np.amax(inpseq_stack[0])

    min_all= np.amin(inpseq_stack[0])




    for i in range(1,len(inpseq_stack)):

        max_all= max(np.amax(inpseq_stack[i]),max_all)

        min_all= min(np.amin(inpseq_stack[i]),min_all)
    '''





    for i in range(0,len(inpseq_stack)):

        inpseq_stack[i][:, :, :] -= inpseq_stack[i][0, 1, :]




        gcn_desc=np.zeros((20,inpseq_stack[i].shape[0],3))

        inp_graph=np.zeros((20,3))


        for k in range(1,inpseq_stack[i].shape[0]):

            for j in range(0,20):


                inp_graph[j,0]=inpseq_stack[i][k,j,0]
                inp_graph[j,1]=inpseq_stack[i][k,j,1]
                inp_graph[j,2]=inpseq_stack[i][k,j,2]


            feat_gcn=graph(   torch.from_numpy(inp_graph).type('torch.FloatTensor')  ).detach().cpu().numpy()

            gcn_desc[:,k,0] = feat_gcn[0:20]
            gcn_desc[:,k,1] = feat_gcn[20:40]
            gcn_desc[:,k,2] = feat_gcn[40:60]

        

        
        gcn_desc[:,:,0] = np.floor(255*( gcn_desc[:,:,0]- np.amin(gcn_desc[:,:,0]) )/(np.amax(gcn_desc[:,:,0])-np.amin(gcn_desc[:,:,0])))
        gcn_desc[:,:,1] = np.floor(255*( gcn_desc[:,:,1]- np.amin(gcn_desc[:,:,1]) )/(np.amax(gcn_desc[:,:,1])-np.amin(gcn_desc[:,:,1])))
        gcn_desc[:,:,2] = np.floor(255*( gcn_desc[:,:,2]- np.amin(gcn_desc[:,:,2]) )/(np.amax(gcn_desc[:,:,2])-np.amin(gcn_desc[:,:,2])))


        img=Image.fromarray(gcn_desc.astype(np.uint8))
        img=img.resize((img_size,img_size))
        img.save(path_gcn+str(i)+'_'+str(labels_stack[i])+'.png')
 
        

      
     
      
        ### Joint distance  



        dis_xyz=np.zeros((32))
        distance_desc=np.zeros((32,inpseq_stack[i].shape[0]))

        glob_limbs_dist=0

        glob_limbs_dist+=Distance3D(inpseq_stack[i][0,0,0], inpseq_stack[i][0,0,1], inpseq_stack[i][0,0,2], inpseq_stack[i][0,1,0], inpseq_stack[i][0,1,1], inpseq_stack[i][0,1,2])
        glob_limbs_dist+=Distance3D(inpseq_stack[i][0,1,0], inpseq_stack[i][0,1,1], inpseq_stack[i][0,1,2], inpseq_stack[i][0,2,0], inpseq_stack[i][0,2,1], inpseq_stack[i][0,2,2])
        glob_limbs_dist+=Distance3D(inpseq_stack[i][0,2,0], inpseq_stack[i][0,2,1], inpseq_stack[i][0,2,2], inpseq_stack[i][0,3,0], inpseq_stack[i][0,3,1], inpseq_stack[i][0,3,2])
        glob_limbs_dist+=Distance3D(inpseq_stack[i][0,2,0], inpseq_stack[i][0,2,1], inpseq_stack[i][0,2,2], inpseq_stack[i][0,3,0], inpseq_stack[i][0,3,1], inpseq_stack[i][0,3,2])
        glob_limbs_dist+=Distance3D(inpseq_stack[i][0,0,0], inpseq_stack[i][0,0,1], inpseq_stack[i][0,0,2], inpseq_stack[i][0,12,0], inpseq_stack[i][0,12,1], inpseq_stack[i][0,12,2])
        glob_limbs_dist+=Distance3D(inpseq_stack[i][0,12,0], inpseq_stack[i][0,12,1], inpseq_stack[i][0,12,2], inpseq_stack[i][0,13,0], inpseq_stack[i][0,13,1], inpseq_stack[i][0,13,2])
        glob_limbs_dist+=Distance3D(inpseq_stack[i][0,13,0], inpseq_stack[i][0,13,1], inpseq_stack[i][0,13,2], inpseq_stack[i][0,14,0], inpseq_stack[i][0,14,1], inpseq_stack[i][0,14,2])
        glob_limbs_dist+=Distance3D(inpseq_stack[i][0,14,0], inpseq_stack[i][0,14,1], inpseq_stack[i][0,14,2], inpseq_stack[i][0,15,0], inpseq_stack[i][0,15,1], inpseq_stack[i][0,15,2])
        glob_limbs_dist+=Distance3D(inpseq_stack[i][0,0,0], inpseq_stack[i][0,0,1], inpseq_stack[i][0,0,2], inpseq_stack[i][0,16,0], inpseq_stack[i][0,16,1], inpseq_stack[i][0,16,2])
        glob_limbs_dist+=Distance3D(inpseq_stack[i][0,16,0], inpseq_stack[i][0,16,1], inpseq_stack[i][0,16,2], inpseq_stack[i][0,17,0], inpseq_stack[i][0,17,1], inpseq_stack[i][0,17,2])
        glob_limbs_dist+=Distance3D(inpseq_stack[i][0,17,0], inpseq_stack[i][0,17,1], inpseq_stack[i][0,17,2], inpseq_stack[i][0,18,0], inpseq_stack[i][0,18,1], inpseq_stack[i][0,18,2])
        glob_limbs_dist+=Distance3D(inpseq_stack[i][0,18,0], inpseq_stack[i][0,18,1], inpseq_stack[i][0,18,2], inpseq_stack[i][0,19,0], inpseq_stack[i][0,19,1], inpseq_stack[i][0,19,2])
        glob_limbs_dist+=Distance3D(inpseq_stack[i][0,2,0], inpseq_stack[i][0,2,1], inpseq_stack[i][0,2,2], inpseq_stack[i][0,8,0], inpseq_stack[i][0,8,1], inpseq_stack[i][0,8,2])
        glob_limbs_dist+=Distance3D(inpseq_stack[i][0,8,0], inpseq_stack[i][0,8,1], inpseq_stack[i][0,8,2], inpseq_stack[i][0,9,0], inpseq_stack[i][0,9,1], inpseq_stack[i][0,9,2])
        glob_limbs_dist+=Distance3D(inpseq_stack[i][0,9,0], inpseq_stack[i][0,9,1], inpseq_stack[i][0,9,2], inpseq_stack[i][0,10,0], inpseq_stack[i][0,10,1], inpseq_stack[i][0,10,2])
        glob_limbs_dist+=Distance3D(inpseq_stack[i][0,10,0], inpseq_stack[i][0,10,1], inpseq_stack[i][0,10,2], inpseq_stack[i][0,11,0], inpseq_stack[i][0,11,1], inpseq_stack[i][0,11,2])
        glob_limbs_dist+=Distance3D(inpseq_stack[i][0,2,0], inpseq_stack[i][0,2,1], inpseq_stack[i][0,2,2], inpseq_stack[i][0,4,0], inpseq_stack[i][0,4,1], inpseq_stack[i][0,4,2])
        glob_limbs_dist+=Distance3D(inpseq_stack[i][0,4,0], inpseq_stack[i][0,4,1], inpseq_stack[i][0,4,2], inpseq_stack[i][0,5,0], inpseq_stack[i][0,5,1], inpseq_stack[i][0,5,2])
        glob_limbs_dist+=Distance3D(inpseq_stack[i][0,5,0], inpseq_stack[i][0,5,1], inpseq_stack[i][0,5,2], inpseq_stack[i][0,6,0], inpseq_stack[i][0,6,1], inpseq_stack[i][0,6,2])
        glob_limbs_dist+=Distance3D(inpseq_stack[i][0,6,0], inpseq_stack[i][0,6,1], inpseq_stack[i][0,6,2], inpseq_stack[i][0,7,0], inpseq_stack[i][0,7,1], inpseq_stack[i][0,7,2])
      

        if glob_limbs_dist==0:
            glob_limbs_dist=1







        for k in range(0,inpseq_stack[i].shape[0]):

            dis_xyz[0] = Distance3D(inpseq_stack[i][k,5,0], inpseq_stack[i][k,5,1], inpseq_stack[i][k,5,2], inpseq_stack[i][k,9,0], inpseq_stack[i][k,9,1], inpseq_stack[i][k,9,2])/glob_limbs_dist
            dis_xyz[1] = Distance3D(inpseq_stack[i][k,6,0], inpseq_stack[i][k,6,1], inpseq_stack[i][k,6,2], inpseq_stack[i][k,10,0], inpseq_stack[i][k,10,1], inpseq_stack[i][k,10,2])/glob_limbs_dist
            dis_xyz[2] = Distance3D(inpseq_stack[i][k,13,0], inpseq_stack[i][k,13,1], inpseq_stack[i][k,13,2], inpseq_stack[i][k,17,0], inpseq_stack[i][k,17,1], inpseq_stack[i][k,17,2])/glob_limbs_dist
            dis_xyz[3] = Distance3D(inpseq_stack[i][k,14,0], inpseq_stack[i][k,14,1], inpseq_stack[i][k,14,2], inpseq_stack[i][k,18,0], inpseq_stack[i][k,18,1], inpseq_stack[i][k,18,2])/glob_limbs_dist
            dis_xyz[4] = Distance3D(inpseq_stack[i][k,10,0], inpseq_stack[i][k,10,1], inpseq_stack[i][k,10,2], inpseq_stack[i][k,16,0], inpseq_stack[i][k,16,1], inpseq_stack[i][k,16,2])/glob_limbs_dist
            dis_xyz[5] = Distance3D(inpseq_stack[i][k,10,0], inpseq_stack[i][k,10,1], inpseq_stack[i][k,10,2], inpseq_stack[i][k,17,0], inpseq_stack[i][k,17,1], inpseq_stack[i][k,17,2])/glob_limbs_dist
            dis_xyz[6] = Distance3D(inpseq_stack[i][k,10,0], inpseq_stack[i][k,10,1], inpseq_stack[i][k,10,2], inpseq_stack[i][k,18,0], inpseq_stack[i][k,18,1], inpseq_stack[i][k,18,2])/glob_limbs_dist
            dis_xyz[7] = Distance3D(inpseq_stack[i][k,6,0], inpseq_stack[i][k,6,1], inpseq_stack[i][k,6,2], inpseq_stack[i][k,12,0], inpseq_stack[i][k,12,1], inpseq_stack[i][k,12,2])/glob_limbs_dist
            dis_xyz[8] = Distance3D(inpseq_stack[i][k,6,0], inpseq_stack[i][k,6,1], inpseq_stack[i][k,6,2], inpseq_stack[i][k,13,0], inpseq_stack[i][k,13,1], inpseq_stack[i][k,13,2])/glob_limbs_dist
            dis_xyz[9] = Distance3D(inpseq_stack[i][k,6,0], inpseq_stack[i][k,6,1], inpseq_stack[i][k,6,2], inpseq_stack[i][k,14,0], inpseq_stack[i][k,14,1], inpseq_stack[i][k,14,2])/glob_limbs_dist
            dis_xyz[10] = Distance3D(inpseq_stack[i][k,3,0], inpseq_stack[i][k,3,1], inpseq_stack[i][k,3,2], inpseq_stack[i][k,18,0], inpseq_stack[i][k,18,1], inpseq_stack[i][k,18,2])/glob_limbs_dist
            dis_xyz[11] = Distance3D(inpseq_stack[i][k,3,0], inpseq_stack[i][k,3,1], inpseq_stack[i][k,3,2], inpseq_stack[i][k,14,0], inpseq_stack[i][k,14,1], inpseq_stack[i][k,14,2])/glob_limbs_dist
            dis_xyz[12] = Distance3D(inpseq_stack[i][k,0,0], inpseq_stack[i][k,0,1], inpseq_stack[i][k,0,2], inpseq_stack[i][k,13,0], inpseq_stack[i][k,13,1], inpseq_stack[i][k,13,2])/glob_limbs_dist
            dis_xyz[13] = Distance3D(inpseq_stack[i][k,12,0], inpseq_stack[i][k,12,1], inpseq_stack[i][k,12,2], inpseq_stack[i][k,14,0], inpseq_stack[i][k,14,1], inpseq_stack[i][k,14,2])/glob_limbs_dist
            dis_xyz[14] = Distance3D(inpseq_stack[i][k,0,0], inpseq_stack[i][k,0,1], inpseq_stack[i][k,0,2], inpseq_stack[i][k,17,0], inpseq_stack[i][k,17,1], inpseq_stack[i][k,17,2])/glob_limbs_dist
            dis_xyz[15] = Distance3D(inpseq_stack[i][k,16,0], inpseq_stack[i][k,16,1], inpseq_stack[i][k,16,2], inpseq_stack[i][k,18,0], inpseq_stack[i][k,18,1], inpseq_stack[i][k,18,2])/glob_limbs_dist
            dis_xyz[16] = Distance3D(inpseq_stack[i][k,10,0], inpseq_stack[i][k,10,1], inpseq_stack[i][k,10,2], inpseq_stack[i][k,8,0], inpseq_stack[i][k,8,1], inpseq_stack[i][k,8,2])/glob_limbs_dist
            dis_xyz[17] = Distance3D(inpseq_stack[i][k,9,0], inpseq_stack[i][k,9,1], inpseq_stack[i][k,9,2], inpseq_stack[i][k,2,0], inpseq_stack[i][k,2,1], inpseq_stack[i][k,2,2])/glob_limbs_dist
            dis_xyz[18] = Distance3D(inpseq_stack[i][k,8,0], inpseq_stack[i][k,8,1], inpseq_stack[i][k,8,2], inpseq_stack[i][k,3,0], inpseq_stack[i][k,3,1], inpseq_stack[i][k,3,2])/glob_limbs_dist
            dis_xyz[19] = Distance3D(inpseq_stack[i][k,3,0], inpseq_stack[i][k,3,1], inpseq_stack[i][k,3,2], inpseq_stack[i][k,4,0], inpseq_stack[i][k,4,1], inpseq_stack[i][k,4,2])/glob_limbs_dist
            dis_xyz[20] = Distance3D(inpseq_stack[i][k,2,0], inpseq_stack[i][k,2,1], inpseq_stack[i][k,2,2], inpseq_stack[i][k,5,0], inpseq_stack[i][k,5,1], inpseq_stack[i][k,5,2])/glob_limbs_dist
            dis_xyz[21] = Distance3D(inpseq_stack[i][k,4,0], inpseq_stack[i][k,4,1], inpseq_stack[i][k,4,2], inpseq_stack[i][k,6,0], inpseq_stack[i][k,6,1], inpseq_stack[i][k,6,2])/glob_limbs_dist
            dis_xyz[22] = Distance3D(inpseq_stack[i][k,17,0], inpseq_stack[i][k,17,1], inpseq_stack[i][k,17,2], inpseq_stack[i][k,14,0], inpseq_stack[i][k,14,1], inpseq_stack[i][k,14,2])/glob_limbs_dist
            dis_xyz[23] = Distance3D(inpseq_stack[i][k,13,0], inpseq_stack[i][k,13,1], inpseq_stack[i][k,13,2], inpseq_stack[i][k,18,0], inpseq_stack[i][k,18,1], inpseq_stack[i][k,18,2])/glob_limbs_dist
            dis_xyz[24] = Distance3D(inpseq_stack[i][k,9,0], inpseq_stack[i][k,9,1], inpseq_stack[i][k,9,2], inpseq_stack[i][k,6,0], inpseq_stack[i][k,6,1], inpseq_stack[i][k,6,2])/glob_limbs_dist
            dis_xyz[25] = Distance3D(inpseq_stack[i][k,5,0], inpseq_stack[i][k,5,1], inpseq_stack[i][k,5,2], inpseq_stack[i][k,10,0], inpseq_stack[i][k,10,1], inpseq_stack[i][k,10,2])
            dis_xyz[26] = Distance3D(inpseq_stack[i][k,3,0], inpseq_stack[i][k,3,1], inpseq_stack[i][k,3,2], inpseq_stack[i][k,13,0], inpseq_stack[i][k,13,1], inpseq_stack[i][k,13,2])/glob_limbs_dist
            dis_xyz[27] = Distance3D(inpseq_stack[i][k,3,0], inpseq_stack[i][k,3,1], inpseq_stack[i][k,3,2], inpseq_stack[i][k,6,0], inpseq_stack[i][k,6,1], inpseq_stack[i][k,6,2])/glob_limbs_dist
            dis_xyz[28] = Distance3D(inpseq_stack[i][k,3,0], inpseq_stack[i][k,3,1], inpseq_stack[i][k,3,2], inpseq_stack[i][k,5,0], inpseq_stack[i][k,5,1], inpseq_stack[i][k,5,2])/glob_limbs_dist
            dis_xyz[29] = Distance3D(inpseq_stack[i][k,3,0], inpseq_stack[i][k,3,1], inpseq_stack[i][k,3,2], inpseq_stack[i][k,17,0], inpseq_stack[i][k,17,1], inpseq_stack[i][k,17,2])/glob_limbs_dist
            dis_xyz[30] = Distance3D(inpseq_stack[i][k,3,0], inpseq_stack[i][k,3,1], inpseq_stack[i][k,3,2], inpseq_stack[i][k,10,0], inpseq_stack[i][k,10,1], inpseq_stack[i][k,10,2])/glob_limbs_dist
            dis_xyz[31] = Distance3D(inpseq_stack[i][k,3,0], inpseq_stack[i][k,3,1], inpseq_stack[i][k,3,2], inpseq_stack[i][k,9,0], inpseq_stack[i][k,9,1], inpseq_stack[i][k,9,2])/glob_limbs_dist


            distance_desc[:,k] = dis_xyz[:]

        
        


        distance_desc[:,:] = np.floor(255*( distance_desc[:,:]- np.amin(distance_desc[:,:]) )/(np.amax(distance_desc[:,:])-np.amin(distance_desc[:,:])))


        
        img=Image.fromarray(distance_desc.astype(np.uint8),'L')
        img=img.resize((img_size,img_size))
        img1=img
        img.save(path_disj+str(i)+'_'+str(labels_stack[i])+'.png')
   




        ### Joint distance frames 


        dis_frms=np.zeros((20))
        distance_desc_frms=np.zeros((20,inpseq_stack[i].shape[0]-1))


        for k in range(0,inpseq_stack[i].shape[0]-1):



            dis_frms[0] = Distance3D(inpseq_stack[i][k,0,0], inpseq_stack[i][k,0,1], inpseq_stack[i][k,0,2], inpseq_stack[i][k+1,0,0], inpseq_stack[i][k+1,0,1], inpseq_stack[i][k+1,0,2])
            dis_frms[1] = Distance3D(inpseq_stack[i][k,1,0], inpseq_stack[i][k,1,1], inpseq_stack[i][k,1,2], inpseq_stack[i][k+1,1,0], inpseq_stack[i][k+1,1,1], inpseq_stack[i][k+1,1,2])
            dis_frms[2] = Distance3D(inpseq_stack[i][k,2,0], inpseq_stack[i][k,2,1], inpseq_stack[i][k,2,2], inpseq_stack[i][k+1,2,0], inpseq_stack[i][k+1,2,1], inpseq_stack[i][k+1,2,2])
            dis_frms[3] = Distance3D(inpseq_stack[i][k,3,0], inpseq_stack[i][k,3,1], inpseq_stack[i][k,3,2], inpseq_stack[i][k+1,3,0], inpseq_stack[i][k+1,3,1], inpseq_stack[i][k+1,3,2])
            dis_frms[4] = Distance3D(inpseq_stack[i][k,4,0], inpseq_stack[i][k,4,1], inpseq_stack[i][k,4,2], inpseq_stack[i][k+1,4,0], inpseq_stack[i][k+1,4,1], inpseq_stack[i][k+1,4,2])
            dis_frms[5] = Distance3D(inpseq_stack[i][k,5,0], inpseq_stack[i][k,5,1], inpseq_stack[i][k,5,2], inpseq_stack[i][k+1,5,0], inpseq_stack[i][k+1,5,1], inpseq_stack[i][k+1,5,2])
            dis_frms[6] = Distance3D(inpseq_stack[i][k,6,0], inpseq_stack[i][k,6,1], inpseq_stack[i][k,6,2], inpseq_stack[i][k+1,6,0], inpseq_stack[i][k+1,6,1], inpseq_stack[i][k+1,6,2])
            dis_frms[7] = Distance3D(inpseq_stack[i][k,7,0], inpseq_stack[i][k,7,1], inpseq_stack[i][k,7,2], inpseq_stack[i][k+1,7,0], inpseq_stack[i][k+1,7,1], inpseq_stack[i][k+1,7,2])
            dis_frms[8] = Distance3D(inpseq_stack[i][k,8,0], inpseq_stack[i][k,8,1], inpseq_stack[i][k,8,2], inpseq_stack[i][k+1,8,0], inpseq_stack[i][k+1,8,1], inpseq_stack[i][k+1,8,2])
            dis_frms[9] = Distance3D(inpseq_stack[i][k,9,0], inpseq_stack[i][k,9,1], inpseq_stack[i][k,9,2], inpseq_stack[i][k+1,9,0], inpseq_stack[i][k+1,9,1], inpseq_stack[i][k+1,9,2])
            dis_frms[10] = Distance3D(inpseq_stack[i][k,10,0], inpseq_stack[i][k,10,1], inpseq_stack[i][k,10,2], inpseq_stack[i][k+1,10,0], inpseq_stack[i][k+1,10,1], inpseq_stack[i][k+1,10,2])
            dis_frms[11] = Distance3D(inpseq_stack[i][k,11,0], inpseq_stack[i][k,11,1], inpseq_stack[i][k,11,2], inpseq_stack[i][k+1,11,0], inpseq_stack[i][k+1,11,1], inpseq_stack[i][k+1,11,2])
            dis_frms[12] = Distance3D(inpseq_stack[i][k,12,0], inpseq_stack[i][k,12,1], inpseq_stack[i][k,12,2], inpseq_stack[i][k+1,12,0], inpseq_stack[i][k+1,12,1], inpseq_stack[i][k+1,12,2])
            dis_frms[13] = Distance3D(inpseq_stack[i][k,13,0], inpseq_stack[i][k,13,1], inpseq_stack[i][k,13,2], inpseq_stack[i][k+1,13,0], inpseq_stack[i][k+1,13,1], inpseq_stack[i][k+1,13,2])
            dis_frms[14] = Distance3D(inpseq_stack[i][k,14,0], inpseq_stack[i][k,14,1], inpseq_stack[i][k,14,2], inpseq_stack[i][k+1,14,0], inpseq_stack[i][k+1,14,1], inpseq_stack[i][k+1,14,2])
            dis_frms[15] = Distance3D(inpseq_stack[i][k,15,0], inpseq_stack[i][k,15,1], inpseq_stack[i][k,15,2], inpseq_stack[i][k+1,15,0], inpseq_stack[i][k+1,15,1], inpseq_stack[i][k+1,15,2])
            dis_frms[16] = Distance3D(inpseq_stack[i][k,16,0], inpseq_stack[i][k,16,1], inpseq_stack[i][k,16,2], inpseq_stack[i][k+1,16,0], inpseq_stack[i][k+1,16,1], inpseq_stack[i][k+1,16,2])
            dis_frms[17] = Distance3D(inpseq_stack[i][k,17,0], inpseq_stack[i][k,17,1], inpseq_stack[i][k,17,2], inpseq_stack[i][k+1,17,0], inpseq_stack[i][k+1,17,1], inpseq_stack[i][k+1,17,2])
            dis_frms[18] = Distance3D(inpseq_stack[i][k,18,0], inpseq_stack[i][k,18,1], inpseq_stack[i][k,18,2], inpseq_stack[i][k+1,18,0], inpseq_stack[i][k+1,18,1], inpseq_stack[i][k+1,18,2])
            dis_frms[19] = Distance3D(inpseq_stack[i][k,19,0], inpseq_stack[i][k,19,1], inpseq_stack[i][k,19,2], inpseq_stack[i][k+1,19,0], inpseq_stack[i][k+1,19,1], inpseq_stack[i][k+1,19,2])
            

            distance_desc_frms[:,k] = dis_frms[:]

        

        if (distance_desc_frms.shape[1]==0):

            distance_desc_frms=np.zeros((20,inpseq_stack[i].shape[0]-1))

        else:
           
            distance_desc_frms[:,:] = np.floor(255*( distance_desc_frms[:,:]- np.amin(distance_desc_frms[:,:]) )/(np.amax(distance_desc_frms[:,:])-np.amin(distance_desc_frms[:,:])))

            
        img=Image.fromarray(distance_desc_frms.astype(np.uint8),'L')
        img=img.resize((img_size,img_size))
        img2=img
        img.save(path_disf+str(i)+'_'+str(labels_stack[i])+'.png')
        


        ###  Angles  

        ang=np.zeros((14))
        angle_desc=np.zeros((14,inpseq_stack[i].shape[0]))

        for k in range(0,inpseq_stack[i].shape[0]):

            ang[0] = angle_3D(inpseq_stack[i][k,12,:],inpseq_stack[i][k,13,:], inpseq_stack[i][k,0,:])
            ang[1] = angle_3D(inpseq_stack[i][k,13,:],inpseq_stack[i][k,14,:], inpseq_stack[i][k,12,:])
            ang[2] = angle_3D(inpseq_stack[i][k,16,:],inpseq_stack[i][k,0,:], inpseq_stack[i][k,17,:])
            ang[3] = angle_3D(inpseq_stack[i][k,17,:],inpseq_stack[i][k,16,:], inpseq_stack[i][k,18,:])
            ang[4] = angle_3D(inpseq_stack[i][k,9,:],inpseq_stack[i][k,8,:], inpseq_stack[i][k,10,:])
            ang[5] = angle_3D(inpseq_stack[i][k,8,:],inpseq_stack[i][k,2,:], inpseq_stack[i][k,9,:])
            ang[6] = angle_3D(inpseq_stack[i][k,2,:],inpseq_stack[i][k,8,:], inpseq_stack[i][k,3,:])
            ang[7] = angle_3D(inpseq_stack[i][k,2,:],inpseq_stack[i][k,3,:], inpseq_stack[i][k,4,:]) 
            ang[8] = angle_3D(inpseq_stack[i][k,4,:],inpseq_stack[i][k,5,:], inpseq_stack[i][k,2,:])
            ang[9] = angle_3D(inpseq_stack[i][k,5,:],inpseq_stack[i][k,6,:], inpseq_stack[i][k,4,:])
            ang[10] = angle_3D(inpseq_stack[i][k,18,:],inpseq_stack[i][k,19,:], inpseq_stack[i][k,17,:])
            ang[11] = angle_3D(inpseq_stack[i][k,14,:],inpseq_stack[i][k,13,:], inpseq_stack[i][k,15,:])
            ang[12] = angle_3D(inpseq_stack[i][k,10,:],inpseq_stack[i][k,9,:], inpseq_stack[i][k,11,:])
            ang[13] = angle_3D(inpseq_stack[i][k,6,:],inpseq_stack[i][k,5,:], inpseq_stack[i][k,7,:])

            angle_desc[:,k] = ang[:]

  
        angle_desc[:,:] = np.floor(255*( angle_desc[:,:]- np.amin(angle_desc[:,:]) )/(np.amax(angle_desc[:,:])-np.amin(angle_desc[:,:])))

        
        img=Image.fromarray(angle_desc.astype(np.uint8),'L')
        img=img.resize((img_size,img_size))
        img3=img
        img.save(path_ang+str(i)+'_'+str(labels_stack[i])+'.png')

        

        glob_desc=Image.merge("RGB", (img1,img2,img3))

        glob_desc.save(path_desc+str(i)+'_'+str(labels_stack[i])+'.png')

        


        
    return 0







if __name__ == "__main__":



    inplist_skel, outlist_skel= data_UCLA_skel(mod=args.action, ds_path=args.dataset_path)

    path_disj=   args.descrip_path+ 'ucla_JDD_'+ args.action+'/'
    path_disf=   args.descrip_path+ 'ucla_ADD_'+ args.action+'/'
    path_ang=    args.descrip_path+ 'ucla_LAD_'+ args.action+'/'
    path_fuse=   args.descrip_path+ 'ucla_FDD_'+ args.action+'/'
    path_gcn=    args.descrip_path+ 'ucla_GD_'+ args.action+'/'

    img_size=112

    create_descriptors(inplist_skel, outlist_skel, path_gcn,path_fuse,path_disj,path_disf,path_ang,img_size)




