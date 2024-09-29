

import numpy as np
import os
import glob
import sys 
import time
from matplotlib import pyplot as plt
import cv2
import torch.nn as nn
import math 
import scipy.sparse as sp

from natsort import natsorted

from einops import rearrange, repeat

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from arguments import args

# data.keys()      dict_keys(['file_name', 'nbodys', 'njoints', 'skel_body0', 'rgb_body0', 'depth_body0', 'skel_body1', 'rgb_body1', 'depth_body1'])

# SsssCcccPpppRrrrAaaa,  sss is the setup number, ccc is the camera ID, ppp is the performer (subject) ID, rrr is the replication number (1 or 2), and aaa is the action class label.

# ---------------  nturgb 60 
# cross-subject evaluation:  subject 1 2 4 5 8 9 13 14 15 16 17 18 19 25 27 28 31 34 35 38 (training),  the rest for testing
# cross-view evaluation: camera 2, 3 for training. camera 1 for testing

# ---------------  nturgb 120 
# cross-subject evaluation: 1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35,38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78,
# 80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103. (106, 53 previous for training, the rest 53 for testing)
# cross-setup evaluation: even setup IDs for training, odd for testing



def data_ntu_skel(mod,dataset,eval_protocol,npy_path):

    # ntu60, ntu120

    ntu60train_list=[1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
    ntu120train_list=[1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35,38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78, 80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103 ]
    inplist_skel=[]
    outlist_skel=[]


    if dataset=='ntu120':

        datalist120 = os.listdir(npy_path)

        datalist120.sort()

        if eval_protocol=='csub':

            for file in datalist120:

                data = np.load(npy_path+file,allow_pickle=True).item()

                if mod=='train':

                    if int(file[9:12]) in ntu120train_list:

                        if 'skel_body1' in data:

                            skelseq_all=np.zeros((len(data['skel_body0']),50,3))
                            for k in range(0,len(data['skel_body0'])):

                                skelseq_all[k,:,:]=np.concatenate((data['skel_body0'][k],data['skel_body1'][k]),axis=0)
                        else:

                            skelseq_all=np.zeros((len(data['skel_body0']),25,3))

                            for k in range(0,len(data['skel_body0'])):
                                skelseq_all[k,:,:]=data['skel_body0'][k]
                        # zero center
                        skelseq_all[:, :, :] -= skelseq_all[0, 1, :]
  
                        inplist_skel.append(skelseq_all)
                        outlist_skel.append(int(file[17:20]))
                if mod=='test':

                    if int(file[9:12]) not in ntu120train_list:

                        # remove joints
                        # based on the paper dataset (remove strating counting frpm 0)
                        # remove:  15: left foot,  19:right foot,  2: neck,  7: left hand, 21:tip of the left hand, 22: left thumb,  11: right hand, 23:tip of the right hand , 24: right thumb
                    
                        if 'skel_body1' in data:
                            skelseq_all=np.zeros((len(data['skel_body0']),50,3))
                
                            for k in range(0,len(data['skel_body0'])):
                                

                                skelseq_all[k,:,:]=np.concatenate((data['skel_body0'][k],data['skel_body1'][k]),axis=0)
                          
                        
                        else:
                        
                     
                            skelseq_all=np.zeros((len(data['skel_body0']),25,3))

                            for k in range(0,len(data['skel_body0'])):
                        
                                skelseq_all[k,:,:]=data['skel_body0'][k]

   
                        # zero center

                        skelseq_all[:, :, :] -= skelseq_all[0, 1, :]
                 


                        inplist_skel.append(skelseq_all)
                        outlist_skel.append(int(file[17:20]))


        if eval_protocol=='csetup':
        
            for file in datalist120:

                data = np.load(npy_path+file,allow_pickle=True).item()

                if mod=='train':

                
                    if (int(file[1:4]) % 2) ==0:

                        if 'skel_body1' in data:

                            skelseq_all=np.zeros((len(data['skel_body0']),50,3))
                 
                            for k in range(0,len(data['skel_body0'])):
                                


                                skelseq_all[k,:,:]=np.concatenate((data['skel_body0'][k],data['skel_body1'][k]),axis=0)
                          
                         
                        else:
                    

                                skelseq_all=np.zeros((len(data['skel_body0']),25,3))

                                for k in range(0,len(data['skel_body0'])):

                            
                                    skelseq_all[k,:,:]=data['skel_body0'][k]

                        # zero center

                        skelseq_all[:, :, :] -= skelseq_all[0, 1, :]
     

                        inplist_skel.append(skelseq_all)
                        outlist_skel.append(int(file[17:20]))


                if mod=='test':

                
                    if (int(file[1:4]) % 2) !=0:

                        # remove joints
                        # based on the paper dataset (remove strating counting frpm 0)
                        # remove:  15: left foot,  19:right foot,  2: neck,  7: left hand, 21:tip of the left hand, 22: left thumb,  11: right hand, 23:tip of the right hand , 24: right thumb

                    
                        if 'skel_body1' in data:

                            skelseq_all=np.zeros((len(data['skel_body0']),50,3))
                 
                            for k in range(0,len(data['skel_body0'])):
                                
                                skelseq_all[k,:,:]=np.concatenate((data['skel_body0'][k],data['skel_body1'][k]),axis=0)
                          
                         
                        else:
                    

                                skelseq_all=np.zeros((len(data['skel_body0']),25,3))

                                for k in range(0,len(data['skel_body0'])):

                                    skelseq_all[k,:,:]=data['skel_body0'][k]
                
                        # zero center

                        skelseq_all[:, :, :] -= skelseq_all[0, 1, :]

                        inplist_skel.append(skelseq_all)
                        outlist_skel.append(int(file[17:20]))
                    

    if dataset=='ntu60':

        datalist60 = os.listdir(npy_path)

        datalist60.sort()

        if eval_protocol=='csub':
        
            for file in datalist60:

                data = np.load(npy_path+file,allow_pickle=True).item()

                if mod=='train':

                    if int(file[9:12]) in ntu60train_list:

                        if 'skel_body1' in data:

                            skelseq_all=np.zeros((len(data['skel_body0']),50,3))
                 
                            for k in range(0,len(data['skel_body0'])):

                                skelseq_all[k,:,:]=np.concatenate((data['skel_body0'][k],data['skel_body1'][k]),axis=0)
                          
                        else:
                    
                            skelseq_all=np.zeros((len(data['skel_body0']),25,3))

                            for k in range(0,len(data['skel_body0'])):

                        
                                skelseq_all[k,:,:]=data['skel_body0'][k]

                        # zero center

                        skelseq_all[:, :, :] -= skelseq_all[0, 1, :]

                        inplist_skel.append(skelseq_all)
                        outlist_skel.append(int(file[17:20]))


                if mod=='test':

                
                    if int(file[9:12]) not in ntu60train_list:

                        if ('skel_body1' in data) :

                            #print(int(file[17:20]))

                            skelseq_all=np.zeros((len(data['skel_body0']),50,3))

                 
                            for k in range(0,len(data['skel_body0'])):

                                skelseq_all[k,:,:]=np.concatenate((data['skel_body0'][k],data['skel_body1'][k]),axis=0)
                          
                        else:
                        
                     
                            skelseq_all=np.zeros((len(data['skel_body0']),25,3))

                            for k in range(0,len(data['skel_body0'])):

                                skelseq_all[k,:,:]=data['skel_body0'][k]

                        # zero center

                        skelseq_all[:, :, :] -= skelseq_all[0, 1, :]

                        inplist_skel.append(skelseq_all)
                        outlist_skel.append(int(file[17:20]))


        if eval_protocol=='cview':
        
            for file in datalist60:

                data = np.load(npy_path+file,allow_pickle=True).item()

                if mod=='train':

                
                    if ((int(file[5:8])  == 2) or  (int(file[5:8])  == 3)):

                        if 'skel_body1' in data:

                            skelseq_all=np.zeros((len(data['skel_body0']),50,3))
                 
                            for k in range(0,len(data['skel_body0'])):

                                skelseq_all[k,:,:]=np.concatenate((data['skel_body0'][k],data['skel_body1'][k]),axis=0)
                          
                        else:

                                skelseq_all=np.zeros((len(data['skel_body0']),25,3))

                                for k in range(0,len(data['skel_body0'])):

                                    skelseq_all[k,:,:]=data['skel_body0'][k]

                        # zero center

                        skelseq_all[:, :, :] -= skelseq_all[0, 1, :]

                        inplist_skel.append(skelseq_all)
                        outlist_skel.append(int(file[17:20]))


                if mod=='test':

                
                    if (int(file[5:8])  == 1):

                      
                        if 'skel_body1' in data:

                            skelseq_all=np.zeros((len(data['skel_body0']),50,3))
                 
                            for k in range(0,len(data['skel_body0'])):

                                skelseq_all[k,:,:]=np.concatenate((data['skel_body0'][k],data['skel_body1'][k]),axis=0)
                          
                        else:

                                skelseq_all=np.zeros((len(data['skel_body0']),25,3))

                                for k in range(0,len(data['skel_body0'])):

                                    skelseq_all[k,:,:]=data['skel_body0'][k]

                        # zero center

                        skelseq_all[:, :, :] -= skelseq_all[0, 1, :]
     

                        inplist_skel.append(skelseq_all)
                        outlist_skel.append(int(file[17:20]))

    return inplist_skel, outlist_skel



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


    edges = torch.tensor([[0, 1], [1, 20], [20, 2], [2, 3],
                        [0, 12], [12, 13], [13, 14], [14, 15],
                        [0, 16], [16, 17], [17, 18], [18, 19],
                        [20, 4], [4, 5], [5, 6], [6, 7],[7,21],[7,22],
                        [20, 8], [8, 9], [9, 10], [10, 11], [11,23], [11,24]], dtype=torch.long)

    
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

    adj=adj_mx_from_edges(25)

    kernel = adj.size(0)
   

    x=torch.unsqueeze(x,0)
    x=torch.unsqueeze(x,0)

    x=x.transpose(1,3)
    x=x.transpose(2,3)

    #x = nn.Conv2d(in_channels, out_channels , kernel_size=(1, 1))(x)

    x=torch.squeeze(x)

    x = torch.matmul(x, adj)

    return torch.flatten(x)


def create_descriptors(inpseq_stack, labels_stack, path_gcn, path_fuse, path_disj, path_disf, path_ang, img_size):

    max_all= np.amax(inpseq_stack[0])
    min_all= np.amin(inpseq_stack[0])

    for i in range(1,len(inpseq_stack)):

        max_all= max(np.amax(inpseq_stack[i]),max_all)
        min_all= min(np.amin(inpseq_stack[i]),min_all)

    for i in range(0,len(inpseq_stack)):


        ###  gcn
 
        if inpseq_stack[i].shape[1]==25: 
            gcn_desc=np.zeros((25,inpseq_stack[i].shape[0],3))
            inp_graph=np.zeros((25,3))


            for k in range(1,inpseq_stack[i].shape[0]):
                for j in range(0,25):
                    inp_graph[j,0]=inpseq_stack[i][k,j,0]
                    inp_graph[j,1]=inpseq_stack[i][k,j,1]
                    inp_graph[j,2]=inpseq_stack[i][k,j,2]
                feat_gcn=graph(   torch.from_numpy(inp_graph).type('torch.FloatTensor')  ).detach().cpu().numpy()
                gcn_desc[:,k,0] = feat_gcn[0:25]
                gcn_desc[:,k,1] = feat_gcn[25:50]
                gcn_desc[:,k,2] = feat_gcn[50:75]
        else:
            gcn_desc=np.zeros((50,inpseq_stack[i].shape[0],3))
            inp_graph1=np.zeros((25,3))
            inp_graph2=np.zeros((25,3))

            for k in range(1,inpseq_stack[i].shape[0]):
                for j in range(0,25):
                    inp_graph1[j,0]=inpseq_stack[i][k,j,0]
                    inp_graph1[j,1]=inpseq_stack[i][k,j,1]
                    inp_graph1[j,2]=inpseq_stack[i][k,j,2]
                for j in range(0,25):
                    inp_graph2[j,0]=inpseq_stack[i][k,j+25,0]
                    inp_graph2[j,1]=inpseq_stack[i][k,j+25,1]
                    inp_graph2[j,2]=inpseq_stack[i][k,j+25,2]


                feat_gcn_body1=graph( torch.from_numpy(inp_graph1).type('torch.FloatTensor') ).detach().cpu().numpy()
                feat_gcn_body2=graph( torch.from_numpy(inp_graph2).type('torch.FloatTensor') ).detach().cpu().numpy()
                feat_gcn=np.concatenate((feat_gcn_body1,feat_gcn_body2))

                gcn_desc[:,k,0] = np.concatenate((feat_gcn[0:25],feat_gcn[75:100]))
                gcn_desc[:,k,1] = np.concatenate((feat_gcn[25:50],feat_gcn[100:125]))
                gcn_desc[:,k,2] = np.concatenate((feat_gcn[50:75],feat_gcn[125:150]))


        gcn_desc[:,:,0] = np.floor(255*( gcn_desc[:,:,0]- np.amin(gcn_desc[:,:,0]) )/(np.amax(gcn_desc[:,:,0])-np.amin(gcn_desc[:,:,0])))
        gcn_desc[:,:,1] = np.floor(255*( gcn_desc[:,:,1]- np.amin(gcn_desc[:,:,1]) )/(np.amax(gcn_desc[:,:,1])-np.amin(gcn_desc[:,:,1])))
        gcn_desc[:,:,2] = np.floor(255*( gcn_desc[:,:,2]- np.amin(gcn_desc[:,:,2]) )/(np.amax(gcn_desc[:,:,2])-np.amin(gcn_desc[:,:,2])))

        img=Image.fromarray(gcn_desc.astype(np.uint8))
        img=img.resize((img_size,img_size))
        img.save(path_gcn+str(i)+'_'+str(labels_stack[i])+'.png')


        ### Joint distance  

        if inpseq_stack[i].shape[1]==25: 
            dis_xyz=np.zeros((32))
            distance_desc=np.zeros((32,inpseq_stack[i].shape[0]))

            glob_limbs_dist=0

            glob_limbs_dist+=Distance3D(inpseq_stack[i][0,0,0], inpseq_stack[i][0,0,1], inpseq_stack[i][0,0,2], inpseq_stack[i][0,1,0], inpseq_stack[i][0,1,1], inpseq_stack[i][0,1,2])
            glob_limbs_dist+=Distance3D(inpseq_stack[i][0,1,0], inpseq_stack[i][0,1,1], inpseq_stack[i][0,1,2], inpseq_stack[i][0,20,0], inpseq_stack[i][0,20,1], inpseq_stack[i][0,20,2])
            glob_limbs_dist+=Distance3D(inpseq_stack[i][0,20,0], inpseq_stack[i][0,20,1], inpseq_stack[i][0,20,2], inpseq_stack[i][0,2,0], inpseq_stack[i][0,2,1], inpseq_stack[i][0,2,2])
            glob_limbs_dist+=Distance3D(inpseq_stack[i][0,2,0], inpseq_stack[i][0,2,1], inpseq_stack[i][0,2,2], inpseq_stack[i][0,3,0], inpseq_stack[i][0,3,1], inpseq_stack[i][0,3,2])
            glob_limbs_dist+=Distance3D(inpseq_stack[i][0,0,0], inpseq_stack[i][0,0,1], inpseq_stack[i][0,0,2], inpseq_stack[i][0,12,0], inpseq_stack[i][0,12,1], inpseq_stack[i][0,12,2])
            glob_limbs_dist+=Distance3D(inpseq_stack[i][0,12,0], inpseq_stack[i][0,12,1], inpseq_stack[i][0,12,2], inpseq_stack[i][0,13,0], inpseq_stack[i][0,13,1], inpseq_stack[i][0,13,2])
            glob_limbs_dist+=Distance3D(inpseq_stack[i][0,13,0], inpseq_stack[i][0,13,1], inpseq_stack[i][0,13,2], inpseq_stack[i][0,14,0], inpseq_stack[i][0,14,1], inpseq_stack[i][0,14,2])
            glob_limbs_dist+=Distance3D(inpseq_stack[i][0,14,0], inpseq_stack[i][0,14,1], inpseq_stack[i][0,14,2], inpseq_stack[i][0,15,0], inpseq_stack[i][0,15,1], inpseq_stack[i][0,15,2])
            glob_limbs_dist+=Distance3D(inpseq_stack[i][0,0,0], inpseq_stack[i][0,0,1], inpseq_stack[i][0,0,2], inpseq_stack[i][0,16,0], inpseq_stack[i][0,16,1], inpseq_stack[i][0,16,2])
            glob_limbs_dist+=Distance3D(inpseq_stack[i][0,16,0], inpseq_stack[i][0,16,1], inpseq_stack[i][0,16,2], inpseq_stack[i][0,17,0], inpseq_stack[i][0,17,1], inpseq_stack[i][0,17,2])
            glob_limbs_dist+=Distance3D(inpseq_stack[i][0,17,0], inpseq_stack[i][0,17,1], inpseq_stack[i][0,17,2], inpseq_stack[i][0,18,0], inpseq_stack[i][0,18,1], inpseq_stack[i][0,18,2])
            glob_limbs_dist+=Distance3D(inpseq_stack[i][0,18,0], inpseq_stack[i][0,18,1], inpseq_stack[i][0,18,2], inpseq_stack[i][0,19,0], inpseq_stack[i][0,19,1], inpseq_stack[i][0,19,2])
            glob_limbs_dist+=Distance3D(inpseq_stack[i][0,20,0], inpseq_stack[i][0,20,1], inpseq_stack[i][0,20,2], inpseq_stack[i][0,4,0], inpseq_stack[i][0,4,1], inpseq_stack[i][0,4,2])
            glob_limbs_dist+=Distance3D(inpseq_stack[i][0,4,0], inpseq_stack[i][0,4,1], inpseq_stack[i][0,4,2], inpseq_stack[i][0,5,0], inpseq_stack[i][0,5,1], inpseq_stack[i][0,5,2])
            glob_limbs_dist+=Distance3D(inpseq_stack[i][0,5,0], inpseq_stack[i][0,5,1], inpseq_stack[i][0,5,2], inpseq_stack[i][0,6,0], inpseq_stack[i][0,6,1], inpseq_stack[i][0,6,2])
            glob_limbs_dist+=Distance3D(inpseq_stack[i][0,6,0], inpseq_stack[i][0,6,1], inpseq_stack[i][0,6,2], inpseq_stack[i][0,7,0], inpseq_stack[i][0,7,1], inpseq_stack[i][0,7,2])
            glob_limbs_dist+=Distance3D(inpseq_stack[i][0,7,0], inpseq_stack[i][0,7,1], inpseq_stack[i][0,7,2], inpseq_stack[i][0,21,0], inpseq_stack[i][0,21,1], inpseq_stack[i][0,21,2])
            glob_limbs_dist+=Distance3D(inpseq_stack[i][0,7,0], inpseq_stack[i][0,7,1], inpseq_stack[i][0,7,2], inpseq_stack[i][0,22,0], inpseq_stack[i][0,22,1], inpseq_stack[i][0,22,2])
            glob_limbs_dist+=Distance3D(inpseq_stack[i][0,20,0], inpseq_stack[i][0,20,1], inpseq_stack[i][0,20,2], inpseq_stack[i][0,8,0], inpseq_stack[i][0,8,1], inpseq_stack[i][0,8,2])
            glob_limbs_dist+=Distance3D(inpseq_stack[i][0,8,0], inpseq_stack[i][0,8,1], inpseq_stack[i][0,8,2], inpseq_stack[i][0,9,0], inpseq_stack[i][0,9,1], inpseq_stack[i][0,9,2])
            glob_limbs_dist+=Distance3D(inpseq_stack[i][0,9,0], inpseq_stack[i][0,9,1], inpseq_stack[i][0,9,2], inpseq_stack[i][0,10,0], inpseq_stack[i][0,10,1], inpseq_stack[i][0,10,2])
            glob_limbs_dist+=Distance3D(inpseq_stack[i][0,10,0], inpseq_stack[i][0,10,1], inpseq_stack[i][0,10,2], inpseq_stack[i][0,11,0], inpseq_stack[i][0,11,1], inpseq_stack[i][0,11,2])
            glob_limbs_dist+=Distance3D(inpseq_stack[i][0,11,0], inpseq_stack[i][0,11,1], inpseq_stack[i][0,11,2], inpseq_stack[i][0,23,0], inpseq_stack[i][0,23,1], inpseq_stack[i][0,23,2])
            glob_limbs_dist+=Distance3D(inpseq_stack[i][0,11,0], inpseq_stack[i][0,11,1], inpseq_stack[i][0,11,2], inpseq_stack[i][0,24,0], inpseq_stack[i][0,24,1], inpseq_stack[i][0,24,2])


            if glob_limbs_dist==0:
                glob_limbs_dist=1


            for k in range(0,inpseq_stack[i].shape[0]):

                dis_xyz[0] = Distance3D(inpseq_stack[i][k,5,0], inpseq_stack[i][k,5,1], inpseq_stack[i][k,5,2], inpseq_stack[i][k,9,0], inpseq_stack[i][k,9,1], inpseq_stack[i][k,9,2])/glob_limbs_dist
                dis_xyz[1] = Distance3D(inpseq_stack[i][k,6,0], inpseq_stack[i][k,6,1], inpseq_stack[i][k,6,2], inpseq_stack[i][k,10,0], inpseq_stack[i][k,10,1], inpseq_stack[i][k,10,2])/glob_limbs_dist
                dis_xyz[2] = Distance3D(inpseq_stack[i][k,13,0], inpseq_stack[i][k,13,1], inpseq_stack[i][k,13,2], inpseq_stack[i][k,17,0], inpseq_stack[i][k,17,1], inpseq_stack[i][k,17,2])/glob_limbs_dist
                dis_xyz[3] = Distance3D(inpseq_stack[i][k,14,0], inpseq_stack[i][k,14,1], inpseq_stack[i][k,14,2], inpseq_stack[i][k,18,0], inpseq_stack[i][k,18,1], inpseq_stack[i][k,18,2])/glob_limbs_dist
                dis_xyz[4] = Distance3D(inpseq_stack[i][k,6,0], inpseq_stack[i][k,6,1], inpseq_stack[i][k,6,2], inpseq_stack[i][k,12,0], inpseq_stack[i][k,12,1], inpseq_stack[i][k,12,2])/glob_limbs_dist
                dis_xyz[5] = Distance3D(inpseq_stack[i][k,6,0], inpseq_stack[i][k,6,1], inpseq_stack[i][k,6,2], inpseq_stack[i][k,13,0], inpseq_stack[i][k,13,1], inpseq_stack[i][k,13,2])/glob_limbs_dist
                dis_xyz[6] = Distance3D(inpseq_stack[i][k,6,0], inpseq_stack[i][k,6,1], inpseq_stack[i][k,6,2], inpseq_stack[i][k,14,0], inpseq_stack[i][k,14,1], inpseq_stack[i][k,14,2])/glob_limbs_dist
                dis_xyz[7] = Distance3D(inpseq_stack[i][k,10,0], inpseq_stack[i][k,10,1], inpseq_stack[i][k,10,2], inpseq_stack[i][k,16,0], inpseq_stack[i][k,16,1], inpseq_stack[i][k,16,2])/glob_limbs_dist
                dis_xyz[8] = Distance3D(inpseq_stack[i][k,10,0], inpseq_stack[i][k,10,1], inpseq_stack[i][k,10,2], inpseq_stack[i][k,17,0], inpseq_stack[i][k,17,1], inpseq_stack[i][k,17,2])/glob_limbs_dist
                dis_xyz[9] = Distance3D(inpseq_stack[i][k,10,0], inpseq_stack[i][k,10,1], inpseq_stack[i][k,10,2], inpseq_stack[i][k,18,0], inpseq_stack[i][k,18,1], inpseq_stack[i][k,18,2])/glob_limbs_dist
                dis_xyz[10] = Distance3D(inpseq_stack[i][k,3,0], inpseq_stack[i][k,3,1], inpseq_stack[i][k,3,2], inpseq_stack[i][k,14,0], inpseq_stack[i][k,14,1], inpseq_stack[i][k,14,2])/glob_limbs_dist
                dis_xyz[11] = Distance3D(inpseq_stack[i][k,3,0], inpseq_stack[i][k,3,1], inpseq_stack[i][k,3,2], inpseq_stack[i][k,18,0], inpseq_stack[i][k,18,1], inpseq_stack[i][k,18,2])/glob_limbs_dist

                dis_xyz[12] = Distance3D(inpseq_stack[i][k,0,0], inpseq_stack[i][k,0,1], inpseq_stack[i][k,0,2], inpseq_stack[i][k,17,0], inpseq_stack[i][k,17,1], inpseq_stack[i][k,17,2])/glob_limbs_dist
                dis_xyz[13] = Distance3D(inpseq_stack[i][k,16,0], inpseq_stack[i][k,16,1], inpseq_stack[i][k,16,2], inpseq_stack[i][k,18,0], inpseq_stack[i][k,18,1], inpseq_stack[i][k,18,2])/glob_limbs_dist
                dis_xyz[14] = Distance3D(inpseq_stack[i][k,0,0], inpseq_stack[i][k,0,1], inpseq_stack[i][k,0,2], inpseq_stack[i][k,13,0], inpseq_stack[i][k,13,1], inpseq_stack[i][k,13,2])/glob_limbs_dist
                dis_xyz[15] = Distance3D(inpseq_stack[i][k,12,0], inpseq_stack[i][k,12,1], inpseq_stack[i][k,12,2], inpseq_stack[i][k,14,0], inpseq_stack[i][k,14,1], inpseq_stack[i][k,14,2])/glob_limbs_dist
                dis_xyz[16] = Distance3D(inpseq_stack[i][k,4,0], inpseq_stack[i][k,4,1], inpseq_stack[i][k,4,2], inpseq_stack[i][k,6,0], inpseq_stack[i][k,6,1], inpseq_stack[i][k,6,2])/glob_limbs_dist
                dis_xyz[17] = Distance3D(inpseq_stack[i][k,20,0], inpseq_stack[i][k,20,1], inpseq_stack[i][k,20,2], inpseq_stack[i][k,5,0], inpseq_stack[i][k,5,1], inpseq_stack[i][k,5,2])/glob_limbs_dist
                dis_xyz[18] = Distance3D(inpseq_stack[i][k,2,0], inpseq_stack[i][k,2,1], inpseq_stack[i][k,2,2], inpseq_stack[i][k,4,0], inpseq_stack[i][k,4,1], inpseq_stack[i][k,4,2])/glob_limbs_dist
                dis_xyz[19] = Distance3D(inpseq_stack[i][k,2,0], inpseq_stack[i][k,2,1], inpseq_stack[i][k,2,2], inpseq_stack[i][k,8,0], inpseq_stack[i][k,8,1], inpseq_stack[i][k,8,2])/glob_limbs_dist
                dis_xyz[20] = Distance3D(inpseq_stack[i][k,20,0], inpseq_stack[i][k,20,1], inpseq_stack[i][k,20,2], inpseq_stack[i][k,9,0], inpseq_stack[i][k,9,1], inpseq_stack[i][k,17,2])/glob_limbs_dist
                dis_xyz[21] = Distance3D(inpseq_stack[i][k,8,0], inpseq_stack[i][k,8,1], inpseq_stack[i][k,8,2], inpseq_stack[i][k,10,0], inpseq_stack[i][k,10,1], inpseq_stack[i][k,9,2])/glob_limbs_dist

                dis_xyz[22] = Distance3D(inpseq_stack[i][k,13,0], inpseq_stack[i][k,13,1], inpseq_stack[i][k,13,2], inpseq_stack[i][k,18,0], inpseq_stack[i][k,18,1], inpseq_stack[i][k,18,2])/glob_limbs_dist
                dis_xyz[23] = Distance3D(inpseq_stack[i][k,14,0], inpseq_stack[i][k,14,1], inpseq_stack[i][k,14,2], inpseq_stack[i][k,17,0], inpseq_stack[i][k,17,1], inpseq_stack[i][k,17,2])/glob_limbs_dist
                dis_xyz[24] = Distance3D(inpseq_stack[i][k,5,0], inpseq_stack[i][k,5,1], inpseq_stack[i][k,5,2], inpseq_stack[i][k,10,0], inpseq_stack[i][k,10,1], inpseq_stack[i][k,10,2])/glob_limbs_dist
                dis_xyz[25] = Distance3D(inpseq_stack[i][k,6,0], inpseq_stack[i][k,6,1], inpseq_stack[i][k,6,2], inpseq_stack[i][k,9,0], inpseq_stack[i][k,9,1], inpseq_stack[i][k,9,2])/glob_limbs_dist
                dis_xyz[26] = Distance3D(inpseq_stack[i][k,3,0], inpseq_stack[i][k,3,1], inpseq_stack[i][k,3,2], inpseq_stack[i][k,17,0], inpseq_stack[i][k,17,1], inpseq_stack[i][k,17,2])/glob_limbs_dist
                dis_xyz[27] = Distance3D(inpseq_stack[i][k,3,0], inpseq_stack[i][k,3,1], inpseq_stack[i][k,3,2], inpseq_stack[i][k,10,0], inpseq_stack[i][k,10,1], inpseq_stack[i][k,10,2])/glob_limbs_dist
                dis_xyz[28] = Distance3D(inpseq_stack[i][k,3,0], inpseq_stack[i][k,3,1], inpseq_stack[i][k,3,2], inpseq_stack[i][k,9,0], inpseq_stack[i][k,9,1], inpseq_stack[i][k,9,2])/glob_limbs_dist
                dis_xyz[29] = Distance3D(inpseq_stack[i][k,3,0], inpseq_stack[i][k,3,1], inpseq_stack[i][k,3,2], inpseq_stack[i][k,13,0], inpseq_stack[i][k,13,1], inpseq_stack[i][k,13,2])/glob_limbs_dist
                dis_xyz[30] = Distance3D(inpseq_stack[i][k,3,0], inpseq_stack[i][k,3,1], inpseq_stack[i][k,3,2], inpseq_stack[i][k,6,0], inpseq_stack[i][k,6,1], inpseq_stack[i][k,6,2])/glob_limbs_dist
                dis_xyz[31] = Distance3D(inpseq_stack[i][k,3,0], inpseq_stack[i][k,3,1], inpseq_stack[i][k,3,2], inpseq_stack[i][k,5,0], inpseq_stack[i][k,5,1], inpseq_stack[i][k,5,2])/glob_limbs_dist


                distance_desc[:,k] = dis_xyz[:]


        else:

            dis_xyz=np.zeros((64))

            distance_desc=np.zeros((64,inpseq_stack[i].shape[0]))



            glob_limbs_dist=0

            glob_limbs_dist+=Distance3D(inpseq_stack[i][0,0,0], inpseq_stack[i][0,0,1], inpseq_stack[i][0,0,2], inpseq_stack[i][0,1,0], inpseq_stack[i][0,1,1], inpseq_stack[i][0,1,2])
            glob_limbs_dist+=Distance3D(inpseq_stack[i][0,1,0], inpseq_stack[i][0,1,1], inpseq_stack[i][0,1,2], inpseq_stack[i][0,20,0], inpseq_stack[i][0,20,1], inpseq_stack[i][0,20,2])
            glob_limbs_dist+=Distance3D(inpseq_stack[i][0,20,0], inpseq_stack[i][0,20,1], inpseq_stack[i][0,20,2], inpseq_stack[i][0,2,0], inpseq_stack[i][0,2,1], inpseq_stack[i][0,2,2])
            glob_limbs_dist+=Distance3D(inpseq_stack[i][0,2,0], inpseq_stack[i][0,2,1], inpseq_stack[i][0,2,2], inpseq_stack[i][0,3,0], inpseq_stack[i][0,3,1], inpseq_stack[i][0,3,2])
            glob_limbs_dist+=Distance3D(inpseq_stack[i][0,0,0], inpseq_stack[i][0,0,1], inpseq_stack[i][0,0,2], inpseq_stack[i][0,12,0], inpseq_stack[i][0,12,1], inpseq_stack[i][0,12,2])
            glob_limbs_dist+=Distance3D(inpseq_stack[i][0,12,0], inpseq_stack[i][0,12,1], inpseq_stack[i][0,12,2], inpseq_stack[i][0,13,0], inpseq_stack[i][0,13,1], inpseq_stack[i][0,13,2])
            glob_limbs_dist+=Distance3D(inpseq_stack[i][0,13,0], inpseq_stack[i][0,13,1], inpseq_stack[i][0,13,2], inpseq_stack[i][0,14,0], inpseq_stack[i][0,14,1], inpseq_stack[i][0,14,2])
            glob_limbs_dist+=Distance3D(inpseq_stack[i][0,14,0], inpseq_stack[i][0,14,1], inpseq_stack[i][0,14,2], inpseq_stack[i][0,15,0], inpseq_stack[i][0,15,1], inpseq_stack[i][0,15,2])
            glob_limbs_dist+=Distance3D(inpseq_stack[i][0,0,0], inpseq_stack[i][0,0,1], inpseq_stack[i][0,0,2], inpseq_stack[i][0,16,0], inpseq_stack[i][0,16,1], inpseq_stack[i][0,16,2])
            glob_limbs_dist+=Distance3D(inpseq_stack[i][0,16,0], inpseq_stack[i][0,16,1], inpseq_stack[i][0,16,2], inpseq_stack[i][0,17,0], inpseq_stack[i][0,17,1], inpseq_stack[i][0,17,2])
            glob_limbs_dist+=Distance3D(inpseq_stack[i][0,17,0], inpseq_stack[i][0,17,1], inpseq_stack[i][0,17,2], inpseq_stack[i][0,18,0], inpseq_stack[i][0,18,1], inpseq_stack[i][0,18,2])
            glob_limbs_dist+=Distance3D(inpseq_stack[i][0,18,0], inpseq_stack[i][0,18,1], inpseq_stack[i][0,18,2], inpseq_stack[i][0,19,0], inpseq_stack[i][0,19,1], inpseq_stack[i][0,19,2])
            glob_limbs_dist+=Distance3D(inpseq_stack[i][0,20,0], inpseq_stack[i][0,20,1], inpseq_stack[i][0,20,2], inpseq_stack[i][0,4,0], inpseq_stack[i][0,4,1], inpseq_stack[i][0,4,2])
            glob_limbs_dist+=Distance3D(inpseq_stack[i][0,4,0], inpseq_stack[i][0,4,1], inpseq_stack[i][0,4,2], inpseq_stack[i][0,5,0], inpseq_stack[i][0,5,1], inpseq_stack[i][0,5,2])
            glob_limbs_dist+=Distance3D(inpseq_stack[i][0,5,0], inpseq_stack[i][0,5,1], inpseq_stack[i][0,5,2], inpseq_stack[i][0,6,0], inpseq_stack[i][0,6,1], inpseq_stack[i][0,6,2])
            glob_limbs_dist+=Distance3D(inpseq_stack[i][0,6,0], inpseq_stack[i][0,6,1], inpseq_stack[i][0,6,2], inpseq_stack[i][0,7,0], inpseq_stack[i][0,7,1], inpseq_stack[i][0,7,2])
            glob_limbs_dist+=Distance3D(inpseq_stack[i][0,7,0], inpseq_stack[i][0,7,1], inpseq_stack[i][0,7,2], inpseq_stack[i][0,21,0], inpseq_stack[i][0,21,1], inpseq_stack[i][0,21,2])
            glob_limbs_dist+=Distance3D(inpseq_stack[i][0,7,0], inpseq_stack[i][0,7,1], inpseq_stack[i][0,7,2], inpseq_stack[i][0,22,0], inpseq_stack[i][0,22,1], inpseq_stack[i][0,22,2])
            glob_limbs_dist+=Distance3D(inpseq_stack[i][0,20,0], inpseq_stack[i][0,20,1], inpseq_stack[i][0,20,2], inpseq_stack[i][0,8,0], inpseq_stack[i][0,8,1], inpseq_stack[i][0,8,2])
            glob_limbs_dist+=Distance3D(inpseq_stack[i][0,8,0], inpseq_stack[i][0,8,1], inpseq_stack[i][0,8,2], inpseq_stack[i][0,9,0], inpseq_stack[i][0,9,1], inpseq_stack[i][0,9,2])
            glob_limbs_dist+=Distance3D(inpseq_stack[i][0,9,0], inpseq_stack[i][0,9,1], inpseq_stack[i][0,9,2], inpseq_stack[i][0,10,0], inpseq_stack[i][0,10,1], inpseq_stack[i][0,10,2])
            glob_limbs_dist+=Distance3D(inpseq_stack[i][0,10,0], inpseq_stack[i][0,10,1], inpseq_stack[i][0,10,2], inpseq_stack[i][0,11,0], inpseq_stack[i][0,11,1], inpseq_stack[i][0,11,2])
            glob_limbs_dist+=Distance3D(inpseq_stack[i][0,11,0], inpseq_stack[i][0,11,1], inpseq_stack[i][0,11,2], inpseq_stack[i][0,23,0], inpseq_stack[i][0,23,1], inpseq_stack[i][0,23,2])
            glob_limbs_dist+=Distance3D(inpseq_stack[i][0,11,0], inpseq_stack[i][0,11,1], inpseq_stack[i][0,11,2], inpseq_stack[i][0,24,0], inpseq_stack[i][0,24,1], inpseq_stack[i][0,24,2])

            glob_limbs_dist2=0

            glob_limbs_dist2+=Distance3D(inpseq_stack[i][0,0+25,0], inpseq_stack[i][0,0+25,1], inpseq_stack[i][0,0+25,2], inpseq_stack[i][0,1+25,0], inpseq_stack[i][0,1+25,1], inpseq_stack[i][0,1+25,2])
            glob_limbs_dist2+=Distance3D(inpseq_stack[i][0,1+25,0], inpseq_stack[i][0,1+25,1], inpseq_stack[i][0,1+25,2], inpseq_stack[i][0,20+25,0], inpseq_stack[i][0,20+25,1], inpseq_stack[i][0,20+25,2])
            glob_limbs_dist2+=Distance3D(inpseq_stack[i][0,20+25,0], inpseq_stack[i][0,20+25,1], inpseq_stack[i][0,20+25,2], inpseq_stack[i][0,2+25,0], inpseq_stack[i][0,2+25,1], inpseq_stack[i][0,2+25,2])
            glob_limbs_dist2+=Distance3D(inpseq_stack[i][0,2+25,0], inpseq_stack[i][0,2+25,1], inpseq_stack[i][0,2+25,2], inpseq_stack[i][0,3+25,0], inpseq_stack[i][0,3+25,1], inpseq_stack[i][0,3+25,2])
            glob_limbs_dist2+=Distance3D(inpseq_stack[i][0,0+25,0], inpseq_stack[i][0,0+25,1], inpseq_stack[i][0,0+25,2], inpseq_stack[i][0,12+25,0], inpseq_stack[i][0,12+25,1], inpseq_stack[i][0,12+25,2])
            glob_limbs_dist2+=Distance3D(inpseq_stack[i][0,12+25,0], inpseq_stack[i][0,12+25,1], inpseq_stack[i][0,12+25,2], inpseq_stack[i][0,13+25,0], inpseq_stack[i][0,13+25,1], inpseq_stack[i][0,13+25,2])
            glob_limbs_dist2+=Distance3D(inpseq_stack[i][0,13+25,0], inpseq_stack[i][0,13+25,1], inpseq_stack[i][0,13+25,2], inpseq_stack[i][0,14+25,0], inpseq_stack[i][0,14+25,1], inpseq_stack[i][0,14+25,2])
            glob_limbs_dist2+=Distance3D(inpseq_stack[i][0,14+25,0], inpseq_stack[i][0,14+25,1], inpseq_stack[i][0,14+25,2], inpseq_stack[i][0,15+25,0], inpseq_stack[i][0,15+25,1], inpseq_stack[i][0,15+25,2])
            glob_limbs_dist2+=Distance3D(inpseq_stack[i][0,0+25,0], inpseq_stack[i][0,0+25,1], inpseq_stack[i][0,0+25,2], inpseq_stack[i][0,16+25,0], inpseq_stack[i][0,16+25,1], inpseq_stack[i][0,16+25,2])
            glob_limbs_dist2+=Distance3D(inpseq_stack[i][0,16+25,0], inpseq_stack[i][0,16+25,1], inpseq_stack[i][0,16+25,2], inpseq_stack[i][0,17+25,0], inpseq_stack[i][0,17+25,1], inpseq_stack[i][0,17+25,2])
            glob_limbs_dist2+=Distance3D(inpseq_stack[i][0,17+25,0], inpseq_stack[i][0,17+25,1], inpseq_stack[i][0,17+25,2], inpseq_stack[i][0,18+25,0], inpseq_stack[i][0,18+25,1], inpseq_stack[i][0,18+25,2])
            glob_limbs_dist2+=Distance3D(inpseq_stack[i][0,18+25,0], inpseq_stack[i][0,18+25,1], inpseq_stack[i][0,18+25,2], inpseq_stack[i][0,19+25,0], inpseq_stack[i][0,19+25,1], inpseq_stack[i][0,19+25,2])
            glob_limbs_dist2+=Distance3D(inpseq_stack[i][0,20+25,0], inpseq_stack[i][0,20+25,1], inpseq_stack[i][0,20+25,2], inpseq_stack[i][0,4+25,0], inpseq_stack[i][0,4+25,1], inpseq_stack[i][0,4+25,2])
            glob_limbs_dist2+=Distance3D(inpseq_stack[i][0,4+25,0], inpseq_stack[i][0,4+25,1], inpseq_stack[i][0,4+25,2], inpseq_stack[i][0,5+25,0], inpseq_stack[i][0,5+25,1], inpseq_stack[i][0,5+25,2])
            glob_limbs_dist2+=Distance3D(inpseq_stack[i][0,5+25,0], inpseq_stack[i][0,5+25,1], inpseq_stack[i][0,5+25,2], inpseq_stack[i][0,6+25,0], inpseq_stack[i][0,6+25,1], inpseq_stack[i][0,6+25,2])
            glob_limbs_dist2+=Distance3D(inpseq_stack[i][0,6+25,0], inpseq_stack[i][0,6+25,1], inpseq_stack[i][0,6+25,2], inpseq_stack[i][0,7+25,0], inpseq_stack[i][0,7+25,1], inpseq_stack[i][0,7+25,2])
            glob_limbs_dist2+=Distance3D(inpseq_stack[i][0,7+25,0], inpseq_stack[i][0,7+25,1], inpseq_stack[i][0,7+25,2], inpseq_stack[i][0,21+25,0], inpseq_stack[i][0,21+25,1], inpseq_stack[i][0,21+25,2])
            glob_limbs_dist2+=Distance3D(inpseq_stack[i][0,7+25,0], inpseq_stack[i][0,7+25,1], inpseq_stack[i][0,7+25,2], inpseq_stack[i][0,22+25,0], inpseq_stack[i][0,22+25,1], inpseq_stack[i][0,22+25,2])
            glob_limbs_dist2+=Distance3D(inpseq_stack[i][0,20+25,0], inpseq_stack[i][0,20+25,1], inpseq_stack[i][0,20+25,2], inpseq_stack[i][0,8+25,0], inpseq_stack[i][0,8+25,1], inpseq_stack[i][0,8+25,2])
            glob_limbs_dist2+=Distance3D(inpseq_stack[i][0,8+25,0], inpseq_stack[i][0,8+25,1], inpseq_stack[i][0,8+25,2], inpseq_stack[i][0,9+25,0], inpseq_stack[i][0,9+25,1], inpseq_stack[i][0,9+25,2])
            glob_limbs_dist2+=Distance3D(inpseq_stack[i][0,9+25,0], inpseq_stack[i][0,9+25,1], inpseq_stack[i][0,9+25,2], inpseq_stack[i][0,10+25,0], inpseq_stack[i][0,10+25,1], inpseq_stack[i][0,10+25,2])
            glob_limbs_dist2+=Distance3D(inpseq_stack[i][0,10+25,0], inpseq_stack[i][0,10+25,1], inpseq_stack[i][0,10+25,2], inpseq_stack[i][0,11+25,0], inpseq_stack[i][0,11+25,1], inpseq_stack[i][0,11+25,2])
            glob_limbs_dist2+=Distance3D(inpseq_stack[i][0,11+25,0], inpseq_stack[i][0,11+25,1], inpseq_stack[i][0,11+25,2], inpseq_stack[i][0,23+25,0], inpseq_stack[i][0,23+25,1], inpseq_stack[i][0,23+25,2])
            glob_limbs_dist2+=Distance3D(inpseq_stack[i][0,11+25,0], inpseq_stack[i][0,11+25,1], inpseq_stack[i][0,11+25,2], inpseq_stack[i][0,24+25,0], inpseq_stack[i][0,24+25,1], inpseq_stack[i][0,24+25,2])

            if glob_limbs_dist==0:
                glob_limbs_dist=1
            
            if glob_limbs_dist2==0:
                glob_limbs_dist2=1


            for k in range(0,inpseq_stack[i].shape[0]):

                dis_xyz[0] = Distance3D(inpseq_stack[i][k,5,0], inpseq_stack[i][k,5,1], inpseq_stack[i][k,5,2], inpseq_stack[i][k,9,0], inpseq_stack[i][k,9,1], inpseq_stack[i][k,9,2])/glob_limbs_dist
                dis_xyz[1] = Distance3D(inpseq_stack[i][k,6,0], inpseq_stack[i][k,6,1], inpseq_stack[i][k,6,2], inpseq_stack[i][k,10,0], inpseq_stack[i][k,10,1], inpseq_stack[i][k,10,2])/glob_limbs_dist
                dis_xyz[2] = Distance3D(inpseq_stack[i][k,13,0], inpseq_stack[i][k,13,1], inpseq_stack[i][k,13,2], inpseq_stack[i][k,17,0], inpseq_stack[i][k,17,1], inpseq_stack[i][k,17,2])/glob_limbs_dist
                dis_xyz[3] = Distance3D(inpseq_stack[i][k,14,0], inpseq_stack[i][k,14,1], inpseq_stack[i][k,14,2], inpseq_stack[i][k,18,0], inpseq_stack[i][k,18,1], inpseq_stack[i][k,18,2])/glob_limbs_dist
                dis_xyz[4] = Distance3D(inpseq_stack[i][k,6,0], inpseq_stack[i][k,6,1], inpseq_stack[i][k,6,2], inpseq_stack[i][k,12,0], inpseq_stack[i][k,12,1], inpseq_stack[i][k,12,2])/glob_limbs_dist
                dis_xyz[5] = Distance3D(inpseq_stack[i][k,6,0], inpseq_stack[i][k,6,1], inpseq_stack[i][k,6,2], inpseq_stack[i][k,13,0], inpseq_stack[i][k,13,1], inpseq_stack[i][k,13,2])/glob_limbs_dist
                dis_xyz[6] = Distance3D(inpseq_stack[i][k,6,0], inpseq_stack[i][k,6,1], inpseq_stack[i][k,6,2], inpseq_stack[i][k,14,0], inpseq_stack[i][k,14,1], inpseq_stack[i][k,14,2])/glob_limbs_dist
                dis_xyz[7] = Distance3D(inpseq_stack[i][k,10,0], inpseq_stack[i][k,10,1], inpseq_stack[i][k,10,2], inpseq_stack[i][k,16,0], inpseq_stack[i][k,16,1], inpseq_stack[i][k,16,2])/glob_limbs_dist
                dis_xyz[8] = Distance3D(inpseq_stack[i][k,10,0], inpseq_stack[i][k,10,1], inpseq_stack[i][k,10,2], inpseq_stack[i][k,17,0], inpseq_stack[i][k,17,1], inpseq_stack[i][k,17,2])/glob_limbs_dist
                dis_xyz[9] = Distance3D(inpseq_stack[i][k,10,0], inpseq_stack[i][k,10,1], inpseq_stack[i][k,10,2], inpseq_stack[i][k,18,0], inpseq_stack[i][k,18,1], inpseq_stack[i][k,18,2])/glob_limbs_dist
                dis_xyz[10] = Distance3D(inpseq_stack[i][k,3,0], inpseq_stack[i][k,3,1], inpseq_stack[i][k,3,2], inpseq_stack[i][k,14,0], inpseq_stack[i][k,14,1], inpseq_stack[i][k,14,2])/glob_limbs_dist
                dis_xyz[11] = Distance3D(inpseq_stack[i][k,3,0], inpseq_stack[i][k,3,1], inpseq_stack[i][k,3,2], inpseq_stack[i][k,18,0], inpseq_stack[i][k,18,1], inpseq_stack[i][k,18,2])/glob_limbs_dist

                dis_xyz[12] = Distance3D(inpseq_stack[i][k,0,0], inpseq_stack[i][k,0,1], inpseq_stack[i][k,0,2], inpseq_stack[i][k,17,0], inpseq_stack[i][k,17,1], inpseq_stack[i][k,17,2])/glob_limbs_dist
                dis_xyz[13] = Distance3D(inpseq_stack[i][k,16,0], inpseq_stack[i][k,16,1], inpseq_stack[i][k,16,2], inpseq_stack[i][k,18,0], inpseq_stack[i][k,18,1], inpseq_stack[i][k,18,2])/glob_limbs_dist
                dis_xyz[14] = Distance3D(inpseq_stack[i][k,0,0], inpseq_stack[i][k,0,1], inpseq_stack[i][k,0,2], inpseq_stack[i][k,13,0], inpseq_stack[i][k,13,1], inpseq_stack[i][k,13,2])/glob_limbs_dist
                dis_xyz[15] = Distance3D(inpseq_stack[i][k,12,0], inpseq_stack[i][k,12,1], inpseq_stack[i][k,12,2], inpseq_stack[i][k,14,0], inpseq_stack[i][k,14,1], inpseq_stack[i][k,14,2])/glob_limbs_dist
                dis_xyz[16] = Distance3D(inpseq_stack[i][k,4,0], inpseq_stack[i][k,4,1], inpseq_stack[i][k,4,2], inpseq_stack[i][k,6,0], inpseq_stack[i][k,6,1], inpseq_stack[i][k,6,2])/glob_limbs_dist
                dis_xyz[17] = Distance3D(inpseq_stack[i][k,20,0], inpseq_stack[i][k,20,1], inpseq_stack[i][k,20,2], inpseq_stack[i][k,5,0], inpseq_stack[i][k,5,1], inpseq_stack[i][k,5,2])/glob_limbs_dist
                dis_xyz[18] = Distance3D(inpseq_stack[i][k,2,0], inpseq_stack[i][k,2,1], inpseq_stack[i][k,2,2], inpseq_stack[i][k,4,0], inpseq_stack[i][k,4,1], inpseq_stack[i][k,4,2])/glob_limbs_dist
                dis_xyz[19] = Distance3D(inpseq_stack[i][k,2,0], inpseq_stack[i][k,2,1], inpseq_stack[i][k,2,2], inpseq_stack[i][k,8,0], inpseq_stack[i][k,8,1], inpseq_stack[i][k,8,2])/glob_limbs_dist
                dis_xyz[20] = Distance3D(inpseq_stack[i][k,20,0], inpseq_stack[i][k,20,1], inpseq_stack[i][k,20,2], inpseq_stack[i][k,9,0], inpseq_stack[i][k,9,1], inpseq_stack[i][k,17,2])/glob_limbs_dist
                dis_xyz[21] = Distance3D(inpseq_stack[i][k,8,0], inpseq_stack[i][k,8,1], inpseq_stack[i][k,8,2], inpseq_stack[i][k,10,0], inpseq_stack[i][k,10,1], inpseq_stack[i][k,9,2])/glob_limbs_dist

                dis_xyz[22] = Distance3D(inpseq_stack[i][k,13,0], inpseq_stack[i][k,13,1], inpseq_stack[i][k,13,2], inpseq_stack[i][k,18,0], inpseq_stack[i][k,18,1], inpseq_stack[i][k,18,2])/glob_limbs_dist
                dis_xyz[23] = Distance3D(inpseq_stack[i][k,14,0], inpseq_stack[i][k,14,1], inpseq_stack[i][k,14,2], inpseq_stack[i][k,17,0], inpseq_stack[i][k,17,1], inpseq_stack[i][k,17,2])/glob_limbs_dist
                dis_xyz[24] = Distance3D(inpseq_stack[i][k,5,0], inpseq_stack[i][k,5,1], inpseq_stack[i][k,5,2], inpseq_stack[i][k,10,0], inpseq_stack[i][k,10,1], inpseq_stack[i][k,10,2])/glob_limbs_dist
                dis_xyz[25] = Distance3D(inpseq_stack[i][k,6,0], inpseq_stack[i][k,6,1], inpseq_stack[i][k,6,2], inpseq_stack[i][k,9,0], inpseq_stack[i][k,9,1], inpseq_stack[i][k,9,2])/glob_limbs_dist
                dis_xyz[26] = Distance3D(inpseq_stack[i][k,3,0], inpseq_stack[i][k,3,1], inpseq_stack[i][k,3,2], inpseq_stack[i][k,17,0], inpseq_stack[i][k,17,1], inpseq_stack[i][k,17,2])/glob_limbs_dist
                dis_xyz[27] = Distance3D(inpseq_stack[i][k,3,0], inpseq_stack[i][k,3,1], inpseq_stack[i][k,3,2], inpseq_stack[i][k,10,0], inpseq_stack[i][k,10,1], inpseq_stack[i][k,10,2])/glob_limbs_dist
                dis_xyz[28] = Distance3D(inpseq_stack[i][k,3,0], inpseq_stack[i][k,3,1], inpseq_stack[i][k,3,2], inpseq_stack[i][k,9,0], inpseq_stack[i][k,9,1], inpseq_stack[i][k,9,2])/glob_limbs_dist
                dis_xyz[29] = Distance3D(inpseq_stack[i][k,3,0], inpseq_stack[i][k,3,1], inpseq_stack[i][k,3,2], inpseq_stack[i][k,13,0], inpseq_stack[i][k,13,1], inpseq_stack[i][k,13,2])/glob_limbs_dist
                dis_xyz[30] = Distance3D(inpseq_stack[i][k,3,0], inpseq_stack[i][k,3,1], inpseq_stack[i][k,3,2], inpseq_stack[i][k,6,0], inpseq_stack[i][k,6,1], inpseq_stack[i][k,6,2])/glob_limbs_dist
                dis_xyz[31] = Distance3D(inpseq_stack[i][k,3,0], inpseq_stack[i][k,3,1], inpseq_stack[i][k,3,2], inpseq_stack[i][k,5,0], inpseq_stack[i][k,5,1], inpseq_stack[i][k,5,2])/glob_limbs_dist


                dis_xyz[32] = Distance3D(inpseq_stack[i][k,5+25,0], inpseq_stack[i][k,5+25,1], inpseq_stack[i][k,5+25,2], inpseq_stack[i][k,9+25,0], inpseq_stack[i][k,9+25,1], inpseq_stack[i][k,9+25,2])/glob_limbs_dist2
                dis_xyz[33] = Distance3D(inpseq_stack[i][k,6+25,0], inpseq_stack[i][k,6+25,1], inpseq_stack[i][k,6+25,2], inpseq_stack[i][k,10+25,0], inpseq_stack[i][k,10+25,1], inpseq_stack[i][k,10+25,2])/glob_limbs_dist2
                dis_xyz[34] = Distance3D(inpseq_stack[i][k,13+25,0], inpseq_stack[i][k,13+25,1], inpseq_stack[i][k,13+25,2], inpseq_stack[i][k,17+25,0], inpseq_stack[i][k,17+25,1], inpseq_stack[i][k,17+25,2])/glob_limbs_dist2
                dis_xyz[35] = Distance3D(inpseq_stack[i][k,14+25,0], inpseq_stack[i][k,14+25,1], inpseq_stack[i][k,14+25,2], inpseq_stack[i][k,18+25,0], inpseq_stack[i][k,18+25,1], inpseq_stack[i][k,18+25,2])/glob_limbs_dist2
                dis_xyz[36] = Distance3D(inpseq_stack[i][k,6+25,0], inpseq_stack[i][k,6+25,1], inpseq_stack[i][k,6+25,2], inpseq_stack[i][k,12+25,0], inpseq_stack[i][k,12+25,1], inpseq_stack[i][k,12+25,2])/glob_limbs_dist2
                dis_xyz[37] = Distance3D(inpseq_stack[i][k,6+25,0], inpseq_stack[i][k,6+25,1], inpseq_stack[i][k,6+25,2], inpseq_stack[i][k,13+25,0], inpseq_stack[i][k,13+25,1], inpseq_stack[i][k,13+25,2])/glob_limbs_dist2
                dis_xyz[38] = Distance3D(inpseq_stack[i][k,6+25,0], inpseq_stack[i][k,6+25,1], inpseq_stack[i][k,6+25,2], inpseq_stack[i][k,14+25,0], inpseq_stack[i][k,14+25,1], inpseq_stack[i][k,14+25,2])/glob_limbs_dist2
                dis_xyz[39] = Distance3D(inpseq_stack[i][k,10+25,0], inpseq_stack[i][k,10+25,1], inpseq_stack[i][k,10+25,2], inpseq_stack[i][k,16+25,0], inpseq_stack[i][k,16+25,1], inpseq_stack[i][k,16+25,2])/glob_limbs_dist2
                dis_xyz[40] = Distance3D(inpseq_stack[i][k,10+25,0], inpseq_stack[i][k,10+25,1], inpseq_stack[i][k,10+25,2], inpseq_stack[i][k,17+25,0], inpseq_stack[i][k,17+25,1], inpseq_stack[i][k,17+25,2])/glob_limbs_dist2
                dis_xyz[41] = Distance3D(inpseq_stack[i][k,10+25,0], inpseq_stack[i][k,10+25,1], inpseq_stack[i][k,10+25,2], inpseq_stack[i][k,18+25,0], inpseq_stack[i][k,18+25,1], inpseq_stack[i][k,18+25,2])/glob_limbs_dist2
                dis_xyz[42] = Distance3D(inpseq_stack[i][k,3+25,0], inpseq_stack[i][k,3+25,1], inpseq_stack[i][k,3+25,2], inpseq_stack[i][k,14+25,0], inpseq_stack[i][k,14+25,1], inpseq_stack[i][k,14+25,2])/glob_limbs_dist2
                dis_xyz[43] = Distance3D(inpseq_stack[i][k,3+25,0], inpseq_stack[i][k,3+25,1], inpseq_stack[i][k,3+25,2], inpseq_stack[i][k,18+25,0], inpseq_stack[i][k,18+25,1], inpseq_stack[i][k,18+25,2])/glob_limbs_dist2

                dis_xyz[44] = Distance3D(inpseq_stack[i][k,0+25,0], inpseq_stack[i][k,0+25,1], inpseq_stack[i][k,0+25,2], inpseq_stack[i][k,17+25,0], inpseq_stack[i][k,17+25,1], inpseq_stack[i][k,17+25,2])/glob_limbs_dist2
                dis_xyz[45] = Distance3D(inpseq_stack[i][k,16+25,0], inpseq_stack[i][k,16+25,1], inpseq_stack[i][k,16+25,2], inpseq_stack[i][k,18+25,0], inpseq_stack[i][k,18+25,1], inpseq_stack[i][k,18+25,2])/glob_limbs_dist2
                dis_xyz[46] = Distance3D(inpseq_stack[i][k,0+25,0], inpseq_stack[i][k,0+25,1], inpseq_stack[i][k,0+25,2], inpseq_stack[i][k,13+25,0], inpseq_stack[i][k,13+25,1], inpseq_stack[i][k,13+25,2])/glob_limbs_dist2
                dis_xyz[47] = Distance3D(inpseq_stack[i][k,12+25,0], inpseq_stack[i][k,12+25,1], inpseq_stack[i][k,12+25,2], inpseq_stack[i][k,14+25,0], inpseq_stack[i][k,14+25,1], inpseq_stack[i][k,14+25,2])/glob_limbs_dist2
                dis_xyz[48] = Distance3D(inpseq_stack[i][k,4+25,0], inpseq_stack[i][k,4+25,1], inpseq_stack[i][k,4+25,2], inpseq_stack[i][k,6+25,0], inpseq_stack[i][k,6+25,1], inpseq_stack[i][k,6+25,2])/glob_limbs_dist2
                dis_xyz[48] = Distance3D(inpseq_stack[i][k,20+25,0], inpseq_stack[i][k,20+25,1], inpseq_stack[i][k,20+25,2], inpseq_stack[i][k,5+25,0], inpseq_stack[i][k,5+25,1], inpseq_stack[i][k,5+25,2])/glob_limbs_dist2
                dis_xyz[50] = Distance3D(inpseq_stack[i][k,2+25,0], inpseq_stack[i][k,2+25,1], inpseq_stack[i][k,2+25,2], inpseq_stack[i][k,4+25,0], inpseq_stack[i][k,4+25,1], inpseq_stack[i][k,4+25,2])/glob_limbs_dist2
                dis_xyz[51] = Distance3D(inpseq_stack[i][k,2+25,0], inpseq_stack[i][k,2+25,1], inpseq_stack[i][k,2+25,2], inpseq_stack[i][k,8+25,0], inpseq_stack[i][k,8+25,1], inpseq_stack[i][k,8+25,2])/glob_limbs_dist2
                dis_xyz[52] = Distance3D(inpseq_stack[i][k,20+25,0], inpseq_stack[i][k,20+25,1], inpseq_stack[i][k,20+25,2], inpseq_stack[i][k,9+25,0], inpseq_stack[i][k,9+25,1], inpseq_stack[i][k,17+25,2])/glob_limbs_dist2
                dis_xyz[53] = Distance3D(inpseq_stack[i][k,8+25,0], inpseq_stack[i][k,8+25,1], inpseq_stack[i][k,8+25,2], inpseq_stack[i][k,10+25,0], inpseq_stack[i][k,10+25,1], inpseq_stack[i][k,9+25,2])/glob_limbs_dist2

                dis_xyz[54] = Distance3D(inpseq_stack[i][k,13+25,0], inpseq_stack[i][k,13+25,1], inpseq_stack[i][k,13+25,2], inpseq_stack[i][k,18+25,0], inpseq_stack[i][k,18+25,1], inpseq_stack[i][k,18+25,2])/glob_limbs_dist2
                dis_xyz[55] = Distance3D(inpseq_stack[i][k,14+25,0], inpseq_stack[i][k,14+25,1], inpseq_stack[i][k,14+25,2], inpseq_stack[i][k,17+25,0], inpseq_stack[i][k,17+25,1], inpseq_stack[i][k,17+25,2])/glob_limbs_dist2
                dis_xyz[56] = Distance3D(inpseq_stack[i][k,5+25,0], inpseq_stack[i][k,5+25,1], inpseq_stack[i][k,5+25,2], inpseq_stack[i][k,10+25,0], inpseq_stack[i][k,10+25,1], inpseq_stack[i][k,10+25,2])/glob_limbs_dist2
                dis_xyz[57] = Distance3D(inpseq_stack[i][k,6+25,0], inpseq_stack[i][k,6+25,1], inpseq_stack[i][k,6+25,2], inpseq_stack[i][k,9+25,0], inpseq_stack[i][k,9+25,1], inpseq_stack[i][k,9+25,2])/glob_limbs_dist2
                dis_xyz[58] = Distance3D(inpseq_stack[i][k,3+25,0], inpseq_stack[i][k,3+25,1], inpseq_stack[i][k,3+25,2], inpseq_stack[i][k,17+25,0], inpseq_stack[i][k,17+25,1], inpseq_stack[i][k,17+25,2])/glob_limbs_dist2
                dis_xyz[59] = Distance3D(inpseq_stack[i][k,3+25,0], inpseq_stack[i][k,3+25,1], inpseq_stack[i][k,3+25,2], inpseq_stack[i][k,10+25,0], inpseq_stack[i][k,10+25,1], inpseq_stack[i][k,10+25,2])/glob_limbs_dist2
                dis_xyz[60] = Distance3D(inpseq_stack[i][k,3+25,0], inpseq_stack[i][k,3+25,1], inpseq_stack[i][k,3+25,2], inpseq_stack[i][k,9+25,0], inpseq_stack[i][k,9+25,1], inpseq_stack[i][k,9+25,2])/glob_limbs_dist2
                dis_xyz[61] = Distance3D(inpseq_stack[i][k,3+25,0], inpseq_stack[i][k,3+25,1], inpseq_stack[i][k,3+25,2], inpseq_stack[i][k,13+25,0], inpseq_stack[i][k,13+25,1], inpseq_stack[i][k,13+25,2])/glob_limbs_dist2
                dis_xyz[62] = Distance3D(inpseq_stack[i][k,3+25,0], inpseq_stack[i][k,3+25,1], inpseq_stack[i][k,3+25,2], inpseq_stack[i][k,6+25,0], inpseq_stack[i][k,6+25,1], inpseq_stack[i][k,6+25,2])/glob_limbs_dist2
                dis_xyz[63] = Distance3D(inpseq_stack[i][k,3+25,0], inpseq_stack[i][k,3+25,1], inpseq_stack[i][k,3+25,2], inpseq_stack[i][k,5+25,0], inpseq_stack[i][k,5+25,1], inpseq_stack[i][k,5+25,2])/glob_limbs_dist2


                distance_desc[:,k] = dis_xyz[:]


        distance_desc[:,:] = np.floor(255*( distance_desc[:,:]- np.amin(distance_desc[:,:]) )/(np.amax(distance_desc[:,:])-np.amin(distance_desc[:,:])))

        img=Image.fromarray(distance_desc.astype(np.uint8),'L')
        img=img.resize((img_size,img_size))
        img1=img
        img.save(path_disj+str(i)+'_'+str(labels_stack[i])+'.png')
   


        ### Joint distance frames 

        if inpseq_stack[i].shape[1]==25: 
            dis_frms=np.zeros((25))
            distance_desc_frms=np.zeros((25,inpseq_stack[i].shape[0]-1))

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
                dis_frms[20] = Distance3D(inpseq_stack[i][k,20,0], inpseq_stack[i][k,20,1], inpseq_stack[i][k,20,2], inpseq_stack[i][k+1,20,0], inpseq_stack[i][k+1,20,1], inpseq_stack[i][k+1,20,2])
                dis_frms[21] = Distance3D(inpseq_stack[i][k,21,0], inpseq_stack[i][k,21,1], inpseq_stack[i][k,21,2], inpseq_stack[i][k+1,21,0], inpseq_stack[i][k+1,21,1], inpseq_stack[i][k+1,21,2])
                dis_frms[22] = Distance3D(inpseq_stack[i][k,22,0], inpseq_stack[i][k,22,1], inpseq_stack[i][k,22,2], inpseq_stack[i][k+1,22,0], inpseq_stack[i][k+1,22,1], inpseq_stack[i][k+1,22,2])
                dis_frms[23] = Distance3D(inpseq_stack[i][k,23,0], inpseq_stack[i][k,23,1], inpseq_stack[i][k,23,2], inpseq_stack[i][k+1,23,0], inpseq_stack[i][k+1,23,1], inpseq_stack[i][k+1,23,2])
                dis_frms[24] = Distance3D(inpseq_stack[i][k,24,0], inpseq_stack[i][k,24,1], inpseq_stack[i][k,24,2], inpseq_stack[i][k+1,24,0], inpseq_stack[i][k+1,24,1], inpseq_stack[i][k+1,24,2])
            

                distance_desc_frms[:,k] = dis_frms[:]

        else:

            dis_frms=np.zeros((50))

            distance_desc_frms=np.zeros((50,inpseq_stack[i].shape[0]-1))


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
                dis_frms[20] = Distance3D(inpseq_stack[i][k,20,0], inpseq_stack[i][k,20,1], inpseq_stack[i][k,20,2], inpseq_stack[i][k+1,20,0], inpseq_stack[i][k+1,20,1], inpseq_stack[i][k+1,20,2])
                dis_frms[21] = Distance3D(inpseq_stack[i][k,21,0], inpseq_stack[i][k,21,1], inpseq_stack[i][k,21,2], inpseq_stack[i][k+1,21,0], inpseq_stack[i][k+1,21,1], inpseq_stack[i][k+1,21,2])
                dis_frms[22] = Distance3D(inpseq_stack[i][k,22,0], inpseq_stack[i][k,22,1], inpseq_stack[i][k,22,2], inpseq_stack[i][k+1,22,0], inpseq_stack[i][k+1,22,1], inpseq_stack[i][k+1,22,2])
                dis_frms[23] = Distance3D(inpseq_stack[i][k,23,0], inpseq_stack[i][k,23,1], inpseq_stack[i][k,23,2], inpseq_stack[i][k+1,23,0], inpseq_stack[i][k+1,23,1], inpseq_stack[i][k+1,23,2])
                dis_frms[24] = Distance3D(inpseq_stack[i][k,24,0], inpseq_stack[i][k,24,1], inpseq_stack[i][k,24,2], inpseq_stack[i][k+1,24,0], inpseq_stack[i][k+1,24,1], inpseq_stack[i][k+1,24,2])
                
                dis_frms[25] = Distance3D(inpseq_stack[i][k,0+25,0], inpseq_stack[i][k,0+25,1], inpseq_stack[i][k,0+25,2], inpseq_stack[i][k+1,0+25,0], inpseq_stack[i][k+1,0+25,1], inpseq_stack[i][k+1,0+25,2])
                dis_frms[26] = Distance3D(inpseq_stack[i][k,1+25,0], inpseq_stack[i][k,1+25,1], inpseq_stack[i][k,1+25,2], inpseq_stack[i][k+1,1+25,0], inpseq_stack[i][k+1,1+25,1], inpseq_stack[i][k+1,1+25,2])
                dis_frms[27] = Distance3D(inpseq_stack[i][k,2+25,0], inpseq_stack[i][k,2+25,1], inpseq_stack[i][k,2+25,2], inpseq_stack[i][k+1,2+25,0], inpseq_stack[i][k+1,2+25,1], inpseq_stack[i][k+1,2+25,2])
                dis_frms[28] = Distance3D(inpseq_stack[i][k,3+25,0], inpseq_stack[i][k,3+25,1], inpseq_stack[i][k,3+25,2], inpseq_stack[i][k+1,3+25,0], inpseq_stack[i][k+1,3+25,1], inpseq_stack[i][k+1,3+25,2])
                dis_frms[29] = Distance3D(inpseq_stack[i][k,4+25,0], inpseq_stack[i][k,4+25,1], inpseq_stack[i][k,4+25,2], inpseq_stack[i][k+1,4+25,0], inpseq_stack[i][k+1,4+25,1], inpseq_stack[i][k+1,4+25,2])
                dis_frms[30] = Distance3D(inpseq_stack[i][k,5+25,0], inpseq_stack[i][k,5+25,1], inpseq_stack[i][k,5+25,2], inpseq_stack[i][k+1,5+25,0], inpseq_stack[i][k+1,5+25,1], inpseq_stack[i][k+1,5+25,2])
                dis_frms[31] = Distance3D(inpseq_stack[i][k,6+25,0], inpseq_stack[i][k,6+25,1], inpseq_stack[i][k,6+25,2], inpseq_stack[i][k+1,6+25,0], inpseq_stack[i][k+1,6+25,1], inpseq_stack[i][k+1,6+25,2])
                dis_frms[32] = Distance3D(inpseq_stack[i][k,7+25,0], inpseq_stack[i][k,7+25,1], inpseq_stack[i][k,7+25,2], inpseq_stack[i][k+1,7+25,0], inpseq_stack[i][k+1,7+25,1], inpseq_stack[i][k+1,7+25,2])
                dis_frms[33] = Distance3D(inpseq_stack[i][k,8+25,0], inpseq_stack[i][k,8+25,1], inpseq_stack[i][k,8+25,2], inpseq_stack[i][k+1,8+25,0], inpseq_stack[i][k+1,8+25,1], inpseq_stack[i][k+1,8+25,2])
                dis_frms[34] = Distance3D(inpseq_stack[i][k,9+25,0], inpseq_stack[i][k,9+25,1], inpseq_stack[i][k,9+25,2], inpseq_stack[i][k+1,9+25,0], inpseq_stack[i][k+1,9+25,1], inpseq_stack[i][k+1,9+25,2])
                dis_frms[35] = Distance3D(inpseq_stack[i][k,10+25,0], inpseq_stack[i][k,10+25,1], inpseq_stack[i][k,10+25,2], inpseq_stack[i][k+1,10+25,0], inpseq_stack[i][k+1,10+25,1], inpseq_stack[i][k+1,10+25,2])
                dis_frms[36] = Distance3D(inpseq_stack[i][k,11+25,0], inpseq_stack[i][k,11+25,1], inpseq_stack[i][k,11+25,2], inpseq_stack[i][k+1,11+25,0], inpseq_stack[i][k+1,11+25,1], inpseq_stack[i][k+1,11+25,2])
                dis_frms[37] = Distance3D(inpseq_stack[i][k,12+25,0], inpseq_stack[i][k,12+25,1], inpseq_stack[i][k,12+25,2], inpseq_stack[i][k+1,12+25,0], inpseq_stack[i][k+1,12+25,1], inpseq_stack[i][k+1,12+25,2])
                dis_frms[38] = Distance3D(inpseq_stack[i][k,13+25,0], inpseq_stack[i][k,13+25,1], inpseq_stack[i][k,13+25,2], inpseq_stack[i][k+1,13+25,0], inpseq_stack[i][k+1,13+25,1], inpseq_stack[i][k+1,13+25,2])
                dis_frms[39] = Distance3D(inpseq_stack[i][k,14+25,0], inpseq_stack[i][k,14+25,1], inpseq_stack[i][k,14+25,2], inpseq_stack[i][k+1,14+25,0], inpseq_stack[i][k+1,14+25,1], inpseq_stack[i][k+1,14+25,2])
                dis_frms[40] = Distance3D(inpseq_stack[i][k,15+25,0], inpseq_stack[i][k,15+25,1], inpseq_stack[i][k,15+25,2], inpseq_stack[i][k+1,15+25,0], inpseq_stack[i][k+1,15+25,1], inpseq_stack[i][k+1,15+25,2])
                dis_frms[41] = Distance3D(inpseq_stack[i][k,16+25,0], inpseq_stack[i][k,16+25,1], inpseq_stack[i][k,16+25,2], inpseq_stack[i][k+1,16+25,0], inpseq_stack[i][k+1,16+25,1], inpseq_stack[i][k+1,16+25,2])
                dis_frms[42] = Distance3D(inpseq_stack[i][k,17+25,0], inpseq_stack[i][k,17+25,1], inpseq_stack[i][k,17+25,2], inpseq_stack[i][k+1,17+25,0], inpseq_stack[i][k+1,17+25,1], inpseq_stack[i][k+1,17+25,2])
                dis_frms[43] = Distance3D(inpseq_stack[i][k,18+25,0], inpseq_stack[i][k,18+25,1], inpseq_stack[i][k,18+25,2], inpseq_stack[i][k+1,18+25,0], inpseq_stack[i][k+1,18+25,1], inpseq_stack[i][k+1,18+25,2])
                dis_frms[44] = Distance3D(inpseq_stack[i][k,19+25,0], inpseq_stack[i][k,19+25,1], inpseq_stack[i][k,19+25,2], inpseq_stack[i][k+1,19+25,0], inpseq_stack[i][k+1,19+25,1], inpseq_stack[i][k+1,19+25,2])
                dis_frms[45] = Distance3D(inpseq_stack[i][k,20+25,0], inpseq_stack[i][k,20+25,1], inpseq_stack[i][k,20+25,2], inpseq_stack[i][k+1,20+25,0], inpseq_stack[i][k+1,20+25,1], inpseq_stack[i][k+1,20+25,2])
                dis_frms[46] = Distance3D(inpseq_stack[i][k,21+25,0], inpseq_stack[i][k,21+25,1], inpseq_stack[i][k,21+25,2], inpseq_stack[i][k+1,21+25,0], inpseq_stack[i][k+1,21+25,1], inpseq_stack[i][k+1,21+25,2])
                dis_frms[47] = Distance3D(inpseq_stack[i][k,22+25,0], inpseq_stack[i][k,22+25,1], inpseq_stack[i][k,22+25,2], inpseq_stack[i][k+1,22+25,0], inpseq_stack[i][k+1,22+25,1], inpseq_stack[i][k+1,22+25,2])
                dis_frms[48] = Distance3D(inpseq_stack[i][k,23+25,0], inpseq_stack[i][k,23+25,1], inpseq_stack[i][k,23+25,2], inpseq_stack[i][k+1,23+25,0], inpseq_stack[i][k+1,23+25,1], inpseq_stack[i][k+1,23+25,2])
                dis_frms[49] = Distance3D(inpseq_stack[i][k,24+25,0], inpseq_stack[i][k,24+25,1], inpseq_stack[i][k,24+25,2], inpseq_stack[i][k+1,24+25,0], inpseq_stack[i][k+1,24+25,1], inpseq_stack[i][k+1,24+25,2])


                distance_desc_frms[:,k] = dis_frms[:]



        distance_desc_frms[:,:] = np.floor(255*( distance_desc_frms[:,:]- np.amin(distance_desc_frms[:,:]) )/(np.amax(distance_desc_frms[:,:])-np.amin(distance_desc_frms[:,:])))

        
        img=Image.fromarray(distance_desc_frms.astype(np.uint8),'L')
        img=img.resize((img_size,img_size))
        img2=img
        img.save(path_disf+str(i)+'_'+str(labels_stack[i])+'.png')
        


        ###  Angles  


        if inpseq_stack[i].shape[1]==25: 
            ang=np.zeros((14))
            angle_desc=np.zeros((14,inpseq_stack[i].shape[0]))

            for k in range(0,inpseq_stack[i].shape[0]):

                ang[0] = angle_3D(inpseq_stack[i][k,16,:],inpseq_stack[i][k,17,:], inpseq_stack[i][k,0,:])
                ang[1] = angle_3D(inpseq_stack[i][k,17,:],inpseq_stack[i][k,18,:], inpseq_stack[i][k,16,:])
                ang[2] = angle_3D(inpseq_stack[i][k,12,:],inpseq_stack[i][k,0,:], inpseq_stack[i][k,13,:])
                ang[3] = angle_3D(inpseq_stack[i][k,13,:],inpseq_stack[i][k,12,:], inpseq_stack[i][k,14,:])
                ang[4] = angle_3D(inpseq_stack[i][k,5,:],inpseq_stack[i][k,4,:], inpseq_stack[i][k,6,:])
                ang[5] = angle_3D(inpseq_stack[i][k,4,:],inpseq_stack[i][k,20,:], inpseq_stack[i][k,5,:])
                ang[6] = angle_3D(inpseq_stack[i][k,20,:],inpseq_stack[i][k,4,:], inpseq_stack[i][k,2,:])
                ang[7] = angle_3D(inpseq_stack[i][k,20,:],inpseq_stack[i][k,2,:], inpseq_stack[i][k,8,:]) 
                ang[8] = angle_3D(inpseq_stack[i][k,8,:],inpseq_stack[i][k,9,:], inpseq_stack[i][k,20,:])
                ang[9] = angle_3D(inpseq_stack[i][k,9,:],inpseq_stack[i][k,10,:], inpseq_stack[i][k,8,:])
                ang[10] = angle_3D(inpseq_stack[i][k,14,:],inpseq_stack[i][k,15,:], inpseq_stack[i][k,13,:])
                ang[11] = angle_3D(inpseq_stack[i][k,18,:],inpseq_stack[i][k,17,:], inpseq_stack[i][k,19,:])
                ang[12] = angle_3D(inpseq_stack[i][k,7,:],inpseq_stack[i][k,22,:], inpseq_stack[i][k,21,:])
                ang[13] = angle_3D(inpseq_stack[i][k,11,:],inpseq_stack[i][k,23,:], inpseq_stack[i][k,24,:])

                angle_desc[:,k] = ang[:]
    
        else: 

            ang=np.zeros((28))
            angle_desc=np.zeros((28,inpseq_stack[i].shape[0]))

            for k in range(0,inpseq_stack[i].shape[0]):

                ang[0] = angle_3D(inpseq_stack[i][k,16,:],inpseq_stack[i][k,17,:], inpseq_stack[i][k,0,:])
                ang[1] = angle_3D(inpseq_stack[i][k,17,:],inpseq_stack[i][k,18,:], inpseq_stack[i][k,16,:])
                ang[2] = angle_3D(inpseq_stack[i][k,12,:],inpseq_stack[i][k,0,:], inpseq_stack[i][k,13,:])
                ang[3] = angle_3D(inpseq_stack[i][k,13,:],inpseq_stack[i][k,12,:], inpseq_stack[i][k,14,:])
                ang[4] = angle_3D(inpseq_stack[i][k,5,:],inpseq_stack[i][k,4,:], inpseq_stack[i][k,7,:])
                ang[5] = angle_3D(inpseq_stack[i][k,4,:],inpseq_stack[i][k,20,:], inpseq_stack[i][k,5,:])
                ang[6] = angle_3D(inpseq_stack[i][k,20,:],inpseq_stack[i][k,4,:], inpseq_stack[i][k,2,:])
                ang[7] = angle_3D(inpseq_stack[i][k,20,:],inpseq_stack[i][k,2,:], inpseq_stack[i][k,8,:]) 
                ang[8] = angle_3D(inpseq_stack[i][k,8,:],inpseq_stack[i][k,9,:], inpseq_stack[i][k,20,:])
                ang[9] = angle_3D(inpseq_stack[i][k,9,:],inpseq_stack[i][k,10,:], inpseq_stack[i][k,8,:])
                ang[10] = angle_3D(inpseq_stack[i][k,14,:],inpseq_stack[i][k,15,:], inpseq_stack[i][k,13,:])
                ang[11] = angle_3D(inpseq_stack[i][k,18,:],inpseq_stack[i][k,17,:], inpseq_stack[i][k,19,:])
                ang[12] = angle_3D(inpseq_stack[i][k,7,:],inpseq_stack[i][k,22,:], inpseq_stack[i][k,21,:])
                ang[13] = angle_3D(inpseq_stack[i][k,11,:],inpseq_stack[i][k,23,:], inpseq_stack[i][k,24,:])
                ang[14]= angle_3D(inpseq_stack[i][k,16+25,:],inpseq_stack[i][k,17+25,:], inpseq_stack[i][k,0+25,:])
                ang[15]= angle_3D(inpseq_stack[i][k,17+25,:],inpseq_stack[i][k,18+25,:], inpseq_stack[i][k,16+25,:])
                ang[16]= angle_3D(inpseq_stack[i][k,12+25,:],inpseq_stack[i][k,0+25,:], inpseq_stack[i][k,13+25,:])
                ang[17]= angle_3D(inpseq_stack[i][k,13+25,:],inpseq_stack[i][k,12+25,:], inpseq_stack[i][k,14+25,:])
                ang[18]= angle_3D(inpseq_stack[i][k,5+25,:],inpseq_stack[i][k,4+25,:], inpseq_stack[i][k,7+25,:])
                ang[19]= angle_3D(inpseq_stack[i][k,4+25,:],inpseq_stack[i][k,20+25,:], inpseq_stack[i][k,5+25,:])
                ang[20]= angle_3D(inpseq_stack[i][k,20+25,:],inpseq_stack[i][k,4+25,:], inpseq_stack[i][k,2+25,:])
                ang[21]= angle_3D(inpseq_stack[i][k,20+25,:],inpseq_stack[i][k,2+25,:], inpseq_stack[i][k,8+25,:]) 
                ang[22]= angle_3D(inpseq_stack[i][k,8+25,:],inpseq_stack[i][k,9+25,:], inpseq_stack[i][k,20+25,:])
                ang[23]= angle_3D(inpseq_stack[i][k,9+25,:],inpseq_stack[i][k,10+25,:], inpseq_stack[i][k,8+25,:])
                ang[24] = angle_3D(inpseq_stack[i][k,14+25,:],inpseq_stack[i][k,15+25,:], inpseq_stack[i][k,13+25,:])
                ang[25] = angle_3D(inpseq_stack[i][k,18+25,:],inpseq_stack[i][k,17+25,:], inpseq_stack[i][k,19+25,:])
                ang[26] = angle_3D(inpseq_stack[i][k,7+25,:],inpseq_stack[i][k,22+25,:], inpseq_stack[i][k,21+25,:])
                ang[27] = angle_3D(inpseq_stack[i][k,11+25,:],inpseq_stack[i][k,23+25,:], inpseq_stack[i][k,24+25,:])

                angle_desc[:,k] = ang[:]
 
        angle_desc[:,:] = np.floor(255*( angle_desc[:,:]- np.amin(angle_desc[:,:]) )/(np.amax(angle_desc[:,:])-np.amin(angle_desc[:,:])))
        
        img=Image.fromarray(angle_desc.astype(np.uint8),'L')
        img=img.resize((img_size,img_size))
        img3=img
        img.save(path_ang+str(i)+'_'+str(labels_stack[i])+'.png')

        glob_desc=Image.merge("RGB", (img1,img2,img3))

        glob_desc.save(path_fuse+str(i)+'_'+str(labels_stack[i])+'.png')

        
    return 0





if __name__ == "__main__":

    dataset= args.dataset 
    eval_protocol= args.protocol         #   ntu120: csub  csetup   ntu60: csub  cview 
    npy_path = args.dataset_path         #  '.data/ntu/npy120skel/'   '.data/ntu/npy60skel/'
  
    inplist_skel, outlist_skel = data_ntu_skel(args.action , dataset, eval_protocol, npy_path )  


    path_disj=   args.descrip_path+ args.dataset+'_JDD_'+ args.action+'_'+ args.protocol +'/' 
    path_disf=   args.descrip_path+ args.dataset+'_ADD_'+ args.action+'_'+ args.protocol  +'/'
    path_ang=    args.descrip_path+ args.dataset+'_LAD_'+ args.action+'_'+ args.protocol  +'/'
    path_fuse=   args.descrip_path+ args.dataset+'_FDD_'+ args.action+'_'+ args.protocol  +'/'
    path_gcn=    args.descrip_path+ args.dataset+'_GD_'+ args.action+'_'+ args.protocol  +'/'

    img_size=112

    create_descriptors(inplist_skel, outlist_skel, path_gcn,path_fuse,path_disj,path_disf,path_ang,img_size)




