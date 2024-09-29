

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


def get_descriptors (path_disj, path_disf, path_ang, path_desc, path_gcn):

    disj_stack=[]
    disf_stack=[]
    ang_stack=[]
    desc_stack=[]
    gcn_stack=[]

    label_stack=[]



    for image_name in natsorted( os.listdir(path_disj)  ):


        if args.dataset=='ntu60':
            hot_lab=np.zeros(60)
        if args.dataset=='ntu120':
            hot_lab=np.zeros(120)   
        if args.dataset=='ucla':
            hot_lab=np.zeros(10)          

        hot_lab[:]=0.1

        path = os.path.join(path_disj,image_name)
        img  = cv2.imread(path)
        disj_stack.append(img)
  
        path = os.path.join(path_disf,image_name)
        img  = cv2.imread(path)
        disf_stack.append(img)

        path = os.path.join(path_ang,image_name)
        img  = cv2.imread(path)
        ang_stack.append(img)

        path = os.path.join(path_desc,image_name)
        img  = cv2.imread(path)
        desc_stack.append(img)

        path = os.path.join(path_gcn,image_name)
        img  = cv2.imread(path)
        gcn_stack.append(img)
   

        hot_lab[int(image_name.split("_",1)[1][:-4])-1]=0.8
        label_stack.append(hot_lab)


    return  gcn_stack,  desc_stack, disj_stack, disf_stack, ang_stack,  label_stack









class ActionDataset(Dataset):

    def __init__(self, inn_disj, inn_disf, inn_ang, inn_gcn, inn_desc, out ): 

        self.inn_gcn = gcn_stack
        self.inn_desc = desc_stack
        self.inn_disj = disj_stack
        self.inn_disf = disf_stack
        self.inn_ang = ang_stack
   
        self.out = label_stack
  
    
    def __len__(self):

        return len(self.out)

    def __getitem__(self, idx):

        inp_gcn = self.inn_gcn[idx]
        inp_desc = self.inn_desc[idx]
        inp_disj = self.inn_disj[idx]
        inp_disf = self.inn_disf[idx]
        inp_ang = self.inn_ang[idx]

        outp= self.out[idx]

        sample = {"GD": inp_gcn, "FDD": inp_desc,"JDD": inp_disj, "ADD": inp_disf,"LAD": inp_ang,  "Outp": outp}

        return sample





if args.dataset=='ucla':

    desc1_train_path=args.descrip_path+args.dataset+'_JDD_train/'
    desc2_train_path=args.descrip_path+args.dataset+'_ADD_train/'
    desc3_train_path=args.descrip_path+args.dataset+'_LAD_train/'
    desc4_train_path=args.descrip_path+args.dataset+'_FDD_train/'
    desc5_train_path=args.descrip_path+args.dataset+'_GD_train/'

    desc1_test_path=args.descrip_path+args.dataset+'_JDD_test/'
    desc2_test_path=args.descrip_path+args.dataset+'_ADD_test/'
    desc3_test_path=args.descrip_path+args.dataset+'_LAD_test/'
    desc4_test_path=args.descrip_path+args.dataset+'_FDD_test/'
    desc5_test_path=args.descrip_path+args.dataset+'_GD_test/'

else:

    desc1_train_path=args.descrip_path+args.dataset+'_JDD_train_'+args.protocol+'/'
    desc2_train_path=args.descrip_path+args.dataset+'_ADD_train_'+args.protocol+'/'
    desc3_train_path=args.descrip_path+args.dataset+'_LAD_train_'+args.protocol+'/'
    desc4_train_path=args.descrip_path+args.dataset+'_FDD_train_'+args.protocol+'/'
    desc5_train_path=args.descrip_path+args.dataset+'_GD_train_'+args.protocol+'/'

    desc1_test_path=args.descrip_path+args.dataset+'_JDD_test_'+args.protocol+'/'
    desc2_test_path=args.descrip_path+args.dataset+'_ADD_test_'+args.protocol+'/'
    desc3_test_path=args.descrip_path+args.dataset+'_LAD_test_'+args.protocol+'/'
    desc4_test_path=args.descrip_path+args.dataset+'_FDD_test_'+args.protocol+'/'
    desc5_test_path=args.descrip_path+args.dataset+'_GD_test_'+args.protocol+'/'


if args.action=='train':

    gcn_stack,  desc_stack, disj_stack, disf_stack, ang_stack,  label_stack = get_descriptors (desc1_train_path,  desc2_train_path, desc3_train_path, desc4_train_path ,desc5_train_path)
    input_output_df =({'input_gcn': gcn_stack, 'input_desc': desc_stack, 'input_disj': disj_stack, 'input_disf': disf_stack, 'input_ang': ang_stack,'output': label_stack })
    TDtrain = ActionDataset(input_output_df['input_gcn'],input_output_df['input_desc'],input_output_df['input_disj'],input_output_df['input_disf'],input_output_df['input_ang'],input_output_df['output'])

    gcn_stack,  desc_stack, disj_stack, disf_stack, ang_stack,  label_stack = get_descriptors (desc1_test_path,  desc2_test_path, desc3_test_path, desc4_test_path ,desc5_test_path)
    input_output_df =({'input_gcn': gcn_stack, 'input_desc': desc_stack, 'input_disj': disj_stack, 'input_disf': disf_stack, 'input_ang': ang_stack,'output': label_stack })
    TDtest = ActionDataset(input_output_df['input_gcn'],input_output_df['input_desc'],input_output_df['input_disj'],input_output_df['input_disf'],input_output_df['input_ang'],input_output_df['output'])

if args.action=='test':

    gcn_stack,  desc_stack, disj_stack, disf_stack, ang_stack,  label_stack = get_descriptors (desc1_test_path,  desc2_test_path, desc3_test_path, desc4_test_path ,desc5_test_path)
    input_output_df =({'input_gcn': gcn_stack, 'input_desc': desc_stack, 'input_disj': disj_stack, 'input_disf': disf_stack, 'input_ang': ang_stack,'output': label_stack })
    TDtest = ActionDataset(input_output_df['input_gcn'],input_output_df['input_desc'],input_output_df['input_disj'],input_output_df['input_disf'],input_output_df['input_ang'],input_output_df['output'])


















