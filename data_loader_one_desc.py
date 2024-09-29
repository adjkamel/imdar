

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



def get_descriptors (desc_path):

    desc_stack=[]

    label_stack=[]

    for image_name in natsorted( os.listdir(desc_path)  ):

        if args.dataset=='ntu60':
            hot_lab=np.zeros(60)
        if args.dataset=='ntu120':
            hot_lab=np.zeros(120)   
        if args.dataset=='ucla':
            hot_lab=np.zeros(10)           

        hot_lab[:]=0.1

        path = os.path.join(desc_path,image_name)
        img  = cv2.imread(path)
        desc_stack.append(img)


        hot_lab[int(image_name.split("_",1)[1][:-4])-1]=0.8
        label_stack.append(hot_lab)

        #label_stack.append(int(image_name.split("_",1)[1][:-4])-1)
    
    return   desc_stack, label_stack


class ActionDataset(Dataset):

    def __init__(self,  inn_desc, out ): 

        #self.inn_gcn = gcn_stack
        self.inn_desc = desc_stack
        self.out = label_stack
  
    
    def __len__(self):

        return len(self.out)

        # , len(self.output3Depth), len(self.output3DGlobal), len(self.output3DLocal), len(self.output3DAngles),

    def __getitem__(self, idx):
        inp_desc = self.inn_desc[idx]
        outp= self.out[idx]
        sample = { "Inp_desc": inp_desc,  "Outp": outp}
        return sample




desc_train_path=args.descrip_path+args.dataset+'_'+args.descriptor+'_train_'+args.protocol+'/'
desc_test_path=args.descrip_path+args.dataset+'_'+args.descriptor+'_test_'+args.protocol+'/'


if args.action=='train':

    desc_stack,  label_stack = get_descriptors (desc_train_path)
    input_output_df =({ 'input_desc': desc_stack,'output': label_stack })
    TDtrain = ActionDataset(input_output_df['input_desc'],input_output_df['output'])

    desc_stack, label_stack = get_descriptors (desc_test_path)
    input_output_df =({'input_desc': desc_stack, 'output': label_stack })
    TDtest = ActionDataset(input_output_df['input_desc'],input_output_df['output'])


if args.action=='test':

    desc_stack, label_stack = get_descriptors (desc_test_path)
    input_output_df =({'input_desc': desc_stack, 'output': label_stack })
    TDtest = ActionDataset(input_output_df['input_desc'],input_output_df['output'])




