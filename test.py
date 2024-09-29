# !/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
import torch.nn as nn
import torch
from models import ResNet
from models import ResNet_c
from models import ResNet_c_feat
from models import Fuse5
import numpy as np
from torch.utils.data import DataLoader,RandomSampler
import matplotlib.pyplot as plt
from arguments import args
from data_loader_all_desc import  TDtest


if torch.cuda.is_available():
    device = torch.device("cuda:"+ str(args.gpu))
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")



def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal(m.weight)


def test():
  
    model_gcn = ResNet_c(num_classes=num_cls).to(device)
    model_desc= ResNet_c(num_classes=num_cls).to(device)
    model_disj = ResNet_c(num_classes=num_cls).to(device)
    model_disf= ResNet_c(num_classes=num_cls).to(device)
    model_ang = ResNet_c(num_classes=num_cls).to(device)
    model_fuse5 = Fuse5(num_classes=num_cls).to(device)

    

    model_gcn.load_state_dict(torch.load( path + 'GD_epoch_best.tar' , map_location=device))
    model_desc.load_state_dict(torch.load( path + 'FDD_epoch_best.tar' , map_location=device))
    model_disj.load_state_dict(torch.load( path + 'JDD_epoch_best.tar' , map_location=device))
    model_disf.load_state_dict(torch.load( path + 'ADD_epoch_best.tar' , map_location=device))
    model_ang.load_state_dict(torch.load( path + 'LAD_epoch_best.tar' , map_location=device))
    model_fuse5.load_state_dict(torch.load( path + 'all_epoch_best.tar' , map_location=device))
    
        
    model_gcn.eval()
    model_desc.eval()
    model_disj.eval()
    model_disf.eval()
    model_ang.eval()
    model_fuse5.eval()
 
   
    test_loader = DataLoader(dataset=TDtest, batch_size=batch_size , shuffle=True)


    with torch.no_grad():

        epoch_val_accuracy = 0

        ct=0

        for k, datatest in enumerate(test_loader):

            input_gcn = datatest['GD'].to(device)
            input_desc = datatest['FDD'].to(device)
            input_disj = datatest['JDD'].to(device)
            input_disf = datatest['ADD'].to(device)
            input_ang = datatest['LAD'].to(device)

            #input_dist = datatest['Inp_dist'].to(device)

            label = datatest['Outp'].to(device)


            input_gcn= torch.permute(input_gcn,(0,3,1,2)).float()
            input_desc= torch.permute(input_desc,(0,3,1,2)).float()
            input_disj= torch.permute(input_disj,(0,3,1,2)).float()
            input_disf= torch.permute(input_disf,(0,3,1,2)).float()
            input_ang= torch.permute(input_ang,(0,3,1,2)).float()



            out_gcn=model_gcn(input_gcn)
            out_desc=model_desc(input_desc)
            out_disj=model_disj(input_disj)
            out_disf=model_disf(input_disf)
            out_ang=model_ang(input_ang)

            out_fus5, output1,output2,output3,output4,output5,output12,output345, f1, f2, f3, f4, f5, f12, f345, fall5 =model_fuse5(input_gcn,input_desc,input_disj,input_disf,input_ang)

            
            
            
            acc=0

            list_pred=[]

            for i in range(0,label.size(0)):
            
                


                larg=[]
                      
                larg.append(out_fus5[i,:].argmax())
                larg.append(out_gcn[i,:].argmax())
                larg.append(out_desc[i,:].argmax())
                larg.append(out_disj[i,:].argmax())
                larg.append(out_disf[i,:].argmax())
                larg.append(out_ang[i,:].argmax())


                ### Voting and Priority Score Selection  (voting + priority based on individual descriptor accuracy)

                if (larg[0]==larg[1]==larg[2]==larg[3]==larg[4]==larg[5]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[0]==larg[1]==larg[2]==larg[3]==larg[4]==label[i,:].argmax()):
                    acc+=1
                elif (larg[0]==larg[1]==larg[2]==larg[3]==larg[5]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[0]==larg[1]==larg[2]==larg[4]==larg[5]==label[i,:].argmax()):
                    acc+=1
                elif (larg[0]==larg[1]==larg[3]==larg[4]==larg[5]==label[i,:].argmax()):
                    acc+=1
                elif (larg[0]==larg[2]==larg[3]==larg[4]==larg[5]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[1]==larg[2]==larg[3]==larg[4]==larg[5]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[0]==larg[1]==larg[2]==larg[3]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[0]==larg[1]==larg[2]==larg[4]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[0]==larg[1]==larg[2]==larg[5]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[0]==larg[1]==larg[3]==larg[4]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[0]==larg[1]==larg[3]==larg[5]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[0]==larg[1]==larg[4]==larg[5]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[0]==larg[2]==larg[3]==larg[4]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[0]==larg[2]==larg[3]==larg[5]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[0]==larg[2]==larg[4]==larg[5]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[0]==larg[3]==larg[4]==larg[5]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[1]==larg[2]==larg[3]==larg[4]==label[i,:].argmax()):
                    acc+=1
                elif (larg[1]==larg[2]==larg[3]==larg[5]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[1]==larg[2]==larg[4]==larg[5]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[1]==larg[3]==larg[4]==larg[5]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[2]==larg[3]==larg[4]==larg[5]==label[i,:].argmax()):
                    acc+=1
                elif (larg[0]==larg[1]==larg[2]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[0]==larg[1]==larg[3]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[0]==larg[1]==larg[4]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[0]==larg[1]==larg[5]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[0]==larg[2]==larg[3]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[0]==larg[2]==larg[4]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[0]==larg[2]==larg[5]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[0]==larg[3]==larg[4]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[0]==larg[3]==larg[5]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[0]==larg[4]==larg[5]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[1]==larg[2]==larg[3]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[1]==larg[2]==larg[4]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[1]==larg[2]==larg[5]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[1]==larg[3]==larg[4]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[1]==larg[3]==larg[5]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[1]==larg[4]==larg[5]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[2]==larg[3]==larg[4]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[2]==larg[3]==larg[5]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[2]==larg[4]==larg[5]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[3]==larg[4]==larg[5]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[0]==larg[1]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[0]==larg[2]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[0]==larg[3]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[0]==larg[4]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[0]==larg[5]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[1]==larg[2]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[1]==larg[3]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[1]==larg[4]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[1]==larg[5]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[2]==larg[3]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[2]==larg[4]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[2]==larg[5]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[3]==larg[4]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[3]==larg[5]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[4]==larg[5]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[0]!=larg[1]==label[i,:].argmax()): 
                     acc+=1
                elif (larg[0]!=larg[2]==label[i,:].argmax()): 
                     acc+=1
                elif (larg[0]!=larg[3]==label[i,:].argmax()): 
                    acc+=1 
                elif (larg[0]!=larg[4]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[0]!=larg[5]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[1]!=larg[2]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[1]!=larg[3]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[1]!=larg[4]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[1]!=larg[5]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[2]!=larg[3]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[2]!=larg[4]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[2]!=larg[5]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[3]!=larg[4]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[3]!=larg[5]==label[i,:].argmax()): 
                    acc+=1
                elif (larg[4]!=larg[5]==label[i,:].argmax()): 
                    acc+=1
                else:
                    pass


            acc=acc/label.size(0)
            
            epoch_val_accuracy += acc / len(test_loader)

    print(ct)

    print("accuracy = ",epoch_val_accuracy)
   

if __name__ == '__main__':

    args.action = 'test'

    batch_size = 32
    epochs = 30
    lr = 0.0001
    gamma = 0.5
    step_size=4


    if args.dataset=='ntu60':
        num_cls=60
    if args.dataset=='ntu120':
        num_cls=120
    if args.dataset=='ucla':
        num_cls=10


    if args.dataset=='ucla':
        path='./checkpoints/'+ args.dataset+'_'
    else:
        path='./checkpoints/'+ args.dataset+'_'+args.protocol+'_'

    test()
    
