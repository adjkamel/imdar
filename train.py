# !/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import 
from __future__ import print_function
import torch.nn as nn
import torch
import time
from models import ResNet_c
from models import Fuse5
from torch.utils.data import DataLoader,RandomSampler
import matplotlib.pyplot as plt
from arguments import args


if args.descriptor=='all':
    from data_loader_all_desc import TDtrain, TDtest
else:
    from data_loader_one_desc import TDtrain, TDtest



if torch.cuda.is_available():
    device = torch.device("cuda:"+ str(args.gpu))
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")



def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal(m.weight)


def train_fuse5(step_size,gamma):


    model = Fuse5(num_classes=num_cls).to(device)
    model.apply(weight_init)
    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if args.dataset!='ucla':
        scheduler =torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    train_loader = DataLoader(dataset=TDtrain, batch_size=batch_size , shuffle=True)
    test_loader = DataLoader(dataset=TDtest, batch_size=batch_size , shuffle=True)



    for epoch in range(epochs):

        print(f'Starting epoch {epoch + 1}')

        epoch_loss = 0
        epoch_accuracy = 0

        for i, data in enumerate(train_loader):


            input_gcn = data['GD'].to(device)
            input_desc = data['FDD'].to(device)
            input_disj = data['JDD'].to(device)
            input_disf = data['ADD'].to(device)
            input_ang = data['LAD'].to(device)

            label = data['Outp'].to(device)
         
            
            input_gcn= torch.permute(input_gcn,(0,3,1,2)).float()
            input_desc= torch.permute(input_desc,(0,3,1,2)).float()
            input_disj= torch.permute(input_disj,(0,3,1,2)).float()
            input_disf= torch.permute(input_disf,(0,3,1,2)).float()
            input_ang= torch.permute(input_ang,(0,3,1,2)).float()

            optimizer.zero_grad()

            output, output1,output2,output3,output4,output5,output12,output345, f1, f2, f3, f4, f5, f12, f345, fall5 = model(input_gcn,input_desc,input_disj,input_disf,input_ang)
        
            loss = loss_function(output, label) 
            loss1 = loss_function(output1, label) 
            loss2 = loss_function(output2, label) 
            loss3 = loss_function(output3, label) 
            loss4 = loss_function(output4, label) 
            loss5 = loss_function(output5, label) 
            loss12 = loss_function(output12, label) 
            loss345 = loss_function(output345, label) 


            lossg=loss+loss1+loss2+loss3+loss4+loss5+loss12+loss345

            lossg.backward()
  
            optimizer.step()

            acc = (output.argmax(dim=1) == label.argmax(dim=1)).float().mean()

            epoch_accuracy += acc / len(train_loader)
            epoch_loss += lossg / len(train_loader)

        with torch.no_grad():

            epoch_val_accuracy = 0
            epoch_val_accuracy1 = 0
            epoch_val_accuracy2 = 0
            epoch_val_accuracy3 = 0
            epoch_val_accuracy4 = 0
            epoch_val_accuracy5 = 0
            epoch_val_accuracy12 = 0
            epoch_val_accuracy345 = 0


            epoch_val_loss = 0

            for k, datatest in enumerate(test_loader):

                input_gcn = datatest['GD'].to(device)
                input_desc = datatest['FDD'].to(device)
                input_disj = datatest['JDD'].to(device)
                input_disf = datatest['ADD'].to(device)
                input_ang = datatest['LAD'].to(device)

                label = datatest['Outp'].to(device)
            
                
                input_gcn= torch.permute(input_gcn,(0,3,1,2)).float()
                input_desc= torch.permute(input_desc,(0,3,1,2)).float()
                input_disj= torch.permute(input_disj,(0,3,1,2)).float()
                input_disf= torch.permute(input_disf,(0,3,1,2)).float()
                input_ang= torch.permute(input_ang,(0,3,1,2)).float()

        
                val_output, val_output1,val_output2,val_output3,val_output4,val_output5,val_output12,val_output345, f1, f2, f3, f4, f5, f12, f345, fall5 = model(input_gcn,input_desc,input_disj,input_disf,input_ang)

                val_loss = loss_function(val_output, label)
                val_loss1 = loss_function(val_output1, label) 
                val_loss2 = loss_function(val_output2, label) 
                val_loss3 = loss_function(val_output3, label) 
                val_loss4 = loss_function(val_output4, label) 
                val_loss5 = loss_function(val_output5, label) 
                val_loss12 = loss_function(val_output12, label) 
                val_loss345 = loss_function(val_output345, label) 


                val_lossg=val_loss+val_loss1+val_loss2+val_loss3+val_loss4+val_loss5+val_loss12+val_loss345
 

                acc = (val_output.argmax(dim=1) == label.argmax(dim=1)).float().mean()
                epoch_val_accuracy += acc / len(test_loader)

                acc1 = (val_output1.argmax(dim=1) == label.argmax(dim=1)).float().mean()
                epoch_val_accuracy1 += acc1 / len(test_loader)

                acc2 = (val_output2.argmax(dim=1) == label.argmax(dim=1)).float().mean()
                epoch_val_accuracy2 += acc2 / len(test_loader)

                acc3 = (val_output3.argmax(dim=1) == label.argmax(dim=1)).float().mean()
                epoch_val_accuracy3 += acc3 / len(test_loader)

                acc4 = (val_output4.argmax(dim=1) == label.argmax(dim=1)).float().mean()
                epoch_val_accuracy4 += acc4 / len(test_loader)

                acc5 = (val_output5.argmax(dim=1) == label.argmax(dim=1)).float().mean()
                epoch_val_accuracy5 += acc5 / len(test_loader)

                acc12 = (val_output12.argmax(dim=1) == label.argmax(dim=1)).float().mean()
                epoch_val_accuracy12 += acc12 / len(test_loader)

                acc345 = (val_output345.argmax(dim=1) == label.argmax(dim=1)).float().mean()
                epoch_val_accuracy345 += acc345 / len(test_loader)

                
                epoch_val_loss += val_lossg / len(test_loader)

        if args.dataset!='ucla':

            if (epoch+1)==5:
                    gamma=0.5
                    step_size=1
                    scheduler =torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            if (epoch+1)==8:
                    gamma=0.5
                    step_size=3
                    scheduler =torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
                

            scheduler.step()

            print("Lr:", scheduler.get_lr())

        print(f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n")
   
        print("epoch_val_accuracy1:", epoch_val_accuracy1)
        print("epoch_val_accuracy2:", epoch_val_accuracy2)
        print("epoch_val_accuracy3:", epoch_val_accuracy3)
        print("epoch_val_accuracy4:", epoch_val_accuracy4)
        print("epoch_val_accuracy5:", epoch_val_accuracy5)
        print("epoch_val_accuracy12:", epoch_val_accuracy12)
        print("epoch_val_accuracy345:", epoch_val_accuracy345)

        #if (epoch % 50) or (epoch_loss <= 0.005): 
        torch.save(model.state_dict(),  path+ str(epoch + 1) + "_epoch.tar")


def train_one(step_size,gamma):

    model = ResNet_c(num_classes=num_cls).to(device)
 
    model.apply(weight_init)

    loss_function = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if args.dataset!='ucla':

        scheduler =torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    train_loader = DataLoader(dataset=TDtrain, batch_size=batch_size , shuffle=True)

    test_loader = DataLoader(dataset=TDtest, batch_size=batch_size , shuffle=True)


    class_list=[0 for c in range(0 , 60)]

    for epoch in range(epochs):

        print(f'Starting epoch {epoch + 1}')

        epoch_loss = 0
        epoch_accuracy = 0


        for i, data in enumerate(train_loader):


            input_desc = data['Inp_desc'].to(device)

            label = data['Outp'].to(device)
         
            input_desc= torch.permute(input_desc,(0,3,1,2)).float()

            #print(inputs.size())            
            optimizer.zero_grad()
   
            output = model(input_desc)
            
            loss = loss_function(output, label) 

            loss.backward()
  
            optimizer.step()


            acc = (output.argmax(dim=1) == label.argmax(dim=1)).float().mean()

            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)

        with torch.no_grad():

            epoch_val_accuracy = 0
            epoch_val_loss = 0

            for k, datatest in enumerate(test_loader):

                input_desc = datatest['Inp_desc'].to(device)

                label = datatest['Outp'].to(device)

                input_desc= torch.permute(input_desc,(0,3,1,2)).float()

                val_output = model(input_desc)
                
                val_loss = loss_function(val_output, label)

                acc = (val_output.argmax(dim=1) == label.argmax(dim=1)).float().mean()

                epoch_val_accuracy += acc / len(test_loader)

                
                epoch_val_loss += val_loss / len(test_loader)

        if args.dataset!= 'ucla':

            if (epoch+1)==5:
                    gamma=0.5
                    step_size=1
                    scheduler =torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            if (epoch+1)==8:
                    gamma=0.5
                    step_size=3
                    scheduler =torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

            scheduler.step()

        print(f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n")

        #if (epoch % 50) or (epoch_loss <= 0.005):
        torch.save(model.state_dict(),  path+ str(epoch + 1) + "_epoch.tar")




if __name__ == '__main__':


    args.action = 'train'

   
    batch_size = 32
    epochs = 500
    lr = 0.0001
    gamma = 0.5
    step_size=5

     


    if args.dataset=='ntu60':
        num_cls=60
    if args.dataset=='ntu120':
        num_cls=120
    if args.dataset=='ucla':
        num_cls=10




    if args.dataset=='ucla':
        path='./checkpoints/'+ args.dataset+'_'+args.descriptor+'_'
    else:
        path='./checkpoints/'+ args.dataset+'_'+args.protocol+'_'+args.descriptor+'_'

    
    if args.descriptor=='all':
        train_fuse5(step_size,gamma)
    else:
        train_one(step_size,gamma)

    


    
