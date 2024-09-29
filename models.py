import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn import Linear
import pywt
from arguments import args
 
 
if torch.cuda.is_available():
    device = torch.device("cuda:"+ str(args.gpu))
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device("cpu")


class ResBlock(nn.Module):

    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()

        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:

            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
            
    def forward(self, x):

        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)
        
        return out



class Fuse5(nn.Module):

    def __init__(self, num_classes=60):
        super(Fuse5, self).__init__()
        self.inchannel = 256
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResBlock, 512, 2, stride=1)
        self.layer2 = self.make_layer(ResBlock, 1024, 2, stride=2)
        self.layer3 = self.make_layer(ResBlock, 512, 2, stride=2)        
        self.layer4 = self.make_layer(ResBlock, 1024, 2, stride=2)        
        self.fc = nn.Linear(1024, num_classes)
        self.convf = nn.Conv2d(5120, num_classes, kernel_size=1,  bias=False)
        self.convf1 = nn.Conv2d(1024, num_classes, kernel_size=1,  bias=False)
        self.convf12 = nn.Conv2d(2048, num_classes, kernel_size=1,  bias=False)
        self.convf345 = nn.Conv2d(3072, num_classes, kernel_size=1,  bias=False)


    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    def forward(self, x1,x2,x3,x4,x5):


        out1 = self.conv1(x1)
        out1 = F.avg_pool2d(out1, 4) 
        out1 = self.layer1(out1)
        out1 = F.avg_pool2d(out1, 4) 
        out1 = self.layer2(out1)
        out1 = F.avg_pool2d(out1, 4) 
        feat1=out1

        out2 = self.conv1(x2)
        out2 = F.avg_pool2d(out2, 4) 
        out2 = self.layer1(out2)
        out2 = F.avg_pool2d(out2, 4) 
        out2 = self.layer2(out2)
        out2 = F.avg_pool2d(out2, 4) 
        feat2=out2

        out3 = self.conv1(x3)
        out3 = F.avg_pool2d(out3, 4) 
        out3 = self.layer1(out3)
        out3 = F.avg_pool2d(out3, 4) 
        out3 = self.layer2(out3)
        out3 = F.avg_pool2d(out3, 4) 
        feat3=out3

        out4 = self.conv1(x4)
        out4 = F.avg_pool2d(out4, 4) 
        out4 = self.layer1(out4)
        out4 = F.avg_pool2d(out4, 4) 
        out4 = self.layer2(out4)
        out4 = F.avg_pool2d(out4, 4) 
        feat4=out4

        out5 = self.conv1(x5)
        out5 = F.avg_pool2d(out5, 4) 
        out5 = self.layer1(out5)
        out5 = F.avg_pool2d(out5, 4) 
        out5 = self.layer2(out5)
        out5 = F.avg_pool2d(out5, 4) 
        feat5=out5

        out =  torch.cat((out1,out2,out3,out4,out5),1)
        featall5=out
        out = self.convf(out)
        out = out.view(out.size(0), -1)
 
        out12 =  torch.cat((out1,out2),1)
        feat12=out12
        out12 = self.convf12(out12)
        out12 = out12.view(out12.size(0), -1)

        out345 =  torch.cat((out3,out4,out5),1)
        feat345=out345
        out345 = self.convf345(out345)
        out345 = out345.view(out345.size(0), -1)

        out1 = self.convf1(out1)
        out1 = out1.view(out1.size(0), -1)

        out2 = self.convf1(out2)
        out2 = out2.view(out2.size(0), -1)

        out3 = self.convf1(out3)
        out3 = out3.view(out3.size(0), -1)

        out4 = self.convf1(out4)
        out4 = out4.view(out4.size(0), -1)

        out5 = self.convf1(out5)
        out5 = out5.view(out5.size(0), -1)

        return out, out1,out2,out3,out4,out5,out12,out345, feat1, feat2, feat3, feat4, feat5, feat12, feat345, featall5


class ResNet_c(nn.Module):

    def __init__(self, num_classes=120):
        super(ResNet_c, self).__init__()
        self.inchannel = 256
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResBlock, 512, 2, stride=1)
        self.layer2 = self.make_layer(ResBlock, 1024, 2, stride=2)
        self.layer3 = self.make_layer(ResBlock, 512, 2, stride=2)        
        self.layer4 = self.make_layer(ResBlock, 1024, 2, stride=2)        
        self.fc = nn.Linear(1024, num_classes)
        self.convf = nn.Conv2d(1024, num_classes, kernel_size=1,  bias=False)
  
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    def forward(self, x):

        out = self.conv1(x)
        out = F.avg_pool2d(out, 4) 
        out = self.layer1(out)
        out = F.avg_pool2d(out, 4) 
        out = self.layer2(out)
        out = F.avg_pool2d(out, 4) 
        out = self.convf(out)
        out = out.view(out.size(0), -1)
        
        return out


class ResNet_c_feat(nn.Module):
    def __init__(self, num_classes=60):
        super(ResNet_c_feat, self).__init__()
        self.inchannel = 256
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResBlock, 512, 2, stride=1)
        self.layer2 = self.make_layer(ResBlock, 1024, 2, stride=2)
        self.layer3 = self.make_layer(ResBlock, 512, 2, stride=2)        
        self.layer4 = self.make_layer(ResBlock, 1024, 2, stride=2)        
        self.fc = nn.Linear(1024, num_classes)
        self.convf = nn.Conv2d(1024, num_classes, kernel_size=1,  bias=False)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    def forward(self, x):

        out = self.conv1(x)
        out = F.avg_pool2d(out, 4) 
        out = self.layer1(out)
        out = F.avg_pool2d(out, 4) 
        feat_layer2 = out
        out = self.layer2(out)
        out = F.avg_pool2d(out, 4) 
        out = self.convf(out)
        out = out.view(out.size(0), -1)

        return out, feat_layer2


class ResNet(nn.Module):
    def __init__(self, num_classes=60):
        super(ResNet, self).__init__()
        self.inchannel = 128
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResBlock, 128, 2, stride=1)
        self.layer2 = self.make_layer(ResBlock, 256, 2, stride=2)
        self.layer3 = self.make_layer(ResBlock, 512, 2, stride=2)        
        self.layer4 = self.make_layer(ResBlock, 1024, 2, stride=2)        
        self.fc = nn.Linear(1024, num_classes)
        self.convf = nn.Conv2d(1024, num_classes, kernel_size=1,  bias=False)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
     
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = F.avg_pool2d(out, 4) 
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

