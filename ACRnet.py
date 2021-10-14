# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 16:36:14 2020



@author: wby
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#---------------------------------模型-----------------------------------------------------
#定义make_layer
def make_layer(in_channel, out_channel, block_num, stride=1):
    shortcut = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 1, stride),
        nn.BatchNorm2d(out_channel)
    )
    
    layers = list()
    
    layers.append(ResBlock(in_channel, out_channel, stride, shortcut))

    for i in range(1, block_num):
        layers.append(ResBlock(out_channel, out_channel))
    return nn.Sequential(*layers)



class ACRBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, shortcut=None, shortcut2=None):
        super(ACRBlock, self).__init__()
        
        self.left1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel*2, 3, stride, 1, bias=False),
            nn.BatchNorm2d(in_channel*2),
            nn.ReLU(True),
            
        )
        self.left2 = nn.Sequential(
            nn.Conv2d(in_channel*2, in_channel*4, 1, 1),
            nn.BatchNorm2d(in_channel*4),
          #  nn.ReLU(True),    
                )
        self.left3 = nn.Sequential(
            nn.Conv2d(in_channel*4, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
         #   nn.ReLU(True),   
                )
        self.left4 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel*2, 1, 1),
            nn.BatchNorm2d(out_channel*2),
        #    nn.ReLU(True),
            
        )
        self.left5 = nn.Sequential(
            nn.Conv2d(out_channel*2, out_channel*4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel*4),
        #    nn.ReLU(True),    
                )
        self.left6 = nn.Sequential(
            nn.Conv2d(out_channel*4, out_channel,1, 1),
            nn.BatchNorm2d(out_channel),
          #  nn.ReLU(True),   
                )
        
        self.right = shortcut
       
        
        self.se = nn.Sequential(  
            nn.AdaptiveAvgPool2d((1,1)),    
            nn.Conv2d(in_channel,out_channel//4,kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_channel//4,in_channel*4,kernel_size=1),
            nn.Sigmoid(),
            
        )
        self.se2 = nn.Sequential(  
            nn.AdaptiveAvgPool2d((1,1)),    
            nn.Conv2d(in_channel*4,in_channel//4,kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channel//4,out_channel*2,kernel_size=1),
            nn.Sigmoid(),
            
        )
        self.se3 = nn.Sequential( 
            nn.AdaptiveAvgPool2d((1,1)),    
            nn.Conv2d(out_channel*2,out_channel//4,kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_channel//4,out_channel,kernel_size=1),
            nn.Sigmoid(),
            
        )
    def forward(self, x):
        x1 = self.se(x)    #*******
        
        x2 = self.left1(x)  #**********
        x2 = self.left2(x2) #**********
        x1a = self.se2(x2)
        out1 = x1*x2         #**************
        
        x2 = self.left3(out1)     #*********
        residual = x if self.right is None else self.right(x) #****************
        
        x2 += residual  #*************
        x2 = self.left4(x2)
        
        x1b = self.se3(x2)
        out2 = x1a*x2
        x2 = self.left5(out2)
        x2 = self.left6(x2)
        x4 = x1b*x2
                
        x4 += residual
        
        return F.relu(x4)



class ACRnet(nn.Module):
    def __init__(self,num_classes=100):
        super(ACRnet, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 15, 7, 2, 3, bias=False), 
            nn.BatchNorm2d(15),
            nn.ReLU(True), 
            nn.MaxPool2d(3, 2, 1)
        )
        self.layer1 = make_layer(15, 15, 2, stride=1)
        self.layer2 = make_layer(15, 30, 3, stride=2)
        self.layer3 = make_layer(30, 45, 4, stride=2)
        self.layer4 = make_layer(45, 50, 4, stride=2)
     #   self.avg = nn.AvgPool2d(7)
        self.classifier = nn.Sequential(nn.Linear(50*7*7, num_classes))

    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
  #     x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
  #     x = F.softmax(x, dim = 1)
        return x

if __name__=='__main__':

    model = ACRet()
    print(model)