# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 18:00:40 2020

@author: Administrator
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo
import numpy as np


class DSPNet(nn.Module):
    def __init__(self):
        super(DSPNet,self).__init__()
        
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)    #downsample noise img
        self.nam = NAM()    #Noise Estimation Network 
        self.fpn = FPN()    #DeGamma Noise Network
        
    def forward(self,x):
        
        x_128 = x
        
        enlm = self.nam(x)    #Estimation Noise Level Map
        concat_img = torch.cat([x,enlm], dim=1)
        res_128 = self.fpn(concat_img)    #Assume the size of the input image is (128,128).
        out_128 = res_128 + x_128
        
        return enlm,out_128

class ASPP(nn.Module):
    def __init__(self,features=256, out_features=256):
        super(ASPP,self).__init__()
        
        self.conv_1x1_1 = nn.Sequential(
                nn.Conv2d(256,64,kernel_size=1),
                nn.BatchNorm2d(64),
                nn.ReLU()
                )
        
        self.conv_3x3_6 = nn.Sequential(
                nn.Conv2d(256,64,kernel_size=3, stride=1, padding=2, dilation=2),
                nn.BatchNorm2d(64),
                nn.ReLU()
                )

        self.conv_3x3_12 = nn.Sequential(
                nn.Conv2d(256,64,kernel_size=3, stride=1, padding=3, dilation=3),
                nn.BatchNorm2d(64),
                nn.ReLU()
                )

        self.conv_3x3_18 = nn.Sequential(
                nn.Conv2d(256,64,kernel_size=3, stride=1, padding=6, dilation=6),
                nn.BatchNorm2d(64),
                nn.ReLU()
                )
        
        self.conv_out = nn.Sequential(
                nn.Conv2d(320,256,kernel_size=1),
                nn.BatchNorm2d(256),
                nn.ReLU()
                )

        
        self.avg_pool = nn.AvgPool2d(kernel_size=3,stride=2,padding=1)
        
    def forward(self,x):
        h,w = x.size(2),x.size(3)
        out_1x1    = self.conv_1x1_1(x)
        out_3x3_6  = self.conv_3x3_6(x)
        out_3x3_12 = self.conv_3x3_12(x)
        out_3x3_18 = self.conv_3x3_18(x)
        out_img = self.avg_pool(x)       
        out_img = self.conv_1x1_1(out_img)
        out_img = F.upsample(out_img,size=(h,w),mode = 'bilinear')
        
        out = torch.cat([out_1x1, out_3x3_6, out_3x3_12, out_3x3_18, out_img], 1)
        out = self.conv_out(out)
        
        return out

class AS3_64(nn.Module):
    def __init__(self,features=64, out_features=64):
        super(AS3_64,self).__init__()

        self.conv_1x1_1 = nn.Sequential(
                nn.Conv2d(64,32,kernel_size=1, stride=1, padding=1, dilation=1),nn.BatchNorm2d(32),nn.ReLU()
                )        

        self.conv_3x3_2 = nn.Sequential(
                nn.Conv2d(64,32,kernel_size=3, stride=1, padding=3, dilation=3),nn.BatchNorm2d(32),nn.ReLU()
                )

        self.conv_3x3_3 = nn.Sequential(
                nn.Conv2d(64,32,kernel_size=3, stride=1, padding=6, dilation=6),nn.BatchNorm2d(32),nn.ReLU()
                )
        
        self.conv_out = nn.Sequential(
                nn.Conv2d(96,64,kernel_size=1),nn.BatchNorm2d(64),nn.ReLU()
                )
        self.avg_pool = nn.AvgPool2d(kernel_size=3,stride=2,padding=1)
        
        
    def forward(self,x):
        h,w = x.size(2),x.size(3)        
        out_3x3_2 = self.conv_3x3_2(x)
        out_3x3_3 = self.conv_3x3_3(x)

        out_avg = self.avg_pool(x)       
        out_avg = self.conv_1x1_1(out_avg)
        out_avg = F.upsample(out_avg,size=(h,w),mode = 'bilinear')
        
        out = torch.cat([out_avg,out_3x3_2, out_3x3_3], 1)
        out = self.conv_out(out)
        
        return out

class NAM(nn.Module):
    def __init__(self):
        super(NAM,self).__init__()
        
        self.conv_in = nn.Sequential(
                nn.Conv2d(1, 64,kernel_size=3,stride=1,padding=1,bias=False),nn.BatchNorm2d(64),nn.ReLU()
                )
        
        self.conv_mid = nn.Sequential(
                nn.Conv2d(64, 64,kernel_size=3,stride=1,padding=1,bias=False),nn.BatchNorm2d(64),nn.ReLU()              
                )
        
        self.conv_sigmoid = nn.Sequential(
                nn.Conv2d(64,1,kernel_size=3,stride=1,padding=1,bias=False),nn.Sigmoid()
                )
        
        self.conv_out = nn.Sequential(
                nn.Conv2d(64, 64,kernel_size=3,stride=1,padding=1,bias=False),nn.BatchNorm2d(64),nn.ReLU()
                )
        
        self.as3 = AS3_64()
        
    def forward(self,x_in):
        x = self.conv_in(x_in)
        #x = self.as3(x)
        x = self.conv_mid(x)
        x = self.conv_mid(x)
        x = self.conv_mid(x) 
        x = self.conv_mid(x)        
        nam = self.conv_sigmoid(x) * x_in + x_in
        
        return nam
               

class FPN(nn.Module):
    def __init__(self):
        super(FPN,self).__init__()
        
        self.conv_in = nn.Conv2d(2,64,kernel_size=3,stride=1,padding=1,bias=False)        
        self.conv_64_64 = nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1,bias=False)       
        self.conv_64_128 = nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1,bias=False)       
        self.conv_128_128 = nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1,bias=False)     
        self.conv_128_256 = nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1,bias=False)      
        self.conv_256_256 = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1,bias=False)    
        self.conv_256_128 = nn.Conv2d(256,128,kernel_size=3,stride=1,padding=1,bias=False)
        self.conv_128_64 = nn.Conv2d(128,64,kernel_size=3,stride=1,padding=1,bias=False)        
        self.conv_64_1 = nn.Conv2d(64,1,kernel_size=3,stride=1,padding=1,bias=False)
        
        self.latlayer_64 = nn.Conv2d(64,64,kernel_size=1,stride=1,padding=0)
        self.latlayer_128 = nn.Conv2d(128,128,kernel_size=1,stride=1,padding=0)
        self.latlayer_256 = nn.Conv2d(256,256,kernel_size=1,stride=1,padding=0)
        
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.aspp = ASPP()
        
    def _upsample_add(self,x,y):
        _,_,H,W = y.size()
        return F.upsample(x,size=(H,W),mode='bilinear')+y 
    
    def forward(self,x):
        
        #Left Conv1
        x = self.conv_in(x)
        x = self.relu(x)       
        x = self.conv_64_64(x)   
        x_out128 = self.relu(x)

        #MaxPool DownSample:128-->64
        x = self.maxpool(x_out128)
        
        #Left Conv2
        x = self.conv_64_128(x)
        x = self.relu(x)
        x = self.conv_128_128(x)
        x = self.relu(x)
        x = self.conv_128_128(x)
        x_out64 = self.relu(x)
        
        #MaxPool DownSample:64-->32
        x = self.maxpool(x_out64)
        
        #Left Conv3
        x = self.conv_128_256(x)
        x = self.relu(x)    
        x = self.conv_256_256(x)
        x = self.relu(x)
        x = self.conv_256_256(x)
        x_out32 = self.relu(x)
     
        #ASPP
        x = self.aspp(x_out32)   

        x = self.conv_256_256(x)
        x = self.relu(x)
        x = self.conv_256_256(x)
        x = self.relu(x)
        x = self.conv_256_256(x)
        x = self.relu(x)
        
        #Reduce the number of feature maps:256-128    
        x = self.conv_256_128(x)
        
        #Right Conv2
        #Add high resolution feature map and low resolution feature map and up_sample on low resolution image      
        x_right64 = self._upsample_add(x,self.latlayer_128(x_out64))#64:128-128
        x = self.relu(x_right64)
        x = self.conv_128_128(x)
        x = self.relu(x)
        x = self.conv_128_128(x)
        x = self.relu(x)

        #Reduce the number of feature maps:128-64           
        x = self.conv_128_64(x)

        #Right Conv1
        #Add high resolution feature map and low resolution feature map and up_sample on low resolution image         
        x_right128 = self._upsample_add(x,self.latlayer_64(x_out128))#128:64-64
        x = self.relu(x_right128)    
        x = self.conv_64_64(x)
        x = self.relu(x)        
        x_out = self.conv_64_1(x)
        
        return x_out



        
        
        
        
