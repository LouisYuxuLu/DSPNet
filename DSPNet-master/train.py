# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 17:34:04 2020

@author: Administrator
"""

import os, time, scipy.io, shutil
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
#from tensorboardX import SummaryWriter
import numpy as np
import cv2
import scipy.misc
from DSPNet_model import *
from makedataset import Dataset
import utils_train
import SSIM
from utils.noise import *
from utils.common import *
from utils.loss import *


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_checkpoint(checkpoint_dir, num_input_channels):
    if num_input_channels ==3:
        
        if os.path.exists(checkpoint_dir + 'checkpoint.pth.tar'):
            # load existing model
            model_info = torch.load(checkpoint_dir + 'checkpoint.pth.tar')
            print('==> loading existing model:', checkpoint_dir + 'checkpoint.pth.tar')
            net = DSPNet()
            device_ids = [0]
            model = nn.DataParallel(net, device_ids=device_ids).cuda()
            model.load_state_dict(model_info['state_dict'])
            optimizer = torch.optim.Adam(model.parameters())
            optimizer.load_state_dict(model_info['optimizer'])
            cur_epoch = model_info['epoch']
            
        else:
            # create model
            net = DSPNet()
            device_ids = [0]
            model = nn.DataParallel(net, device_ids=device_ids).cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            cur_epoch = 0

    else:
        if os.path.exists(checkpoint_dir + 'checkpoint.pth.tar'):
            # load existing model
            model_info = torch.load(checkpoint_dir + 'checkpoint.pth.tar')
            print('==> loading existing model:', checkpoint_dir + 'checkpoint.pth.tar')
            net = DSPNet()
            device_ids = [0]
            model = nn.DataParallel(net, device_ids=device_ids).cuda()
            model.load_state_dict(model_info['state_dict'])
            optimizer = torch.optim.Adam(model.parameters())
            optimizer.load_state_dict(model_info['optimizer'])
            cur_epoch = model_info['epoch']
            
        else:
            # create model
            net = DSPNet()
            device_ids = [0]
            model = nn.DataParallel(net, device_ids=device_ids).cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            cur_epoch = 0


    return model, optimizer,cur_epoch

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
	torch.save(state, checkpoint_dir + 'checkpoint.pth.tar')
	if is_best:
		shutil.copyfile(checkpoint_dir + 'checkpoint.pth.tar',checkpoint_dir + 'model_best.pth.tar')

def adjust_learning_rate(optimizer, epoch, lr_update_freq):
	if not epoch % lr_update_freq and epoch:
		for param_group in optimizer.param_groups:
			param_group['lr'] = param_group['lr'] * 0.1
			print( param_group['lr'])
	return optimizer

def train_psnr(log_train_in,log_train_out):
    train_in = torch.exp(-log_train_in)
    train_out = torch.exp(-log_train_out)
    psnr = utils_train.batch_psnr(train_in,train_out,1.)
    return psnr

def train_synthetic(data):
    
    imgo_train =data
    img_train = -torch.log(data+1e-3)
    noiseimg = torch.zeros(img_train.size())
    noiselevel = torch.zeros(img_train.size())
    for nx in range(imgo_train.shape[0]):
        noiseimg[nx,:,:,:] = torch.from_numpy(AddRealNoise(img_train[nx, :, :, :].numpy()))
        noiselevel[nx,:,:,:] = noiseimg[nx,:,:,:]-img_train[nx,:,:,:]
    
    input_var = Variable(noiseimg.cuda(), volatile=True)
    target_var = Variable(img_train.cuda(), volatile=True)
    noise_level_var = Variable(noiselevel.cuda(), volatile=True)
    
    return input_var, target_var,noise_level_var

def train_real(data,num_input_channels):
    c = num_input_channels
    imgc_train = -torch.log(data[:,0:c,:,:]+1e-3)
    imgn_train = -torch.log(data[:,c:2*c,:,:]+1e-3)
    noiselevel = torch.zeros(imgc_train.size())
    for nx in range(imgc_train.shape[0]):
        noiselevel[nx,:,:,:] = imgn_train[nx,:,:,:]-imgc_train[nx,:,:,:]
   
    input_var = Variable(imgn_train.cuda(), volatile=True)
    target_var = Variable(imgc_train.cuda(), volatile=True)
    noise_level_var = Variable(noiselevel.cuda(), volatile=True)  
      
    return input_var, target_var, noise_level_var

def test_synthetic(test_syn,result_syn, model,epoch,num_input_channels):
    '''test synthetic gamma images and set noiselevel(15,30,50,75)'''
    
    noiselevel = [1,5,10,15,20]
    files = os.listdir(test_syn)
    for i in range(len(noiselevel)):
        ssim=0.0; psnr=0.0
        for j in range(len(files)):   
            model.eval()
            with torch.no_grad():
                img_c =  cv2.imread(test_syn + '/' + files[j])
                if num_input_channels == 3:
                    img_cc = img_c[:,:,::-1] / 255.0
                    clear_img = -np.log(img_cc+1e-3)
                    clear_img = np.array(clear_img).astype('float32')
                    noise = np.zeros(clear_img.shape)
                    w,h,c = noise.shape
                    noise_img = clear_img + (-np.log(np.random.gamma(shape=noiselevel[i],scale = 1/noiselevel[i],size =(w,h,c))+1e-3))
                    noise_img_chw = hwc_to_chw(noise_img)
                    input_var = torch.from_numpy(noise_img_chw.copy()).type(torch.FloatTensor).unsqueeze(0)
                    
                    input_var = input_var.cuda()
                    _,output = model(input_var)            
                    output_np = output.squeeze().cpu().detach().numpy()
                    output_np = chw_to_hwc(output_np)
                else:
                    img_cc =  cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY) / 255.0
                    clear_img = -np.log(img_cc+1e-3)
                    clear_img = np.array(clear_img).astype('float32')
                    noise = np.zeros(clear_img.shape)
                    w,h = noise.shape
                    noise_img = clear_img + (-np.log(np.random.gamma(shape=noiselevel[i],scale = 1/noiselevel[i],size =(w,h))+1e-3))
                    input_var = torch.from_numpy(noise_img.copy()).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
                    input_var = input_var.cuda()
                    _,output = model(input_var)            
                    output_np = output.squeeze().cpu().detach().numpy()
                                 
                output_np = np.exp(-output_np)
                SSIM_val = SSIM.compute_ssim(np.clip(255*img_cc,0.0,255.0),np.clip(255*output_np,0.0,255.0),num_input_channels)
                PSNR_val = SSIM.cal_psnr(np.clip(255*img_cc,0.0,255.0),np.clip(255*output_np,0.0,255.0))     
                ssim+=SSIM_val;psnr+=PSNR_val
                temp = np.concatenate((img_cc,np.exp(-noise_img), output_np), axis=1)      
                if num_input_channels==3:
                    cv2.imwrite(result_syn + '/' + files[j][:-4] +'_%d_%d_Mix_%.4f_%.4f'%(epoch,noiselevel[i],SSIM_val,PSNR_val)+'.png',np.clip(temp[:,:,::-1]*255,0.0,255.0))
                else:
                    cv2.imwrite(result_syn + '/' + files[j][:-4] +'_%d_%d_Mix_%.4f_%.4f'%(noiselevel[i],epoch,SSIM_val,PSNR_val)+'.png',np.clip(temp*255,0.0,255.0))            
        print('Synthetic Images Test: NoiseLevel is %d, SSIM is :%6f and PSNR is :%6f'%(noiselevel[i],ssim/len(files),psnr/len(files)))
        with open('./log/syntext.txt','a') as f:
            f.writelines('Synthetic Images Test: Epoch is %d,  NoiseLevel is %d, SSIM is :%6f and PSNR is :%6f'%(epoch,noiselevel[i],ssim/len(files),psnr/len(files)))
            f.writelines('\r\n')
           
        
if __name__ == '__main__':
    checkpoint_dir = './checkpoint/'
    test_syn = './dataset/test/synthetic'
    result_syn = './result/test/synthetic'

 
    maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
    
    print('> Loading dataset ...')
    dataset_train_syn = Dataset(trainrgb=False,trainsyn = True, shuffle=True)
    loader_train_syn = DataLoader(dataset=dataset_train_syn, num_workers=0, batch_size=32, shuffle=True)
    
    
    num_input_channels = 1
    lr_update_freq = 15

    model, optimizer, cur_epoch = load_checkpoint(checkpoint_dir,num_input_channels)

    mix_loss = fixed_loss()
    msssim_loss = mix_loss.cuda()
    L1_loss = torch.nn.L1Loss(reduce=True, size_average=True)
    L1_loss = L1_loss.cuda()
    L2_loss = torch.nn.MSELoss(reduce=True, size_average=True)
    L2_loss = L2_loss.cuda()

    for epoch in range(cur_epoch, 50):
        #losses = AverageMeter()
        optimizer = adjust_learning_rate(optimizer, epoch, lr_update_freq)
        learnrate = optimizer.param_groups[-1]['lr']
        model.train()
        #train
        for i,data in enumerate(loader_train_syn,0):
            input_var, target_var,noise_level_var =train_synthetic(data)   
            target_var_128 = target_var
            
            enml,outputvar_128 = model(input_var)                         
            loss = L2_loss(enml, noise_level_var)+ 0.8*mix_loss(outputvar_128, target_var_128) + 0.2*L1_loss(outputvar_128, target_var_128)
            
            #losses.update(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()   
            psnr = train_psnr(target_var_128,outputvar_128)
            print("[Synthetic_Epoch %d][%d/%d] lr :%f loss: %.4f PSNR_train: %.4f" %(epoch+1, i+1, len(loader_train_syn), learnrate, loss.item(), psnr))
        
        test_synthetic(test_syn,result_syn, model,epoch,num_input_channels)  
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict()}, is_best=0)
