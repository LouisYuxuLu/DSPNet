import os, time, scipy.io, shutil
import torch
import torch.nn as nn
import numpy as np
import cv2
from DSPNet_model import *


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



def load_checkpoint(checkpoint_dir):


    model_info = torch.load(checkpoint_dir + 'checkpoint.pth.tar')
    print('==> loading existing model:', checkpoint_dir + 'checkpoint.pth.tar')
    net = DSPNet()
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(model_info['state_dict'])
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(model_info['optimizer'])
    cur_epoch = model_info['epoch']

    return model, optimizer,cur_epoch

def test_model(test_syn,result_syn, model):

    files = os.listdir(test_syn)
    for i in range(len(files)):   
        model.eval()
        with torch.no_grad():
            img_c =  cv2.imread(test_syn + '/' + files[i])
            img_cc =  cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY) / 255.0
            clear_img = -np.log(img_cc+1e-3)
            clear_img = np.array(clear_img).astype('float32')
            noise = np.zeros(clear_img.shape)
            w,h = noise.shape
            noise_img = clear_img + (-np.log(np.random.gamma(shape=5,scale = 1/5,size =(w,h))))
            input_var = torch.from_numpy(noise_img.copy()).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0)
            input_var = input_var.cuda()
            _,output = model(input_var)            
            output_np = np.exp(-output.squeeze().cpu().detach().numpy())

            cv2.imwrite(result_syn + '/' + files[i][:-4] + '_DSPNet.png',np.clip(output_np*255,0.0,255.0)) 


        
if __name__ == '__main__':
    checkpoint_dir = './checkpoint/'
    test_syn = './dataset/test' 
    result_syn = './result' 


    model, optimizer, cur_epoch = load_checkpoint(checkpoint_dir)
    test_model(test_syn,result_syn, model)  

