
import numpy as np

def AddRealNoise(x):
    
    sigma_c = np.random.uniform(5, 20)
    w, h, c = x.shape
    temp_x = x
    noise_c = np.zeros((w, h, c))
    noise_c = -np.log(np.random.gamma(shape=sigma_c,scale = 1/sigma_c,size =(w,h,c))+1e-3)
    temp_x_n = temp_x + noise_c
    
    return temp_x_n
