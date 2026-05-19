import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy import sparse
from matplotlib.colors import LinearSegmentedColormap
import torch
import torch.nn.functional as F
import cv2
import argparse
import os


def str2bool(v):
    '''
    This function is used to convert string to boolean in the argument parser.
    '''

    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def weights_init_normal(m):
    '''
    Custom weights initialization called on netG and netD
    '''

    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class LambdaLR():
    '''
    Create a learning rate scheduler that decays linearly from 1 to 0.5
    '''

    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return (1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch))*0.5
    


def check_legality(opt):
    '''
    Check the legality of the input parameters and create the result root directory if it does not exist.
    '''

    #===== Create the Result Root =====#
    if not os.path.exists(opt.resultRoot):
        os.makedirs(opt.resultRoot)
        print(f"Directory {opt.resultRoot} created")


    #===== Load the data =====#
    data = scipy.io.loadmat(opt.dataRoot)

    # SRI for MSI region
    SRI_M = data['SRI_M'] 
    [IM,JM,KH] = SRI_M.shape

    # MSI
    MSI = data['MSI']    
    KM = MSI.shape[2]

    # HSI 
    HSI = data['HSI']
    [IH,JH,_] = HSI.shape

    # SRI for HSI region
    SRI_H = data['SRI_H']

    # PM
    PM = data['Q']
    PM = PM.astype(np.float32)

    # number of materials (R)
    R = opt.numMaterial

    #===== Check the legality of dimensions =====#
    if IM != MSI.shape[0] or JM != MSI.shape[1]:
        raise ValueError('The spatial size of MSI and SRI_M should be equal')
    
    if KH != HSI.shape[2] or KH != SRI_H.shape[2]:
        raise ValueError('The spectral size of HSI,SRI_H and SRI_M should be equal')

    #===== Check the legality of super-resolution ratio =====#
    if opt.sr == 0:
        if IM/IH != JM/JH:
            raise ValueError('IH/JH should be equal to IM/JM if you don\'t set the downsample rate')
        opt. sr = int(IM/IH)
        print('The downsample rate is set to:', opt.sr)


    #===== Check the legality of the patchSize =====#
    if opt.patchSize > IM or opt.patchSize > JM:
        raise ValueError('The patch size should be smaller than the spatial size of the image')
    
    # if opt.patchSize * opt.sr < 48:
    #     raise ValueError('The high-resolution patch size (opt.patchSize * sr) should be larger than 48')
    
    if (opt.patchSize * opt.sr) % 16 != 0:
        raise ValueError('To fit the network architecture, the high-resolution patch size (opt.patchSize * sr) should be divisible by 16')
    

    #===== Check the overlap =====#
    if (IH - opt.patchSize) % (opt.patchSize - opt.overlap) !=0 or (JH - opt.patchSize) % (opt.patchSize - opt.overlap) !=0:
        raise ValueError('The patch size and overlap should be well designed to fit the image size, please read the comments in the argument.')
    

    return data,IM, JM, IH, JH, R

    