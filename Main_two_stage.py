import argparse
import itertools
import os
from xmlrpc.client import boolean
from torch.autograd import Variable
import torch.nn as nn
import torch
import numpy as np
import scipy.io
import torch.nn.functional as F

from Models_Unet_ver2 import SR_Unet
from Models_Unet_ver2 import Generator_down
from Models_Unet_ver2 import Discriminator_M


print('torch_version:', torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

import Algorithms_train as alg_train
import Algorithms_BTD as alg_BTD
import utils

import Evaluation as eva
import wandb
# breast/ neuroendocrine / GLAS
import time


print(torch.__version__)
print(torch.version.cuda)




parser = argparse.ArgumentParser()

#========================== Root Preparation ===================================#
parser.add_argument('--dataRoot', type=str, default='/scratch/Jiahui/Ph.D_Program/HSR_ours/Dataset/Abun_Pavia_Center.mat',help='path to MSI, HSI, SRI data')
    # data should be .mat file with the following keys:
        # SRI_M: SRI for MSI region
        # HSI: HSI
        # MSI: MSI 
        # Q: PM matrix with shape (KM, KH)
        # SRI_H: SRI for HSI region (SRI_H should have the same size as SRI_M)
    # SRI_M, HSI, MSI, SRI_H should be normalized to [0,1], saved as float

parser.add_argument('--resultRoot',type=str,default='/scratch/Jiahui/Ph.D_Program/HSR_ours/Result_end2end/',help='path to save the results')

parser.add_argument('--firstStageResult',type=str,default = '/scratch/Jiahui/Ph.D_Program/HSR_ours/Result_end2end/',help='The result of the first stage (used only if Stage_1 is False)')



#========================== Training Parameter  ===================================#
parser.add_argument('--numMaterial', type=int, default=4, help='number of materials in the data (R)')

parser.add_argument('--Stage1',type=utils.str2bool,default=True,help='Run the first stage? (coupled block tensor decomposition)')
# if False, the abundance maps and the endmembers should be prepared in advance, saved in ${resultRoot}/First_Stage_Result.mat with the following keys:
    # SM: abundance maps for MSI region (IM, JM, R)
    # SH: abundance maps for HSI region (IH, JH, R)
    # C: endmembers (K, R)

parser.add_argument('--Stage2',type=utils.str2bool, default=True, help='Run the second stage? (diversified distribution matching)')

parser.add_argument('--cuda', type=utils.str2bool, default=True, help='use GPU computation')

parser.add_argument('--sr', type=int, default=4, help='super-resolution factor (downsampling rate), if set to 0, then we use IM/IH as the downsampling rate')


#========== First Stage: Coupled Block Tensor Decomposition ===========#
parser.add_argument('--iterationCBTD',type=int, default=1200, help='The iterations of the first stage')

parser.add_argument('--s2o_weight',type=float,default=0.01,help='The weight of the soft sum2one constraint')



#========== Second Stage: Diversified Distribution Matching ===========#
parser.add_argument('--nEpochs', type=int, default=2000, help='number of epochs of training')

parser.add_argument('--batchSize', type=int, default=6, help='size of the batches sampled in each iteration')

parser.add_argument('--iterEachEpoch', type=int, default=20, help='number of iterations in each epoch')
    # we randomly sample patches in each iteration so there is no strictly defined epoch.
    # we use epoch to group up some iterations to be more clear in the log

parser.add_argument('--patchSize', type=int, default=12, help='The size of low-resolution patches (height = width)')

parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for gan')

parser.add_argument('--decayEpoch', type=int, default=1000, help='epoch to start linearly decaying the learning rate to 0 (for cyclegan)')

parser.add_argument('--sideSampling',type=float, default=0.1, help='chance of sampling at the side of the image (0-1)')

parser.add_argument('--scale_weight',type=float, default=1.0, help='weight of Scale Loss')

parser.add_argument('--gan_weight', type=float, default=1.0, help='weight of GAN Loss')

parser.add_argument('--inverse_weight', type=float, default=1.0, help='weight of Inverse Loss')

parser.add_argument('--scaleUp', type=float, default=2.0, help='Scale up the value of data')

parser.add_argument('--gp', type=utils.str2bool, default=False, help='Use gradient penalty?')


#========================== Evaluation ===================================#
parser.add_argument('--eval_Stage1',type=utils.str2bool, default=True, help='evaluate the first stage? (we will compute the RMSE,PSNR,SSIM,SAM,ERGAS metrics)')

parser.add_argument('--eval_Stage2',type =utils.str2bool, default=False, help='evaluate through training (only for second stage)? (we will compute the PSNR,SSIM,SAM,ERGAS,CC,FID metrics)')

parser.add_argument('--eval_interval',type = int, default=10, help='evaluate at every eval_interval epochs')

parser.add_argument('--FID',type = utils.str2bool, default=False, help='evaluate FID during the training?')

parser.add_argument('--RGB',type = str, default=False, help='RGB bands for evaluating FID and LPIPS (comma-separated values, e.g., "25,18,7")')

parser.add_argument('--overlap',type=int, default=15, help='the overlap between patches when reconstructing the image')
    # when we reconstruct the high-resolution abundance maps for HSI region, we apply a convolution like stratgy. 
    # This is used to decide the overlap between patches when reconstructing the image.
    # Normally we set overlap = patchSize - 1 to average the patches and aviod the artifacts.
    # This can be set to values s.t. (IH - patchSize) % (patchSize - overlap) == 0 and (JH - patchSize) % (patchSize - overlap) == 0

#============================== Log =======================================#
parser.add_argument('--wandb', type=utils.str2bool, default=True, help='Use wandb for logging?')

parser.add_argument('--project_name', type=str, default='MELD', help='wandb project name (only used if wandb is True)')

parser.add_argument('--textLog', type=utils.str2bool, default=True, help='Use text log for logging?')

parser.add_argument('--run_name', type=str, default='Unreg_HSR', help='run name of the wandb log')


#===== Print the arguments =====#
opt = parser.parse_args()

for arg in vars(opt):
    print(arg, getattr(opt, arg))


#===== Initialize the wandb =====#
if opt.wandb:
    wandb.init(project=opt.project_name, name=opt.run_name)
    wandb.config.update(opt)



#===== Define the Device =====#
device = torch.device('cuda' if (torch.cuda.is_available() and opt.cuda) else 'cpu')

#===== Check Legality =====#
data, IM, JM, IH, JH, R = utils.check_legality(opt)


#===== Prepare the evaluator =====#
if opt.eval_Stage1 or opt.eval_Stage2:
    evaluator = eva.Evaluator(opt,data,device)


#======= First Stage Start: Coupled Block Tensor Decomposition =======#
'''
The first stage is to estimate the abundance maps and endmembers for the MSI and HSI regions.
'''

if opt.Stage1:
    MSR_start_time = time.time()
    #===== Initialize the parameters =====#
    trainer_CBTD = alg_BTD.BSCLL1_soft_constraint(data, R, Lambda=opt.s2o_weight, maxIter=opt.iterationCBTD, e=1e-9)
    SM3, SH3, CH = trainer_CBTD.PGD_BSCLL1()
    MSR_end_time = time.time()

    print(f'First stage finished. Time cost: {(MSR_end_time-MSR_start_time)/60:.2f} minutes')

    #===== Evaluate the results  =====#
    if opt.eval_Stage1:
        SRI_M_reconstructed = evaluator.reconstruct_SRI_M(SM3, CH)
        eva_M_dict = evaluator.Evaluate_SRI_M(SRI_M_reconstructed)

        flag = alg_BTD.Evaluation_and_Visualization(SM3,SH3,CH,flag_eva=True,flag_visual = True, result_root = opt.resultRoot,size_M = [IM,JM],size_H = [IH,JH],R=R)

        #===== Print the evaluation metrics and record them to text and wandb  =====#
        eva.print_dict(eva_M_dict,epoch = -1)

        if opt.textLog:
            evaluator.write_dict_to_file(os.path.join(opt.resultRoot,f'Evaluation_{opt.run_name}.txt'),eva_M_dict,stage=1,epoch=-1)
        
        if opt.wandb:
            evaluator.write_dict_to_wandb(eva_M_dict,stage='MSR',SRI_M_reconstructed=SRI_M_reconstructed)

    #===== Save the results  =====#
    SM = np.reshape(SM3,(IM,JM,R),order='F') # first reshape the abundance maps to the tensor form (I,J,R)
    SH = np.reshape(SH3,(IH,JH,R),order='F')

    saveresult = {'SM':SM,'SH':SH,'CH':CH,'SRI_M_reconstructed':SRI_M_reconstructed}
    scipy.io.savemat(os.path.join(opt.resultRoot,'decomposition_result.mat'),saveresult)



#===== If not run the first stage, we read the abundance maps and endmembers from .mat file =====#
else:
    first_stage_result = scipy.io.loadmat(opt.firstStageResult)
    SM = first_stage_result['SM']
    SH = first_stage_result['SH']
    CH = first_stage_result['CH']





#======= Second Stage Start: Diversified Distribution Matching =======#
if opt.Stage2:

    #===== Define the network =====#
    netG_H2M = SR_Unet(n_channels=1, n_classes=1,scale_factor=opt.sr)
    netG_M2H = Generator_down(input_nc=1, output_nc=1, inter_nc=64, n_blocks=2,scale_factor=opt.sr)
    netD_M = Discriminator_M(num_domains=opt.numMaterial)

    #===== Define the optimizer =====#
    optimizer_G = torch.optim.Adam(itertools.chain(netG_H2M.parameters(), netG_M2H.parameters()),lr=opt.lr, betas=(0.5,0.999))
    optimizer_D_M = torch.optim.Adam(netD_M.parameters(), lr=opt.lr, betas=(0.5,0.999))
    
    #===== Define the learning rate scheduler =====#
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=utils.LambdaLR(opt.nEpochs, 0, opt.decayEpoch).step)
    lr_scheduler_D_M = torch.optim.lr_scheduler.LambdaLR(optimizer_D_M, lr_lambda=utils.LambdaLR(opt.nEpochs, 0, opt.decayEpoch).step)

    #===== Put the networks to the device =====#
    netG_H2M.to(device)
    netG_M2H.to(device)
    netD_M.to(device)

    # #===== Initialize the networks =====#
    netG_H2M.apply(utils.weights_init_normal)
    netG_M2H.apply(utils.weights_init_normal)
    netD_M.apply(utils.weights_init_normal)




    #===== Initialize the tranier =====#
    trainer_DDM = alg_train.Trainer(opt,SH,SM,device)



    #===== Start the training =====#
    print('Start Training')
    for epoch in range(opt.nEpochs+1):
        epoch_start_time = time.time()
        for i in range(opt.iterEachEpoch):
            #===== Update G =====#    
            optimizer_G.zero_grad()
            loss_G,loss_gan_H2M, loss_inverse_HMH, loss_inverse_MHM, loss_scale = trainer_DDM.generator_loss(netG_H2M,netG_M2H,netD_M)
            loss_G.backward()
            optimizer_G.step()

            #===== Update DM =====#
            optimizer_D_M.zero_grad()
            if opt.gp and i % 5 == 0:
                loss_D_M = trainer_DDM.discriminator_loss_penalty(netD_M)
            else:
                loss_D_M = trainer_DDM.discriminator_loss(netD_M)

            loss_D_M.backward()
            optimizer_D_M.step()

            
            #===== Print the loss =====#
            eva.record_loss({'Loss_cyclegan':loss_G.item(),
                        'Loss_D_M':loss_D_M.item(),
                        'Loss_GAN_H2M':loss_gan_H2M.item(),
                        'Loss_inverse_HMH':loss_inverse_HMH.item(),
                        'Loss_inverse_MHM':loss_inverse_MHM.item(),
                        'Loss_scale':loss_scale.item()},epoch=epoch,iter=i,log_wandb=opt.wandb)



        #===== Update the learning rate =====#
        lr_scheduler_G.step()
        lr_scheduler_D_M.step()

        epoch_end_time = time.time()
        print(f'Epoch {epoch}/{opt.nEpochs} finished. Time cost: {(epoch_end_time-epoch_start_time):.2f} seconds')

        #===== Evaluate the results=====#
        if opt.eval_Stage2:
            if epoch % opt.eval_interval == 0:
                SRI_H_reconstructed, SR_SH= evaluator.reconstruct_SRI_H(netG_H2M,SH,CH)

                eva_H_dict = evaluator.Evaluate_SRI_H(SRI_H_reconstructed)


                #===== Print the evaluation metrics and record them to text and wandb  =====#
                eva.print_dict(eva_H_dict,epoch = epoch)

                if opt.textLog:
                    evaluator.write_dict_to_file(os.path.join(opt.resultRoot,f'Evaluation_{opt.run_name}.txt'),eva_H_dict,stage=2,epoch=epoch)

                if opt.wandb:
                    evaluator.write_dict_to_wandb(eva_H_dict,stage='HSR',SRI_H_reconstructed=SRI_H_reconstructed)
                

    #===== Save the results =====#
    saveresult = {'SRI_H_reconstructed':SRI_H_reconstructed,'SR_SH':SR_SH}
    scipy.io.savemat(os.path.join(opt.resultRoot,f'DDM_result_{opt.run_name}.mat'),saveresult)



if opt.wandb:
    wandb.finish()
