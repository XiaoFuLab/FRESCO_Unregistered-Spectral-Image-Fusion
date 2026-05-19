import argparse
import itertools
import os
from xmlrpc.client import boolean
from torch.autograd import Variable
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import torch.autograd as autograd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"



class Trainer:
    '''
    This class is used for patch-to-patch super-resolution training.
    It mainly contains the patch sampling, augmentation, and the loss calculation.
    '''

    def __init__(self,opt,SH,SM,device):

        #===== store the parameters =====#
        self.down_rate = opt.sr
        self.R = opt.numMaterial
        self.batchSize = opt.batchSize
        self.patchSize_H = opt.patchSize
        self.patchSize_M = int(opt.patchSize * opt.sr)
        self.sideSampling = opt.sideSampling
        self.scaleUp = opt.scaleUp
        self.gan_weight = opt.gan_weight
        self.inverse_weight = opt.inverse_weight
        self.scale_weight = opt.scale_weight

        self.device = device
        self.IH, self.JH = SH.shape[0], SH.shape[1]
        self.IM, self.JM = SM.shape[0], SM.shape[1]

        self.SH = torch.from_numpy(SH).float().to(device)
        self.SM = torch.from_numpy(SM).float().to(device)

        #===== define the loss =====#
        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_scale = nn.MSELoss()


        #===== define the labels for real and fake patchs =====#
        self.target_real = torch.ones(self.batchSize, device=device, requires_grad=False)
        self.target_fake = torch.zeros(self.batchSize, device=device, requires_grad=False)
        
        #===== define the index to assist multi-domain discriminator =====#
        self.index = torch.arange(self.R, device=device).unsqueeze(1).repeat(1, self.batchSize).long()


    
    
    def generator_loss(self,netG_H2M,netG_M2H,netD_M):
        '''
        This function is used to calculate the generator loss, which contains the following three parts:
        1. GAN loss for the netG_H2M
            We only do distribution matching for the netG_H2M. 

        2. Inverse loss for the netG_H2M
            The inverse loss is used to force the invertability.

        3. Scale loss for the netG_H2M and netG_M2H
            The scale loss is used to force the generated patches to have the similar mean value as the original patches.
        '''

        #===== Randomly sample the training patches =====#
        self.crop_data()


        #===== Initialize the generated patches =====#
        self.fake_M_patches = torch.zeros(self.batchSize,self.R,self.patchSize_M,self.patchSize_M).to(self.device)

        #===== Initialize the losses =====#
        loss_gan_H2M = torch.tensor([0.0],requires_grad=False).to(self.device)  # GAN loss for the netG_H2M
        loss_inverse_HMH = torch.tensor([0.0],requires_grad=False).to(self.device) # Inverse loss on H \to M \to H direction
        loss_inverse_MHM = torch.tensor([0.0],requires_grad=False).to(self.device) # Inverse loss on M \to H \to M direction
        loss_scale = torch.tensor([0.0],requires_grad=False).to(self.device) # Scale loss for the netG_H2M and netG_M2H
        
        for r in range(self.R):
            M_patch_r = self.M_patches[:,r,:,:].unsqueeze(1)
            H_patch_r = self.H_patches[:,r,:,:].unsqueeze(1)
            
            #===== Generate the fake patches =====#
            fake_M_patch_r = netG_H2M(H_patch_r)
            fake_H_patch_r = netG_M2H(M_patch_r)


            #===== GAN_H2M loss =====#
            pred_fake = netD_M(fake_M_patch_r,self.index[r,:]) #judge the fake patch by using the r-th domain discriminator output.

            loss_gan_H2M += 1/self.R * self.criterion_GAN(pred_fake, self.target_real) 
                                       

            #===== Inverse loss =====#
            recovered_M_patch_r = netG_H2M(fake_H_patch_r)
            loss_inverse_MHM += 1/self.R * self.criterion_cycle(recovered_M_patch_r, M_patch_r)

            recovered_H_patch_r = netG_M2H(fake_M_patch_r)
            loss_inverse_HMH += 1/self.R * self.criterion_cycle(recovered_H_patch_r, H_patch_r)


            #===== Scale loss =====#
            mean_M_patch  = torch.mean(M_patch_r,dim=[1,2,3])
            mean_H_patch = torch.mean(H_patch_r,dim=[1,2,3])
            mean_fake_M_patch = torch.mean(fake_M_patch_r,dim=[1,2,3])
            mean_fake_H_patch = torch.mean(fake_H_patch_r,dim=[1,2,3])

            loss_scale += 1/self.R * (self.criterion_scale(mean_M_patch,mean_fake_H_patch) + self.criterion_scale(mean_H_patch,mean_fake_M_patch))
        

            #===== Sum up to get the total loss =====#
            loss_G = self.gan_weight * loss_gan_H2M + self.inverse_weight * (loss_inverse_HMH + loss_inverse_MHM) + self.scale_weight * loss_scale / self.scaleUp


            #===== Save the fake images =====#
            self.fake_M_patches[:,r,:,:] = fake_M_patch_r.squeeze()

        return loss_G, loss_gan_H2M, loss_inverse_HMH, loss_inverse_MHM, loss_scale


    def discriminator_loss(self,netD_M):
        '''
        This function is the classical multi-domain discriminator loss.
        '''

        loss_D = torch.tensor([0.0],requires_grad=False).to(self.device)


        for r in range(self.R):
            #===== Get the sampled and generated patches for the r-th abundance map =====#
            M_patch_r = self.M_patches[:,r,:,:].unsqueeze(1).detach()

            fake_M_patch_r = self.fake_M_patches[:,r,:,:].unsqueeze(1).detach()

            pred_real_DM = netD_M(M_patch_r,self.index[r,:])
            pred_fake_DM = netD_M(fake_M_patch_r,self.index[r,:])


            #===== Calculate the loss =====#
            loss_D += 1/self.R * 0.5 * (self.criterion_GAN(pred_real_DM, self.target_real) 
                                        + self.criterion_GAN(pred_fake_DM, self.target_fake))


        return loss_D



    def r1_penalty(self, d_out, x_in):
        """
        R1 regularization: ||∇_x D(x)||^2 on real samples.
        d_out: discriminator output on real, shape [B, ...]
        x_in: real input tensor, shape [B, C, H, W], requires_grad=True
        """
        grad = autograd.grad(
            outputs=d_out.sum(),
            inputs=x_in,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        grad = grad.view(grad.size(0), -1)              # [B, CHW]
        return (grad.pow(2).sum(dim=1)).mean()          # E[||grad||^2]
    

    def discriminator_loss_penalty(self, netD_M, r1_gamma=2.0):
        loss_D = torch.tensor(0.0, device=self.device)

        for r in range(self.R):
            # real / fake patches
           
            M_patch_r = self.M_patches[:, r, :, :].unsqueeze(1)
            fake_M_patch_r = self.fake_M_patches[:, r, :, :].unsqueeze(1).detach()

            # R1 needs grad wrt real input
            M_patch_r = M_patch_r.detach()              # detach from upstream graph (e.g., data pipeline)
            M_patch_r.requires_grad_(True)              # but enable gradient wrt input for R1

            pred_real_DM = netD_M(M_patch_r, self.index[r, :])
            pred_fake_DM = netD_M(fake_M_patch_r, self.index[r, :])

            # GAN loss (your original)
            loss_gan = 0.5 * (
                self.criterion_GAN(pred_real_DM, self.target_real) +
                self.criterion_GAN(pred_fake_DM, self.target_fake)
            )

            # R1 gradient penalty on real
            gp = self.r1_penalty(pred_real_DM, M_patch_r)
            loss_r1 = 0.5 * r1_gamma * gp
            loss_D = loss_D + (1.0 / self.R) * (loss_gan + loss_r1)

        return loss_D



    def crop_data(self):
        '''
        In each iteration, we randomly sample patches from the high-resolution and low-resolution abundance maps.
        The patches are saved in self.M_patches and self.H_patches. Shape: (B, R, H, W), where B is the batch size.
        '''
        
        self.M_patches = torch.zeros(self.batchSize,self.R,self.patchSize_M,self.patchSize_M).to(self.device)
        self.H_patches= torch.zeros(self.batchSize,self.R,self.patchSize_H,self.patchSize_H).to(self.device)


        #===== Randomly sample the training patches according to different material abundance maps =====#
        for r in range(self.R):
            self.M_patches[:,r,:,:] = self.get_blocks_aug(self.SM[:,:,r], self.patchSize_M) * self.scaleUp
            self.H_patches[:,r,:,:] = self.get_blocks_aug(self.SH[:,:,r], self.patchSize_H) * self.scaleUp

        



    def agumentation(self,bigPatch, angle, patchSize, flip_up_down, flip_left_right):
        '''
        This function is used to do random rotation and flip on the patches.
        The rotation angle is randomly selected from [0, 360).
        The flip is randomly selected from {0, 1} with two directions: up-down and left-right. flip=0 means no flip, flip=1 means flip.
        The patch is cropped from the middle of the agumented big-patch. 
        '''

        #===== Rotate the patch =====#
        angle_rad = torch.tensor(angle * np.pi / 180, dtype=torch.float32, device=self.device)
        grid = F.affine_grid(torch.tensor([[torch.cos(angle_rad), -torch.sin(angle_rad), 0],
                                        [torch.sin(angle_rad), torch.cos(angle_rad), 0]], dtype=torch.float32,device=self.device).unsqueeze(0),
                            bigPatch.unsqueeze(0).unsqueeze(0).size(), align_corners=False)
        rotated_bigPatch = F.grid_sample(bigPatch.unsqueeze(0).unsqueeze(0), grid, align_corners=False).squeeze()


        #===== Flip the patch =====#
        if flip_up_down:
            rotated_bigPatch = torch.flip(rotated_bigPatch, [0])

        if flip_left_right:
            rotated_bigPatch = torch.flip(rotated_bigPatch, [1])


        #===== Crop the middle part to get the patch =====#
        center_index = (rotated_bigPatch.size(0) - patchSize) // 2
        patch_agumented = rotated_bigPatch[center_index:center_index+patchSize, center_index:center_index+patchSize]

        return patch_agumented

    def get_blocks_sides(self,img,patchSize):
        '''
        This function is used to do data agumentation for the patches sampled on the boundary of the whole image.
        For this type of side-patches, we can only rotate them by 0, 90, 180, 270 degrees and flip them, since other angles will cause the patch to be out of the image and create blank pixels.
        The flip is randomly selected from {0, 1} with two directions: up-down and left-right. flip=0 means no flip, flip=1 means flip.
        '''

        #===== Randomly select the patch to sample from which boundary of the image, 1: Left, 2: Right, 3: Top, 4: Bottom =====#
        H,W = img.shape
        sides = torch.randint(1, 5, (1,)).item()


        #===== Randomly sample the patch locations on the boundary =====#
        if sides == 1:
            leftTop_H = torch.randint(0, int(H - patchSize), (1,)).item()
            leftTop_W = 0
        
        elif sides == 2:
            leftTop_H = torch.randint(0, int(H - patchSize), (1,)).item()
            leftTop_W = W - patchSize
        
        elif sides == 3:
            leftTop_H = 0
            leftTop_W = torch.randint(0, int(W - patchSize), (1,)).item()

       
        else:
            leftTop_H = H - patchSize
            leftTop_W = torch.randint(0, int(W - patchSize), (1,)).item()

        
        #===== Crop the patch =====#
        patch = img[leftTop_H:leftTop_H + patchSize, leftTop_W:leftTop_W + patchSize]


        #===== Random rotate 0,90, 180, 270 and random flip =====#
        angle = torch.randint(0, 4, (1,)).item()
        flip_up_down = torch.randint(0, 2, (1,)).item()
        flip_left_right = torch.randint(0, 2, (1,)).item()

        patch_augmented = self.agumentation(patch, angle*90, patchSize, flip_up_down=flip_up_down, flip_left_right=flip_left_right)

        return patch_augmented

    def get_blocks_aug(self,img,patchSize):
        '''
        This function is used to sample patches away from the boundary.
        It contains two types of sampling:
        1. Randomly sample patches around the center of the image.
            Since we need to the random rotations to the patches, we hope this rotation will not cause any blank pixels in the patch.
            The strategy is that we first crop a big patch (H' = \sqrt(2) * H, W' = \sqrt(2) * W), and then randomly rotate this big patch by any angle.
            Finally, we crop the desired patch from the middle of the rotated big patch. 
            In this case we can ensure that no blank pixels will occur in the patch with any rotation angle.
            However, to crop this big patch, the patch locations can only be selected away from the boundary of the image. 

        2. Randomly sample patches around the boundary of the image.
            For this type of side-patches, we can only rotate them by 0, 90, 180, 270 degrees and flip them, since other angles will cause the patch to be out of the image and create blank pixels.

        We use a weight to control the probability of the two types of sampling. The weight is set to 0.1 by default, meaning that 10% of the patches are sampled from the boundary of the image.
        '''


        H, W = img.shape
        patches = torch.zeros((self.batchSize, patchSize, patchSize), dtype=img.dtype, device=self.device)

        #===== The size of the big patch =====#
        cropSize = int(np.ceil(patchSize * np.sqrt(2))) + 2

        #===== Randomly sample the patch locations =====#
        for i in range(self.batchSize):
            #===== If random number < self.sideSampling, we take samples from the boundary =====#
            random_number = torch.rand(1).item()
            if random_number < self.sideSampling:
                patch_agumented = self.get_blocks_sides(img,patchSize)

            #===== Else we take samples away from the boundary =====#
            else:
                leftTop_H = torch.randint(0, int(H - cropSize), (1,)).item()
                leftTop_W = torch.randint(0, int(W - cropSize), (1,)).item()


                bigPatch = img[leftTop_H:leftTop_H + cropSize, leftTop_W:leftTop_W + cropSize]


                #===== Randomly rotate and flip the big patch =====#
                angle = torch.randint(0, 360, (1,)).item()
                flip_up_down = torch.randint(0, 2, (1,)).item()
                flip_left_right = torch.randint(0, 2, (1,)).item()

                patch_agumented = self.agumentation(bigPatch, angle, patchSize, flip_up_down=flip_up_down, flip_left_right=flip_left_right)
                
            #===== Save the patch =====#
            patches[i,:, :] = patch_agumented

        return patches




