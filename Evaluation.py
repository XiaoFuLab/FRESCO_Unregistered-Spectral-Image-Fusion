
from xmlrpc.client import boolean
import os
import sys
sys.path.append("..")
from torch.autograd import Variable
import torch
from PIL import Image
import numpy as np
import cv2
import scipy.io
from skimage.metrics import structural_similarity as compare_ssim
from torchvision.models import inception_v3,resnet18
from torchvision.models.inception import Inception_V3_Weights

from scipy import linalg
from torchvision.transforms import Normalize, ToTensor, Compose
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import lpips
import wandb

class Evaluator:
    '''
    This class is used to evaluate the performance of the algorithm.
    It contains the PSNR, SSIM, SAM, ERGAS, FID and LPIPS metrics for both SRI_H and SRI_M evaluations.
    '''

    def __init__(self,opt,data,device):
        '''
        Initialize the data, parameters for evaluation.

        input:
            opt: the arguments.
            data: the data dictionary, which should contain the {SRI_M, HSI, MSI, Q, SRI_H}
            sr: the super-resolution scale.
            devices: the devices used for training and evaluation.
        '''
        #===== store the parameters =====
        self.opt = opt
        self.sr = opt.sr
        self.device = device
        self.scaleUp = opt.scaleUp
        self.R = opt.numMaterial
        self.patchSize = opt.patchSize
        self.overlap = opt.overlap
        #===== initialize the parameters used to compute FID =====#
        self.mu_M_GT = None
        self.sigma_M_GT = None
        self.mu_H_GT = None
        self.sigma_H_GT = None

        #===== load the data =====#
        self.SRI_M = data['SRI_M']
        self.MSI = data['MSI']
        self.HSI = data['HSI']
        self.SRI_H = data['SRI_H']
        self.PM = data['Q']
        self.IH, self.JH, self.KH = self.HSI.shape
        self.IM, self.JM, self.KM = self.MSI.shape

        #===== load the RGB bands =====#
        try:
            if not opt.RGB:  # check if RGB is provided
                print('RGB is not provided. Now evaluating on all bands.')
                self.RGB = [] # use all bands
            else:
                self.RGB = np.array([int(x) for x in opt.RGB.split(',')])  # convert to list of integers
        except ValueError:
            print('RGB should be a list of integers, e.g. "25,16,3". Now evaluating on all bands.')
            self.RGB = []  # use all bands


        #===== initialize the mu and sigma used to compute the FID (if opt.FID == True) =====#
        if self.opt.FID:
            print('Computing the mu and sigma on ground-truth SRIs for FID...')
            if self.opt.Stage1:
                self.mu_M_GT,self.sigma_M_GT = compute_mu_sigma(self.SRI_M,self.RGB,device)
            
            if self.opt.Stage2:
                self.mu_H_GT,self.sigma_H_GT = compute_mu_sigma(self.SRI_H,self.RGB,device)
    

    def Evaluate_SRI_M(self,SRI_M_reconstructed):
        '''
        Evaluate the M-SRI task.
        input:
            SRI_M_reconstructed: the reconstructed M-SRI.

        output:
            psnr_M: the PSNR of the M-SRI.
            ssim_M: the SSIM of the M-SRI.
            sam_M: the SAM of the M-SRI.
            ergas_M: the ERGAS of the M-SRI.
            lpips_M: the LPIPS of the M-SRI.
            fid_M: the FID of the M-SRI.
        '''

        #===== compute the metrics =====#
        psnr_M = calculate_bandwise_psnr(SRI_M_reconstructed, self.SRI_M)
        ssim_M = calculate_bandwise_ssim(SRI_M_reconstructed, self.SRI_M)
        sam_M = calculate_sam(SRI_M_reconstructed,self.SRI_M)
        ergas_M = calculate_bandwise_ergas(SRI_M_reconstructed, self.SRI_M, self.sr)


        #===== compute the LPIPS =====#
        lpips_M = 0
        if len(self.RGB) == 3:
            lpips_M = calculate_lpips(SRI_M_reconstructed, self.SRI_M, self.RGB,self.device, net='alex')


        #===== compute the FID =====#
        fid_M = 0
        if self.opt.FID:
            #===== compute the mu and sigma =====#
            mu_M_reconstructed,sigma_M_reconstructed = compute_mu_sigma(SRI_M_reconstructed,self.RGB,self.device)

            #===== compute the FID =====#
            fid_M = compute_fid(self.mu_M_GT,self.sigma_M_GT,mu_M_reconstructed,sigma_M_reconstructed)


        #===== return the metrics as a dictionary=====#
        return {'psnr_M': psnr_M,'ssim_M': ssim_M,'sam_M': sam_M,'ergas_M': ergas_M,'lpips_M': lpips_M,'fid_M': fid_M}

    def Evaluate_SRI_H(self,SRI_H_reconstructed):
        '''
        Evaluate the H-SRI task.
        input:
            SRI_H_reconstructed: the reconstructed H-SRI.

        output:
            psnr_H: the PSNR of the H-SRI.
            ssim_H: the SSIM of the H-SRI.
            sam_H: the SAM of the H-SRI.
            ergas_H: the ERGAS of the H-SRI.
            lpips_H: the LPIPS of the H-SRI.
            fid_H: the FID of the H-SRI.
        '''

        #===== compute the metrics =====#
        psnr_H = calculate_bandwise_psnr(SRI_H_reconstructed, self.SRI_H)
        ssim_H = calculate_bandwise_ssim(SRI_H_reconstructed, self.SRI_H)
        sam_H = calculate_sam(SRI_H_reconstructed,self.SRI_H)
        ergas_H = calculate_bandwise_ergas(SRI_H_reconstructed, self.SRI_H, self.sr)


        #===== compute the LPIPS =====#
        lpips_H = 0
        if len(self.RGB) == 3:
            lpips_H = calculate_lpips(SRI_H_reconstructed, self.SRI_H, self.RGB,self.device, net='alex')


        #===== compute the FID =====#
        fid_H = 0
        if self.opt.FID:
            #===== compute the mu and sigma =====#
            mu_H_reconstructed,sigma_H_reconstructed = compute_mu_sigma(SRI_H_reconstructed,self.RGB,self.device)

            #===== compute the FID =====#
            fid_H = compute_fid(self.mu_H_GT,self.sigma_H_GT,mu_H_reconstructed,sigma_H_reconstructed)


        #===== return the metrics =====#
        return {'psnr_H': psnr_H,'ssim_H': ssim_H,'sam_H': sam_H,'ergas_H': ergas_H,'lpips_H': lpips_H,'fid_H': fid_H}

    def reconstruct_SRI_M(self,SM3,CH):
        '''
        Reconstruct the SRI_M from the SM3 and CH.
        '''

        #===== reconstruct the SRI_M =====#
        SRI_M_reconstructed = np.dot(SM3,CH.T).reshape((self.IM,self.JM,self.KH),order='F')


        #===== clip the SRI_M to [0,1] =====#
        SRI_M_reconstructed = np.maximum(0, np.minimum(1, SRI_M_reconstructed))

        return SRI_M_reconstructed
    


    def reconstruct_SRI_H(self,netG_H2M,SH,CH):
        '''
        Reconstruct the SRI_H from the SH and CH and the trained super-resolution network netG_H2M.
        The idea is to use a sliding window (with stride = patchSize - overlap) to get a lot of patches from the SH.
        Then, we use the trained super-resolution network to get the high-resolution patches.
        Finally, we combine the high-resolution patches to get the high-resolution abundance maps f(SH), and combine with CH to recover SRI_H. 
        
        Note: Here for the normalization issue in the network, do not turn on the eval mode.
        '''


        netG_H2M = netG_H2M.to(self.device)

        with torch.no_grad():
            
            #===== Scale the abundance maps and endmembers =====#
            SH = SH * self.scaleUp
            CH = CH/self.scaleUp


            #===== Initialize the high-resolution abundance maps =====#
            SR_SH = np.zeros((int(self.IH * self.sr), int(self.JH * self.sr),self.R))


            for r in range(self.R):
                #===== Split the SH into overlapped patches =====#
                SH_r = SH[:,:,r]
                patches, indices = split_image_with_overlap(SH_r, self.patchSize, self.overlap)
                patches = np.expand_dims(patches, axis=1)
                patches = torch.from_numpy(patches).float().to(self.device)


                #===== Initialize the high-resolution patches =====#
                SR_H_patches = torch.zeros(patches.shape[0],1,int(self.patchSize * self.sr), int(self.patchSize * self.sr)).to(self.device)


                #===== There might be a lot of overlapped patches, which will cause "out of memory" if we send all of them to the network =====#
                #=====                  so we randomly shuufle them and send at most 50 patches to the network at once                    =====#
                index_input = np.arange(patches.shape[0])
                np.random.shuffle(index_input)

                if patches.shape[0] <= 50:
                     SR_H_patches = netG_H2M(patches)

                #===== send 50 patches to the network, use selected_index to remember the locations of the patch =====#
                else:
                    for k in range(patches.shape[0]//50):
                        if k==patches.shape[0]//50-1:
                            selected_index = index_input[k*50:]
                            SR_H_patches_selected = netG_H2M(patches[selected_index])
                            SR_H_patches[selected_index] = SR_H_patches_selected

                        else:
                            selected_index = index_input[k*50:(k+1)*50]
                            SR_H_patches_selected = netG_H2M(patches[selected_index])
                            SR_H_patches[selected_index] = SR_H_patches_selected

                SR_H_patches = np.squeeze(SR_H_patches.cpu().detach().numpy())


                #===== Combine the high-resolution patches to get the high-resolution abundance maps =====#
                SR_SH[:,:,r] = combine_blocks_with_overlap(SR_H_patches, indices,[self.IH,self.JH], self.patchSize, self.sr, self.overlap)


        #==== Reconstruct the SRI_H from the high-resolution abundance maps and endmembers =====#
        SR_SH3 = SR_SH.reshape((-1,self.R),order='F')
        SRI_H_reconstructed = np.dot(SR_SH3,CH.T).reshape((self.IM,self.JM,self.KH),order='F')
        

        #===== Clip the SRI_H to [0,1] =====#
        SRI_H_reconstructed = np.maximum(0, np.minimum(1, SRI_H_reconstructed))
    
        return SRI_H_reconstructed,SR_SH
    



    def write_dict_to_file(self,file_path,data_dict,stage=1, epoch = -1, separator = '; ', end='\n'):
        """
        Write a dictionary to a file, with each key-value pair on a new line.

        inputs:
            dict (dict): The dictionary to write.
            file_path (str): The path to the file where the dictionary will be written.
            separator (str): The separator between key and value. Default is '; '.
            end (str): The string appended after the last line. Default is '\n'.
        """
        with open(file_path, 'a') as f:
            if stage == 1:
                f.write('Evaluation for SRI_M: \n')
            elif stage ==2 and epoch == 0:
                f.write('Evaluation for SRI_H: \n')

            if epoch != -1:
                f.write(f'Epoch: {epoch}{separator}')
            for key, value in data_dict.items():
                f.write(f'{key}: {value}{separator}')
            f.write(end)
            f.write('\n')




    def write_dict_to_wandb(self,data_dict,stage='MSR',SRI_M_reconstructed=None,SRI_H_reconstructed=None):
        """
        Write a dictionary to wandb, with each key-value pair on a new line.

        inputs:
            dict (dict): The dictionary to write.
            stage (int): The stage of the evaluation. 1 for SRI_M, 2 for SRI_H.
        """
        RGB_show = self.RGB
        if len(self.RGB) != 3:
            RGB_show = self.SRI_H.shape[2] //2

        if wandb.run is not None:
            if stage == 'MSR':
                #===== write the metrics to wandb =====#
                table = wandb.Table(columns=["Metric", "Value"])
                for key, value in data_dict.items():
                    table.add_data(key, value)

                wandb.log({"MSR Metric": table})

                wandb.log({"MSR:":[ wandb.Image(self.SRI_M[:,:,RGB_show], caption="GT_SRI_M"),
                                    wandb.Image(SRI_M_reconstructed[:,:,RGB_show], caption="Est_SRI_M"),
                                ]
                            })

            elif stage == 'HSR':
                wandb.log(data_dict)
                wandb.log({"HSR:":[ wandb.Image(self.SRI_H[:,:,RGB_show], caption="GT_SRI_H"),
                                    wandb.Image(SRI_H_reconstructed[:,:,RGB_show], caption="Est_SRI_H"),
                                ]
                            })
            
            elif stage == 'Loss':
                wandb.log(data_dict)




def print_dict(data_dict, epoch = -1):
    """
    Print a dictionary to the console, with each key-value pair on a new line.

    inputs:
        dict (dict): The dictionary to print.
        epoch (int): The epoch number. Default is -1.
    """
    if epoch != -1:
        print(f'Epoch: {epoch}')
    for key, value in data_dict.items():
        print(f'{key}: {value}; ')

    print('\n')




def record_loss(data_dict, epoch=0, iter=0, log_wandb=False):
    """
    Write a dictionary to wandb, with each key-value pair on a single line.

    inputs:
        data_dict (dict): The dictionary to write.
        epoch (int): The current epoch number.
        iter (int): The current iteration number.
        log_wandb (bool): Whether to log the data to wandb.
    """
    #===== Print epoch and iteration =====#
    print(f'Epoch: {epoch}, Iter: {iter}; ', end='')

    #===== Combine all key-value pairs into a single string =====#
    loss_str = "; ".join([f'{key}: {value}' for key, value in data_dict.items()])
    print(loss_str + '\n')  # Print the combined string

    #===== Log to wandb if enabled =====#
    if log_wandb and wandb.run is not None:
        wandb.log(data_dict)





def split_image_with_overlap(image, patchSize, overlap):
    '''
    Split the image into small patches with overlap.
    input:
        image: the input image, shape: (H,W).
        patchSize: the size of the patch.
        overlap: the overlap size.
    output:
        patches: the small patches, shape: (num_patches, patchSize, patchSize).
        indices: the top-left coordinate of each block, shape: (num_patches, 2).
    '''

    #===== Initialize the patches and indices =====#
    patches = [] # store small patches
    indices = [] # store the top-left coordinate of each block


    #===== Split the images into overlapped patches =====#
    h, w = image.shape
    step = patchSize - overlap
    for i in range(0, h - overlap, step):
        for j in range(0, w - overlap, step):
            block = image[i:i+patchSize, j:j+patchSize]

            #===== Save the patches themselves and their coordinates =====#
            patches.append(block)
            indices.append((i, j))
   
    return np.array(patches), indices





def combine_blocks_with_overlap(SR_H_patches, indices,spatialShape, patchSize, scale_factor, overlap):
    '''
    Combine the small patches into a large image with overlap.
    For the overlapped area, the average value is taken to smooth the image and avoid the artifacts.

    input:
        SR_H_patches: the small patches, shape: (num_patches, patchSize, patchSize).
        indices: the top-left coordinate of each block, shape: (num_patches, 2).
        spatialShape: the shape of the whole image.
        patchSize: the size of the small block.
        scale_factor: the scale factor.
        overlap: the overlap size.
    output:
        combined_image: the combined image, shape: (H*scale_factor, W*scale_factor).
    '''


    #===== Initialize the combined image and weight matrix =====#
    h, w = spatialShape
    SR_patchSize = int(patchSize * scale_factor)
    combined_image = np.zeros((h * scale_factor, w * scale_factor), dtype=np.float32)
    weight_matrix = np.zeros((h * scale_factor, w * scale_factor), dtype=np.float32)
    

    #===== Combine the patches into the large image =====#
    for k, (i, j) in enumerate(indices):
        SR_H_patch = SR_H_patches[k]
        x_start = int(i * scale_factor)
        y_start = int(j * scale_factor)
        combined_image[x_start:x_start+SR_patchSize, y_start:y_start+SR_patchSize] += SR_H_patch
        weight_matrix[x_start:x_start+SR_patchSize, y_start:y_start+SR_patchSize] += 1


    #===== Average the overlapped area =====#
    combined_image /= weight_matrix


    return combined_image



def calculate_bandwise_psnr(est, ref):
    """
    Calculate bandwise PSNR and take the mean.

    input: 
            est: estimated image, shape [IM, JM, KH], values in [0, 1].
            ref: reference image, shape [IM, JM, KH], values in [0, 1].

    output: float: mean PSNR value over all bands.
    """

    KH = est.shape[2]
    psnr_values = np.zeros(KH)

    for i in range(KH):
        max_val = np.max(ref[:, :, i])
        mse = np.mean((est[:, :, i] - ref[:, :, i]) ** 2)
        
        if mse == 0:
            psnr_values[i] = float('inf')
        else:
            psnr_values[i] = 10 * np.log10((max_val ** 2) / mse)

    return np.mean(psnr_values)



def calculate_bandwise_ssim(est,ref):
    """
    Calculate bandwise SSIM and take the mean.

    input: est: estimated image, shape [IM, JM, KH], values in [0, 1].
           ref: reference image, shape [IM, JM, KH], values in [0, 1].

    output: float: mean SSIM value over all bands.
    """
    KH = est.shape[2]
    ssim_values = np.zeros(KH)

    for i in range(KH):
        ssim_values[i] = compare_ssim(est[:, :, i], ref[:, :, i], data_range=1.0)

    return np.mean(ssim_values)




def calculate_bandwise_ergas(est,ref, sr):
    """
    Calculate bandwise ERGAS and take the mean.
    input: est: estimated image, shape [IM, JM, KH], values in [0, 1].
           ref: reference image, shape [IM, JM, KH], values in [0, 1].
           sr: super-resolution scale factor.

    output: mean ERGAS value over all bands.
    """
    KH = est.shape[2]
    ergas_bandwise = np.zeros(KH)

    for i in range(KH):
        mean_value = np.mean(ref[:, :, i])
        mse_value = np.mean((est[:, :, i] - ref[:, :, i]) ** 2)
        ergas_bandwise[i] = 100 / sr * np.sqrt(mse_value / (mean_value ** 2))

    return np.mean(ergas_bandwise)


def calculate_sam(est, ref):
    """
    Compute the Spectral Angle Mapper (SAM)

    input: 
            est: estimated image, values in [0, 1].
            ref: reference image, values in [0, 1].

    output: The mean SAM value (in degrees) over all valid pixels.

    """

    #===== Compute dot product for each pixel =====#
    dot_product = np.sum(est * ref, axis=2)


    #===== Compute the L2 norm of each pixel spectrum =====#
    norm_est = np.linalg.norm(est, axis=2)
    norm_ref = np.linalg.norm(ref, axis=2)


    #===== Prevent division by zero (add a small epsilon) =====# 
    eps = 1e-8
    norm_est = np.maximum(norm_est, eps)
    norm_ref = np.maximum(norm_ref, eps)


    #===== Compute the cosine of the spectral angle =====#
    cos_theta = dot_product / (norm_est * norm_ref)


    #===== Ensure values are within the valid range [-1, 1] to avoid NaN =====#
    cos_theta = np.clip(cos_theta, -1.0, 1.0)


    #===== Compute SAM in **degrees** =====#
    sam_map = np.arccos(cos_theta) * (180 / np.pi)  # Convert radians to degrees


    #===== Compute mean SAM value =====#
    mean_sam = np.nanmean(sam_map)

    return mean_sam


def calculate_lpips(est, ref, RGB,device, net='alex'):
    """
    Calculate LPIPS distance between two images using a specified network.

    input:
        est (numpy array): Estimated image, shape [H, W, C], values in [0, 1].
        ref (numpy array): Reference image, shape [H, W, C], values in [0, 1].
        RGB (list): Indices of RGB channels.
        net (str): Network to use ('alex', 'vgg', or 'squeeze').
        cuda (bool): Whether to use GPU for computation.

    output: LPIPS distance.
    """

    #===== Check if RGB is provided =====#
    RGB_est = est[:, :, RGB]
    RGB_ref = ref[:, :, RGB]

    assert RGB_est.shape == RGB_ref.shape, "Input images must have the same shape"
    assert RGB_est.shape[2] == 3, "Input images must have 3 channels (RGB)"
    

    #===== Convert to torch tensor and reshape to [1, 3, H, W] =====#
    def to_tensor(img_np):
        img_tensor = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0).float()  # [1, 3, H, W]
        img_tensor = img_tensor * 2 - 1  # Normalize from [0,1] to [-1,1]
        return img_tensor

    est_tensor = to_tensor(RGB_est)
    ref_tensor = to_tensor(RGB_ref)


    #===== Load LPIPS model =====#
    loss_fn = lpips.LPIPS(net=net)  # 'alex' | 'vgg' | 'squeeze'


    #===== Move model and tensors to device =====#
    loss_fn = loss_fn.to(device)
    est_tensor = est_tensor.to(device)
    ref_tensor = ref_tensor.to(device)


    #===== Compute LPIPS distance =====#
    with torch.no_grad():
        dist = loss_fn(est_tensor, ref_tensor)

    return dist.item()



    


        
def compute_mu_sigma(SRI,RGB,device):
    '''
    This function is used to compute the mu and sigma used to compute the FID.
    We only evaluate the FID on the RGB bands.
    We randomly sample 1000 blocks with size 80 \times 80 \times 3 from the image.
    The mu and sigma are computed for 10 times and the average is taken.

    input:
        SRI: the hyperspectral image to be sampled.
        RGB: the RGB bands of the image, if RGB == [], then we random sample bands as well.
    '''
    #===== Initialize the mu and sigma =====#
    mu_list = np.zeros((3,2048))
    sigma_list = np.zeros((3,2048,2048))


    for i in range(3):
        #===== get the blocks =====#
        blocks = get_blocks_FID(SRI,80,1000,RGB)

        #===== compute the mu and sigma =====#
        mu_list[i],sigma_list[i] = compute_feature_FID(blocks,device)


    return mu_list,sigma_list



def get_blocks_FID(img,blockSize,numSamples,RGB):
    '''
    This function is used to random sample patches from the RGB bands of a hyperspectral image.

    input:
        img: the hyperspectral image to be sampled.
        blockSize: the size of the patch.
        numSamples: the number of patches to be sampled.
        RGB: the RGB bands of the image, if RGB == [], then we random sample bands as well.

    output:
        patch: the sampled patches, shape: (numSamples, 3, blockSize, blockSize).
    '''


    H,W,K= img.shape
    patch = np.zeros((numSamples,3,blockSize,blockSize))
    
    for i in range(numSamples):
        #===== get the random left-top point =====#
        leftTop_H = np.random.randint(0,H-blockSize+1)
        leftTop_W = np.random.randint(0,W-blockSize+1)
        bandStart = np.random.randint(0,K-2)

        #===== get the block =====#
        if len(RGB) ==3:
            patch[i] = img[leftTop_H:leftTop_H+blockSize, leftTop_W:leftTop_W+blockSize,RGB].transpose(2,0,1)
        else:
            patch[i] = img[leftTop_H:leftTop_H+blockSize, leftTop_W:leftTop_W+blockSize,bandStart:bandStart+3].transpose(2,0,1)
            
    return patch



def compute_feature_FID(batch_data,device):
    '''
    This function is used to compute the mu and sigma of feature of the patches using the InceptionV3 model.
    
    input:
        batch_data: the input batch data, shape: (numSamples * 3 * blockSize * blockSize).
        cuda: whether to use cuda or not.
    
    output:
        mu: the mean of the features, shape: (2048,).
        sigma: the covariance of the features, shape: (2048, 2048).
    '''

    #===== Normalize the image =====#
    batch_data = np.float32(batch_data)
    global_min = batch_data.min()
    global_max = batch_data.max()
    batch_normalized = (batch_data - global_min) / (global_max - global_min)


    #===== convert the batch to tensor =====#
    batch_tensor = torch.from_numpy(batch_normalized)

    #===== normalize the batch along the channel dimension =====#
    custom_normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    batch_normalized_final = custom_normalize(batch_tensor)

    #===== Load InceptionV3 model =====#
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    model.fc = torch.nn.Identity()

    model = model.to(device)
    batch_normalized_final = batch_normalized_final.to(device)

    model.eval()

    #===== Forward pass through InceptionV3 =====#
    with torch.no_grad():
        features = model(batch_normalized_final)
    
    #===== Convert features to numpy array =====#
    features = features.detach().cpu().numpy()

    #===== Compute mu and sigma =====#
    mu, sigma = features.mean(axis=0), np.cov(features, rowvar=False)

    return mu, sigma


def compute_fid(mu1_list,sigma1_list,mu2_list,sigma2_list):
    '''
    Compute the FID between two distributions.
    input:
        mu1: the mean of the first distribution.
        sigma1: the covariance of the first distribution.

        mu2: the mean of the second distribution.
        sigma2: the covariance of the second distribution.

    output:
        fid: the FID between the two distributions.

    '''

    FID_list = np.zeros(mu1_list.shape[0])

    for i in range(mu1_list.shape[0]):
        #===== Take the mean and covariance of the two distributions =====#
        mu1 = mu1_list[i]
        sigma1 = sigma1_list[i]
        mu2 = mu2_list[i]
        sigma2 = sigma2_list[i]

        #===== Compute the FID =====#
        ssdiff = np.sum((mu1 - mu2)**2.0)
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        FID_list[i] = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)

    #===== Take the mean of the FID =====#
    return np.mean(FID_list)


   

