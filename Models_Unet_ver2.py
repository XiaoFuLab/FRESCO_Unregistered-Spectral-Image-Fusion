import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import functools

import torch.nn as nn
import torch.nn.functional as F



#======= Generator from high-resolution patches to low-resolution patches =======#
class Generator_down(nn.Module):
    '''
    Generator from high-resolution patch to low-resolution patch
    The network currently only support downsampling by 2, 3, or 4, but one can modify the code to support other downsampling factors.
    We also have tried to call x = F.interpolate(x, scale_factor=1/self.scale_factor, mode='bilinear', align_corners=False) to support other downsampling factors, but it does not work that well.
    '''

    def __init__(self, input_nc=1, output_nc=1, inter_nc=64, n_blocks=6,scale_factor=4):
        super(Generator_down, self).__init__()
        
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.inter_nc = inter_nc
        self.n_blocks = n_blocks
        self.scale_factor = scale_factor
        
        # Input blocks
        InBlock = []
        
        InBlock += [nn.Conv2d(input_nc, inter_nc, kernel_size=1, stride=1, padding=0),
                    nn.LeakyReLU(0.2)]
        
        # ResnetBlocks
        ResnetBlocks = []
        
        for i in range(n_blocks):
            ResnetBlocks += [ResnetBlock(inter_nc)]
        
        # Output block
        OutBlock = []
        if self.scale_factor%2==0:
            for r in range(int(math.log2(self.scale_factor))):
                OutBlock += [nn.ReflectionPad2d(1),
                            nn.Conv2d(inter_nc, inter_nc, kernel_size=3, stride=2, padding=0),
                            nn.LeakyReLU(0.2)]

        elif self.scale_factor==3:
            OutBlock += [nn.ReflectionPad2d(1),
                         nn.Conv2d(self.inter_nc, self.inter_nc, kernel_size=3, stride=3, padding=0),
                         nn.LeakyReLU(0.2)]
            
        else:
            raise ValueError('The current downsampling generator only support downsampling factor = 2^k or 3, one can modify the architecture to support other downsampling factors.')
        
        OutBlock += [nn.ReflectionPad2d(1),
                     nn.Conv2d(inter_nc, output_nc, kernel_size=3, stride=1, padding=0),
                     nn.LeakyReLU(0.2)]
        
        self.InBlock = nn.Sequential(*InBlock)
        self.ResnetBlocks = nn.Sequential(*ResnetBlocks)
        self.OutBlock = nn.Sequential(*OutBlock)

    def forward(self,x):
        out = self.InBlock(x)
        out = self.ResnetBlocks(out)
        out = self.OutBlock(out)
        return out
    


class ResnetBlock(nn.Module):

    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        
        conv_block = []
        
        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, padding=0), 
                       nn.BatchNorm2d(dim),
                       nn.LeakyReLU(0.2)]
        
        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, padding=0), 
                       nn.BatchNorm2d(dim), 
                       nn.LeakyReLU(0.2)]
        
        self.conv_block = nn.Sequential(*conv_block)
        
    def forward(self, x):
        out = self.conv_block(x)
        # Skip connection
        out = out + x
        return out
    

    

# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ConvBlock, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.LeakyReLU(inplace=True),
#         )

#     def forward(self, x):
#         return self.conv(x)
    

# class Encoder(nn.Module):
#     def __init__(self, in_channels, out_channels) -> None:
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.MaxPool2d(2),
#             ConvBlock(in_channels, out_channels)
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         return x
    

# class Decoder(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(Decoder, self).__init__()
#         self.conv = nn.Sequential(
#             nn.UpsamplingBilinear2d(scale_factor=2),
#             nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.LeakyReLU(),
#         )
#         self.conv_block = ConvBlock(in_channels, out_channels)

#     def forward(self, x, skip):
#         x = self.conv(x)
#         x = torch.concat([x, skip], dim=1)
#         x = self.conv_block(x)
#         return x

# class FinalOutput(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(FinalOutput, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
#             #nn.Tanh()
#             nn.LeakyReLU(0.2)
#         )

#     def forward(self, x):
#         return self.conv(x)

# class FirstFeature(nn.Module):
#     '''
#     Implementation of UNET with Skip connections
#     '''
#     def __init__(self, in_channels, out_channels):
#         super(FirstFeature, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
#             nn.LeakyReLU()
#         )

#     def forward(self, x):
#         return self.conv(x)


# class SR_Unet(nn.Module):
#     def __init__(
#             self, n_channels=1, n_classes=1,scale_factor=4
#     ):
#         super(SR_Unet, self).__init__()
#         self.scale_factor = scale_factor
#         self.n_channels = n_channels
#         self.n_classes = n_classes

#         self.in_conv1 = FirstFeature(n_channels, 64)
#         self.in_conv2 = ConvBlock(64, 64)

#         self.enc_1 = Encoder(64, 128)
#         self.enc_2 = Encoder(128, 256)
#         self.enc_3 = Encoder(256, 512)
#         self.enc_4 = Encoder(512, 1024)

#         self.dec_1 = Decoder(1024, 512)
#         self.dec_2 = Decoder(512, 256)
#         self.dec_3 = Decoder(256, 128)
#         self.dec_4 = Decoder(128, 64)

#         self.out_conv = FinalOutput(64, n_classes)

#     def forward(self, x):
#         x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear')
#         x = self.in_conv1(x)
#         x1 = self.in_conv2(x)

#         x2 = self.enc_1(x1)
#         x3 = self.enc_2(x2)
#         x4 = self.enc_3(x3)
#         x5 = self.enc_4(x4)

#         x = self.dec_1(x5, x4)
#         x = self.dec_2(x, x3)
#         x = self.dec_3(x, x2)
#         x = self.dec_4(x, x1)

#         x = self.out_conv(x)
#         return x
    
#     def init_weights(self):
#         # for m in self.modules():
#         #     if isinstance(m, nn.Conv2d):
#         #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#         #         if m.bias is not None:
#         #             nn.init.constant_(m.bias, 0)
#         #     elif isinstance(m, nn.BatchNorm2d):
#         #         nn.init.constant_(m.weight, 1)
#         #         nn.init.constant_(m.bias, 0)
#         classname = self.__class__.__name__
#         if classname.find('Conv') != -1:
#             torch.nn.init.normal_(self.weight.data, 0.0, 0.02)
#         elif classname.find('BatchNorm2d') != -1:
#             torch.nn.init.normal_(self.weight.data, 1.0, 0.02)
#             torch.nn.init.constant_(self.bias.data, 0.0)
#         elif classname.find('Linear') != -1:
#             torch.nn.init.normal_(self.weight.data, 0.0, 0.02)
#             if self.bias is not None:
#                 torch.nn.init.constant_(self.bias.data, 0.0)






#======= Generator from low-resolution patches to high-resolution patches =======#
class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x


class SR_Unet(nn.Module):
    '''
    Implementation of a super-resolution Unet with Skip connections
    It can fit any scale factor, but the input patch size (after interpolation) should be larger or equal to 48.
    '''

    def __init__(self, n_channels=1, n_classes=1, scale_factor=4):
        super().__init__()
        self.scale_factor = scale_factor

        # Inline FirstFeature
        self.in_conv1 = nn.Sequential(
            nn.Conv2d(n_channels, 64, 1, 1, 0, bias=False),
            nn.LeakyReLU(inplace=True)
        )

        # Inline ConvBlock
        self.in_conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
        )

        self.enc_1 = Encoder(64, 128)
        self.enc_2 = Encoder(128, 256)
        self.enc_3 = Encoder(256, 512)
        self.enc_4 = Encoder(512, 1024)

        self.dec_1 = Decoder(1024, 512)
        self.dec_2 = Decoder(512, 256)
        self.dec_3 = Decoder(256, 128)
        self.dec_4 = Decoder(128, 64)

        # Inline FinalOutput
        self.out_conv = nn.Sequential(
            nn.Conv2d(64, n_classes, 1, 1, 0, bias=False),
            #nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        inp = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        x = self.in_conv1(inp)
        x1 = self.in_conv2(x)

        x2 = self.enc_1(x1)
        x3 = self.enc_2(x2)
        x4 = self.enc_3(x3)
        x5 = self.enc_4(x4)

        x = self.dec_1(x5, x4)
        x = self.dec_2(x, x3)
        x = self.dec_3(x, x2)
        x = self.dec_4(x, x1)

        return self.out_conv(x)







#======= Multi--domain discriminator for high-resolution patches =======#
class Discriminator_M(nn.Module):
    '''
    Multi-domain discriminator for high-resolution patches.
    '''

    def __init__(self,num_domains=4):
        super(Discriminator_M, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, num_domains, kernel_size=1)
        )


    def forward(self, x, y):
        x = self.net(x)
        # Average pooling and flatten
        out=  F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], x.size()[1])
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        out = out[idx, y]  # (batch)
        return out



