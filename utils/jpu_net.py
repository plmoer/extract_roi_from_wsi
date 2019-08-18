'''
Author: Linmin
Update: May, 03, 2019
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
from .enc_module.customize import JPU
from .enc_module.encnet_1 import EncHead
from .enc_module.base import BaseNet
from .enc_module.fcn import FCNHead

up_kwargs = {'mode': 'trilinear', 'align_corners': True}


class JPUNet(nn.Module):
    def __init__(self, n_channels, n_classes,  ndf=16, ngf=128, bVAE=0, nX=32, nY=32, nZ=32):
        super(JPUNet, self).__init__()
        #self.auxlayer = FCNHead(ndf*4, n_classes, norm_layer=None)
        self.ndf = ndf
        self.ngf = ngf
        self.bVAE = bVAE
        self.inc = inConv(n_channels, ndf)
        self.resU_0 = resUnit(ndf, ndf)
        self.down_1 = downConv(ndf, ndf*2)
        self.resU_1 = resUnit(ndf*2, ndf*2)
        self.down_2 = downConv(ndf*2, ndf*4)
        self.resU_2 = resUnit(ndf*4, ndf*4)
        self.down_3 = downConv(ndf*4, ndf*8)
        self.resU_3 = resUnit(ndf*8, ndf*8)
        self.down_4 = downConv(ndf*8, ndf*16)
        self.resU_4 = resUnit(ndf*16, ndf*16)
        self.ave = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(ndf*16, ngf)
        self.fc2 = nn.Linear(ngf, n_classes)

        # self.up_3 = upConv(ndf*8, ndf*4)
        # self.up_2 = upConv(ndf*4, ndf*2)
        # self.up_1 = upConv(ndf*2, ndf)
        # self.final_out_1 = outConv(ndf, n_classes)
        # self.final_out_2 = outConv(ndf+ndf, n_classes)
        # # self.final_out_res = inConv(ndf, n_channels)

        # self.vd = VDConv(ndf*8, 16)
        # self.vu = VUConv(16, ndf*8)
        # self.fc0 = nn.Linear(16*nX*nY * nZ//(8*8*8*8), ngf*2)  # 32*32*32
        # self.fc11 = nn.Linear(ngf*2, ngf)  # 32*32*32
        # self.fc12 = nn.Linear(ngf*2, ngf)  # 32*32*32
        # self.fc2 = nn.Linear(ngf, 16*nX*nY * nZ//(8*8*8*8))  # 32*32*32
        # self.vae_up_1 = UP_VAE(ndf*8, ndf*4)
        # self.vae_up_2 = UP_VAE(ndf*4, ndf*2)
        # self.vae_up_3 = UP_VAE(ndf*2, ndf)
        # self.vend = nn.Conv3d(ndf, n_channels, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight, gain=np.sqrt(2.0))
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                init.constant_(m.bias, 0)
        self.jpu = JPU([ndf*4, ndf*8, ndf*16], width=ndf*4,
                       norm_layer=None, up_kwargs=up_kwargs)
        self.head = EncHead([ndf*2, ndf*4, ndf*8, ndf*16], n_classes, lateral=False,
                            se_loss=True, norm_layer=None, up_kwargs=up_kwargs)

    def getDimension(self, x):
        [nB, nC, nX, nY, nZ] = x.shape
        self.nB = nB
        self.nC = nC
        self.nX = nX
        self.nY = nY
        self.nZ = nZ

    # def reparameterize(self, mu, logvar):
    #     std = torch.exp(0.5*logvar)
    #     eps = torch.randn_like(std)
    #     # std = logvar.mul(0.5).exp_()
    #     # eps = torch.cuda.FloatTensor(std.size()).normal_()
    #     # eps = Variable(eps)p
    #     return eps.mul(std).add_(mu)

    # def decode(self, z):
    #     z = self.fc2(z)
    #     z = z.view(z.size(0), 16, (self.nX)//16, (self.nY)//16, (self.nZ)//16)
    #     z = self.vu(z)
    #     z = self.vae_up_1(z)
    #     z = self.vae_up_2(z)
    #     z = self.vae_up_3(z)
    #     z = self.vend(z)
    #     return z

    # def vae(self, x):
    #     x = self.vd(x)
    #     x = x.view(x.size(0), -1)
    #     x = F.relu(self.fc0(x))
    #     mu, logvar = self.fc11(x), self.fc12(x)
    #     z = self.reparameterize(mu, logvar)
    #     res = self.decode(z)
    #     return res, mu, logvar

    def forward(self, x):
        self.getDimension(x)
        imsize = x.size()[2:]
        x = self.inc(x)
        x = self.resU_0(x)
        en_block1 = self.down_1(x)
        en_block1 = self.resU_1(en_block1)
        en_block2 = self.down_2(en_block1)
        en_block2 = self.resU_2(en_block2)
        en_block3 = self.down_3(en_block2)
        en_block3 = self.resU_3(en_block3)
        en_block4 = self.down_4(en_block3)
        en_block4 = self.resU_4(en_block4)
        en_block4 = self.down_4(en_block3)
        features = self.jpu(en_block1, en_block2, en_block3, en_block4)

        y = list(self.head(*features))

        x = self.ave(features[-1])
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x, y[-1]


class inConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1):
        super(inConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(
                in_ch, out_ch,  kernel_size=3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.LeakyReLU(0.01)
        )
        # init.xavier_uniform_(self.conv.weight, gain=np.sqrt(2.0))  # or gain=1
        # init.constant_(self.conv.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        return x


class resUnit(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(resUnit, self).__init__()
        self.resConv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.LeakyReLU(0.01),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.LeakyReLU(0.01),
            # nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        )
        self.bridge = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.LeakyReLU(0.01),
            # nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        )

    def forward(self, x):
        out = self.resConv(x)
        identity = self.bridge(x)
        out.add_(identity)
        return out


class downConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(downConv, self).__init__()
        self.downSample = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.LeakyReLU(0.01)
            # nn.MaxPool3d(2)
        )
        self.resUnit = resUnit(out_ch, out_ch)

    def forward(self, x):
        out = self.downSample(x)
        out = self.resUnit(out)
        return out


class upConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(upConv, self).__init__()
        self.upSample = nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2),
            nn.GroupNorm(4, out_ch),
            nn.LeakyReLU(0.01)
        )
        self.resUnit = resUnit(in_ch, out_ch)

    def forward(self, x, y):
        temp = self.upSample(x)
        out = torch.cat([temp, y], dim=1)
        out = self.resUnit(out)
        return out


class outConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outConv, self).__init__()
        self.finalConv = nn.Sequential(
            # nn.GroupNorm(4, in_ch),
            # nn.LeakyReLU(0.01),
            nn.Conv3d(in_ch, out_ch, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.finalConv(x)
        return out


class VDConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(VDConv, self).__init__()
        self.vdConv = nn.Sequential(
            nn.GroupNorm(4, in_ch),
            nn.LeakyReLU(0.01),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x):
        out = self.vdConv(x)
        return out


class VUConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(VUConv, self).__init__()
        self.vuConv = nn.Sequential(
            nn.LeakyReLU(0.01),
            nn.ConvTranspose3d(in_ch, out_ch, 1),
        )

    def forward(self, x):

        out = self.vuConv(x)
        out = nn.functional.interpolate(
            out, scale_factor=2, mode='trilinear', align_corners=False)
        return out


class UP_VAE(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UP_VAE, self).__init__()
        self.VAE_UP = nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, 1),
        )
        self.resUnit = resUnit(out_ch, out_ch)

    def forward(self, x):
        x = self.VAE_UP(x)
        x = nn.functional.interpolate(
            x, scale_factor=2, mode='trilinear', align_corners=False)
        out = self.resUnit(x)
        return out


# class SeparableConv3d(nn.Module):
#     def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, BatchNorm=nn.BatchNorm3d):
#         super(SeparableConv3d, self).__init__()

#         self.conv1 = nn.Conv3d(inplanes, inplanes, kernel_size,
#                                stride, padding, dilation, groups=inplanes, bias=bias)
#         self.bn = BatchNorm(inplanes)
#         self.pointwise = nn.Conv3d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn(x)
#         x = self.pointwise(x)
#         return x
