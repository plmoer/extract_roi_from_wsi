'''
Author: Linmin
Update: May, 03, 2019
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes,  ndf=16, ngf=128, bVAE=0, nX=32, nY=32, nZ=32):
        super(UNet, self).__init__()
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
        self.up_3 = upConv(ndf*8, ndf*4)
        self.up_2 = upConv(ndf*4, ndf*2)
        self.up_1 = upConv(ndf*2, ndf)
        self.final_out_1 = outConv(ndf, n_classes)
        self.final_out_2 = outConv(ndf+ndf, n_classes)
        # self.final_out_res = inConv(ndf, n_channels)

        self.vd = VDConv(ndf*8, 16)
        self.vu = VUConv(16, ndf*8)
        self.fc0 = nn.Linear(16*nX*nY * nZ//(8*8*8*8), ngf*2)  # 32*32*32
        self.fc11 = nn.Linear(ngf*2, ngf)  # 32*32*32
        self.fc12 = nn.Linear(ngf*2, ngf)  # 32*32*32
        self.fc2 = nn.Linear(ngf, 16*nX*nY * nZ//(8*8*8*8))  # 32*32*32
        self.vae_up_1 = UP_VAE(ndf*8, ndf*4)
        self.vae_up_2 = UP_VAE(ndf*4, ndf*2)
        self.vae_up_3 = UP_VAE(ndf*2, ndf)
        self.vend = nn.Conv3d(ndf, n_channels, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight, gain=np.sqrt(2.0))
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                init.constant_(m.bias, 0)

    def getDimension(self, x):
        [nB, nC, nX, nY, nZ] = x.shape
        self.nB = nB
        self.nC = nC
        self.nX = nX
        self.nY = nY
        self.nZ = nZ

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        # std = logvar.mul(0.5).exp_()
        # eps = torch.cuda.FloatTensor(std.size()).normal_()
        # eps = Variable(eps)p
        return eps.mul(std).add_(mu)

    def decode(self, z):
        z = self.fc2(z)
        z = z.view(z.size(0), 16, (self.nX)//16, (self.nY)//16, (self.nZ)//16)
        z = self.vu(z)
        z = self.vae_up_1(z)
        z = self.vae_up_2(z)
        z = self.vae_up_3(z)
        z = self.vend(z)
        return z

    def vae(self, x):
        x = self.vd(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc0(x))
        mu, logvar = self.fc11(x), self.fc12(x)
        z = self.reparameterize(mu, logvar)
        res = self.decode(z)
        return res, mu, logvar

    def forward(self, x):
        self.getDimension(x)
        init_conv = self.inc(x)
        en_block0 = self.resU_0(init_conv)
        en_block1 = self.down_1(en_block0)
        en_block1 = self.resU_1(en_block1)
        en_block2 = self.down_2(en_block1)
        en_block2 = self.resU_2(en_block2)
        en_block3 = self.down_3(en_block2)
        en_block3 = self.resU_3(en_block3)
        en_block3 = self.resU_3(en_block3)
        en_block3 = self.resU_3(en_block3)
        de_block2 = self.up_3(en_block3, en_block2)
        de_block1 = self.up_2(de_block2, en_block1)
        de_block0 = self.up_1(de_block1, en_block0)

        res, mu, logvar = self.vae(en_block3)
        if self.bVAE == 2:
            output = torch.cat([de_block0, res], 1)
            output = self.final_out_2(output)
        else:
            output = self.final_out_1(de_block0)

        # res = self.final_out_res(F.relu(res))
        return output, res, mu, logvar


class inConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1):
        super(inConv, self).__init__()
        self.conv = nn.Conv3d(
            in_ch, out_ch,  kernel_size=3, padding=1)
        init.xavier_uniform_(self.conv.weight, gain=np.sqrt(2.0))  # or gain=1
        init.constant_(self.conv.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        return x


class resUnit(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(resUnit, self).__init__()
        self.resConv = nn.Sequential(
            nn.GroupNorm(4, in_ch),
            nn.LeakyReLU(0.01),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.LeakyReLU(0.01),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        )
        self.bridge = nn.Sequential(
            nn.GroupNorm(4, in_ch),
            nn.LeakyReLU(0.01),
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
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
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
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
            nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
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
