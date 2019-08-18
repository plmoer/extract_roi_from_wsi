import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoding import Encoding
from .syncbn import BatchNorm3d
from .customize import Mean


class EncModule(nn.Module):
    def __init__(self, in_channels, nclass, ncodes=32, se_loss=True, norm_layer=None):
        super(EncModule, self).__init__()
        self.se_loss = se_loss
        self.encoding = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 1),
            nn.GroupNorm(4, in_channels),
            nn.LeakyReLU(0.01),
            Encoding(D=in_channels, K=ncodes),
            nn.GroupNorm(4, ncodes),
            # BatchNorm3d(ncodes),
            # encoding.nn.Encoding(D=in_channels, K=ncodes),
            # encoding.nn.BatchNorm1d(ncodes),
            nn.LeakyReLU(0.01),
            Mean(dim=1))
        # encoding.nn.Mean(dim=1))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid())
        if self.se_loss:
            self.selayer = nn.Linear(in_channels, nclass)

    def forward(self, x):
        en = self.encoding(x)
        b, c, _, _, _ = x.size()
        gamma = self.fc(en)
        y = gamma.view(b, c, 1, 1, 1)
        outputs = [F.relu_(x + x * y)]
        outputs.append(en)
        if self.se_loss:
            outputs.append(self.selayer(en))
        return tuple(outputs)


class EncHead(nn.Module):
    def __init__(self, in_channels, out_channels, se_loss=True, jpu=True, lateral=False,
                 norm_layer=None, up_kwargs=None):
        super(EncHead, self).__init__()
        self.se_loss = se_loss
        self.lateral = lateral
        self.up_kwargs = up_kwargs
        self.conv5 = nn.Sequential(nn.Conv3d(in_channels[-1], in_channels[0], 1),
                                   nn.GroupNorm(4, in_channels[0]),
                                   #    norm_layer(512),
                                   nn.LeakyReLU(0.01))
        nn.Sequential(nn.Conv3d(in_channels[-1], in_channels[0], 3, padding=1),
                      nn.GroupNorm(4, in_channels[0]),
                      #   norm_layer(512),
                      nn.LeakyReLU(0.01))
        if lateral:
            self.connect = nn.ModuleList([
                nn.Sequential(
                    nn.Conv3d(in_channels[0], in_channels[0],
                              kernel_size=1, stride=2),
                    nn.GroupNorm(4, in_channels[0]),
                    # norm_layer(512),
                    nn.LeakyReLU(0.01)),
                nn.Sequential(
                    nn.Conv3d(in_channels[1], in_channels[0], kernel_size=1),
                    nn.GroupNorm(4, in_channels[0]),
                    # norm_layer(512),
                    nn.LeakyReLU(0.01)),
                nn.Sequential(
                    # nn.Conv3d(in_channels[1], in_channels[0], kernel_size=1),
                    nn.ConvTranspose3d(
                        in_channels[2], in_channels[0], 2, stride=2),
                    nn.GroupNorm(4, in_channels[0]),
                    # norm_layer(512),
                    nn.LeakyReLU(0.01)),
            ])
            self.fusion = nn.Sequential(
                nn.Conv3d(4*in_channels[0], in_channels[0],
                          kernel_size=3, padding=1),
                nn.GroupNorm(4, in_channels[0]),
                # norm_layer(512),
                nn.LeakyReLU(0.01))
        self.encmodule = EncModule(in_channels[0], out_channels, ncodes=32,
                                   se_loss=se_loss, norm_layer=None)
        self.conv6 = nn.Sequential(nn.Dropout3d(0.1, False),
                                   nn.Conv3d(in_channels[0], out_channels, 1))

    def forward(self, *inputs):
        feat = self.conv5(inputs[-1])
        if self.lateral:
            c1 = self.connect[0](inputs[0])
            c2 = self.connect[1](inputs[1])
            c3 = self.connect[2](inputs[2])
            feat = self.fusion(torch.cat([feat, c1, c2, c3], 1))
        outs = list(self.encmodule(feat))
        outs[0] = self.conv6(outs[0])
        return tuple(outs)
