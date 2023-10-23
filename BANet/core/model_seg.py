import torch
import torch.nn as nn
import torch.nn.functional as F
from .segformer_head import SegFormerHead, SegFormerFuse
from . import mix_transformer
from segmentation_models_pytorch.base import ClassificationHead # add by cyx
from segmentation_models_pytorch.encoders._utils import patch_first_conv # add by cyx, used for replacing in_channels


# CNN model: from unsupervised segmentation
class MyNet(nn.Module):
    def __init__(self, input_dim=3, out_dim=32, nChannel=100, nConv=2):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, nChannel, kernel_size=3, stride=1, padding=1 )
        self.bn1 = nn.BatchNorm2d(nChannel)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(nConv-1):
            self.conv2.append( nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append( nn.BatchNorm2d(nChannel) )
        self.conv3 = nn.Conv2d(nChannel, out_dim, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(out_dim)
        self.nConv = nConv

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        for i in range(self.nConv-1):
            x = self.conv2[i](x)
            x = F.relu( x )
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x


# 2022.11.7: add unseg module
class Mitcls_CAM_unseg(nn.Module):
    def __init__(self, backbone="mit_b2", num_classes=20,
                 embedding_dim=256, pretrained=None, in_chans = 3,
                 pooling="max", stride=(4, 2, 2, 1)):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
        # self.in_channels = [32, 64, 160, 256]
        # self.in_channels = [64, 128, 320, 512]

        self.encoder = getattr(mix_transformer, backbone)(stride)
        self.in_channels = self.encoder.embed_dims
        ## initilize encoder
        if pretrained:
            state_dict = torch.load('pretrained/' + backbone + '.pth')
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            self.encoder.load_state_dict(state_dict, )

        patch_first_conv(self.encoder, in_chans, pretrained=pretrained is not None)

        # self.decoder = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.in_channels,
        #                              embedding_dim=self.embedding_dim, num_classes=self.num_classes)
        self.pooling = nn.AdaptiveAvgPool2d(1) if pooling == "avg" else nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Conv2d(in_channels=self.in_channels[-1],
                                    out_channels=self.num_classes, kernel_size=1, bias=False)
        # self.classifier = ClassificationHead(in_channels=self.in_channels[-1], classes=num_classes,
        #                                     pooling=pooling, dropout=dropout, activation=activation)
        self.unseg = MyNet(input_dim=in_chans, out_dim=32)

    def get_param_groups(self):

        param_groups = [[], [], []]  #

        for name, param in list(self.encoder.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)
        # add by cyx
        for param in list(self.classifier.parameters()):
            param_groups[2].append(param)
        # add by cyx
        for param in list(self.unseg.parameters()):
            param_groups[2].append(param)
        return param_groups

    def forward(self, x, cam_only=False):
        H, W = x.shape[2:]
        _, _, _, x4 = self.encoder(x)

        features = self.classifier(x4)
        # add by cyx
        if cam_only:
            return features

        cls = self.pooling(features)
        cls = cls.squeeze(-1).squeeze(-1)

        # unsupervised segmentation
        features = F.interpolate(features, size=(H//2, W//2),mode='bilinear', align_corners=False)
        x = F.interpolate(x, size=(H//2, W//2),mode='bilinear', align_corners=False)
        mask = self.unseg(x) # N (C1+C2) H W

        # projection
        maskobj = torch.argmax(mask, dim=1) # N H W
        n,c,h,w = features.shape
        features = features.reshape((n, c,-1))
        maskobj = maskobj.reshape((n, -1))
        for i in range(n):
            klist = torch.unique(maskobj[i])
            for k in klist:
                pos = torch.where(maskobj[i] == k)[0]
                features[i, :, pos] = torch.mean(features[i, :, pos], dim=-1, keepdim=True)
        features = features.reshape((n, c, h, w))
        cls_fea = self.pooling(features)
        cls_fea = cls_fea.squeeze(-1).squeeze(-1)

        # # add by cyx
        # if cam_only:
        #     return features

        return cls, cls_fea, features, mask

