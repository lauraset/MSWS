import torch
import torch.nn as nn
import torch.nn.functional as F
from .segformer_head import SegFormerHead, SegFormerFuse
from . import mix_transformer
from segmentation_models_pytorch.base import ClassificationHead # add by cyx
from segmentation_models_pytorch.encoders._utils import patch_first_conv # add by cyx, used for replacing in_channels


# four layer
class Mitcls_CAM_multi4(nn.Module):
    def __init__(self, backbone="mit_b2", num_classes=20, embedding_dim=256, pretrained=None, in_chans = 3,
                 pooling="max", stride=(4, 2, 2, 2)):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]

        self.encoder = getattr(mix_transformer, backbone)(stride)
        self.in_channels = self.encoder.embed_dims
        # self.in_channels = [64, 128, 320, 512] # for mixtransformer

        ## initilize encoder
        if pretrained:
            state_dict = torch.load('pretrained/' + backbone + '.pth')
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            self.encoder.load_state_dict(state_dict, )

        patch_first_conv(self.encoder, in_chans, pretrained=pretrained is not None)

        self.decoder = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.in_channels,
                                     embedding_dim=self.embedding_dim, num_classes=self.num_classes)
        self.pooling = nn.AdaptiveAvgPool2d(1) if pooling == "avg" else nn.AdaptiveMaxPool2d(1)
        self.classifier1 = nn.Conv2d(in_channels=self.in_channels[0],
                                    out_channels=self.num_classes, kernel_size=1, bias=False)
        self.classifier2 = nn.Conv2d(in_channels=self.in_channels[1],
                                    out_channels=self.num_classes, kernel_size=1, bias=False)
        self.classifier3 = nn.Conv2d(in_channels=self.in_channels[2],
                                    out_channels=self.num_classes, kernel_size=1, bias=False)
        self.classifier4 = nn.Conv2d(in_channels=self.in_channels[3],
                                    out_channels=self.num_classes, kernel_size=1, bias=False)

        # branch: class boundary detection
        # self.fc_edge1 = nn.Sequential(
        #     nn.Conv2d(self.in_channels[0], 32, 1, bias=False),nn.GroupNorm(4, 32),
        #     # nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),  # add by yinxcao, 2021.11.06
        #     nn.ReLU(inplace=True),
        # )
        # self.fc_edge2 = nn.Sequential(
        #     nn.Conv2d(self.in_channels[1], 32, 1, bias=False),nn.GroupNorm(4, 32),
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #     nn.ReLU(inplace=True),
        # )
        # self.fc_edge3 = nn.Sequential(
        #     nn.Conv2d(self.in_channels[2], 32, 1, bias=False),nn.GroupNorm(4, 32),
        #     nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
        #     nn.ReLU(inplace=True),
        # )
        # self.fc_edge4 = nn.Sequential(
        #     nn.Conv2d(self.in_channels[3], 32, 1, bias=False),nn.GroupNorm(4, 32),
        #     nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
        #     nn.ReLU(inplace=True),
        # )
        # self.fc_edge6 = nn.Conv2d(32*4, 1, 1, bias=True)

    def get_param_groups(self):

        param_groups = [[], [], []]  #

        for name, param in list(self.encoder.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        # add by cyx: classifier1,2,3,4
        for param in list(self.classifier1.parameters()):
            param_groups[2].append(param)
        for param in list(self.classifier2.parameters()):
            param_groups[2].append(param)
        for param in list(self.classifier3.parameters()):
            param_groups[2].append(param)
        for param in list(self.classifier4.parameters()):
            param_groups[2].append(param)

        return param_groups

    def forward(self, x, cam_only=False, with_cam = True):

        x1, x2, x3, x4 = self.encoder(x)

        # add by cyx
        if cam_only:
            cam_s1 = F.conv2d(x1, self.classifier1.weight).detach()
            cam_s2 = F.conv2d(x2, self.classifier2.weight).detach()
            cam_s3 = F.conv2d(x3, self.classifier3.weight).detach()
            cam_s4 = F.conv2d(x4, self.classifier4.weight).detach()
            return cam_s1, cam_s2, cam_s3, cam_s4

        if with_cam:
            f1, f2, f3, f4 = self.classifier1(x1), self.classifier2(x2), self.classifier3(x3), self.classifier4(x4)
            cls1, cls2, cls3, cls4 = self.pooling(f1), self.pooling(f2), self.pooling(f3), self.pooling(f4)

            cls1 = cls1.squeeze(-1).squeeze(-1)
            cls2 = cls2.squeeze(-1).squeeze(-1)
            cls3 = cls3.squeeze(-1).squeeze(-1)
            cls4 = cls4.squeeze(-1).squeeze(-1)

            return cls1, cls2, cls3, cls4, f1, f2, f3, f4

        else:
            cls1, cls2, cls3, cls4 = self.pooling(x1), self.pooling(x2), self.pooling(x3), self.pooling(x4)
            cls1, cls2, cls3, cls4 = self.classifier1(cls1), self.classifier2(cls2), self.classifier3(cls3), self.classifier4(cls4)

            cls1 = cls1.squeeze(-1).squeeze(-1)
            cls2 = cls2.squeeze(-1).squeeze(-1)
            cls3 = cls3.squeeze(-1).squeeze(-1)
            cls4 = cls4.squeeze(-1).squeeze(-1)

            return cls1, cls2, cls3, cls4


class ConvBnRelu(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)