'''
2022.11.17 design change detection network
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .segformer_head import SegFormerHead, SegFormerHeadEdge
from . import mix_transformer
from segmentation_models_pytorch.encoders._utils import patch_first_conv  # add by cyx, used for replacing in_channels


# add stride and in_chans by cyx, 2022.9.28
# designed for change detection
class WeTrCD(nn.Module):
    def __init__(self, backbone, num_classes=20, embedding_dim=256, pretrained=None,
                 stride=(4, 2, 2, 2), in_chans=3, pooling="gmp"):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
        # self.in_channels = [32, 64, 160, 256]
        # self.in_channels = [64, 128, 320, 512]
        self.in_chans = in_chans
        self.encoder = getattr(mix_transformer, backbone)(stride)
        self.out_channels = self.encoder.embed_dims
        ## initilize encoder
        if pretrained:
            state_dict = torch.load('pretrained/' + backbone + '.pth')
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            self.encoder.load_state_dict(state_dict, )

        patch_first_conv(self.encoder, in_chans, pretrained=pretrained is not None)

        self.decoder = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.out_channels,
                                     embedding_dim=self.embedding_dim, num_classes=self.num_classes)
        # for cls
        # self.pooling = nn.AdaptiveAvgPool2d(1) if pooling == "avg" else nn.AdaptiveMaxPool2d(1)
        # self.classifier = nn.Conv2d(in_channels=self.in_channels[-1], out_channels=self.num_classes,
        #                             kernel_size=1, bias=False)

    def get_param_groups(self):

        param_groups = [[], [], []]  #

        for name, param in list(self.encoder.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        for param in list(self.decoder.parameters()):
            param_groups[2].append(param)

        # param_groups[2].append(self.classifier.weight)

        return param_groups

    def forward(self, x):
        h, w = x.shape[2:]

        _x1 = self.encoder(x[:, :self.in_chans])
        _x2 = self.encoder(x[:, self.in_chans:])

        # add by cyx
        diff_fea = [torch.abs(i-j) for i, j in zip(_x1, _x2)]
        mask = self.decoder(diff_fea)
        mask = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)

        return mask
