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
                 stride=(4, 2, 2, 2), in_chans=3):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
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

# return feature
class WeTrCDfea(nn.Module):
    def __init__(self, backbone, num_classes=20, embedding_dim=256, pretrained=None,
                 stride=(4, 2, 2, 2), in_chans=3):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
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

    def get_param_groups(self):

        param_groups = [[], [], []]  #

        for name, param in list(self.encoder.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        for param in list(self.decoder.parameters()):
            param_groups[2].append(param)

        return param_groups

    def forward(self, x):
        h, w = x.shape[2:]

        _x1 = self.encoder(x[:, :self.in_chans])
        _x2 = self.encoder(x[:, self.in_chans:])

        # add by cyx
        diff_fea = [torch.abs(i-j) for i, j in zip(_x1, _x2)]
        mask = self.decoder(diff_fea)
        mask = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)

        return mask, _x1[-1], _x2[-1]


# consider change direction consistency
class WeTrCDdirec(nn.Module):
    def __init__(self, backbone, num_classes=20, embedding_dim=256, pretrained=None,
                 stride=(4, 2, 2, 2), in_chans=3):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
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

        self.decoder1 = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.out_channels,
                                     embedding_dim=self.embedding_dim, num_classes=self.num_classes)
        self.decoder2 = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.out_channels,
                                     embedding_dim=self.embedding_dim, num_classes=self.num_classes)

    def get_param_groups(self):

        param_groups = [[], [], []]  #

        for name, param in list(self.encoder.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        for param in list(self.decoder1.parameters()):
            param_groups[2].append(param)
        for param in list(self.decoder2.parameters()):
            param_groups[2].append(param)

        return param_groups

    def forward(self, x):
        h, w = x.shape[2:]

        _x1 = self.encoder(x[:, :self.in_chans])
        _x2 = self.encoder(x[:, self.in_chans:])

        # neg direction
        diff1 = [(i-j) for i, j in zip(_x1, _x2)]
        mask = self.decoder1(diff1)
        mask = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)

        # pos direction
        diff2 = [(j-i) for i, j in zip(_x1, _x2)]
        mask2 = self.decoder2(diff2)
        mask2 = F.interpolate(mask2, size=(h, w), mode='bilinear', align_corners=False)

        return mask, mask2


class WeTrCat(nn.Module):
    def __init__(self, backbone, num_classes=20, embedding_dim=256, pretrained=None,
                 stride=(4, 2, 2, 2), in_chans=3):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
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
        # concate two temporal features
        self.out_channels = [2*i for i in self.out_channels]
        self.decoder = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.out_channels,
                                     embedding_dim=self.embedding_dim, num_classes=self.num_classes)

    def get_param_groups(self):

        param_groups = [[], [], []]  #

        for name, param in list(self.encoder.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        for param in list(self.decoder.parameters()):
            param_groups[2].append(param)

        return param_groups

    def forward(self, x):
        h, w = x.shape[2:]

        _x1 = self.encoder(x[:, :self.in_chans])
        _x2 = self.encoder(x[:, self.in_chans:])

        # temporal symmetry
        fea1 = [torch.cat([i, j], dim=1) for i, j in zip(_x1, _x2)]
        mask1 = self.decoder(fea1)
        mask1 = F.interpolate(mask1, size=(h, w), mode='bilinear', align_corners=False)

        fea2 = [torch.cat([j, i], dim=1) for i, j in zip(_x1, _x2)]
        mask2 = self.decoder(fea2)
        mask2 = F.interpolate(mask2, size=(h, w), mode='bilinear', align_corners=False)

        return mask1, mask2


# consider unchanged buildings
class WeTrCDSem(nn.Module):
    def __init__(self, backbone, num_classes=20, embedding_dim=256, pretrained=None,
                 stride=(4, 2, 2, 2), in_chans=3):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
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

        self.decoder_un = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.out_channels,
                                     embedding_dim=self.embedding_dim, num_classes=self.num_classes)
        self.decoder_cd = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.out_channels,
                                     embedding_dim=self.embedding_dim, num_classes=self.num_classes)

    def get_param_groups(self):

        param_groups = [[], [], []]  #

        for name, param in list(self.encoder.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        for param in list(self.decoder_un.parameters()):
            param_groups[2].append(param)
        for param in list(self.decoder_cd.parameters()):
            param_groups[2].append(param)
        # param_groups[2].append(self.classifier.weight)

        return param_groups

    def forward(self, x):
        h, w = x.shape[2:]

        _x1 = self.encoder(x[:, :self.in_chans])
        _x2 = self.encoder(x[:, self.in_chans:])

        # change
        diff_fea = [torch.abs(i-j) for i, j in zip(_x1, _x2)]
        mask_cd = self.decoder_cd(diff_fea)
        # unchange
        mask1 = self.decoder_un(_x1)
        mask2 = self.decoder_un(_x2)

        mask_cd = F.interpolate(mask_cd, size=(h, w), mode='bilinear', align_corners=False)
        mask1 = F.interpolate(mask1, size=(h, w), mode='bilinear', align_corners=False)
        mask2 = F.interpolate(mask2, size=(h, w), mode='bilinear', align_corners=False)

        return mask_cd, mask1, mask2


class WeTrCDDiff(nn.Module):
    def __init__(self, backbone, num_classes=20, embedding_dim=256, pretrained=None,
                 stride=(4, 2, 2, 2), in_chans=3):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
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

    def get_param_groups(self):

        param_groups = [[], [], []]  #

        for name, param in list(self.encoder.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        for param in list(self.decoder.parameters()):
            param_groups[2].append(param)

        return param_groups

    def forward(self, x):
        h, w = x.shape[2:]

        _x1 = self.encoder(x[:, :self.in_chans]) # t1
        _x2 = self.encoder(x[:, self.in_chans:]) # t2

        fea_abs = [torch.abs(i-j) for i, j in zip(_x1, _x2)]
        mask_abs = self.decoder(fea_abs)

        fea_neg = [F.relu(i-j) for i, j in zip(_x1, _x2)]
        mask_neg = self.decoder(fea_neg)

        fea_pos = [F.relu(j-i) for i, j in zip(_x1, _x2)]
        mask_pos = self.decoder(fea_pos)

        mask_abs = F.interpolate(mask_abs, size=(h, w), mode='bilinear', align_corners=False)
        mask_neg = F.interpolate(mask_neg, size=(h, w), mode='bilinear', align_corners=False)
        mask_pos = F.interpolate(mask_pos, size=(h, w), mode='bilinear', align_corners=False)

        return mask_abs, mask_neg, mask_pos
