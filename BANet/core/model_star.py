import ever as er
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .segformer_head import SegFormerHead, SegFormerHeadEdge
from . import mix_transformer
from segmentation_models_pytorch.encoders._utils import patch_first_conv  # add by cyx, used for replacing in_channels


MAX_TIMES = 50
MASK1 = 'mask1'
VMASK2 = 'virtual_mask2'

class DropConnect(nn.Module):
    def __init__(self, drop_rate):
        super(DropConnect, self).__init__()
        self.p = drop_rate

    def forward(self, inputs):
        """Drop connect.
            Args:
                input (tensor: BCWH): Input of this structure.
                p (float: 0.0~1.0): Probability of drop connection.
                training (bool): The running mode.
            Returns:
                output: Output after drop connection.
        """
        p = self.p
        assert 0 <= p <= 1, 'p must be in range of [0,1]'

        if not self.training:
            return inputs

        batch_size = inputs.shape[0]
        keep_prob = 1 - p

        # generate binary_tensor mask according to probability (p for 0, 1-p for 1)
        random_tensor = keep_prob
        random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
        binary_tensor = torch.floor(random_tensor)

        output = inputs / keep_prob * binary_tensor
        return output

def get_detector(name, **kwargs):
    if 'convs' == name:
        return Conv3x3ReLUBNs(kwargs['in_channels'],
                              kwargs['inner_channels'],
                              kwargs['out_channels'],
                              kwargs['scale'],
                              kwargs['num_convs'],
                              kwargs.get('drop_rate', 0.)
                              )
    raise ValueError(f'{name} is not supported.')

def Conv3x3ReLUBNs(in_channels,
                   inner_channels,
                   out_channels,
                   scale,
                   num_convs,
                   drop_rate=0.):
    layers = [nn.Sequential(
        nn.Conv2d(in_channels, inner_channels, 3, 1, 1),
        nn.ReLU(True),
        nn.BatchNorm2d(inner_channels),
        DropConnect(drop_rate) if drop_rate > 0. else nn.Identity()
    )]
    layers += [nn.Sequential(
        nn.Conv2d(inner_channels, inner_channels, 3, 1, 1),
        nn.ReLU(True),
        nn.BatchNorm2d(inner_channels),
        DropConnect(drop_rate) if drop_rate > 0. else nn.Identity()
    ) for _ in range(num_convs - 1)]

    cls_layer = nn.Conv2d(inner_channels, out_channels, 3, 1, 1)
    layers.append(cls_layer)
    layers.append(nn.UpsamplingBilinear2d(scale_factor=scale))
    return nn.Sequential(*layers)


def generate_target(x1, mask1):
    # x [N, C * 1, H, W]
    # y dict(mask1=tensor[N, H, W], ...)
    # y = {MASK1: y}
    # mask1 = y[MASK1]
    N = x1.size(0)
    org_inds = np.arange(N)
    t = 0
    while True and t <= MAX_TIMES:
        t += 1
        shuffle_inds = org_inds.copy()
        np.random.shuffle(shuffle_inds)

        ok = org_inds == shuffle_inds
        if all(~ok):
            break
    virtual_x2 = x1[shuffle_inds, :, :, :]
    virtual_mask2 = mask1[shuffle_inds, ...]
    x = torch.cat([x1, virtual_x2], dim=1)

    # y[VMASK2] = virtual_mask2
    return x, virtual_mask2


class ChangeMixin(nn.Module):
    def __init__(self, backbone, num_classes=20, embedding_dim=256, pretrained=None,
                 stride=(4, 2, 2, 2), in_chans=3, pooling="gmp"):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
        self.encoder = getattr(mix_transformer, backbone)(stride)
        self.in_channels = self.encoder.embed_dims
        self.in_chans = in_chans
        ## initilize encoder
        if pretrained:
            state_dict = torch.load('pretrained/' + backbone + '.pth')
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            self.encoder.load_state_dict(state_dict, )

        patch_first_conv(self.encoder, in_chans, pretrained=pretrained is not None)

        self.decoder = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.in_channels,
                                     embedding_dim=self.embedding_dim, num_classes=self.num_classes)
        # self.in_channels = [2*i for i in self.in_channels]
        self.decodercd = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.in_channels,
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
        for param in list(self.decodercd.parameters()):
            param_groups[2].append(param)
        return param_groups

    def forward(self, x, y=None):
        h, w = x.shape[2:]
        #if x.size(1) == self.in_chans:
        x, y2 = generate_target(x, y)

        y1_feature = self.encoder(x[:, :self.in_chans]) # t1
        vy2_feature = self.encoder(x[:, self.in_chans:]) # t2

        y1_pred = self.decoder(y1_feature)
        y1_pred = F.interpolate(y1_pred, size=(h, w), mode='bilinear', align_corners=False)

        # extract positive feature
        # cd1 = self.decodercd([torch.cat([i, j], dim=1) for i, j in zip(y1_feature, vy2_feature)])
        # cd2 = self.decodercd([torch.cat([j, i], dim=1) for i, j in zip(y1_feature, vy2_feature)])
        cd1 = self.decodercd([(i-j) for i, j in zip(y1_feature, vy2_feature)])
        cd2 = self.decodercd([(j-i) for i, j in zip(y1_feature, vy2_feature)])
        cd1 = F.interpolate(cd1, size=(h, w), mode='bilinear', align_corners=False)
        cd2 = F.interpolate(cd2, size=(h, w), mode='bilinear', align_corners=False)

        return y1_pred, y2, cd1, cd2

    # return change detection
    def forwardcd(self, x):
        h, w = x.shape[2:]
        # if x.size(1) == self.in_chans:
        # x, y2 = generate_target(x, y)

        y1_feature = self.encoder(x[:, :self.in_chans]) # t1
        vy2_feature = self.encoder(x[:, self.in_chans:]) # t2

        #y1_pred = self.decoder(y1_feature)
        #y1_pred = F.interpolate(y1_pred, size=(h, w), mode='bilinear', align_corners=False)

        # extract positive feature
        # cd1 = self.decodercd([torch.cat([i, j], dim=1) for i, j in zip(y1_feature, vy2_feature)])
        cd1 = self.decodercd([(i-j) for i, j in zip(y1_feature, vy2_feature)])
        #cd2 = self.decodercd([torch.cat([j, i], dim=1) for i, j in zip(y1_feature, vy2_feature)])
        cd1 = F.interpolate(cd1, size=(h, w), mode='bilinear', align_corners=False)
        #cd2 = F.interpolate(cd2, size=(h, w), mode='bilinear', align_corners=False)

        return cd1

