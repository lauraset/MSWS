import torch
import torch.nn as nn
import torch.nn.functional as F
from .segformer_head import SegFormerHead, SegFormerFuse
from . import mix_transformer
from segmentation_models_pytorch.encoders._utils import patch_first_conv # add by cyx, used for replacing in_channels
from onestage.pamr import PAMR


def focal_loss(x, p = 1, c = 0.1):
    return torch.pow(1 - x, p) * torch.log(c + x)


class Mitcls_CAM_seg(nn.Module):
    def __init__(self, backbone="mit_b2", num_classes=20, embedding_dim=256, pretrained=None, in_chans = 3,
                 pooling="max", stride=(4, 2, 2, 2)):
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

        # add fuse module
        self.decoder = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.in_channels,
                                     embedding_dim=self.embedding_dim, num_classes=self.num_classes)
        self.pooling = nn.AdaptiveAvgPool2d(1) if pooling == "avg" else nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Conv2d(in_channels=embedding_dim,
                                    out_channels=self.num_classes, kernel_size=1, bias=False)
        self._aff = PAMR(10, [1, 2, 4, 8, 12, 24])

    def run_pamr(self, im, mask):
        im = F.interpolate(im, mask.size()[-2:], mode="bilinear", align_corners=True)
        masks_dec = self._aff(im, mask)
        return masks_dec

    def get_param_groups(self):

        param_groups = [[], [], []]  #

        for name, param in list(self.encoder.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        for param in list(self.classifier.parameters()):
            param_groups[2].append(param)
        for param in list(self.decoder.parameters()):
            param_groups[2].append(param)

        return param_groups

    def forward(self, x, y_raw=None, labels=None, cam_only=False, with_cam=True):

        x = self.encoder(x)
        x = self.decoder(x)

        bs, c, h, w = x.size()
        masks = F.softmax(x, dim=1)

        # reshaping
        features = x.view(bs, c, -1)
        masks_ = masks.view(bs, c, -1)

        # classification loss
        cls_1 = (features * masks_).sum(-1) / (1.0 + masks_.sum(-1))

        # focal penalty loss
        cls_2 = focal_loss(masks_.mean(-1), p=3, c=0.01)

        # adding the losses together
        cls = cls_1[:, 1:] + cls_2[:, 1:]

        self._mask_logits = x

        # foreground stats
        masks_ = masks_[:, 1:]
        cls_fg = (masks_.mean(-1) * labels).sum(-1) / labels.sum(-1)

        # mask refinement with PAMR
        masks_dec = self.run_pamr(y_raw, masks.detach())

        # upscale the masks & clean
        masks = self._rescale_and_clean(masks, y, labels)
        masks_dec = self._rescale_and_clean(masks_dec, y, labels)

        # create pseudo GT
        pseudo_gt = pseudo_gtmask(masks_dec).detach()
        loss_mask = balanced_mask_loss_ce(self._mask_logits, pseudo_gt, labels)

        return cls, cls_fg, {"cam": masks, "dec": masks_dec}, self._mask_logits, pseudo_gt, loss_mask

