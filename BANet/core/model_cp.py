import torch
import torch.nn as nn
import torch.nn.functional as F 
from .segformer_head import SegFormerHead, SegFormerFuse
from . import mix_transformer
from segmentation_models_pytorch.base import ClassificationHead # add by cyx
from segmentation_models_pytorch.encoders._utils import patch_first_conv # add by cyx, used for replacing in_channels


# add stride and in_chans by cyx, 2022.9.28
class WeTr(nn.Module):
    def __init__(self, backbone, num_classes=20, embedding_dim=256, pretrained=None,
                 stride=(4,2,2,2), in_chans = 3, pooling ="gmp"):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
        #self.in_channels = [32, 64, 160, 256]
        #self.in_channels = [64, 128, 320, 512]

        self.encoder = getattr(mix_transformer, backbone)(stride)
        self.in_channels = self.encoder.embed_dims
        ## initilize encoder
        if pretrained:
            state_dict = torch.load('pretrained/'+backbone+'.pth')
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            self.encoder.load_state_dict(state_dict,)

        patch_first_conv(self.encoder, in_chans, pretrained=pretrained is not None)

        self.decoder = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.in_channels,
                                     embedding_dim=self.embedding_dim, num_classes=self.num_classes)
        # for cls
        # self.pooling = nn.AdaptiveAvgPool2d(1) if pooling == "avg" else nn.AdaptiveMaxPool2d(1)
        # self.classifier = nn.Conv2d(in_channels=self.in_channels[-1], out_channels=self.num_classes,
        #                             kernel_size=1, bias=False)

    def _forward_cam(self, x):
        
        cam = F.conv2d(x, self.classifier.weight)
        cam = F.relu(cam)
        
        return cam

    def get_param_groups(self):

        param_groups = [[], [], []] # 
        
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

        _x = self.encoder(x)
        # _x1, _x2, _x3, _x4 = _x
        # cls = self.classifier(_x4)
        # add by cyx
        mask = self.decoder(_x)
        mask = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)

        return mask


# 2022.10.3, return cls and seg
class Mitcls_seg(nn.Module):
    def __init__(self, backbone, num_classes=20, embedding_dim=256, pretrained=None,
                 stride=(4, 2, 2, 2), in_chans=3, pooling="gmp"):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]

        self.encoder = getattr(mix_transformer, backbone)(stride)
        self.in_channels = self.encoder.embed_dims

        ## initilize encoder
        if pretrained:
            state_dict = torch.load('pretrained/' + backbone + '.pth')
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            self.encoder.load_state_dict(state_dict, )

        patch_first_conv(self.encoder, in_chans, pretrained=pretrained is not None)

        # for seg
        self.decoder = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.in_channels,
                                     embedding_dim=self.embedding_dim, num_classes=self.num_classes)
        # for cls
        if pooling == "gmp":
            self.pooling = nn.AdaptiveMaxPool2d(1)
        elif pooling == "gap":
            self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Conv2d(in_channels=self.in_channels[-1], out_channels=self.num_classes,
                                    kernel_size=1, bias=False)

    def _forward_cam(self, x):

        cam = F.conv2d(x, self.classifier.weight)
        cam = F.relu(cam)

        return cam

    def get_param_groups(self):

        param_groups = [[], [], []]  #

        for name, param in list(self.encoder.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        for param in list(self.decoder.parameters()):
            param_groups[2].append(param)
        # cls
        param_groups[2].append(self.classifier.weight)

        return param_groups

    def forward(self, x, cam_only=False, require_seg=False):
        h, w = x.shape[2:]

        _x = self.encoder(x)

        outputs = [[], []] # cls, seg

        if cam_only:
            cam_s4 = F.conv2d(_x[-1], self.classifier.weight).detach()
            return cam_s4

        # cls
        cls = self.pooling(_x[-1])
        cls = self.classifier(cls)
        cls = cls.view(-1, self.num_classes)
        outputs[0] = cls

        if require_seg:
            seg = self.decoder(_x)
            seg = F.interpolate(seg, size=(h, w), mode='bilinear', align_corners=False)
            outputs[1] = seg

        return outputs


# for classification, cls use fully-connected layer
class Mitcls(nn.Module):
    def __init__(self, backbone="mit_b2", num_classes=20, embedding_dim=256, pretrained=None, in_chans = 3,
                 pooling="avg", dropout=0.2, activation=None):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
        # self.in_channels = [32, 64, 160, 256]
        # self.in_channels = [64, 128, 320, 512]

        self.encoder = getattr(mix_transformer, backbone)()
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

        # self.classifier = nn.Conv2d(in_channels=self.in_channels[-1], out_channels=self.num_classes, kernel_size=1,
        #                             bias=False)
        self.classifier = ClassificationHead(in_channels=self.in_channels[-1], classes=num_classes,
                                            pooling=pooling, dropout=dropout, activation=activation)

    def _forward_cam(self, x):

        cam = F.conv2d(x, self.classifier.weight)
        cam = F.relu(cam)

        return cam

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

        return param_groups

    def forward(self, x):

        _x = self.encoder(x)
        # _x1, _x2, _x3, _x4 = _x
        cls = self.classifier(_x[-1])

        return cls


# add cam, cls use conv layer
class Mitcls_CAM(nn.Module):
    def __init__(self, backbone="mit_b2", num_classes=20, embedding_dim=256, pretrained=None, in_chans = 3,
                 pooling="max", dropout=0.2, activation=None, stride=(4, 2, 2, 2)):
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

    def forward_cam(self, x):
        _x = self.encoder(x)
        cam = F.conv2d(_x[-1], self.classifier.weight)
        cam = F.relu(cam)

        return cam

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

        return param_groups

    def forward(self, x, cam_only=False):

        _x = self.encoder(x)
        # add by cyx
        if cam_only:
            cam_s4 = F.conv2d(_x[-1], self.classifier.weight).detach()
            return cam_s4

        cls = self.pooling(_x[-1])
        cls = self.classifier(cls)
        cls = cls.squeeze(-1).squeeze(-1)
        return cls


# 2022.11.05: add seam
class Mitcls_seam(nn.Module):
    def __init__(self, backbone="mit_b2", num_classes=20, embedding_dim=256, pretrained=None, in_chans = 3,
                 pooling="max", dropout=0.2, activation=None, stride=(4, 2, 2, 2)):
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

        self.f8_3 = torch.nn.Conv2d(self.in_channels[-3], 64, 1, bias=False)
        self.f8_4 = torch.nn.Conv2d(self.in_channels[-2], 128, 1, bias=False)
        self.f9 = torch.nn.Conv2d(192 + in_chans, 192, 1, bias=False)

    def forward_cam(self, x):
        _x = self.encoder(x)
        cam = F.conv2d(_x[-1], self.classifier.weight)
        cam = F.relu(cam)

        return cam

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

        for param in list(self.f8_3.parameters()):
            param_groups[2].append(param)
        for param in list(self.f8_4.parameters()):
            param_groups[2].append(param)
        for param in list(self.f9.parameters()):
            param_groups[2].append(param)

        return param_groups

    def forward(self, x, cam_only=False):
        N, C, H, W = x.size()
        _, x2, x3, x4 = self.encoder(x)
        # add by cyx
        if cam_only:
            cam_s4 = F.conv2d(x4, self.classifier.weight).detach()
            return cam_s4

        cam = self.classifier(x4)

        # seam
        n, c, h, w = cam.size()
        with torch.no_grad():
            cam_d = F.relu(cam.detach())
            cam_d_max = torch.max(cam_d.view(n, c, -1), dim=-1)[0].view(n, c, 1, 1) + 1e-5
            cam_d_norm = F.relu(cam_d - 1e-5) / cam_d_max
            cam_d_norm[:, 0, :, :] = 1 - torch.max(cam_d_norm[:, 1:, :, :], dim=1)[0]
            cam_max = torch.max(cam_d_norm[:, 1:, :, :], dim=1, keepdim=True)[0]
            cam_d_norm[:, 1:, :, :][cam_d_norm[:, 1:, :, :] < cam_max] = 0

        f8_3 = F.relu(self.f8_3(x2.detach()), inplace=True)
        f8_4 = F.relu(self.f8_4(x3.detach()), inplace=True)
        x_s = F.interpolate(x, (h, w), mode='bilinear', align_corners=True)
        f = torch.cat([x_s, f8_3, f8_4], dim=1)
        f = self.f9(f)

        cam_rv = F.interpolate(self.PCM(cam_d_norm, f), (H, W), mode='bilinear', align_corners=True)
        cam = F.interpolate(cam, (H, W), mode='bilinear', align_corners=True)

        return cam, cam_rv

    def PCM(self, cam, f):
        n, _, h, w = f.size()
        cam = F.interpolate(cam, (h, w), mode='bilinear', align_corners=True).view(n, -1, h * w)
        # f = self.f9(f)
        f = f.view(n, -1, h * w)
        f = f / (torch.norm(f, dim=1, keepdim=True) + 1e-5)

        aff = F.relu(torch.matmul(f.transpose(1, 2), f), inplace=True)
        aff = aff / (torch.sum(aff, dim=1, keepdim=True) + 1e-5)
        cam_rv = torch.matmul(cam, aff).view(n, -1, h, w)

        return cam_rv


# 2022.11.2: add multi-layer cls
class Mitcls_CAM_multi(nn.Module):
    def __init__(self, backbone="mit_b2", num_classes=20, embedding_dim=256, pretrained=None, in_chans = 3,
                 pooling="max", dropout=0.2, activation=None, stride=(4, 2, 2, 2)):
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
        self.classifier1 = nn.Conv2d(in_channels=self.in_channels[-1],
                                    out_channels=self.num_classes, kernel_size=1, bias=False)
        self.classifier2 = nn.Conv2d(in_channels=self.in_channels[-2],
                                    out_channels=self.num_classes, kernel_size=1, bias=False)
        self.classifier3 = nn.Conv2d(in_channels=self.in_channels[-3],
                                    out_channels=self.num_classes, kernel_size=1, bias=False)
        # self.classifier = ClassificationHead(in_channels=self.in_channels[-1], classes=num_classes,
        #                                     pooling=pooling, dropout=dropout, activation=activation)

    def forward_cam(self, x):
        _x = self.encoder(x)
        cam = F.conv2d(_x[-1], self.classifier.weight)
        cam = F.relu(cam)

        return cam

    def get_param_groups(self):

        param_groups = [[], [], []]  #

        for name, param in list(self.encoder.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        # add by cyx: classifier1,2,3
        for param in list(self.classifier1.parameters()):
            param_groups[2].append(param)
        for param in list(self.classifier2.parameters()):
            param_groups[2].append(param)
        for param in list(self.classifier3.parameters()):
            param_groups[2].append(param)

        return param_groups

    def forward(self, x, cam_only=False):

        _, x2, x3, x4 = self.encoder(x)

        # add by cyx
        if cam_only:
            cam_s4 = F.conv2d(x4, self.classifier1.weight).detach()
            cam_s3 = F.conv2d(x3, self.classifier2.weight).detach()
            cam_s2 = F.conv2d(x2, self.classifier3.weight).detach()
            return cam_s4, cam_s3, cam_s2

        cls1, cls2, cls3 = self.pooling(x4), self.pooling(x3), self.pooling(x2)
        cls1, cls2, cls3 = self.classifier1(cls1), self.classifier2(cls2), self.classifier3(cls3)

        cls1 = cls1.squeeze(-1).squeeze(-1)
        cls2 = cls2.squeeze(-1).squeeze(-1)
        cls3 = cls3.squeeze(-1).squeeze(-1)
        return cls1, cls2, cls3


# 2022.11.3: add multi-layer cls for the 4 stages and cam
class Mitcls_CAM_multicam(nn.Module):
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

        # self.decoder = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.in_channels,
        #                              embedding_dim=self.embedding_dim, num_classes=self.num_classes)
        self.pooling = nn.AdaptiveAvgPool2d(1) if pooling == "avg" else nn.AdaptiveMaxPool2d(1)
        self.classifier1 = nn.Conv2d(in_channels=self.in_channels[0],
                                    out_channels=self.num_classes, kernel_size=1, bias=False)
        self.classifier2 = nn.Conv2d(in_channels=self.in_channels[1],
                                    out_channels=self.num_classes, kernel_size=1, bias=False)
        self.classifier3 = nn.Conv2d(in_channels=self.in_channels[2],
                                    out_channels=self.num_classes, kernel_size=1, bias=False)
        self.classifier4 = nn.Conv2d(in_channels=self.in_channels[3],
                                    out_channels=self.num_classes, kernel_size=1, bias=False)
        # self.classifier = ClassificationHead(in_channels=self.in_channels[-1], classes=num_classes,
        #                                     pooling=pooling, dropout=dropout, activation=activation)

    def forward_cam(self, x):
        _x = self.encoder(x)
        cam = F.conv2d(_x[-1], self.classifier.weight)
        cam = F.relu(cam)

        return cam

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


# 2022.11.3: add multi-layer cls for the 4 stages and cam
class Mitcls_CAM_multicamv2(nn.Module):
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
        self.decoder = SegFormerFuse(feature_strides=self.feature_strides, in_channels=self.in_channels,
                                     embedding_dim=self.embedding_dim, num_classes=self.num_classes)
        self.pooling = nn.AdaptiveAvgPool2d(1) if pooling == "avg" else nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Conv2d(in_channels=embedding_dim,
                                    out_channels=self.num_classes, kernel_size=1, bias=False)


    def forward_cam(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        cam = F.conv2d(x, self.classifier.weight)
        cam = F.relu(cam)

        return cam

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

    def forward(self, x, cam_only=False, with_cam = True):

        x = self.encoder(x)
        x = self.decoder(x)

        if cam_only:
            cam_s1 = F.conv2d(x, self.classifier.weight).detach()
            return cam_s1

        cls = self.pooling(x)
        cls = self.classifier(cls)
        cls = cls.squeeze(-1).squeeze(-1)

        return cls


# 2022.10.30: add 2 encoder for mux and tlc
# 2022.11.05: add cam consistent loss
class Mitcls_CAM_2enc(nn.Module):
    def __init__(self, backbone="mit_b1", num_classes=20, embedding_dim=256,
                 pretrained=None, in_chans=(4,3),pooling="max",stride=(4, 2, 2, 1)):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
        # self.in_channels = [32, 64, 160, 256]
        # self.in_channels = [64, 128, 320, 512]
        self.in_chans = in_chans
        self.encoder1 = getattr(mix_transformer, backbone)(stride)
        self.encoder2 = getattr(mix_transformer, backbone)(stride)
        self.in_channels = self.encoder1.embed_dims
        ## initilize encoder
        if pretrained:
            state_dict = torch.load('pretrained/' + backbone + '.pth')
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            self.encoder1.load_state_dict(state_dict)
            self.encoder2.load_state_dict(state_dict)

        patch_first_conv(self.encoder1, in_chans[0], pretrained=pretrained is not None)
        patch_first_conv(self.encoder2, in_chans[1], pretrained=pretrained is not None)

        # self.decoder = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.in_channels,
        #                              embedding_dim=self.embedding_dim, num_classes=self.num_classes)
        self.pooling = nn.AdaptiveAvgPool2d(1) if pooling == "avg" else nn.AdaptiveMaxPool2d(1)
        self.classifier1 = nn.Conv2d(in_channels=self.in_channels[-1],
                                    out_channels=self.num_classes, kernel_size=1, bias=False)
        self.classifier2 = nn.Conv2d(in_channels=self.in_channels[-1],
                                    out_channels=self.num_classes, kernel_size=1, bias=False)
        # self.classifier = ClassificationHead(in_channels=self.in_channels[-1], classes=num_classes,
        #                                     pooling=pooling, dropout=dropout, activation=activation)

    def forward_cam(self, x):
        _x1 = self.encoder1(x[:, :self.in_chans[0]])
        _x2 = self.encoder2(x[:, self.in_chans[0]:])
        # merge feature
        _x = torch.cat((_x1[-1], _x2[-1]), dim=1) # N 2C H W

        cam = F.conv2d(_x, self.classifier.weight)
        cam = F.relu(cam)

        return cam

    def get_param_groups(self):

        param_groups = [[], [], []]  #

        for name, param in list(self.encoder1.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)
        # add by cyx
        for name, param in list(self.encoder2.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)
        # add by cyx
        for param in list(self.classifier1.parameters()):
            param_groups[2].append(param)
        for param in list(self.classifier2.parameters()):
            param_groups[2].append(param)

        return param_groups

    def forward(self, x, cam_only=False):

        _, _, _, _x1 = self.encoder1(x[:, :self.in_chans[0]])
        _, _, _, _x2 = self.encoder2(x[:, self.in_chans[0]:])

        # add by cyx
        if cam_only:
            cam_s1 = F.conv2d(_x1, self.classifier1.weight).detach()
            cam_s2 = F.conv2d(_x2, self.classifier2.weight).detach()
            return cam_s1, cam_s2

        # for class1
        f1 = self.classifier1(_x1)
        cls1 = self.pooling(f1)
        cls1 = cls1.squeeze(-1).squeeze(-1)
        # for class2
        f2 = self.classifier2(_x2)
        cls2 = self.pooling(f2)
        cls2 = cls2.squeeze(-1).squeeze(-1)

        return cls1, cls2, f1, f2


# original version
class Mitcls_CAM_2enc_ori(nn.Module):
    def __init__(self, backbone="mit_b1", num_classes=20, embedding_dim=256, pretrained=None, in_chans=(4,3),
                 pooling="max", dropout=0.2, activation=None, stride=(4, 2, 2, 1)):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
        # self.in_channels = [32, 64, 160, 256]
        # self.in_channels = [64, 128, 320, 512]
        self.in_chans = in_chans
        self.encoder1 = getattr(mix_transformer, backbone)(stride)
        self.encoder2 = getattr(mix_transformer, backbone)(stride)
        self.in_channels = self.encoder1.embed_dims
        ## initilize encoder
        if pretrained:
            state_dict = torch.load('pretrained/' + backbone + '.pth')
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            self.encoder1.load_state_dict(state_dict)
            self.encoder2.load_state_dict(state_dict)

        patch_first_conv(self.encoder1, in_chans[0], pretrained=pretrained is not None)
        patch_first_conv(self.encoder2, in_chans[1], pretrained=pretrained is not None)

        # self.decoder = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.in_channels,
        #                              embedding_dim=self.embedding_dim, num_classes=self.num_classes)
        self.pooling = nn.AdaptiveAvgPool2d(1) if pooling == "avg" else nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Conv2d(in_channels=self.in_channels[-1]*2,
                                    out_channels=self.num_classes, kernel_size=1, bias=False)
        # self.classifier = ClassificationHead(in_channels=self.in_channels[-1], classes=num_classes,
        #                                     pooling=pooling, dropout=dropout, activation=activation)

    def forward_cam(self, x):
        _x1 = self.encoder1(x[:, :self.in_chans[0]])
        _x2 = self.encoder2(x[:, self.in_chans[0]:])
        # merge feature
        _x = torch.cat((_x1[-1], _x2[-1]), dim=1) # N 2C H W

        cam = F.conv2d(_x, self.classifier.weight)
        cam = F.relu(cam)

        return cam

    def get_param_groups(self):

        param_groups = [[], [], []]  #

        for name, param in list(self.encoder1.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)
        # add by cyx
        for name, param in list(self.encoder2.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)
        # add by cyx
        for param in list(self.classifier.parameters()):
            param_groups[2].append(param)

        return param_groups

    def forward(self, x, cam_only=False):

        _x1 = self.encoder1(x[:, :self.in_chans[0]])
        _x2 = self.encoder2(x[:, self.in_chans[0]:])
        # merge feature
        _x = torch.cat((_x1[-1], _x2[-1]), dim=1) # N 2C H W
        # add by cyx
        if cam_only:
            cam_s4 = F.conv2d(_x, self.classifier.weight).detach()
            return cam_s4

        cls = self.pooling(_x)
        cls = self.classifier(cls)
        cls = cls.squeeze(-1).squeeze(-1)
        return cls