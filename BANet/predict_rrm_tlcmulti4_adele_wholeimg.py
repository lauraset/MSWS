#!/usr/bin/env python
# coding: utf-8
# predict our method
# label using base model

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import numpy as np
import tifffile as tif
from os.path import join
import math
import time
import rasterio as rio
from core.model import WeTr
import shutil
import argparse
from omegaconf import OmegaConf
from utils.predimg_func import predict_whole_image_over2, predict_whole_image_over

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",default='configs/zy3ba_pos_rrm.yaml',type=str,help="config")
    parser.add_argument("--logname",default='rrm_tlcmulti4_adele', type=str, help="config")
    parser.add_argument("--nchannels",default=7, type=int,help="config")
    parser.add_argument("--logdir",default='E:/yinxcao/weaksup/BANetdata2',type=str) # store

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    cfg.backbone.logname = args.logname
    cfg.dataset.nchannels = args.nchannels
    cfg.dataset.logdir = args.logdir
    return cfg


def load_img(file):
    # t1 to t2
    # file = [os.path.join(city,'img1t.tif'),
    #             os.path.join(city,'img28.tif')]
    ################################################ 1. read image
    t1 = tif.imread(file)
    t1 = t1/255.0

    # reshape to 1 C H W
    if t1.shape[2]>4:
        t1 = np.expand_dims(t1, axis=0)
    else:
        t1 = np.expand_dims(np.transpose(t1,(2,0,1)), axis=0)
    print('t1 shape:')
    print(t1.shape)
    return t1

# 2022.11.1: add tlc image
def load_imgtlc(file, filetlc):
    mux = load_img(file) # 1 C H W
    tlc = load_img(filetlc)
    return np.concatenate((mux, tlc), axis=1) # 1 (C1+C2) H W


def main(city, fname, resdir, cfg):
    ########## define parameters
    # network
    # city = r'Z:\yinxcao\change\shanghai'
    # fcode = 'sh'
    # city = r'Z:\yinxcao\change\beijing'
    # fcode = 'bj'
    num_classes = cfg.dataset.num_classes
    nchannels = cfg.dataset.nchannels
    device = 'cuda'
    logdir = os.path.join(cfg.dataset.logdir, cfg.backbone.logname)
    out_class = 1  # output
    # image
    file = os.path.join(city, fname)
    file_tlc = os.path.join(city, 'tlc'+fname[3:])
    if (not os.path.exists(file)) or (not os.path.exists(file_tlc)):
        print('img or tlc files do not exist, and return')
        return False

    idir = os.path.join(city, resdir)
    os.makedirs(idir, exist_ok=True)
    resname = os.path.join(idir, os.path.basename(logdir)+fname[:-4])
    if os.path.exists(resname+'.tif'):
        return

    # use max pooling
    net = WeTr(backbone=cfg.backbone.config, stride=cfg.backbone.stride,
               num_classes=num_classes, embedding_dim=256,
               pretrained=True, in_chans=nchannels).to(device)

    # print the model
    resume = os.path.join(logdir, 'checkpoint1400.tar') # epoch 10
    try:
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        net.load_state_dict(checkpoint['state_dict'])
        print("=> success '{}' (epoch {})"
                 .format(resume, checkpoint['epoch']))
    except:
        print("resume fails")
        return
    net.eval()

    ########## load images and predict
    t1 = load_imgtlc(file, file_tlc)

    grid = 1024
    stride = grid - 64

    # pad img
    n, b, r, c = t1.shape
    rows = math.ceil((r - grid) / (stride)) * stride + grid
    cols = math.ceil((c - grid) / (stride)) * stride + grid
    print('rows is {}, cols is {}'.format(rows, cols))
    t1pad = np.pad(t1, ((0, 0), (0, 0), (0, rows - r), (0, cols - c),), 'symmetric')
    # print(t1t2.shape)

    print('start predicting==>')
    if out_class>1:
        res = predict_whole_image_over2(net, t1pad, r=r, c=c, num_class=num_classes, grid=grid, stride=stride)
    else:
        res = predict_whole_image_over(net, t1pad, r=r, c=c, num_class=num_classes, grid=grid, stride=stride)

    # 4. convert to [0,1]
    res = torch.from_numpy(res) # C H W
    if num_classes==1:
        res = torch.sigmoid(res).numpy()
    else:
        res = torch.softmax(res, dim=0).squeeze().numpy() # C H W
        res = res[1] # the positive classes

    ############## 3. save
    rastermeta = rio.open(file).profile
    rastermeta.update(dtype='uint8', count=out_class, compress='lzw')

    # seg
    respath = resname+'_seg.tif'
    res_seg = (res > 0.5).astype('uint8') * 255
    with rio.open(respath, mode="w", **rastermeta) as dst:
        if out_class>1:
            for i in range(out_class):
                dst.write(res_seg[i], i+1)
        else:
            dst.write(res_seg, 1)

    # prob: scale from [0,1] to [0,255]
    res = (res * 255).astype('uint8')
    respath = resname+'.tif'
    with rio.open(respath, mode="w", **rastermeta) as dst:
        if out_class>1:
            for i in range(out_class):
                dst.write(res[i], i+1)
        else:
            dst.write(res, 1)

    print('success')


if __name__=="__main__":
    t0 = time.time()
    cfg = get_args()
    # shanghai
    # city = r'Z:\yinxcao\change\shanghai'
    # main(city, 'img18.tif', 'pred_ba', cfg)
    # main(city, 'img28.tif', 'pred_ba', cfg)
    #
    # # beijing
    # city = r'Z:\yinxcao\change\beijing'
    # main(city, 'img18.tif', 'pred_ba', cfg)
    # main(city, 'img28.tif', 'pred_ba', cfg)

    # city = r'Z:\yinxcao\change\kunming'
    # main(city, 'img18.tif', 'pred_ba', cfg)
    # main(city, 'img28.tif', 'pred_ba', cfg)
    #
    # city = r'Z:\yinxcao\change\shenzhen'
    # main(city, 'img18.tif', 'pred_ba', cfg)
    # main(city, 'img28.tif', 'pred_ba', cfg)

    # city = r'Z:\yinxcao\change\xian'
    # main(city, 'img18.tif', 'pred_ba', cfg)
    # main(city, 'img28.tif', 'pred_ba', cfg)

    # 2023.7.30: predict
    city = r'Z:\yinxcao\change\nanjing'
    main(city, 'img18.tif', 'pred_ba', cfg)
    main(city, 'img28.tif', 'pred_ba', cfg)
    city = r'Z:\yinxcao\change\shenyang'
    main(city, 'img18.tif', 'pred_ba', cfg)
    main(city, 'img28.tif', 'pred_ba', cfg)
    city = r'Z:\yinxcao\change\wuhan'
    main(city, 'img18.tif', 'pred_ba', cfg)
    main(city, 'img28.tif', 'pred_ba', cfg)

    # harbin
    # city = r'Y:\yinxcao\change\harbin'
    # main(city, 'img18.tif', 'pred_ba', cfg)
    # main(city, 'img28.tif', 'pred_ba', cfg)
    #
    # city = r'Y:\yinxcao\change\nanjing'
    # main(city, 'img18.tif', 'pred_ba', cfg)
    # main(city, 'img28.tif', 'pred_ba', cfg)

    # city = r'Y:\yinxcao\change\shenyang'
    # main(city, 'img18.tif', 'pred_ba', cfg)
    # main(city, 'img28.tif', 'pred_ba', cfg)
    #
    # city = r'Y:\yinxcao\change\wuhan'
    # main(city, 'img18.tif', 'pred_ba', cfg)
    # main(city, 'img28.tif', 'pred_ba', cfg)

    # city = r'Y:\yinxcao\change\xian'
    # main(city, 'img18.tif', 'pred_ba', cfg)
    # main(city, 'img28.tif', 'pred_ba', cfg)

    # city = r'Z:\yinxcao\change\zhengzhou'
    # main(city, 'img18.tif', 'pred_ba', cfg)
    # main(city, 'img28.tif', 'pred_ba', cfg)

    print('elaps: %.2f'%(time.time()-t0))