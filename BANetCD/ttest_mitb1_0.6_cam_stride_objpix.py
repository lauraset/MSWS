'''
2021.09.06
@Yinxia Cao
@function: used for tuitiantu detection
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import random
import numpy as np
from tqdm import tqdm
from core.model import WeTrCD
from metrics import SegmentationMetric, acc2file, accprint_seg
import cv2
import argparse
from omegaconf import OmegaConf
import pandas as pd
from utils.preprocess import preprocess_t1t2
import shutil


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",default='configs/zy3bacd.yaml',type=str, help="config")
    parser.add_argument("--nchannels",default=7,type=int,help="config")
    # beijing
    # parser.add_argument("--logname",default='objpix_bj_nopre', type=str, help="config")
    # parser.add_argument("--testlist_path", default=r'E:\yinxcao\weaksup\BAdata\imglist_test.csv', type=str)
    # parser.add_argument("--checkpoint", default='checkpoint_last.tar', type=str)
    # shanghai
    parser.add_argument("--logname",default='objpix_sh_nopre', type=str, help="config")
    parser.add_argument("--testlist_path", default=r'E:\yinxcao\weaksup\BAdata_sh\imglist_test.csv', type=str)
    parser.add_argument("--checkpoint", default='checkpoint_last.tar', type=str)
    parser.add_argument("--respath_suffix", default='')
    parser.add_argument("--save", default=True, type=bool)
    parser.add_argument("--num_classes", default=3, type=int, help="eval") # 0, 1:neg, 2: pos, 3:uncert

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    cfg.dataset.nchannels = args.nchannels
    cfg.backbone.logname = args.logname
    cfg.testlist_path = args.testlist_path
    cfg.checkpoint = args.checkpoint
    cfg.save = args.save
    cfg.respath_suffix = args.respath_suffix
    cfg.dataset.num_classes = args.num_classes
    return cfg


def main(cfg):
    # Setup seeds
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    np.random.seed(1337)
    random.seed(1337)

    # Setup datalist
    testlist = pd.read_csv(cfg.testlist_path, header=None, sep=',')

    # Setup parameters
    num_classes = cfg.dataset.num_classes  # 0-unchange, 1-neg, 2-pos
    nchannels = 7
    device = 'cuda'
    logdir = os.path.join(cfg.dataset.logdir, cfg.backbone.logname)

    # use max pooling
    net = WeTrCD(backbone=cfg.backbone.config, stride=cfg.backbone.stride,
               num_classes=num_classes, embedding_dim=256,
               pretrained=True, in_chans=nchannels).to(device)

    # print the model
    start_epoch = 0
    resume = os.path.join(logdir, cfg.checkpoint)
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        net.load_state_dict(checkpoint['state_dict'])
        if 'iter' in checkpoint.keys():
            start_epoch = checkpoint['iter']
        else:
            start_epoch = checkpoint['epoch']
    else:
        print("=> no checkpoint found at resume")
        print("=> Will stop.")
        return

    scale = 1.0
    id = str(start_epoch) + str(scale)
    txtpath = os.path.join(logdir, 'acc' + id + cfg.respath_suffix + '.txt')  # save acc

    if cfg.save:
        respath = os.path.join(logdir, 'pred_' + id + cfg.respath_suffix)
        os.makedirs(respath, exist_ok=True)

    vtest_epoch(net, testlist, device, num_classes, start_epoch, txtpath, cfg.save, respath, scale)


def vtest_epoch(model, testlist, device, classes, epoch, txtpath, issave=False, respath=None, scale=1.0, ismulti=False):
    model.eval()
    acc_total = SegmentationMetric(numClass=classes, device=device)
    num = testlist.shape[0]
    pbar = tqdm(range(num), disable=False)
    with torch.no_grad():
        for idx in range(num):
            # read data
            x, y_true, _ = preprocess_t1t2(testlist, idx, scale)

            x = x.to(device, non_blocking=True)
            y_true = y_true.to(device, non_blocking=True)

            ypred = model.forward(x) # N C H W
            ypred = torch.softmax(ypred, dim=1) # N C H W
            acc_total.addBatch(ypred.argmax(dim=1)>0, y_true)

            f1 = acc_total.F1score()[1]
            iou = acc_total.IntersectionOverUnion()[1]
            pbar.set_description(
                'Test Epoch:{epoch:4}. Iter:{batch:4}|{iter:4}. F1 {f1:.3f}, IOU: {iou:.3f}'.format(
                    epoch=epoch, batch=idx, iter=num, f1=f1, iou=iou))
            pbar.update()
            # save to dir
            if issave:
                imgpath = testlist.iloc[idx,0]
                ibase = os.path.basename(imgpath)[:-4]  #
                pred_name = os.path.join(respath, ibase + "_pred.png")
                predprob_name = os.path.join(respath, ibase + "_predprob.png")
                # predprob and predlabel
                pred = ypred[0].permute([1,2,0]).squeeze().cpu().numpy()
                if pred.shape[-1]<3:
                    cv2.imwrite(predprob_name, pred[...,1] * 255)
                else:
                    cv2.imwrite(predprob_name, pred * 255)  # 0,1 ==> 0,255
                pred = ypred[0].argmax(dim=0).squeeze().cpu().numpy()
                cv2.imwrite(pred_name, pred * 255)  # 0,1 ==> 0,255
                # img
                # idir = os.path.dirname(os.path.dirname(imgpath))
                #img_rgbpath = os.path.join(idir, 'imgc', ibase + '.jpg')
                # shutil.copy(img_rgbpath, respath)

        pbar.close()

    accprint_seg(acc_total)
    acc2file(acc_total, txtpath)
    return True


if __name__ == "__main__":
    cfg = get_args()
    main(cfg)