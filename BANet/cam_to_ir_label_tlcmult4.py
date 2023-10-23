'''
2022.9.28: make irnet label: 1. cam to pseudo label by multithresh; 2. apply crf
2022.10.24: apply for mux+tlc images
2023.6.7: rerun, major revision
'''
import os
from os.path import join
import os.path as osp
import numpy as np
import pydensecrf.densecrf as densecrf
import cv2
from pydensecrf.utils import unary_from_labels
from glob import glob

from multiprocessing import Pool
import time
from functools import partial
from tqdm import tqdm
from skimage.filters import threshold_multiotsu
import pandas as pd

# from irnet
def crf_inference_label(img, labels, t=10, n_labels=21, gt_prob=0.7, HAS_UNK=False):
    h, w = img.shape[:2]
    d = densecrf.DenseCRF2D(w, h, n_labels)
    unary = unary_from_labels(labels, n_labels, gt_prob=gt_prob, zero_unsure=HAS_UNK)
    d.setUnaryEnergy(unary)
    # d.addPairwiseGaussian(sxy=3, compat=3)
    # d.addPairwiseBilateral(sxy=50, srgb=5, rgbim=np.ascontiguousarray(np.copy(img)), compat=10)
    d.addPairwiseGaussian(sxy=3, compat=3, kernel=densecrf.DIAG_KERNEL,
                          normalization=densecrf.NORMALIZE_SYMMETRIC)
    d.addPairwiseBilateral(sxy=10, srgb=5, rgbim=np.ascontiguousarray(np.copy(img)), compat=10,
                           kernel=densecrf.DIAG_KERNEL,
                           normalization=densecrf.NORMALIZE_SYMMETRIC)
    q = d.inference(t)
    MAP = np.argmax(np.array(q).reshape((n_labels, h, w)), axis=0)
    return MAP

def cam_to_label(cams, img):
    '''
    :param cams: normalized to [0,1], H W
    :param img: [0,255], H W C
    :return: pseudo labels
    '''
    if cams.max() == 0:
        conf = np.zeros_like(cams, dtype=np.uint8)
        return conf

    threshs = threshold_multiotsu(image=cams, classes=3) # generate 5 (6-1) threshold
    # print(threshs)
    tl = threshs[0]
    th = threshs[-1]

    fg_conf = np.uint8(cams >= th)
    # fg_conf = crf_inference_label(img, fg_conf, n_labels=2, HAS_UNK=False)

    bg_conf = np.uint8(cams >= tl)
    # bg_conf = crf_inference_label(img, bg_conf, n_labels=2, HAS_UNK=False)

    # 2. combine confident fg & bg
    conf = fg_conf.copy()
    conf[fg_conf == 0] = 255 # ignore
    conf[bg_conf + fg_conf == 0] = 0 # background
    return conf

def run(imgroot, savedir, filep):
    img_name = osp.basename(filep)[:-4]
    resname = os.path.join(savedir, img_name + '_c.png')
    if os.path.exists(resname):
        return
    cams = cv2.imread(filep, cv2.IMREAD_UNCHANGED)
    if (np.unique(cams)).max() == 0:
        conf = np.zeros_like(cams, dtype=np.uint8)
        cv2.imwrite(os.path.join(savedir, img_name + '.png'), conf)
        cv2.imwrite(os.path.join(savedir, img_name + '_c.png'), conf)
        print('equal 0: %s'%img_name)
        return

    imgpath = join(imgroot, img_name+'.jpg')
    img = cv2.imread(imgpath, cv2.IMREAD_UNCHANGED)

    # 1. find confident fg & bg: use multithresh
    threshs = threshold_multiotsu(image=cams, classes=3) # generate 5 (6-1) threshold
    # print(threshs)
    tl = threshs[0]
    th = threshs[-1]

    fg_conf = np.uint8(cams >= th)
    fg_conf = crf_inference_label(img, fg_conf, n_labels=2, HAS_UNK=False)

    bg_conf = np.uint8(cams >= tl)
    bg_conf = crf_inference_label(img, bg_conf, n_labels=2, HAS_UNK=False)

    # 2. combine confident fg & bg
    conf = fg_conf.copy()
    conf[fg_conf == 0] = 255 # ignore
    conf[bg_conf + fg_conf == 0] = 0 # background

    cv2.imwrite(os.path.join(savedir, img_name + '.png'), conf.astype(np.uint8))
    # color
    tmp = np.ones_like(conf, dtype=np.uint8)*127
    tmp[conf == 0] = 0
    tmp[conf == 1] = 255
    cv2.imwrite(os.path.join(savedir, img_name + '_c.png'), tmp)

if __name__=="__main__":
    imgroot = r'E:\yinxcao\ZY3LC\datanew8bit\pos'
    iroot = r'E:\yinxcao\weaksup\BANetdata2'
    idir = 'mit_b1cam_stride_tlcmulti4'
    campath = join(iroot, idir, 'cam')
    tmp = os.listdir(campath)
    filelist = []
    for i in tmp:
        if 'rgb' in i:
            continue
        filelist.append(i)

    camlist = [join(campath, i) for i in filelist]
    savedir = join(iroot, idir, 'irnet')
    os.makedirs(savedir, exist_ok=True)

    # # test on several images
    # for filep in tqdm(camlist):
    #     run(imgroot, savedir, filep)

    # # multiprocessing
    t1 = time.time()
    pfunc = partial(run, imgroot, savedir) # package
    pool = Pool(8)
    pool.map(pfunc, camlist)
    pool.close()
    pool.join()
    print(time.time()-t1)
