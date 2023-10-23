import os
import torch.utils.data as data
import albumentations as A
import torch
import pandas as pd
import numpy as np
import tifffile as tif
import cv2

image_transform = A.Compose([
    A.Flip(p=0.5),
    A.RandomGridShuffle(grid=(2, 2), p=0.5),
    A.Rotate(p=0.5),
]
)
# obtain mux and tlc
class myImageFlodert1t2(data.Dataset):
    def __init__(self, datalist, pseudodir, channels=3, aug=False, num_sample = 0,ignore = 3):
        if isinstance(datalist, list):
            df = pd.read_csv(datalist[0], sep=',', header=None)
            for tmp in datalist[1:]:
                tmplist = pd.read_csv(tmp, sep=',', header=None)
                df = df.append(tmplist, ignore_index=True)
            self.datalist = df
        else:
            self.datalist = pd.read_csv(datalist, sep=',', header=None)

        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        self.channels = channels
        self.pseudodir = os.path.basename(pseudodir) # basename
        self.ignore = ignore

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        idir = os.path.dirname(os.path.dirname(img_path))
        ibase = os.path.basename(img_path)[:-4]
        # mux: h w 4
        img1 = tif.imread(img_path)
        img2 = tif.imread(os.path.join(idir, 'img2', ibase+'.tif'))
        # tlc: h w 3
        tlc1 = tif.imread(os.path.join(idir, 'tlc1', ibase+'.tif'))
        tlc2 = tif.imread(os.path.join(idir, 'tlc2', ibase+'.tif'))
        # concate
        img = np.concatenate([img1, tlc1, img2, tlc2], axis=2)
        # mask
        maskpath = os.path.join(idir, self.pseudodir, 'lab'+ibase[3:]+'.png')
        mask = cv2.imread(maskpath, cv2.IMREAD_UNCHANGED)
        # Augmentation
        if self.aug:
            transformed = image_transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        img = torch.from_numpy(img).permute(2, 0, 1).float() # H W C ==> C H W
        img = img/255.0
        mask = torch.from_numpy(mask).long()# 0, 1:neg, 2: pos, 3:uncert
        #mask[mask == 2] = 1 # add by yinxcao, 2023.6.16
        mask[mask == self.ignore] = 255 # ignore region
        return img, mask

    def __len__(self):
        return len(self.datalist)


# 11.28: add RRM
class myImageFlodert1t2_RRM(data.Dataset):
    def __init__(self, datalist, pseudodir, channels=3, aug=False, num_sample = 0,
                 ignore = 3):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        self.channels = channels
        self.pseudodir = pseudodir
        self.ignore = ignore

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        idir = os.path.dirname(os.path.dirname(img_path))
        ibase = os.path.basename(img_path)[:-4]
        # mux: h w 4
        img1 = tif.imread(img_path)
        img2 = tif.imread(os.path.join(idir, 'img2', ibase+'.tif'))
        # tlc: h w 3
        tlc1 = tif.imread(os.path.join(idir, 'tlc1', ibase+'.tif'))
        tlc2 = tif.imread(os.path.join(idir, 'tlc2', ibase+'.tif'))
        # concate
        img = np.concatenate([img1, tlc1, img2, tlc2], axis=2)
        # mask
        maskpath = os.path.join(self.pseudodir, 'lab'+ibase[3:]+'.png')
        mask = cv2.imread(maskpath, cv2.IMREAD_UNCHANGED)
        # Augmentation
        if self.aug:
            transformed = image_transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        # add for rrm
        ori_img = img.copy() # H W C, numpy, uint8
        ori_img = ori_img.transpose(2, 0, 1).astype('uint8') # C H W
        croppings = np.ones_like(mask, dtype="uint8") # H W

        img = torch.from_numpy(img).permute(2, 0, 1).float() # H W C ==> C H W
        img = img/255.0
        mask = torch.from_numpy(mask) #.long()# 0, 1:neg, 2: pos, 3:uncert
        mask[mask == self.ignore] = 255 # ignore region
        return img, mask, ori_img, croppings

    def __len__(self):
        return len(self.datalist)


# return semantic cd label
class myImageFlodert1t2_sem(data.Dataset):
    def __init__(self, datalist, channels=3, aug=False, num_sample = 0,
                 ignore = 3):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        self.channels = channels
        self.ignore = ignore

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        idir = os.path.dirname(os.path.dirname(img_path))
        ibase = os.path.basename(img_path)[:-4]
        # mux: h w 4
        img1 = tif.imread(img_path)
        img2 = tif.imread(os.path.join(idir, 'img2', ibase+'.tif'))
        # tlc: h w 3
        tlc1 = tif.imread(os.path.join(idir, 'tlc1', ibase+'.tif'))
        tlc2 = tif.imread(os.path.join(idir, 'tlc2', ibase+'.tif'))
        # concate
        img = np.concatenate([img1, tlc1, img2, tlc2], axis=2)
        # mask
        m1 = cv2.imread(os.path.join(idir, 't1seg', 'lab'+ibase[3:]+'.png'), cv2.IMREAD_UNCHANGED)
        m2 = cv2.imread(os.path.join(idir, 't2seg', 'lab' + ibase[3:] + '.png'), cv2.IMREAD_UNCHANGED)
        mcd = cv2.imread(os.path.join(idir, 'cd', 'lab' + ibase[3:] + '.png'), cv2.IMREAD_UNCHANGED)
        mask = np.stack((mcd, m1, m2), axis=2) # H W C
        # Augmentation
        if self.aug:
            transformed = image_transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        img = torch.from_numpy(img).permute(2, 0, 1).float() # H W C ==> C H W
        img = img/255.0
        mask = torch.from_numpy(mask).long()# 0, 1:neg, 2: pos, 3:uncert
        mask[mask == self.ignore] = 255 # ignore region
        return img, mask

    def __len__(self):
        return len(self.datalist)


if __name__=='__main__':
    datalist = ['E:/yinxcao/weaksup/BAdata/imglist_train.csv', 'E:/yinxcao/weaksup/BAdata_sh/imglist_train.csv']
    pseudodir = 'certobjpix'
    loader = myImageFlodert1t2(datalist, pseudodir)
