'''
2021.9.6 tuitiantu
'''
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
# for cls
class myImageFloder(data.Dataset):
    def __init__(self, datalist, channels=3, aug=False, num_sample = 0):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        self.channels = channels

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        img = tif.imread(img_path)[:, :, :self.channels]
        lab = self.datalist.iloc[index, 1]
        # Augmentation
        if self.aug:
            img = image_transform(image=img)["image"]

        img = torch.from_numpy(img).permute(2, 0, 1).float() # H W C ==> C H W
        img = img/255.0
        lab = torch.tensor(lab).long()# [0, C-1], 0,1,2 index
        return img, lab

    def __len__(self):
        return len(self.datalist)

# 2022.10.12: add mux and tlc images
class myImageFloder_muxtlc(data.Dataset):
    def __init__(self, datalist, channels=4, aug=False, num_sample = 0):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        # if self.aug:
        #     self.datalist.sample(frac=1, random_state=0)
        # self.channels = channels

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        img = tif.imread(img_path) # [:, :, :self.channels]
        # tlc
        img_name = os.path.basename(img_path)
        img_dir = os.path.dirname(os.path.dirname(img_path))
        tlc_path = os.path.join(img_dir, 'tlc', 'tlc'+img_name[3:])
        tlc = tif.imread(tlc_path)
        # concat
        img = np.concatenate((img, tlc), axis=2) # N H (C1+C2)
        # lab
        lab = self.datalist.iloc[index, 1]
        # Augmentation
        if self.aug:
            img = image_transform(image=img)["image"]

        img = img/255.0
        img = torch.from_numpy(img).permute(2, 0, 1).float() # H W C ==> C H W

        lab = torch.tensor(lab).long()# [0, C-1], 0,1,2 index

        return img, lab

    def __len__(self):
        return len(self.datalist)


# 2022.10.12: add mux and tlc images
class myImageFloder_tlc(data.Dataset):
    def __init__(self, datalist, channels=3, aug=False, num_sample = 0):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        self.channels = channels

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        # img = tif.imread(img_path)[:, :, :self.channels]
        # tlc
        img_name = os.path.basename(img_path)
        img_dir = os.path.dirname(os.path.dirname(img_path))
        tlc_path = os.path.join(img_dir, 'tlc', 'tlc'+img_name[3:])
        img = tif.imread(tlc_path)
        # concat
        # img = np.concatenate((img, tlc), axis=2) # N H (C1+C2)
        # lab
        lab = self.datalist.iloc[index, 1]
        # Augmentation
        if self.aug:
            img = image_transform(image=img)["image"]

        img = img/255.0
        img = torch.from_numpy(img).permute(2, 0, 1).float() # H W C ==> C H W

        lab = torch.tensor(lab).long()# [0, C-1], 0,1,2 index

        return img, lab

    def __len__(self):
        return len(self.datalist)


# 2022.10.12: add mux and tlc images
class myImageFloder_tlcpath(data.Dataset):
    def __init__(self, datalist, channels=3, aug=False, num_sample = 0):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        self.channels = channels

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        # img = tif.imread(img_path)[:, :, :self.channels]
        # tlc
        img_name = os.path.basename(img_path)
        img_dir = os.path.dirname(os.path.dirname(img_path))
        tlc_path = os.path.join(img_dir, 'tlc', 'tlc'+img_name[3:])
        img = tif.imread(tlc_path)
        # concat
        # img = np.concatenate((img, tlc), axis=2) # N H (C1+C2)
        # lab
        lab = self.datalist.iloc[index, 1]
        # Augmentation
        if self.aug:
            img = image_transform(image=img)["image"]

        img = img/255.0
        img = torch.from_numpy(img).permute(2, 0, 1).float() # H W C ==> C H W

        lab = torch.tensor(lab).long()# [0, C-1], 0,1,2 index

        return img, lab, img_path

    def __len__(self):
        return len(self.datalist)


# 2022.10.12: add mux and tlc images
class myImageFloder_muxtlc_path(data.Dataset):
    def __init__(self, datalist, channels=3, aug=False, num_sample = 0):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        self.channels = channels

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        img = tif.imread(img_path)[:, :, :self.channels]
        # tlc
        img_name = os.path.basename(img_path)
        img_dir = os.path.dirname(os.path.dirname(img_path))
        tlc_path = os.path.join(img_dir, 'tlc', 'tlc'+img_name[3:])
        tlc = tif.imread(tlc_path)
        # concat
        img = np.concatenate((img, tlc), axis=2) # N H (C1+C2)
        # lab
        lab = self.datalist.iloc[index, 1]
        # Augmentation
        if self.aug:
            img = image_transform(image=img)["image"]

        img = img/255.0
        img = torch.from_numpy(img).permute(2, 0, 1).float() # H W C ==> C H W

        lab = torch.tensor(lab).long()# [0, C-1], 0,1,2 index

        return img, lab, img_path

    def __len__(self):
        return len(self.datalist)


# for cls, and return path
class myImageFloder_path(data.Dataset):
    def __init__(self, datalist, channels=3, aug=False, num_sample = 0):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        self.channels = channels

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        img = tif.imread(img_path)[:, :, :self.channels]
        lab = self.datalist.iloc[index, 1]
        # Augmentation
        if self.aug:
            img = image_transform(image=img)["image"]

        img = torch.from_numpy(img).permute(2, 0, 1).float() # H W C ==> C H W
        img = img/255.0
        lab = torch.tensor(lab).long()# [0, C-1], 0,1,2 index
        return img, lab, img_path

    def __len__(self):
        return len(self.datalist)


# for seg with pseudo labels
class myImageFloder_seg(data.Dataset):
    def __init__(self, datalist, pseudodir, channels=3, aug=False, num_sample = 0):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        self.channels = channels
        self.pseudodir = pseudodir # for storing the pseudo labels from CAM

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        img = tif.imread(img_path)[:, :, :self.channels]
        img_name = os.path.basename(img_path)[:-4]
        mask_path = os.path.join(self.pseudodir, img_name+'.png')
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) # png
        # lab = self.datalist.iloc[index, 1]
        # Augmentation
        if self.aug:
            transformed = image_transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        img = torch.from_numpy(img).permute(2, 0, 1).float() # H W C ==> C H W
        img = img/255.0
        # lab = torch.tensor(lab).long()# [0, C-1], 0,1,2 index
        mask = torch.from_numpy(mask).long() # 0, 1, 255 (ignore)
        return img, mask

    def __len__(self):
        return len(self.datalist)


# 2022.11.30: add mux tlc
class myImageFloder_muxtlc_seg(data.Dataset):
    def __init__(self, datalist, pseudodir, channels=7, aug=False, num_sample = 0):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        #self.channels = channels
        self.pseudodir = pseudodir # for storing the pseudo labels from CAM

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        img = tif.imread(img_path)
        # tlc
        img_name = os.path.basename(img_path)
        img_dir = os.path.dirname(os.path.dirname(img_path))
        tlc_path = os.path.join(img_dir, 'tlc', 'tlc'+img_name[3:])
        tlc = tif.imread(tlc_path)
        # concat
        img = np.concatenate((img, tlc), axis=2) # N H (C1+C2)
        # pseudo lab
        mask_path = os.path.join(self.pseudodir, img_name[:-4]+'.png')
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) # png
        # lab = self.datalist.iloc[index, 1]
        # Augmentation
        if self.aug:
            transformed = image_transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        img = torch.from_numpy(img).permute(2, 0, 1).float() # H W C ==> C H W
        img = img/255.0
        # lab = torch.tensor(lab).long()# [0, C-1], 0,1,2 index
        mask = torch.from_numpy(mask).long() # 0, 1, 255 (ignore)
        return img, mask

    def __len__(self):
        return len(self.datalist)


# 2022.11.30: for sec
class myImageFloder_muxtlc_seg_SEC(data.Dataset):
    def __init__(self, datalist, pseudodir, channels=7, aug=False, num_sample = 0, classes=2):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        #self.channels = channels
        self.pseudodir = pseudodir # for storing the pseudo labels from CAM
        self.classes = classes

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        img = tif.imread(img_path)
        # tlc
        img_name = os.path.basename(img_path)
        img_dir = os.path.dirname(os.path.dirname(img_path))
        tlc_path = os.path.join(img_dir, 'tlc', 'tlc'+img_name[3:])
        tlc = tif.imread(tlc_path)
        # concat
        img = np.concatenate((img, tlc), axis=2) # N H (C1+C2)
        # pseudo lab
        mask_path = os.path.join(self.pseudodir, img_name[:-4]+'.png')
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) # png
        # mask: 0-bg, 1-fg, 255-ignore
        # change to : 1-bg, 2-fg, 0-ignore
        mask = mask+1
        lab = 0 # self.datalist.iloc[index, 1]
        # Augmentation
        if self.aug:
            transformed = image_transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        img = torch.from_numpy(img).permute(2, 0, 1).float() # H W C ==> C H W
        img = img/255.0
        # lab
        lab = torch.tensor(lab).long()
        lab_onehot = torch.zeros((self.classes))
        lab_onehot.scatter_(0, 1-lab, 1) # lab: 0-lvwang, 1-negative
        lab_onehot[0] = 1 # the first dim; bg
        lab_onehot = lab_onehot.unsqueeze(0).unsqueeze(1)
        # mask to one-hot
        mask = torch.tensor(mask).long()
        mask_onehot = torch.zeros((self.classes+1, mask.shape[0], mask.shape[1])).long()
        mask_onehot.scatter_(0, mask.unsqueeze(0), 1)
        mask_onehot = mask_onehot[1:, :, :] # delete the first dim, which is ignored
        return img, mask_onehot, lab_onehot, (mask.type(torch.uint8)-1)

    def __len__(self):
        return len(self.datalist)


# for seg test, 2022.10.22
class myImageFloder_segtest(data.Dataset):
    def __init__(self, datalist, channels=3, aug=False, num_sample = 0):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        self.channels = channels

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        img = tif.imread(img_path)[:, :, :self.channels]

        mask_path = self.datalist.iloc[index, 1]
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) # png
        # lab = self.datalist.iloc[index, 1]
        # Augmentation
        if self.aug:
            transformed = image_transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        img = torch.from_numpy(img).permute(2, 0, 1).float() # H W C ==> C H W
        img = img/255.0
        # lab = torch.tensor(lab).long()# [0, C-1], 0,1,2 index
        mask = torch.from_numpy(mask).long() # 0, 1, 255 (ignore)
        return img, mask

    def __len__(self):
        return len(self.datalist)

# for seg test, 2022.10.22
class myImageFloder_muxtlc_segtest(data.Dataset):
    def __init__(self, datalist, channels=3, aug=False, num_sample = 0):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        self.channels = channels

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        img = tif.imread(img_path)[:, :, :self.channels]
        # tlc
        img_name = os.path.basename(img_path)
        img_dir = os.path.dirname(os.path.dirname(img_path))
        tlc_path = os.path.join(img_dir, 'tlc', 'tlc'+img_name[3:])
        tlc = tif.imread(tlc_path)
        # concat
        img = np.concatenate((img, tlc), axis=2) # N H (C1+C2)

        mask_path = self.datalist.iloc[index, 1]
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) # png
        # lab = self.datalist.iloc[index, 1]
        # Augmentation
        if self.aug:
            transformed = image_transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        img = torch.from_numpy(img).permute(2, 0, 1).float() # H W C ==> C H W
        img = img/255.0
        # lab = torch.tensor(lab).long()# [0, C-1], 0,1,2 index
        mask = torch.from_numpy(mask).long() # 0, 1, 255 (ignore)
        return img, mask

    def __len__(self):
        return len(self.datalist)


# for seg test, 2022.10.22
class myImageFloder_tlc_segtest(data.Dataset):
    def __init__(self, datalist, channels=3, aug=False, num_sample = 0):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        self.channels = channels

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        # img = tif.imread(img_path)[:, :, :self.channels]
        # tlc
        img_name = os.path.basename(img_path)
        img_dir = os.path.dirname(os.path.dirname(img_path))
        tlc_path = os.path.join(img_dir, 'tlc', 'tlc'+img_name[3:])
        img = tif.imread(tlc_path)
        # concat
        # img = np.concatenate((img, tlc), axis=2) # N H (C1+C2)

        mask_path = self.datalist.iloc[index, 1]
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) # png
        # lab = self.datalist.iloc[index, 1]
        # Augmentation
        if self.aug:
            transformed = image_transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        img = torch.from_numpy(img).permute(2, 0, 1).float() # H W C ==> C H W
        img = img/255.0
        # lab = torch.tensor(lab).long()# [0, C-1], 0,1,2 index
        mask = torch.from_numpy(mask).long() # 0, 1, 255 (ignore)
        return img, mask

    def __len__(self):
        return len(self.datalist)


# for RRM model, pseudo label from irn
class myImageFloder_RRM_irn(data.Dataset):
    def __init__(self, datalist, pseudodir, channels=3, aug=False, num_sample = 0):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        self.channels = channels
        self.pseudodir = pseudodir # for storing the pseudo labels from CAM

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        img = tif.imread(img_path)[:, :, :self.channels]
        img_name = os.path.basename(img_path)[:-4]
        mask_path = os.path.join(self.pseudodir, img_name+'.png')
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) # png
        # lab = self.datalist.iloc[index, 1]
        # Augmentation
        if self.aug:
            transformed = image_transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        # add for rrm
        ori_img = img[:, :, :3].copy() # numpy, uint8
        ori_img = ori_img.transpose(2, 0, 1).astype('uint8') # C H W
        croppings = np.ones_like(mask, dtype="uint8") # H W

        img = torch.from_numpy(img).permute(2, 0, 1).float() # H W C ==> C H W
        img = img/255.0
        # lab = torch.tensor(lab).long()# [0, C-1], 0,1,2 index
        mask = torch.from_numpy(mask) # 0, 1, 255 (ignore)
        return img, mask, ori_img, croppings

    def __len__(self):
        return len(self.datalist)


# only tlc
class myImageFloder_RRM_irn_onlytlc(data.Dataset):
    def __init__(self, datalist, pseudodir, channels=3, aug=False, num_sample = 0):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        self.channels = channels
        self.pseudodir = pseudodir # for storing the pseudo labels from CAM

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        # img = tif.imread(img_path)[:, :, :self.channels]
        # tlc
        img_name = os.path.basename(img_path)
        img_dir = os.path.dirname(os.path.dirname(img_path))
        tlc_path = os.path.join(img_dir, 'tlc', 'tlc'+img_name[3:])
        img = tif.imread(tlc_path)

        mask_path = os.path.join(self.pseudodir, img_name[:-4]+'.png')
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) # png
        # lab = self.datalist.iloc[index, 1]
        # Augmentation
        if self.aug:
            transformed = image_transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        # add for rrm
        ori_img = img[:, :, :3].copy() # numpy, uint8
        ori_img = ori_img.transpose(2, 0, 1).astype('uint8') # C H W
        croppings = np.ones_like(mask, dtype="uint8") # H W

        img = torch.from_numpy(img).permute(2, 0, 1).float() # H W C ==> C H W
        img = img/255.0
        # lab = torch.tensor(lab).long()# [0, C-1], 0,1,2 index
        mask = torch.from_numpy(mask) # 0, 1, 255 (ignore)
        return img, mask, ori_img, croppings

    def __len__(self):
        return len(self.datalist)


# add pantex labels
class myImageFloder_RRM_irn_pan(data.Dataset):
    def __init__(self, datalist, pseudodir, pandir, channels=3, aug=False, num_sample = 0):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        self.channels = channels
        self.pseudodir = pseudodir # for storing the pseudo labels from CAM
        self.pandir = pandir

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        img = tif.imread(img_path)[:, :, :self.channels]
        img_name = os.path.basename(img_path)[:-4]
        mask_path = os.path.join(self.pseudodir, img_name+'.png')
        pseudo = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) # png
        #
        panpath = os.path.join(self.pandir, img_name+'.png')
        pan = cv2.imread(panpath, cv2.IMREAD_UNCHANGED)
        mask = np.stack((pseudo, pan), axis=2) # H, W, C
        # lab = self.datalist.iloc[index, 1]
        # Augmentation
        if self.aug:
            transformed = image_transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]
        #
        pseudo = mask[:, :, 0]
        pan = mask[:, :, 1]

        # add for rrm
        ori_img = img[:, :, :3].copy() # numpy, uint8
        ori_img = ori_img.transpose(2, 0, 1).astype('uint8') # C H W
        croppings = np.ones_like(pseudo, dtype="uint8") # H W

        img = torch.from_numpy(img).permute(2, 0, 1).float() # H W C ==> C H W
        img = img/255.0
        # lab = torch.tensor(lab).long()# [0, C-1], 0,1,2 index
        pseudo = torch.from_numpy(pseudo) # 0, 1, 255 (ignore)

        panlabel = np.ones_like(pan)*255
        panlabel[pan>0.4*255]=1 # pos
        panlabel[pan<0.2*255]=0 # neg
        panlabel = torch.from_numpy(panlabel)

        return img, pseudo, ori_img, croppings, panlabel

    def __len__(self):
        return len(self.datalist)


# for RRM model, pseudo label from irn
class myImageFloder_RRM_irn_tlc(data.Dataset):
    def __init__(self, datalist, pseudodir, channels=3, aug=False, num_sample = 0):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        self.channels = channels
        self.pseudodir = pseudodir # for storing the pseudo labels from CAM

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        img = tif.imread(img_path)[:, :, :self.channels]
        # tlc
        img_name = os.path.basename(img_path)
        img_dir = os.path.dirname(os.path.dirname(img_path))
        tlc_path = os.path.join(img_dir, 'tlc', 'tlc'+img_name[3:])
        tlc = tif.imread(tlc_path)
        # concat
        img = np.concatenate((img, tlc), axis=2) # N H (C1+C2)

        # mask
        mask_path = os.path.join(self.pseudodir, img_name[:-4]+'.png')
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) # png
        # lab = self.datalist.iloc[index, 1]
        # Augmentation
        if self.aug:
            transformed = image_transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        # add for rrm
        ori_img = img[:, :, :3].copy() # numpy, uint8
        ori_img = ori_img.transpose(2, 0, 1).astype('uint8') # C H W
        croppings = np.ones_like(mask, dtype="uint8") # H W

        img = torch.from_numpy(img).permute(2, 0, 1).float() # H W C ==> C H W
        img = img/255.0
        # lab = torch.tensor(lab).long()# [0, C-1], 0,1,2 index
        mask = torch.from_numpy(mask) # 0, 1, 255 (ignore)
        return img, mask, ori_img, croppings

    def __len__(self):
        return len(self.datalist)


# add edge label from pidinet
class myImageFloder_RRM_irn_tlc_edge(data.Dataset):
    def __init__(self, datalist, pseudodir, edgedir, channels=3, aug=False, num_sample = 0):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        self.channels = channels
        self.pseudodir = pseudodir # for storing the pseudo labels from CAM
        self.edgedir = edgedir

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        img = tif.imread(img_path)[:, :, :self.channels]
        # tlc
        img_name = os.path.basename(img_path)
        img_dir = os.path.dirname(os.path.dirname(img_path))
        tlc_path = os.path.join(img_dir, 'tlc', 'tlc'+img_name[3:])
        tlc = tif.imread(tlc_path)
        # concat
        img = np.concatenate((img, tlc), axis=2) # N H (C1+C2)
        # mask
        mask_path = os.path.join(self.pseudodir, img_name[:-4]+'.png')
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) # png
        # edge
        edge_path = os.path.join(self.edgedir, img_name[:-4]+'.jpg')
        edge = cv2.imread(edge_path, cv2.IMREAD_UNCHANGED)
        # concat
        label = np.stack((mask, edge), axis=2) # 0,1,2
        # Augmentation
        if self.aug:
            transformed = image_transform(image=img, mask=label)
            img = transformed["image"]
            label = transformed["mask"]

        mask = label[:, :, 0]
        edge = label[:, :, 1]
        # add for rrm
        ori_img = img[:, :, :3].copy() # numpy, uint8
        ori_img = ori_img.transpose(2, 0, 1).astype('uint8') # C H W
        croppings = np.ones_like(mask, dtype="uint8") # H W

        img = torch.from_numpy(img).permute(2, 0, 1).float() # H W C ==> C H W
        img = img/255.0
        # lab = torch.tensor(lab).long()# [0, C-1], 0,1,2 index
        mask = torch.from_numpy(mask) # 0, 1, 255 (ignore)
        edge = torch.from_numpy(edge).float()/255.0 # 0-1
        return img, mask, ori_img, croppings, edge

    def __len__(self):
        return len(self.datalist)


# for RRM model, pseudo label from irn
class myImageFloder_RRM_irn_tlco(data.Dataset):
    def __init__(self, datalist, pseudodir, channels=3, aug=False, num_sample = 0):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        self.channels = channels
        self.pseudodir = pseudodir # for storing the pseudo labels from CAM

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        # img = tif.imread(img_path)[:, :, :self.channels]
        # tlc
        img_name = os.path.basename(img_path)
        img_dir = os.path.dirname(os.path.dirname(img_path))
        tlc_path = os.path.join(img_dir, 'tlc', 'tlc'+img_name[3:])
        img = tif.imread(tlc_path)
        # concat
        # img = np.concatenate((img, tlc), axis=2) # N H (C1+C2)

        # mask
        mask_path = os.path.join(self.pseudodir, img_name[:-4]+'.png')
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) # png
        # lab = self.datalist.iloc[index, 1]
        # Augmentation
        if self.aug:
            transformed = image_transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        # add for rrm
        ori_img = img[:, :, :3].copy() # numpy, uint8
        ori_img = ori_img.transpose(2, 0, 1).astype('uint8') # C H W
        croppings = np.ones_like(mask, dtype="uint8") # H W

        img = torch.from_numpy(img).permute(2, 0, 1).float() # H W C ==> C H W
        img = img/255.0
        # lab = torch.tensor(lab).long()# [0, C-1], 0,1,2 index
        mask = torch.from_numpy(mask) # 0, 1, 255 (ignore)
        return img, mask, ori_img, croppings

    def __len__(self):
        return len(self.datalist)


# 2022.10.3: for end-to-end training
class myImageFloder_RRM(data.Dataset):
    def __init__(self, datalist, channels=3, aug=False, num_sample = 0):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        self.channels = channels

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        img = tif.imread(img_path)[:, :, :self.channels]
        lab = self.datalist.iloc[index, 1]
        # Augmentation
        if self.aug:
            img = image_transform(image=img)["image"]

        # add for rrm
        ori_img = img[:, :, :3].copy() # numpy, uint8
        ori_img = ori_img.transpose(2, 0, 1).astype('uint8') # C H W
        h, w = img.shape[:2]
        croppings = np.ones((h, w), dtype="uint8") # H W

        img = torch.from_numpy(img).permute(2, 0, 1).float() # H W C ==> C H W
        img = img/255.0
        lab = torch.tensor(lab).long()# [0, C-1], 0,1,2 index

        return img_path, img, lab, ori_img, croppings

    def __len__(self):
        return len(self.datalist)


# 2022.10.8: for training irnet
class myImageFloder_IRN(data.Dataset):
    def __init__(self, camroot, datalist, channels=3, aug=False, num_sample = 0,
                 suffix ="_irn.png", path_index = None, label_scale = 0.25):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        self.channels = channels
        self.camroot = camroot
        self.suffix = suffix
        self.label_scale = label_scale
        self.extract_aff_lab_func = GetAffinityLabelFromIndices(path_index.src_indices, path_index.dst_indices)

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        ibase = os.path.basename(img_path)[:-4]
        img = tif.imread(img_path) #
        h, w = img.shape[0], img.shape[1]
        if "negative" in img_path:
            mask = np.zeros((h, w), dtype='uint8')
        else:
            mask = cv2.imread(os.path.join(self.camroot, ibase + self.suffix), cv2.IMREAD_UNCHANGED)

        # Augmentation
        if self.aug:
            transformed = image_transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]
        # img
        img = torch.from_numpy(img).permute(2, 0, 1).float() # H W C ==> C H W
        img = img/255.0
        # mask
        # reduced_label = pil_rescale(mask, scale=self.label_scale, order=0) # nearest
        target_size = (int(np.round(h * self.label_scale)), int(np.round(w * self.label_scale)))
        reduced_label = cv2.resize(mask, dsize=target_size, interpolation=cv2.INTER_NEAREST)

        out = {}
        out['img'] = img
        out['aff_bg_pos_label'], out['aff_fg_pos_label'], out['aff_neg_label'] = \
            self.extract_aff_lab_func(reduced_label)
        return out

    def __len__(self):
        return len(self.datalist)


# 2022.11.30: add tlc bands
class myImageFloder_IRN_tlc(data.Dataset):
    def __init__(self, camroot, datalist, channels=7, aug=False, num_sample = 0,
                 suffix ="_irn.png", path_index = None, label_scale = 0.25):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        self.channels = channels
        self.camroot = camroot
        self.suffix = suffix
        self.label_scale = label_scale
        self.extract_aff_lab_func = GetAffinityLabelFromIndices(path_index.src_indices, path_index.dst_indices)

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        img = tif.imread(img_path) #
        h, w = img.shape[0], img.shape[1]
        # tlc
        img_name = os.path.basename(img_path)
        img_dir = os.path.dirname(os.path.dirname(img_path))
        tlc_path = os.path.join(img_dir, 'tlc', 'tlc'+img_name[3:])
        tlc = tif.imread(tlc_path)
        # concat
        img = np.concatenate((img, tlc), axis=2) # N H (C1+C2)

        if "negative" in img_path:
            mask = np.zeros((h, w), dtype='uint8')
        else:
            mask = cv2.imread(os.path.join(self.camroot, img_name[:-4] + self.suffix), cv2.IMREAD_UNCHANGED)

        # Augmentation
        if self.aug:
            transformed = image_transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]
        # img
        img = torch.from_numpy(img).permute(2, 0, 1).float() # H W C ==> C H W
        img = img/255.0
        # mask
        # reduced_label = pil_rescale(mask, scale=self.label_scale, order=0) # nearest
        target_size = (int(np.round(h * self.label_scale)), int(np.round(w * self.label_scale)))
        reduced_label = cv2.resize(mask, dsize=target_size, interpolation=cv2.INTER_NEAREST)

        out = {}
        out['img'] = img
        out['aff_bg_pos_label'], out['aff_fg_pos_label'], out['aff_neg_label'] = \
            self.extract_aff_lab_func(reduced_label)
        return out

    def __len__(self):
        return len(self.datalist)


# 2022.10.25: based on IRN, add labels and cls
class myImageFloder_SANCE(data.Dataset):
    def __init__(self, camroot, datalist, channels=3, aug=False, num_sample = 0,
                 suffix ="_irn.png", path_index = None, label_scale = 0.25):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        self.channels = channels
        self.camroot = camroot
        self.suffix = suffix
        self.label_scale = label_scale
        self.extract_aff_lab_func = GetAffinityLabelFromIndices(path_index.src_indices, path_index.dst_indices)

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        ibase = os.path.basename(img_path)[:-4]
        img = tif.imread(img_path) #
        h, w = img.shape[0], img.shape[1]

        mask = cv2.imread(os.path.join(self.camroot, ibase + self.suffix), cv2.IMREAD_UNCHANGED)

        # Augmentation
        if self.aug:
            transformed = image_transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]
        # img
        img = torch.from_numpy(img).permute(2, 0, 1).float() # H W C ==> C H W
        img = img/255.0
        # mask
        # reduced_label = pil_rescale(mask, scale=self.label_scale, order=0) # nearest
        target_size = (int(np.round(h * self.label_scale)), int(np.round(w * self.label_scale)))
        reduced_label = cv2.resize(mask, dsize=target_size, interpolation=cv2.INTER_NEAREST)

        out = {}
        out['img'] = img
        out['aff_bg_pos_label'], out['aff_fg_pos_label'], out['aff_neg_label'] = \
            self.extract_aff_lab_func(reduced_label)
        # add 2022.10.26
        out['labels'] = torch.from_numpy(reduced_label)
        out['labelcls'] = torch.tensor([1, 1]).float().unsqueeze(1).unsqueeze(1)
        out['box'] = torch.tensor([0, target_size[0], 0, target_size[1]])
        return out

    def __len__(self):
        return len(self.datalist)


class GetAffinityLabelFromIndices():

    def __init__(self, indices_from, indices_to):

        self.indices_from = indices_from
        self.indices_to = indices_to

    def __call__(self, segm_map):

        segm_map_flat = np.reshape(segm_map, -1)

        segm_label_from = np.expand_dims(segm_map_flat[self.indices_from], axis=0)
        segm_label_to = segm_map_flat[self.indices_to]

        valid_label = np.logical_and(np.less(segm_label_from, 21), np.less(segm_label_to, 21))

        equal_label = np.equal(segm_label_from, segm_label_to)

        pos_affinity_label = np.logical_and(equal_label, valid_label)

        bg_pos_affinity_label = np.logical_and(pos_affinity_label, np.equal(segm_label_from, 0)).astype(np.float32)
        fg_pos_affinity_label = np.logical_and(pos_affinity_label, np.greater(segm_label_from, 0)).astype(np.float32)

        neg_affinity_label = np.logical_and(np.logical_not(equal_label), valid_label).astype(np.float32)

        return torch.from_numpy(bg_pos_affinity_label), torch.from_numpy(fg_pos_affinity_label), \
               torch.from_numpy(neg_affinity_label)


# random walk
class myImageFloderclsRW(data.Dataset):
    def __init__(self, datalist, channels=3, aug=False, num_sample = 0):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        self.channels = channels

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        img = tif.imread(img_path)[:, :, :self.channels]
        lab = self.datalist.iloc[index, 1]
        # Augmentation
        if self.aug:
            img = image_transform(image=img)["image"]

        img = img/255.0
        img = np.transpose(img, (2, 0, 1)) # HWC to CHW
        img = np.stack([img, np.flip(img, -1)], axis=0) # 2 C H W

        img = torch.from_numpy(img).float()  #
        # lab = torch.tensor(lab).long()# [0, C-1], 0,1,2 index
        lab = torch.tensor([0, 1]).float().unsqueeze(1).unsqueeze(1)

        out = {"name": img_path, "img": img, "size": (img.shape[2], img.shape[3]),
               "labelcls": lab}
        return out

    def __len__(self):
        return len(self.datalist)


# 2022.10.8, for batch processing
class myImageFloderclsRW_batch(data.Dataset):
    def __init__(self, camroot, datalist, channels=3, aug=False, num_sample = 0, suffix='.jpg'):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        self.channels = channels
        self.camrroot = camroot
        self.suffix = suffix

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        img_name = os.path.basename(img_path)[:-4]

        img = tif.imread(img_path)[:, :, :self.channels]
        # lab = self.datalist.iloc[index, 1]

        img = img/255.0
        img = np.transpose(img, (2, 0, 1)) # HWC to CHW
        # img = np.stack([img, np.flip(img, -1)], axis=0) # 2 C H W

        img = torch.from_numpy(img).float()  #
        # lab = torch.tensor(lab).long()# [0, C-1], 0,1,2 index

        # add cam
        cam_path = os.path.join(self.camrroot, img_name+self.suffix)
        cam = cv2.imread(cam_path, cv2.IMREAD_UNCHANGED)/255.0
        cam = torch.from_numpy(cam).float().squeeze()

        return img_path, img, cam

    def __len__(self):
        return len(self.datalist)


# 2022.10.8, for batch processing
class myImageFloderclsRW_tlc_batch(data.Dataset):
    def __init__(self, camroot, datalist, channels=3, aug=False, num_sample = 0, suffix='.jpg'):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        # self.channels = channels
        self.camrroot = camroot
        self.suffix = suffix

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        img = tif.imread(img_path)
        ######### tlc
        img_name = os.path.basename(img_path)
        img_dir = os.path.dirname(os.path.dirname(img_path))
        tlc_path = os.path.join(img_dir, 'tlc', 'tlc'+img_name[3:])
        tlc = tif.imread(tlc_path)
        ########## concat
        img = np.concatenate((img, tlc), axis=2) # N H (C1+C2)

        img = img/255.0
        img = np.transpose(img, (2, 0, 1)) # HWC to CHW
        # img = np.stack([img, np.flip(img, -1)], axis=0) # 2 C H W

        img = torch.from_numpy(img).float()  #
        # lab = torch.tensor(lab).long()# [0, C-1], 0,1,2 index

        ########### add cam
        cam_path = os.path.join(self.camrroot, img_name[:-4]+self.suffix)
        cam = cv2.imread(cam_path, cv2.IMREAD_UNCHANGED)/255.0
        cam = torch.from_numpy(cam).float().squeeze()

        return img_path, img, cam

    def __len__(self):
        return len(self.datalist)


# 2022.11.30: add tlc bands
class myImageFloderclsRW_tlc(data.Dataset):
    def __init__(self, datalist, channels=7, aug=False, num_sample = 0):
        self.datalist = pd.read_csv(datalist, sep=',', header=None)
        if num_sample>0: # sample
            self.datalist = self.datalist[:num_sample]
        self.aug = aug # augmentation for images
        #self.channels = channels

    def __getitem__(self, index):
        img_path = self.datalist.iloc[index, 0]
        img = tif.imread(img_path)
        # tlc
        img_name = os.path.basename(img_path)
        img_dir = os.path.dirname(os.path.dirname(img_path))
        tlc_path = os.path.join(img_dir, 'tlc', 'tlc'+img_name[3:])
        tlc = tif.imread(tlc_path)
        # concat
        img = np.concatenate((img, tlc), axis=2) # N H (C1+C2)

        # Augmentation
        if self.aug:
            img = image_transform(image=img)["image"]

        img = img/255.0
        img = np.transpose(img, (2, 0, 1)) # HWC to CHW
        img = np.stack([img, np.flip(img, -1)], axis=0) # 2 C H W

        img = torch.from_numpy(img).float()  #
        # lab = torch.tensor(lab).long()# [0, C-1], 0,1,2 index
        lab = torch.tensor([0, 1]).float().unsqueeze(1).unsqueeze(1)

        out = {"name": img_path, "img": img, "size": (img.shape[2], img.shape[3]),
               "labelcls": lab}
        return out

    def __len__(self):
        return len(self.datalist)
