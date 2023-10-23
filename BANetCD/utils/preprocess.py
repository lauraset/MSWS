import os
import tifffile as tif
import numpy as np
import torch
import cv2
import torch.nn.functional as F

# add tlc, and lab
# add scale
def preprocess_tlclab(testlist, idx, scale=1.0):
    img_path = testlist.iloc[idx, 0]
    labpath = testlist.iloc[idx, 1]

    img = tif.imread(img_path)
    # tlc
    img_name = os.path.basename(img_path)
    img_dir = os.path.dirname(os.path.dirname(img_path))
    tlc_path = os.path.join(img_dir, 'tlc', 'tlc' + img_name[3:])
    tlc = tif.imread(tlc_path)
    # concat
    img = np.concatenate((img, tlc), axis=2)  # N H (C1+C2)

    img_norm = np.float32(img)/255.0
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).float()  # H W C ==> C H W
    img_tensor = img_tensor.unsqueeze(0) # N C H W

    # lab
    lab_tensor = cv2.imread(labpath, cv2.IMREAD_UNCHANGED)
    lab_tensor = torch.from_numpy(lab_tensor).unsqueeze(0) # N H W
    #scale
    if scale!=1:
        h,w = img_tensor.shape[2:]
        img_tensor = F.interpolate(img_tensor, size=(int(h*scale), int(w*scale)), mode='bilinear', align_corners=True)
        lab_tensor = F.interpolate(lab_tensor.unsqueeze(0), size=(int(h * scale), int(w * scale)), mode='nearest')
        lab_tensor = lab_tensor[0]
    return img_tensor, lab_tensor, img_norm[:,:,:3]


def preprocess_t1t2(testlist, idx, scale=1.0):
    img_path = testlist.iloc[idx, 0]

    ibase = os.path.basename(img_path)[:-4]
    idir = os.path.dirname(os.path.dirname(img_path))

    img1 = tif.imread(img_path)
    img2 = tif.imread(os.path.join(idir, 'img2', ibase + '.tif'))
    # tlc: h w 3
    tlc1 = tif.imread(os.path.join(idir, 'tlc1', ibase + '.tif'))
    tlc2 = tif.imread(os.path.join(idir, 'tlc2', ibase + '.tif'))
    # concate
    img = np.concatenate([img1, tlc1, img2, tlc2], axis=2)
    img_norm = np.float32(img)/255.0
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).float()  # H W C ==> C H W
    img_tensor = img_tensor.unsqueeze(0) # N C H W

    # mask
    labpath = os.path.join(idir, 'lab', 'lab' + ibase[3:] + '.png')
    lab_tensor = cv2.imread(labpath, cv2.IMREAD_UNCHANGED)
    lab_tensor = torch.from_numpy(lab_tensor).unsqueeze(0) # N H W
    #scale
    if scale!=1:
        h,w = img_tensor.shape[2:]
        img_tensor = F.interpolate(img_tensor, size=(int(h*scale), int(w*scale)), mode='bilinear', align_corners=True)
        lab_tensor = F.interpolate(lab_tensor.unsqueeze(0), size=(int(h * scale), int(w * scale)), mode='nearest')
        lab_tensor = lab_tensor[0]
    return img_tensor, lab_tensor, img_norm[:,:,:3]
