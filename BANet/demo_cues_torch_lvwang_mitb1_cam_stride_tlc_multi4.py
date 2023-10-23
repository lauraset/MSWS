## 2021.10.15
# generate cues based on torch, method: SEC and DSRG
# 1. check probability of cls; 2. generate grad-cam; 3. threhold
# rerun 2023.6.7
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from tqdm import tqdm
import numpy as np
from core.model import Mitcls_CAM_multi4
import tifffile as tif
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
from utils.camutils import multi_scale_cam_multistagev2
from zy3ba_loader import myImageFloder_muxtlc_path

def preprocess(img_path, imgsize=None):
    # img = Image.open(img_path).convert('RGB')
    # img = np.array(img)
    img = tif.imread(img_path)
    img_norm = np.float32(img)/255.0
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).float()  # H W C ==> C H W
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor, img_norm[:,:,:3]

def reshape_transform(tensor):
    b, l, c = tensor.shape
    h = int(np.sqrt(l))
    result = tensor.reshape(b, h, h, c)

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

if __name__=="__main__":

    saveroot = 'E:/yinxcao/weaksup/BANetdata2'

    nchannels = 7 # mux+tlc
    classes = 2
    device = 'cuda'
    trainlist_pos = 'E:/yinxcao/ZY3LC/datanew8bit/datalist_posneg_train_0.6_pos.csv'

    num = 0 #100
    testdataloader = torch.utils.data.DataLoader(
        myImageFloder_muxtlc_path(trainlist_pos, aug=False, channels=nchannels, num_sample=num),
        batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    # use balance model
    backbone = "mit_b1"
    net = Mitcls_CAM_multi4(backbone, num_classes=classes, pretrained=True,
                     in_chans=nchannels, pooling='max', stride=(4, 2, 2, 1)).to(device)
    backbone = backbone+'cam_stride_tlcmulti4'
    weightpath = os.path.join(saveroot, backbone, 'checkpoint.tar')
    weight = torch.load(weightpath)
    print('loading epoch: %d'%weight["epoch"])
    net.load_state_dict(weight["state_dict"])
    net.eval()

    train_cues_dir = os.path.join(saveroot, backbone, 'cam')
    os.makedirs(train_cues_dir, exist_ok=True)

    # scales = [1.0, 0.5, 1.5]
    scales = [1.0]
    # Process by batch
    target = [0]
    for input_tensor,_, imgpathlist in tqdm(testdataloader):
        input_tensor = input_tensor.to(device, non_blocking=True)
        # multi-scale predict
        grayscale_cam = multi_scale_cam_multistagev2(net, input_tensor, scales, weights=(1, 1, 1, 1))
        grayscale_cam = grayscale_cam.cpu().numpy()
        grayscale_cam = grayscale_cam[:, target].squeeze(axis=1)
        # RGB img
        rgb_img = input_tensor[:, :3].detach().cpu().numpy()
        rgb_img = rgb_img.transpose((0, 2, 3, 1)) # N H W C
        # cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True) # cam =0.3
        # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        # cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
        for i, imgpath in enumerate(imgpathlist):
            iname = os.path.basename(imgpath)[:-4]
            icue = os.path.join(train_cues_dir, iname)
            cam = np.uint8(grayscale_cam[i] * 255)
            cam_image = show_cam_on_image(rgb_img[i], grayscale_cam[i], use_rgb=True)
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(icue+'.jpg', cam)
            cv2.imwrite(icue+'_rgb.jpg', cam_image)

        # tif.imwrite(icue + '.tif', grayscale_cam)  #
        # shutil.copy(imgpath, icue + '.png')