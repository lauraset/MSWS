'''
2021.09.06
@Yinxia Cao
@function: used for tuitiantu detection
model: segformer, mit_b1
2022.11.03: cam from multi
2023.06.16: change loss function to "ce_dice"
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import random
import numpy as np
from tqdm import tqdm

from torch.utils import data
from tensorboardX import SummaryWriter #change tensorboardX
from zy3bacd_loader import myImageFlodert1t2
from core.model import WeTrCD
from metrics import SegmentationMetric, AverageMeter

import argparse
from utils.optimizer import PolyWarmupAdamW
from omegaconf import OmegaConf
from collections import OrderedDict
from losses_pytorch.myloss import CE_DICE, CE_DICE_IOU

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",default='configs/zy3bacd.yaml',type=str, help="config")
    parser.add_argument("--nchannels",default=7, type=int,help="config")
    # beijing: obj+pix
    # parser.add_argument("--datapath", default= 'E:/yinxcao/weaksup/BAdata', type=str, help="eval")
    # parser.add_argument("--pseudodir",default='certobjpix',type=str,help="config")
    # parser.add_argument("--logname", default='objpix_bj_nopre', type=str, help="config")
    # obj
    # parser.add_argument("--datapath", default= 'E:/yinxcao/weaksup/BAdata', type=str, help="eval")
    # parser.add_argument("--pseudodir",default='certobj',type=str,help="config")
    # parser.add_argument("--logname", default='obj_bj_nopre', type=str, help="config")
    # pix
    # parser.add_argument("--datapath", default= 'E:/yinxcao/weaksup/BAdata', type=str, help="eval")
    # parser.add_argument("--pseudodir",default='certpix',type=str,help="config")
    # parser.add_argument("--logname", default='pix_bj_nopre', type=str, help="config")
    # shanghai
    parser.add_argument("--datapath", default= 'E:/yinxcao/weaksup/BAdata_sh', type=str, help="eval")
    parser.add_argument("--pseudodir",default='certobjpix',type=str,help="config")
    parser.add_argument("--logname", default='objpix_sh_cdi_nopre', type=str, help="config")

    # old version: pretrain_path = r'E:\yinxcao\weaksup\BANetdata\rrm_tlcmulti4_adele\checkpoint1800.tar'
    # parser.add_argument("--pretrain_path", default=r'E:\yinxcao\weaksup\BANetdata2\rrm_tlcmulti4_adele\checkpoint1400.tar')
    parser.add_argument("--pretrain_path", default=None)
    parser.add_argument("--loss", default="ce_dice",type=str, help="ce, ce_dice, ce_dice_iou")
    parser.add_argument("--max_iters", default=2,type=int,help="max_iters")
    parser.add_argument("--eval_iters", default=1,type=int,help="eval_iters")
    parser.add_argument("--warmup_iter", default=0, type=int, help="eval") # warmup
    parser.add_argument("--batch_size", default=8, type=int, help="eval")
    parser.add_argument("--num_workers", default=4, type=int, help="eval")
    parser.add_argument("--dcrfloss_weight", default=1e-7, type=int, help="eval")
    parser.add_argument("--num_classes", default=3, type=int, help="eval") # 0, 1:neg, 2: pos, 3:uncert

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    cfg.dataset.nchannels = args.nchannels
    cfg.dataset.data_path = args.datapath
    cfg.dataset.pseudodir = args.pseudodir
    cfg.backbone.logname = args.logname
    cfg.pretrain_path = args.pretrain_path

    cfg.train.max_iters = args.max_iters
    cfg.train.eval_iters = args.eval_iters
    cfg.scheduler.warmup_iter = args.warmup_iter
    cfg.dataset.batch_size = args.batch_size
    cfg.dataset.num_workers = args.num_workers
    cfg.dataset.dcrfloss_weight = args.dcrfloss_weight
    cfg.dataset.num_classes = args.num_classes
    cfg.loss = args.loss

    return cfg


def main(cfg):
    # Setup seeds
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    np.random.seed(1337)
    random.seed(1337)
    torch.backends.cudnn.deterministic = True
    device = 'cuda'

    # Setup dataloader
    data_path = cfg.dataset.data_path
    trainlist = os.path.join(data_path, cfg.dataset.trainlist) # training
    testlist = os.path.join(data_path, cfg.dataset.vallist) # validation
    pseudodir = os.path.join(data_path, cfg.dataset.pseudodir) # add, for storing the pseudo labels generated from CAM

    batch_size = cfg.dataset.batch_size
    epochs = cfg.train.max_iters
    epoch_eval = cfg.train.eval_iters
    logdir = os.path.join(cfg.dataset.logdir, cfg.backbone.logname)
    writer = SummaryWriter(log_dir=logdir)
    num_classes = cfg.dataset.num_classes  # 0, 1, 2, 3, 4, 5, 6
    num_workers = cfg.dataset.num_workers
    nchannels = cfg.dataset.nchannels
    global best_acc
    best_acc = 0

    # train & test dataloader
    traindataloader = torch.utils.data.DataLoader(
        myImageFlodert1t2(trainlist, pseudodir, aug=True, channels=nchannels),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    iter_per_epoch = len(traindataloader)
    cfg.scheduler.warmup_iter *= iter_per_epoch
    cfg.train.max_iters *= iter_per_epoch

    # testdataloader = torch.utils.data.DataLoader(
    #     myImageFloder(testlist, aug=False, channels=nchannels),
    #     batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # use max pooling
    net = WeTrCD(backbone=cfg.backbone.config, stride=cfg.backbone.stride,
               num_classes=num_classes, embedding_dim=256,
               pretrained=True, in_chans=nchannels).to(device)
    param_groups = net.get_param_groups()

    # load pretrained weights
    pretrain_path = cfg.pretrain_path
    if (pretrain_path is not None) and (os.path.exists(pretrain_path)):
        pretrain = torch.load(pretrain_path)['state_dict']
        state_dict = net.state_dict()
        newstate = OrderedDict()
        for k, v in pretrain.items():  # loop over pretrained parameters
            if k in state_dict.keys():
                if v.shape == state_dict[k].shape:
                    print(k)
                    newstate[k] = v
        state_dict.update(newstate)
        net.load_state_dict(state_dict)
        print('loading pretrained weights')

    optimizer = PolyWarmupAdamW(
        params=[{"params": param_groups[0],"lr": cfg.optimizer.learning_rate, "weight_decay": cfg.optimizer.weight_decay,},
            {"params": param_groups[1],"lr": cfg.optimizer.learning_rate,"weight_decay": 0.0,},
            {"params": param_groups[2],"lr": cfg.optimizer.learning_rate * 10,"weight_decay": cfg.optimizer.weight_decay,},],
        lr=cfg.optimizer.learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
        betas=cfg.optimizer.betas,
        warmup_iter=cfg.scheduler.warmup_iter,
        max_iter=cfg.train.max_iters,
        warmup_ratio=cfg.scheduler.warmup_ratio,
        power=cfg.scheduler.power
    )

    if cfg.loss == "ce_dice":
        criterion = CE_DICE(ignore_label=cfg.dataset.ignore_index)
    elif cfg.loss == "ce_dice_iou":
        criterion = CE_DICE_IOU(ignore_label=cfg.dataset.ignore_index)
    elif cfg.loss == "ce":
        criterion = torch.nn.CrossEntropyLoss(ignore_index=cfg.dataset.ignore_index)
    else:
        print('not valid loss')
        return
    print('loss is %s'%cfg.loss)

    # resume
    start_iter = 0
    resume = os.path.join(logdir, 'checkpoint2.tar')
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        net.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
                 .format(resume, checkpoint['epoch']))
        start_iter = checkpoint['n_iter']
        best_acc = checkpoint['best_acc']
        optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer.global_step = start_iter
    else:
        print("=> no checkpoint found at resume")
        print("=> Will start from scratch.")
        # return

    # should be placed after weight loading
    # if torch.cuda.device_count() > 1:
    #     net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

    ### start training
    net.train()
    acc_total = SegmentationMetric(numClass=num_classes, device=device)
    # losses = AverageMeter()
    num = cfg.train.max_iters #len(traindataloader)
    pbar = tqdm(range(num), disable=False)
    train_loader_iter = iter(traindataloader)
    ### iteration
    for n_iter in range(cfg.train.max_iters):
        try:
            img, mask = next(train_loader_iter)
        except:
            train_loader_iter = iter(traindataloader)
            img, mask = next(train_loader_iter)

        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        pred = net(img)

        loss = criterion(pred, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # losses.update(loss.item(), img.size(0))

        pred = pred.argmax(1)  # N H W
        mask = mask.to(device)
        acc_total.addBatch(pred[mask < 255], mask[mask < 255])  # valid region

        oa = acc_total.OverallAccuracy()
        miou = acc_total.meanIntersectionOverUnion()
        iou = acc_total.IntersectionOverUnion()
        # acc_total.reset(device) # reset to zeros

        pbar.set_description(
            'Train Iter:{batch:4}|{iter:4}. Loss {loss:.3f}. OA {oa:.3f}, MIOU {miou:.3f}, IOU: {lv:.3f}, {fei:.3f}'.format(
               batch=n_iter, iter=num, loss=loss.item(), oa=oa, miou=miou, lv=iou[1], fei=iou[0]))
        pbar.update()

        if (n_iter % 500 == 0) and (n_iter > 0):
            savefilename = os.path.join(logdir, 'checkpoint' + str(n_iter) + '.tar')
            torch.save({
                'iter': n_iter,
                'state_dict': net.module.state_dict() if hasattr(net, "module") else net.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, savefilename)

        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], n_iter)
        writer.add_scalar('train/1.loss', loss.item(), n_iter)
        writer.add_scalar('train/2.oa', oa, n_iter)
        writer.add_scalar('train/3.iou_pos', iou[1], n_iter)
        writer.add_scalar('train/4.iou_neg', iou[0], n_iter)

    pbar.close()
    writer.close()
    print('training end, and saving the last model')
    savefilename = os.path.join(logdir, 'checkpoint_last.tar')
    torch.save({
        'iter': cfg.train.max_iters,
        'state_dict': net.module.state_dict() if hasattr(net, "module") else net.state_dict(),
        'best_acc': best_acc,
        'optimizer': optimizer.state_dict(),
    }, savefilename)

    return True


# train rrm with CE and denseenergy loss, label from irnet
def train_epoch(net, criterion, dataloader, optimizer, device, epoch, classes,
                logdir):
    net.train()
    acc_total = SegmentationMetric(numClass=classes, device=device)
    losses = AverageMeter()
    num = len(dataloader)
    pbar = tqdm(range(num), disable=False)

    for idx, (img, mask, ori_img, croppings) in enumerate(dataloader):
        img = img.to(device, non_blocking=True)
        # update = update.to(device, non_blocking=True).unsqueeze(1) # N 1 H W

        pred = net(img)

        loss = criterion(ori_img, pred, mask, croppings)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), img.size(0))

        pred = pred.argmax(1) # N H W
        mask = mask.to(device)
        acc_total.addBatch(pred[mask<255], mask[mask<255]) # valid region

        oa = acc_total.OverallAccuracy()
        miou = acc_total.meanIntersectionOverUnion()
        iou = acc_total.IntersectionOverUnion()
        pbar.set_description(
            'Train Epoch:{epoch:4}. Iter:{batch:4}|{iter:4}. Loss {loss:.3f}. OA {oa:.3f}, MIOU {miou:.3f}, IOU: {lv:.3f}, {fei:.3f}'.format(
                epoch=epoch, batch=idx, iter=num, loss=losses.avg, oa=oa, miou=miou, lv=iou[1], fei=iou[0]))
        pbar.update()

        idx = (epoch-1)*num + idx # total iteration
        if idx % 100==0:
            savefilename = os.path.join(logdir, 'checkpoint'+str(idx)+'.tar')
            torch.save({
                'epoch': epoch,
                'iter': idx,
                'state_dict': net.module.state_dict() if hasattr(net, "module") else net.state_dict(),  # multiple GPUs
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, savefilename)

    pbar.close()
    oa = acc_total.OverallAccuracy()
    iou = acc_total.IntersectionOverUnion()
    print('epoch %d, train oa %.3f, miou: %.3f' % (epoch, oa, acc_total.meanIntersectionOverUnion()))
    return losses.avg, oa, iou


def vtest_epoch(model, criterion, dataloader, device, epoch, classes):
    model.eval()
    acc_total = SegmentationMetric(numClass=classes, device=device)
    losses = AverageMeter()
    num = len(dataloader)
    pbar = tqdm(range(num), disable=False)
    with torch.no_grad():
        for idx, (x, y_true) in enumerate(dataloader):
            x = x.to(device, non_blocking = True)
            y_true = y_true.to(device, non_blocking = True) # n c h w
            ypred = model.forward(x)

            loss = criterion(ypred, y_true)

            ypred = ypred.argmax(axis=1)
            # ypred = (torch.sigmoid(ypred)>0.5)
            acc_total.addBatch(ypred, y_true)

            losses.update(loss.item(), x.size(0))
            oa = acc_total.OverallAccuracy()
            iou = acc_total.IntersectionOverUnion()
            miou = acc_total.meanIntersectionOverUnion()
            pbar.set_description(
                'Test Epoch:{epoch:4}. Iter:{batch:4}|{iter:4}. Loss {loss:.3f}. OA {oa:.3f}, MIOU{miou:.3f}, IOU: {lv:.3f}, {fei:.3f}'.format(
                    epoch=epoch, batch=idx, iter=num, loss=losses.avg, oa=oa, miou=miou, lv=iou[1], fei=iou[0]))
            pbar.update()
        pbar.close()

    oa = acc_total.OverallAccuracy()
    iou = acc_total.IntersectionOverUnion()
    print('epoch %d, train oa %.3f, miou: %.3f' % (epoch, oa, acc_total.meanIntersectionOverUnion()))
    return losses.avg, oa, iou


if __name__ == "__main__":
    cfg = get_args()
    main(cfg)