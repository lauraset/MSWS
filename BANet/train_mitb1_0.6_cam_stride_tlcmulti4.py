'''
2021.09.06
@Yinxia Cao
@function: used for tuitiantu detection
2022.10.12: add tlc images
2022.11.2: consider multi-level cams, consider 4 layers
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import random
import numpy as np
from tqdm import tqdm

from torch.utils import data
from tensorboardX import SummaryWriter #change tensorboardX
from zy3ba_loader import myImageFloder_muxtlc
from core.model import Mitcls_CAM_multi4
from metrics import ClassificationMetric, AverageMeter
import shutil
import argparse
from utils.optimizer import PolyWarmupAdamW
from omegaconf import OmegaConf
import torch.nn.functional as F

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",default='configs/zy3ba2.yaml',type=str,help="config")
    parser.add_argument("--backbone",default='mit_b1', type=str,help="config")
    parser.add_argument("--logname",default='mit_b1cam_stride_tlcmulti4',type=str,help="config")
    parser.add_argument("--nchannels",default=7, type=int,help="config")
    parser.add_argument("--max_iters",default=5, type=int, help="epochs")
    parser.add_argument("--eval_iters",default=5, type=int, help="eval")
    parser.add_argument("--warmup_iter", default=0, type=int, help="eval")
    parser.add_argument("--batch_size", default=32, type=int, help="eval")
    parser.add_argument("--num_workers", default=4, type=int, help="eval")
    parser.add_argument("--logdir",default='E:/yinxcao/weaksup/BANetdata2',type=str) # store

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    cfg.exp.backbone = args.backbone
    cfg.exp.logname = args.logname
    cfg.dataset.nchannels = args.nchannels
    cfg.train.max_iters = args.max_iters
    cfg.train.eval_iters = args.eval_iters
    cfg.scheduler.warmup_iter = args.warmup_iter
    cfg.dataset.batch_size = args.batch_size
    cfg.dataset.num_workers = args.num_workers
    cfg.dataset.logdir = args.logdir

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

    batch_size = cfg.dataset.batch_size
    epochs = cfg.train.max_iters
    epoch_eval = cfg.train.eval_iters
    logdir = os.path.join(cfg.dataset.logdir, cfg.exp.logname)
    writer = SummaryWriter(log_dir=logdir)
    classes = cfg.dataset.num_classes  # 0, 1, 2, 3, 4, 5, 6
    num_workers = cfg.dataset.num_workers
    nchannels = cfg.dataset.nchannels
    global best_acc
    best_acc = 0

    # train & test dataloader
    traindataloader = torch.utils.data.DataLoader(
        myImageFloder_muxtlc(trainlist, aug=True, channels=nchannels),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    iter_per_epoch = len(traindataloader)
    cfg.scheduler.warmup_iter *= iter_per_epoch
    cfg.train.max_iters *= iter_per_epoch

    testdataloader = torch.utils.data.DataLoader(
        myImageFloder_muxtlc(testlist, aug=False, channels=nchannels),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    # use max pooling
    net = Mitcls_CAM_multi4(cfg.exp.backbone, num_classes=classes, pretrained=True,
                     in_chans=nchannels, pooling='max', stride=(4, 2, 2, 1)).to(device)
    param_groups = net.get_param_groups()

    optimizer = PolyWarmupAdamW(
        params=[
            {
                "params": param_groups[0],
                "lr": cfg.optimizer.learning_rate,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": param_groups[1],
                "lr": cfg.optimizer.learning_rate,
                "weight_decay": 0.0,
            },
            {
                "params": param_groups[2],
                "lr": cfg.optimizer.learning_rate * 10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
        ],
        lr=cfg.optimizer.learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
        betas=cfg.optimizer.betas,
        warmup_iter=cfg.scheduler.warmup_iter,  # add by cyx
        max_iter=cfg.train.max_iters,
        warmup_ratio=cfg.scheduler.warmup_ratio,
        power=cfg.scheduler.power
    )
    weights = torch.tensor([0.5, 0.5]).to(device) # 1 1 10
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    criterion_cam = torch.nn.L1Loss()

    # print the model
    start_epoch = 0
    resume = os.path.join(logdir, 'checkpoint.tar')
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        net.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
                 .format(resume, checkpoint['epoch']))
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer.global_step = start_epoch*iter_per_epoch
    else:
        print("=> no checkpoint found at resume")
        print("=> Will start from scratch.")
        # return

    # should be placed after weight loading
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

    for epoch in range(epochs-start_epoch):
        epoch = start_epoch + epoch + 1 # current epochs
        lr = optimizer.param_groups[0]['lr']
        print('epoch [%d|%d], lr: %.6f'%(epoch, epochs, lr))
        train_loss, train_oa, train_f1 = train_epoch(net, criterion, criterion_cam, traindataloader,
                                                     optimizer, device, epoch, classes,
                                                     epochs, logdir)
        # save every epoch
        savefilename = os.path.join(logdir, 'checkpoint.tar')
        torch.save({
            'epoch': epoch,
            'state_dict': net.module.state_dict() if hasattr(net, "module") else net.state_dict(),  # multiple GPUs
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, savefilename)
        # write
        writer.add_scalar('lr', lr, epoch)
        writer.add_scalar('train/1.loss', train_loss, epoch)
        writer.add_scalar('train/2.oa', train_oa, epoch)
        writer.add_scalar('train/3.f1_pos', train_f1[0], epoch)
        writer.add_scalar('train/4.f1_neg', train_f1[1], epoch)
    writer.close()


def train_epoch(net, criterion, criterion_cam, dataloader, optimizer, device, epoch, classes,
                max_epoch, logdir):
    net.train()
    acc_total = ClassificationMetric(numClass=classes, device=device)
    losses = AverageMeter()
    num = len(dataloader)
    pbar = tqdm(range(num), disable=False)

    for idx, (x, y_true) in enumerate(dataloader):
        # combine pos and neg
        x = x.to(device, non_blocking=True) # N C H W
        y_true = y_true.to(device, non_blocking=True) # N H W

        y1, y2, y3, y4 = net(x)

        loss = criterion(y1, y_true) + criterion(y2, y_true) + \
               criterion(y3, y_true) + criterion(y4, y_true)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ypred = y4.argmax(1)
        acc_total.addBatch(ypred, y_true)

        losses.update(loss.item(), x.size(0))
        oa = acc_total.OverallAccuracy()
        f1 = acc_total.F1score()
        pbar.set_description('Train Epoch:{epoch:4}. Iter:{batch:4}|{iter:4}. Loss {loss:.3f}. '
                             'OA {oa:.3f}, F1: {pos:.3f}, {neg:.3f}'.format(
                             epoch=epoch, batch=idx, iter=num, loss=losses.avg, oa = oa, pos = f1[0], neg=f1[1]))
        pbar.update()

        # save model
        if idx%400==0:
            savefilename = os.path.join(logdir, 'checkpointtmp.tar')
            torch.save({
                'epoch': epoch,
                'iter': idx,
                'state_dict': net.module.state_dict() if hasattr(net, "module") else net.state_dict(),  # multiple GPUs
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, savefilename)

    pbar.close()
    oa = acc_total.OverallAccuracy()
    f1 = acc_total.F1score()
    return losses.avg, oa, f1


def vtest_epoch(model, criterion, dataloader, device, epoch, classes):
    model.eval()
    acc_total = ClassificationMetric(numClass=classes, device=device)
    losses = AverageMeter()
    num = len(dataloader)
    pbar = tqdm(range(num), disable=False)
    with torch.no_grad():
        for idx, (x, y_true) in enumerate(dataloader):
            x = x.to(device, non_blocking =True)
            y_true = y_true.to(device, non_blocking =True)
            ypred = model.forward(x)
            loss = criterion(ypred, y_true)

            ypred = ypred.argmax(axis=1)
            acc_total.addBatch(ypred, y_true)

            losses.update(loss.item(), x.size(0))
            oa = acc_total.OverallAccuracy()
            f1 = acc_total.F1score()
            pbar.set_description(
                'Test Epoch:{epoch:4}. Iter:{batch:4}|{iter:4}. Loss {loss:.3f}. OA {oa:.3f}, F1: {pos:.3f}, {neg:.3f}'.format(
                    epoch=epoch, batch=idx, iter=num, loss=losses.avg, oa=oa, pos=f1[0], neg=f1[1]))
            pbar.update()
        pbar.close()
    oa = acc_total.OverallAccuracy()
    f1 = acc_total.F1score()
    return losses.avg, oa, f1


if __name__ == "__main__":
    cfg = get_args()
    main(cfg)