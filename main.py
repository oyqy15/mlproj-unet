import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import optim
from tqdm import tqdm as tq

from dataset import Readdata, Cloudset
from model import Unet, DiceLoss, BceDiceLoss
from utils import *


if __name__ == '__main__':
    # config
    path = '/Users/ouyangqianyu/tsinghua/phd1/ML/proj/data'
    model_id = 'test.pt'
    batch_size = 8
    num_workers = 0
    is_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if is_gpu else 'cpu')
    # config
    r = Readdata(path)
    train_set = Cloudset(
        r.train,
        'train',
        r.train_ids,
        r.train_fold,
        training_augmentation()
        # try different augmentation here
    )
    valid_set = Cloudset(
        r.train,
        'valid',
        r.valid_ids,
        r.train_fold,
        validation_augmentation()
        # try different augmentation here
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
    )
    print('training data loaded')
    net = Unet(c_in=3, c_out=4).float()
    if is_gpu:
        net.cuda()
    print('unet built')
    print(net)
    # training 
    initial_lr = 0.03
    criterion = BceDiceLoss(eps=1e-1) # make sure tp=1 at least
    optimizer = optim.Adam(net.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    max_epochs = 32
    valid_loss_min = np.Inf
    # for plot
    train_loss_list = []
    valid_loss_list = []
    lr_list = []
    # start 
    for epoch in range(1, max_epochs + 1):
        train_loss = 0.0
        valid_loss = 0.0
        # train
        net.train()
        bar = tq(train_loader, postfix={'train_loss': np.Inf})
        for img, masks in bar:
            if is_gpu:
                img, masks = img.cuda(), masks.cuda()
            # debug
            print('load data', img.shape, masks.shape)
            masks_pr = net(img)
            # debug
            print('unet calc', masks_pr.shape)
            loss = criterion(masks_pr, masks)
            # debug
            print('loss calc', loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * img.shape[0]
            bar.set_postfix(ordered_dict={'train_loss': loss.item()})
        # validate
        net.eval()
        with torch.no_grad():
            v_bar = tq(valid_loader, postfix={'valid_loss': np.Inf})
            for img, masks in v_bar:
                if is_gpu:
                    img, masks = img.cuda(), masks.cuda()
                masks_pr = net(img)
                loss = criterion(masks_pr, masks)
                valid_loss += loss.item() * img.shape[0]
                v_bar.set_postfix(ordered_dict={'valid_loss': loss.item()})
        train_loss = train_loss / train_loader.dataset.__len__()
        valid_loss = valid_loss / valid_loader.dataset.__len__()
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        lr_list.append([param_group['lr'] for param_group in optimizer.param_groups][0])
        print('epoch: {}, train_loss: {:.6f}, valid_loss: {:.6f}, lr: {:.6f}'.format(epoch, train_loss, valid_loss, lr_list[-1]))
        if valid_loss < valid_loss_min:
            print('model update, saving...')
            torch.save(net.state_dict(), model_id)
            valid_loss_min = valid_loss
        scheduler.step(train_loss)
    # train and validate over

