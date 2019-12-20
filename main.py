import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import optim
from tqdm import tqdm as tq

from dataset import Readdata, Cloudset
from model import Res2Unet, Unet, DiceLoss, BceDiceLoss, RAdam, GDiceLoss
from utils import *


def get_args():
    parser = argparse.ArgumentParser(
        description='Train the UNet on images and target masks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('-mid', '--model-id', metavar='MI', type=str, default='test.pt', help='Load model from a .pt file', dest='model_id')
    parser.add_argument('-e', '--max-epochs', metavar='E', type=int, default=32, help='Number of epochs', dest='max_epochs')
    parser.add_argument('-lr', '--learning-rate', metavar='LR', type=float, default=0.005, help='Learning rate', dest='lr')
    parser.add_argument('-bs', '--batch-size', metavar='B', type=int, default=8, help='Batch size', dest='batch_size')
    parser.add_argument('-res', '--resnet', type=bool, default=False, help='use resnet or not', dest='resnet')
    parser.add_argument('-lo', '--loss', metavar='L', type=str, default='bce', help='type of loss', dest='loss')
    return parser.parse_args()

if __name__ == '__main__':
    # config
    args = get_args()
    path = '../data'
    model_id = os.path.join('checkpoint', args.model_id)
    initial_lr = args.lr
    max_epochs = args.max_epochs
    batch_size = args.batch_size
    is_gpu = torch.cuda.is_available()
    is_resnet = args.resnet
    loss_kind = args.loss
    # config
    print('args:' + str(args))
    print('isgpu?:' + str(is_gpu))
    # print config
    r = Readdata(path)
    train_set = Cloudset(
        r.train,
        'train',
        r.train_ids,
        r.train_fold,
        training_augmentation_kaggle()
        # training_augmentation()
        # try different augmentation here
    )
    valid_set = Cloudset(
        r.train,
        'valid',
        r.valid_ids,
        r.train_fold,
        validation_augmentation_kaggle()
        # validation_augmentation()
        # try different augmentation here
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False
    )
    print('training data loaded')
    if not is_resnet:
        print('vanila unet')
        net = Unet(c_in=3, c_out=4).float()
    else:
        print('resnet 2 unet')
        net = Res2Unet(c_in=3, c_out=4).float()
    if is_gpu:
        # net.cuda()
        net = nn.DataParallel(net)
        net.cuda()
    print('unet built')
    # training 
    criterion = None
    if loss_kind == 'bce':
        print('use loss bce')
        criterion = BceDiceLoss(eps=1e-1) # make sure tp=eps at least
    else:
        print('use loss gdl')
        criterion = GDiceLoss()
    optimizer = RAdam(net.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=2, cooldown=2)
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
            optimizer.zero_grad()
            masks_pr = net(img)
            loss = criterion(masks_pr, masks)
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
        # record & update
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
        scheduler.step(valid_loss)
    # train and validate over
    # record 
    with open(model_id + '.rec', 'w') as fout:
        fout.write('trainloss:\n')
        fout.write(' '.join([str(x) for x in train_loss_list]) + '\n')
        fout.write('validloss:\n')
        fout.write(' '.join([str(x) for x in valid_loss_list]) + '\n')
        fout.write('lr:\n')
        fout.write(' '.join([str(x) for x in lr_list]) + '\n')

    result_fold = create_valid_dir(args.model_id)
    np.save(os.path.join(result_fold, 'ids'), valid_set.ids, allow_pickle=True)
