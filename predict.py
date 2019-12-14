import pandas as pd
import numpy as np
import torch
import os
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm as tq

from dataset import Cloudset, Readdata
from model import Res2Unet, Unet
from utils import *


def get_args():
    parser = argparse.ArgumentParser(
        description='predict masks using model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('-mid', '--model-id', metavar='MI', type=str, default='test.pt', help='Load model from a .pt file', dest='model_id')
    parser.add_argument('-res', '--resnet', type=bool, default=False, help='use resnet or not', dest='resnet')
    return parser.parse_args()

if __name__ == '__main__':
    # config
    args = get_args()
    path = '../data'
    model_id = os.path.join('checkpoint', args.model_id)
    batch_size = 8
    is_gpu = torch.cuda.is_available()
    is_resnet = args.resnet
    device = torch.device('cuda' if is_gpu else 'cpu')
    # config
    print('args:' + str(args))
    print('isgpu?:' + str(is_gpu))
    # print config
    r = Readdata(path)
    test_set = Cloudset(
        r.sub,
        'test',
        r.test_ids,
        r.test_fold,
        validation_augmentation_kaggle()
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False
    )
    print('testing data loaded')
    if not is_resnet:
        print('vanila unet')
        net = Unet(c_in=3, c_out=4).float()
    else:
        print('resnet 2 unet')
        net = Res2Unet(c_in=3, c_out=4).float()
    if is_gpu:
        net.cuda()
    net.load_state_dict(torch.load(model_id, map_location=device))
    print('model loaded')
    # calculation
    t_bar = tq(test_loader)
    net.eval()
    res = []
    with torch.no_grad():
        for img, masks in t_bar:
            if is_gpu:
                img = img.cuda()
            masks_pr = net(img).cpu().detach().numpy() #[batch, 4, 320, 640]
            for batch in masks_pr:
                res.append([resize_f(mask) for mask in batch])
            # break
            # debug
        
    result_fold = create_valid_dir(args.model_id, 'test')
    np.save(os.path.join(result_fold, 'masks'), res)

