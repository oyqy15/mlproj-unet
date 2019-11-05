import pandas as pd
import numpy as np
import torch
import os
import argparse
from torch.utils.data import DataLoader

from dataset import Cloudset, Readdata
from model import Unet
from utils import *


def get_args():
    parser = argparse.ArgumentParser(
        description='predict masks using model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('-mid', '--model-id', metavar='MI', type=str, default='test.pt', help='Load model from a .pt file', dest='model_id')
    parser.add_argument('-t0', '--threshold0', metavar='T', type=float, default=0.5, help='the cell > threshold will be masked (Fish)', dest='threshold0')
    parser.add_argument('-t1', '--threshold1', metavar='T', type=float, default=0.5, help='the cell > threshold will be masked (Flower)', dest='threshold1')
    parser.add_argument('-t2', '--threshold2', metavar='T', type=float, default=0.5, help='the cell > threshold will be masked (Gravel)', dest='threshold2')
    parser.add_argument('-t3', '--threshold3', metavar='T', type=float, default=0.5, help='the cell > threshold will be masked (Sugar)', dest='threshold3')
    return parser.parse_args()

if __name__ == '__main__':
    # config
    args = get_args()
    path = '../data'
    model_id = os.path.join('checkpoint', args.model_id)
    batch_size = 8
    is_gpu = torch.cuda.is_available()
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
        validation_augmentation()
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False
    )
    print('testing data loaded')
    net = Unet(3, 4).float()
    if is_gpu:
        net.cuda()
    net.load_state_dict(torch.load(model_id, map_location=device))
    print('model loaded')
    
    

