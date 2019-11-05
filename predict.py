import pandas as pd
import numpy as np
import torch
import os
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm as tq

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
    thresholds = [args.threshold0, args.threshold1, args.threshold2, args.threshold3]
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
    # prepare
    ans_dict = global_dict()
    image_uid = 0
    answer_label = []
    answer_index = ['Image_Label', 'EncodedPixels']
    # calculation
    t_bar = tq(test_loader)
    net.eval()
    with torch.no_grad():
        for img, masks in t_bar:
            if is_gpu:
                img = img.cuda()
            masks_pr = net(img).cpu().detach().numpy()
            for batch in masks_pr:
                for i, mask in enumerate(batch):
                    # image_uid, i
                    answer_label.append(get_answer(mask, thresholds[i]))
                image_uid += 1
    # submission
    submit_file = os.path.join('checkpoint', model_id + '.csv')
    sub = test_set.df
    sub['EncodedPixels'] = answer_label
    sub.to_csv(submit_file, columns=answer_index, index=False)

