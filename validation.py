import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse
from model import Unet
from dataset import Cloudset, Readdata
from tqdm import tqdm as tq
from utils import *

def get_args():
    parser = argparse.ArgumentParser(
        description='generate validation results using model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('-mid', '--model-id', metavar='MI', type=str, default='test.pt', help='Load model from a .pt file', dest='model_id')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    path = '../data'
    model_id = os.path.join('checkpoint', args.model_id)
    batch_size = 8
    is_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if is_gpu else 'cpu')
    print('args:' + str(args))
    print('isgpu?:' + str(is_gpu))

    result_fold = os.path.join('checkpoint', 'valid', args.model_id)
    if not os.path.exists(result_fold):
        os.makedirs(result_fold)
    
    net = Unet(3, 4).float()
    if is_gpu:
        net = net.cuda()
    r = Readdata(path)
    valid_set = Cloudset(
        r.train,
        'valid',
        r.valid_ids,
        r.train_fold,
        validation_augmentation_kaggle()
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False
    )
    print('validation data loaded')
    net.load_state_dict(torch.load(model_id, map_location=device))
    print('model %s loaded' % model_id)
    
    v_bar = tq(valid_loader)
    net.eval()
    res = []
    with torch.no_grad():
        for img, masks in v_bar:
            if is_gpu:
                img = img.cuda()
            masks_pr = net(img).cpu().detach().numpy()
            for batch in  masks_pr:
                res.append(batch)
             
    res = np.asarray(res, dtype=np.float32)
    print(res.shape)
    res2 = valid_set.ids
    print(len(res2))

    np.save(os.path.join(result_fold, 'masks'), res)
    with open(os.path.join(result_fold, 'ids'), 'w') as fout:
        fout.write('\n'.join(res2))  

    


        