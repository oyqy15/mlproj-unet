import os
import cv2
import numpy as np
import pandas as pd

import albumentations as albu

def create_valid_dir(model_id, kind='valid'):
    result_fold = os.path.join('checkpoint', kind, model_id)
    if not os.path.exists(result_fold):
        os.makedirs(result_fold)
    return result_fold

def global_dict():
    return {0: 'Fish', 1: 'Flower', 2: 'Gravel', 3: 'Sugar'}

def origin_augmentation():
    return albu.Compose([albu.HorizontalFlip(p=0)])

def mask_encode(img, shape=(350, 525)):
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)

def mask_decode(label, shape=(1400, 2100)):
    # return a mask
    s = label.split(' ')
    start, length= [], []
    for i, x in enumerate(s):
        if i % 2 == 0:
            start.append(int(x))
        else:
            length.append(int(x))
    start = np.array(start)
    length = np.array(length)

    start -= 1
    end = start + length

    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for i, x in enumerate(start):
        y = end[i]
        img[x: y] = 1
    return img.reshape(shape, order='F')

def make_mask(df, image_id, shape=(1400, 2100)):
    # return 0,1,2,3 mask
    encoded_masks = df.loc[df['im_id'] == image_id, 'EncodedPixels']
    masks = np.zeros((shape[0], shape[1], 4), dtype=np.float32)

    for i, label in enumerate(encoded_masks.values):
        if not(label is np.nan):
            mask = mask_decode(label)
            masks[:, :, i] = mask
    return masks

def resize_f(x):
    if x.shape != (350, 525):
        x = cv2.resize(x, (525, 350), interpolation=cv2.INTER_LINEAR)
    return x

def get_answer(x, threshold):
    x = resize_f(x)
    mask = cv2.threshold(x, threshold, 1, cv2.THRESH_BINARY)[1].astype('int')
    return mask_encode(mask)
    

