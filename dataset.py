import os
import cv2
import pandas as pd
import numpy as np
import albumentations as albu

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from utils import make_mask


class Readdata:
    def __init__(self, path, seed=42, test_rate=0.1):
        self.path = path
        self.train_fold = os.path.join(path, 'train_images')
        self.test_fold = os.path.join(path, 'test_images')
        train = pd.read_csv(os.path.join(path, 'train.csv'))
        sub = pd.read_csv(os.path.join(path, 'sample_submission.csv'))
        # 5546 train images and 4 masks: Fish, Flower, Gravel, Sugar
        # images: (1400 * 2100)
        # sub:    (350 * 525)
        train['im_id'] = train['Image_Label'].apply(lambda x: x.split('_')[0])
        sub['im_id'] = sub['Image_Label'].apply(lambda x: x.split('_')[0])
        # validation split and test
        id_mask_count = train\
            .loc[train['EncodedPixels'].isnull() == False, 'Image_Label']\
            .apply(lambda x: x.split('_')[0])\
            .value_counts()\
            .reset_index()\
            .rename(columns={'index': 'img_id', 'Image_Label': 'count'})
        train_ids, valid_ids = train_test_split(
            id_mask_count['img_id'].values,
            random_state=seed,
            stratify=id_mask_count['count'],
            test_size=test_rate)
        test_ids = sub['im_id'].drop_duplicates().values
        self.train = train
        self.sub = sub
        self.train_ids = train_ids
        self.valid_ids = valid_ids
        self.test_ids = test_ids


class Cloudset(Dataset):
    def __init__(self,
                 df,
                 data_type,
                 ids,
                 fold,
                 augmentation=albu.Compose([albu.HorizontalFlip(p=0)])
                 ):
        self.df = df
        self.data_type = data_type
        self.ids = ids
        self.image_fold = fold
        self.transform = augmentation

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        image_id = self.ids[item]
        image_path = os.path.join(self.image_fold, image_id)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = make_mask(self.df, image_id)
        augmented = self.transform(image=img, mask=mask)
        img = np.transpose(augmented['image'], [2, 0, 1]).astype(np.float32)
        mask = np.transpose(augmented['mask'], [2, 0, 1]).astype(np.float32)
        # img:  [3, 1400, 2100]
        # mask: [4, 1400, 2100]
        # for torch
        return img, mask

    def gen_item(self, image_id):
        image_path = os.path.join(self.image_fold, image_id)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = make_mask(self.df, image_id)
        augmented = self.transform(image=img, mask=mask)
        img = np.transpose(augmented['image'], [2, 0, 1]).astype('float')
        mask = np.transpose(augmented['mask'], [2, 0, 1]).astype('float')
        return img, mask
