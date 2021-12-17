import os
import glob
import cv2
import numpy as np
from loguru import logger

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


DATA_ROOT = "D:/Dataset/NanoTWICE/"

class NanoDataset(object):
    def __init__(self, root_path=DATA_ROOT, is_train=True,
                 resize=256, cropsize=256):


        self.root_path = root_path
        self.is_train = is_train

        # set transforms
        self.transform_x = T.Compose([T.Resize(resize, Image.ANTIALIAS),
                                      T.CenterCrop(cropsize),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])
        self.transform_mask = T.Compose([T.Resize(resize, Image.NEAREST),
                                         T.CenterCrop(cropsize),
                                         T.ToTensor()])

        self.x, self.mask = self.load_image()
        self.len = len(self.x)

    def load_image(self):
        if self.is_train:
            img_path = os.path.join(self.root_path, "Normal")
            img_list = glob.glob(img_path + "/*.tif")
            mask_list = None
        else:
            img_path = os.path.join(self.root_path, "Anomalous", "images")
            mask_path = os.path.join(self.root_path, "Anomalous", "gt")
            img_list = glob.glob(img_path + "/*.tif")
            mask_list = glob.glob(mask_path + "/*.png")
            img_list.sort()
            mask_list.sort()
            assert len(img_list) == len(mask_list)
        return img_list, mask_list

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x = self.x[idx]
        # x = Image.open(x)
        x = cv2.imread(x)
        # .convert('RGB')
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        x = Image.fromarray(x)
        x = self.transform_x(x)

        if self.is_train:
            mask = torch.Tensor([1])
        else:
            mask = self.mask[idx]
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        return x, mask

if __name__ == "__main__":
    stc = NanoDataset(is_train=False)
    data = stc[0]
    print(data[0].shape)