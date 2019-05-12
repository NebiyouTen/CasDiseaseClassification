'''

Author: nyismaw
Data loader for Casava image files as a part of an in-class kaggle competition
https://www.kaggle.com/c/cassava-disease/

Some code typically used in dataloader and dataset classes are adopted from
the official PyTorch DATA LOADING AND PROCESSING TUTORIAL
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#data-loading-and-processing-tutorial

'''

from __future__ import print_function, division
import argparse
import sys
import os
import numpy as np
import glob
# %matplotlib inline
import matplotlib . pyplot as plt
import matplotlib.image as mpimg
import constants as C
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def generate_train_val(image_dir, image_label):
    cbb_train_images_files = glob.glob(image_dir)
    image_array = []
    image_labels = np.zeros((len(cbb_train_images_files)))
    print (len(cbb_train_images_files)," images in ", image_dir)
    for i, image in enumerate(cbb_train_images_files):
        img = mpimg.imread(image)
        image_array.append(img)
        image_labels[i] = image_label
    return image_array, image_labels

def generate_data_set():
    image_array, image_label = generate_train_val(C.CBB_IMAGES_DIR, C.CBB_LABEL);
    images_all = np.array(image_array)
    labels_all = np.array(image_label)

    image_array, image_label = generate_train_val(C.CBSD_IMAGES_DIR, C.CBSD_LABEL);
    images_all = np.concatenate((images_all,image_array))
    labels_all = np.concatenate((labels_all,image_label))

    image_array, image_label = generate_train_val(C.CGM_IMAGES_DIR, C.CGM_LABEL);
    images_all = np.concatenate((images_all,image_array))
    labels_all = np.concatenate((labels_all,image_label))

    image_array, image_label = generate_train_val(C.CMD_IMAGES_DIR, C.CMD_LABEL);
    images_all = np.concatenate((images_all,image_array))
    labels_all = np.concatenate((labels_all,image_label))

    image_array, image_label = generate_train_val(C.HEALTHY_IMAGES_DIR, C.HEALTHY_LABEL);
    images_all = np.concatenate((images_all,image_array))
    labels_all = np.concatenate((labels_all,image_label))

    print ("Shape of images all ", images_all.shape)
    print ("Shape of labels all ", labels_all.shape)

    train_length = int ( len(images_all) * C.TRAIN_VAL_SPLIT )
    print ("Train length is ", train_length)
    print ("Val length is "  , len(images_all) - train_length)

    np.save(C.TRAIN_DATA_PATH,  images_all[:train_length])
    np.save(C.TRAIN_LABEL_PATH, labels_all[:train_length])
    np.save(C.VAL_DATA_PATH,  images_all[train_length:])
    np.save(C.VAL_LABEL_PATH, labels_all[train_length:])

class CassavaImagesDataset(Dataset):
    """CassavaImagesDataset dataset."""

    def __init__(self, data_path, data_label_path, transform=None):
        """
        Args:
            data_path (string): Path to the train npy image arrays.
            data_label_path (string): Optional transform to be applied
                on a sample.
            transform (callable, optional): Optional transform to be applied
            on a data.
        """
        self.data       = np.load(data_path)
        self.data_label = np.load(data_label_path)
        self.transform   = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_data  = self.data[idx]
        sample_label = self.data_label[idx]
        # if transform call back, tranform image
        if self.transform:
            sample_data = self.transform(sample_data)
        # return sample data and sample label
        return sample_data, sample_label

def collate (batch):
    return NotImplemented

def main(argv):
    print("Data_loader main")
    # generate_data_set()

if __name__ == '__main__':
    main(sys.argv)
