'''
Script to test the created augmented dataset.
'''

# Imports
import numpy as np
import pandas as pd
from glob import glob
import pydicom
import random
import time

# import image manipulation
from PIL import Image

# Import PyTorch
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable

# Import utils for submission
from skimage.morphology import binary_opening, disk, label

# Import rle utils
from mask_functions import rle2mask

def load_data_augmented(datafilepath = '../siim-train-test-augmented/'):
    '''
    Function to load the dataset.
    INPUT:
        datafilepath - path to directory containing the augmented dataset.
    OUTPUT:
        train_fns - train augmented dataset
    '''
    # Load full training and test sets
    train_fns = sorted(glob(datafilepath + 'img/' + '*.png'))

    train_fns_masks = sorted(glob(datafilepath + 'mask/' + '*.png'))

    return train_fns, train_fns_masks

def get_image_by_id(idx, fns, fns_masks):
    '''
    Function to test created augmented images and masks.
    The function returns image and mask by id.
    INPUT:
        idx - id of image in the dataset
        fns - list of filenames of images in the dataset
        fns_masks - list of filenames of masks in the dataset
    '''
    # image height and width
    im_height = 1024
    im_width = 1024
    # image channels
    im_chan = 1

    # get train image and mask
    np_image = np.zeros((im_height, im_width, im_chan), dtype=np.uint8)
    np_mask = np.zeros((im_height, im_width, 1), dtype=np.bool)

    # read png file with image
    image = Image.open(fns[idx])
    image = image.convert('L')

    # read png file with mask
    mask = Image.open(fns_masks[idx])
    mask = mask.convert('L')

    if 'rotated' in (fns[idx].split('/')[-1][:-4]):
        img_id = fns[idx].split('/')[-1][:-4]
        image.save('img_' + img_id + '.png')
        mask.save('mask_' + img_id + '.png')

    return image, mask

def main():

    # Load augmented data
    fns, fns_masks = load_data_augmented()

    for i in range(0,10):
        image , mask = get_image_by_id(i, fns, fns_masks)

if __name__ == '__main__':
    main()
