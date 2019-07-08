'''
Script for creation of augmented dataset for Pneumothorax segmentation competition.
'''
# Imports
import numpy as np
import pandas as pd
from glob import glob
import pydicom
import random
from tqdm import tqdm
from shutil import copy, copyfile, copy2

# import image manipulation
from PIL import Image
import SimpleITK as sitk

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

from mask_functions import rle2mask, mask2rle

# Data loading utility
def load_data(datafilepath = '../siim-train-test/'):
    '''
    Function to load the dataset.
    INPUT:
        datafilepath - path to directory containing the dataset.
    OUTPUT:
        train_fns - train dataset
        train_fns - test dataset
        df_masks - pandas dataframe containing masks for train dataset
    '''
    # Load full training and test sets
    train_fns = sorted(glob(datafilepath + 'dicom-images-train/*/*/*.dcm'))
    test_fns = sorted(glob(datafilepath + 'dicom-images-test/*/*/*.dcm'))
    # Load csv masks
    df_masks = pd.read_csv(datafilepath + 'train-rle.csv', index_col='ImageId')

    return train_fns, test_fns, df_masks

def normalize(arr):
    """
    Function performs the linear normalizarion of the array.
    https://stackoverflow.com/questions/7422204/intensity-normalization-of-image-using-pythonpil-speed-issues
    http://en.wikipedia.org/wiki/Normalization_%28image_processing%29
    INPUT:
        arr - orginal numpy array
    OUTPUT:
        arr - normalized numpy array
    """
    arr = arr.astype('float')
    # Do not touch the alpha channel
    for i in range(1):
        minval = arr[...,i].min()
        maxval = arr[...,i].max()
        if minval != maxval:
            arr[...,i] -= minval
            arr[...,i] *= (255.0/(maxval-minval))
    return arr

def normalize_image(img):
    """
    Function performs the normalization of the image.
    https://stackoverflow.com/questions/7422204/intensity-normalization-of-image-using-pythonpil-speed-issues
    INPUT:
        image - PIL image to be normalized
    OUTPUT:
        new_img - PIL image normalized
    """
    arr = np.array(img)
    new_img = Image.fromarray(normalize(arr).astype('uint8'),'L')
    return new_img

def create_augmented_dataset(train_fns, test_fns, df_masks, filepath = '../siim-train-test-augmented/', no_pneumothorax_examples = 2000, pneumothorax_augmented_examples = 2000):
    '''
    Function to create an augmented dataset:
    INPUT:
        train_fns, test_fns, df_masks - loaded original dataset
        filepath - path to save files of the augmented dataset
        no_pneumothorax_examples - number of healthy images to stay in the dataset
        pneumothorax_augmented_examples - number of augmented pneumothorax images to add to the dataset
    '''
    # image height and width
    im_height = 1024
    im_width = 1024
    # image channels
    im_chan = 1

    filepath_img = filepath + '/img/'
    filepath_mask = filepath + '/mask/'

    # loop over the training set
    for idx, fname in tqdm(enumerate(train_fns)):

        try:
            # If it is a picture without pneumothrax then we only take it
            # to the training dataset with a certain ptobability
            if '-1' in df_masks.loc[fname.split('/')[-1][:-4],' EncodedPixels']:
                # Get sample from uniform distribution and decide to copy the healthy sample
                if random.uniform(0, 1) < (no_pneumothorax_examples / len(train_fns)):
                    np_image = np.zeros((im_height, im_width, im_chan), dtype=np.uint8)
                    data = pydicom.read_file(fname)
                    np_image = np.expand_dims(data.pixel_array, axis=2)
                    image = Image.fromarray(np_image.reshape(im_height, im_width) , 'L')
                    image = normalize_image(image)
                    image.save(filepath_img + fname.split('/')[-1][:-4] + '.png')

                    np_mask = np.zeros((im_height, im_width, 1), dtype=np.bool)
                    mask = Image.fromarray(np_mask.reshape(im_height, im_width).astype(np.uint8) , 'L')
                    mask.save(filepath_mask + fname.split('/')[-1][:-4] + '_mask.png')
            else:
                # If there is pneumothrax on the image then we copy it to the output directory
                np_image = np.zeros((im_height, im_width, im_chan), dtype=np.uint8)
                data = pydicom.read_file(fname)
                np_image = np.expand_dims(data.pixel_array, axis=2)
                image = Image.fromarray(np_image.reshape(im_height, im_width) , 'L')
                image = normalize_image(image)
                image.save(filepath_img + fname.split('/')[-1][:-4] + '.png')

                # Copy the mask to the output directory
                np_mask = np.zeros((im_height, im_width, 1), dtype=np.bool)
                if type(df_masks.loc[fname.split('/')[-1][:-4],' EncodedPixels']) == str:
                    np_mask = np.expand_dims(rle2mask(df_masks.loc[fname.split('/')[-1][:-4],' EncodedPixels'], im_height, im_width), axis=2)
                else:
                    np_mask = np.zeros((im_height, im_width, 1))
                    for x in df_masks.loc[fname.split('/')[-1][:-4],' EncodedPixels']:
                        np_mask =  np_mask + np.expand_dims(rle2mask(x, im_height, im_width), axis=2)

                np_mask = np.transpose(np_mask)
                mask = Image.fromarray(np_mask.reshape(im_height, im_width).astype(np.uint8) , 'L')
                mask.save(filepath_mask + fname.split('/')[-1][:-4] + '_mask.png')

                # Get sample from uniform distribution and decide to include augmented example to the train set
                if random.uniform(0, 1) < (pneumothorax_augmented_examples / len(train_fns)):

                    # augment image and masks
                    angle = random.uniform(0, 10)
                    new_image_id = fname.split('/')[-1][:-4] + '_rotated_' + str(angle)

                    image = TF.rotate(image, angle)
                    mask = TF.rotate(mask, angle)

                    # Save augmented image as png and mask as png
                    image.save(filepath_img + new_image_id + '.png')
                    mask.save(filepath_mask + new_image_id + '_mask.png')
        except:
            pass


def main():
    # Set the number of examples without pneumothorax
    no_pneumothorax_examples = 2000

    # Set the number of examples with pneumothorax to be generated
    pneumothorax_augmented_examples = 1000

    # Load the original data
    train_fns, test_fns, df_masks = load_data()

    # Create augmented dataset
    create_augmented_dataset(train_fns, test_fns, df_masks)

if __name__ == '__main__':
    main()
