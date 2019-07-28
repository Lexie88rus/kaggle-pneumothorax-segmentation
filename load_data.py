'''
Script to load the dataset.
Contains data loading utility and function to create dataloaders.
'''
# Imports
import numpy as np
import pandas as pd
from glob import glob

# Import PyTorch
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# Import dataset for pneumothorax
from dataset import PneumothoraxDataset

# Data loading utility
def load_data(datafilepath = './', healthy_num = 2000):
    '''
    Function to load the dataset.
    INPUT:
        datafilepath - path to directory containing the dataset
        healthy_num - number of healthy images to be used for training
    OUTPUT:
        train_fns - train dataset
        train_fns - test dataset
        df_masks - pandas dataframe containing masks for train dataset
        files_list - list of filenames to be used fo training
    '''
    # Load full training and test sets
    train_fns = sorted(glob(datafilepath + 'dicom-images-train/*/*/*.dcm'))
    test_fns = sorted(glob(datafilepath + 'dicom-images-test/*/*/*.dcm'))
    # Load csv masks
    df_masks = pd.read_csv(datafilepath + 'train-rle.csv', index_col='ImageId')
    # create a list of filenames with images to use

    counter = 0
    files_list = []
    for fname in train_fns:
        try:
            if '-1' in df_masks.loc[fname.split('/')[-1][:-4],' EncodedPixels']:
                if counter <= healthy_num:
                    files_list.append(fname)
                    counter += 1
            else:
                files_list.append(fname)
        except:
            pass

    return train_fns, test_fns, df_masks, files_list

def build_dataloaders(image_size, channels, test_split = .1, batch_size = 16, num_workers = 4, all_data = False):
    '''
    Function to initialize train, test and validation sets and corresponding dataloaders.
    INPUT:
        test_split - train/test split ratio
        image_size - size of images for training
        batch_size - batch size for training
        num_workers - number of workers for training
    OUTPUT:
        trainloader, testloader, validloader - data loaders for training, testing and validation datasets
    '''

    # Setup image size for training
    width = image_size[0]
    height = image_size[1]

    # Load data
    if all_data:
        train_fns, test_fns, df_masks, files_list = load_data(healthy_num = 20000)
    else:
        train_fns, test_fns, df_masks, files_list = load_data(healthy_num = 2000)

    # Create dataset and data loader
    train_ds = PneumothoraxDataset(train_fns, df_masks, files_list, transform=True, size = (height, width), mode = 'train', channels = channels)

    # Creating data indices for training and validation splits:
    dataset_size = len(train_ds)
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    trainloader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers)

    valid_ds = PneumothoraxDataset(test_fns, None, None, transform=False, size = (height, width), mode = 'validation', channels = channels)
    validloader = DataLoader(valid_ds, batch_size=8, shuffle=False, num_workers=num_workers)

    return trainloader, testloader, validloader
