'''
Script to prepare the dataset for training.
'''
# Imports
import numpy as np
import pandas as pd
from glob import glob
import pydicom
import random

# import image manipulation
from PIL import Image

# import image augmentation
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma,OneOf,
    ToFloat, ShiftScaleRotate,GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise,CenterCrop,
    IAAAdditiveGaussianNoise,GaussNoise,OpticalDistortion,RandomSizedCrop
)
from albumentations.pytorch import ToTensor

# Import PyTorch
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# import mask utilities
from mask_functions import rle2mask

# Define the dataset
class PneumothoraxDataset(Dataset):
    '''
    The dataset for pneumothorax segmentation.
    '''

    def __init__(self, fns, df_masks, files_list, transform=True, size = (224, 224), mode = 'train', channels = 3):
        '''
        INPUT:
            fns - glob containing the images
            df_masks - dataframe containing image masks
            files_list - list of files to be used for training
            transform (optional) - enable transforms for the images
            size (optional) - size of images to be used for training
            mode (optional) - training/validation mode
            channels (optional) - number of channels to be used for training
        '''
        self.labels_frame = df_masks
        self.fns = fns
        self.transform = transform
        self.size = size

        self.transforms_mask = transforms.Compose([transforms.Resize(self.size), transforms.ToTensor()])

        if channels == 3:
            self.transforms_image = transforms.Compose([transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        else:
            self.transforms_image = transforms.Compose([transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])])

        self.alb_transforms = Compose([
            HorizontalFlip(p=0.5),
            OneOf([
                RandomContrast(),
                RandomGamma(),
                RandomBrightness(),
            ], p=0.3),
            OneOf([
                ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                GridDistortion(),
                OpticalDistortion(distort_limit=2, shift_limit=0.5),
            ], p=0.3),
            RandomSizedCrop(min_max_height=(156, 224), height=self.size[0], width=self.size[1],p=0.25)
        ],p=1)

        self.mode = mode
        self.files_list = files_list
        self.channels = channels

    def __len__(self):
        if (self.mode == 'validation'):
            return len(self.fns)
        else:
            return len(self.files_list)

    def __getitem__(self, idx):
        '''
        Function to get items from dataset by idx.
        INPUT:
            idx - id of the image in the dataset
        '''
        # image height and width
        im_height = 1024
        im_width = 1024
        # image channels
        im_chan = 1

        # get train image and mask
        np_image = np.zeros((im_height, im_width, im_chan), dtype=np.uint8)
        np_mask = np.zeros((im_height, im_width, 1), dtype=np.bool)

        # if validation then return filename instead of mask
        if self.mode == 'validation':
            # read dcm file with image
            dataset = pydicom.read_file(self.fns[idx])
            np_image = np.expand_dims(dataset.pixel_array, axis=2)

            image = Image.fromarray(np_image.reshape(im_height, im_width) , 'L')

            # if number of channels is 3 then return RGB images
            if (self.channels == 3):
                image = image.convert('RGB')

            image = self.transforms_image(image)
            return [image, self.fns[idx]]

        # read dcm file with image
        dataset = pydicom.read_file(self.files_list[idx])
        np_image = np.expand_dims(dataset.pixel_array, axis=2)

        pneumothorax = False

        # load mask
        try:
            # no pneumothorax
            if '-1' in self.labels_frame.loc[self.files_list[idx].split('/')[-1][:-4],' EncodedPixels']:
                np_mask = np.zeros((im_height, im_width, 1), dtype=np.bool)
            else:
                # there is pneumothorax
                if type(self.labels_frame.loc[self.files_list[idx].split('/')[-1][:-4],' EncodedPixels']) == str:
                    np_mask = np.expand_dims(rle2mask(self.labels_frame.loc[self.files_list[idx].split('/')[-1][:-4],' EncodedPixels'], im_height, im_width), axis=2)
                else:
                    np_mask = np.zeros((im_height, im_width, 1))
                    for x in self.labels_frame.loc[self.files_list[idx].split('/')[-1][:-4],' EncodedPixels']:
                        np_mask =  np_mask + np.expand_dims(rle2mask(x, im_height, im_width), axis=2)

                pneumothorax = True

        except KeyError:
            # couldn't find mask in dataframe
            np_mask = np.zeros((im_height, im_width, 1), dtype=np.bool) # Assume missing masks are empty masks.

        # convert to PIL
        image = Image.fromarray(np_image.reshape(im_height, im_width) , 'L')

        if self.channels == 3:
            image = image.convert('RGB')

        np_mask = np.transpose(np_mask)
        mask = Image.fromarray(np_mask.reshape(im_height, im_width).astype(np.uint8) , 'L')

        if self.transform:
            augmented = self.alb_transforms(image=np.array(image), mask=np.array(mask))
            if self.channels == 3:
                image = Image.fromarray(augmented['image'], 'RGB')
            else:
                image = Image.fromarray(augmented['image'], 'L')

            mask = Image.fromarray(augmented['mask'], 'L')

        # apply required transforms normalization, reshape and convert to tensor
        image = self.transforms_image(image)
        mask = self.transforms_mask(mask)

        # convert to tensor and clip mask
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        mask = np.clip(mask, 0, 1)

        return image, mask
