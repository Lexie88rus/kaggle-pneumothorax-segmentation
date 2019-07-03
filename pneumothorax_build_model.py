'''
Script builds the model for Pneumothorax Segmantation competition.
'''

# Imports
import numpy as np
import pandas as pd
from glob import glob
import pydicom
import random

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

# Import rle utils
from mask_functions import rle2mask

# Import models
from simple_unet import SimpleUNet

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

# Define the dataset
class PneumothoraxDataset(Dataset):
    '''
    The dataset for pneumothorax segmentation.
    '''

    def __init__(self, fns, df_masks, transform=True):
        '''
        INPUT:
            fns - glob containing the images
            df_masks - dataframe containing image masks
            transform (optional) - enable transforms for the images
        '''
        self.labels_frame = df_masks
        self.fns = fns
        self.transform = transform
        self.transforms = transforms.Compose([transforms.ToTensor()])

    def apply_transform(self, image, mask):
        '''
        Apply transforms to the image and mask.
        INPUT:
            image - original PIL image
            mask - original PIL image containing mask
        OUTPUT:
            image - tensor image after transform
            mask - tensor image containing mask after transform
        '''

        # Apply transformations if transform is enabled
        if (self.transform):
            # Resize
            resize = transforms.Resize(size=(1024, 1024))
            image = resize(image)
            mask = resize(mask)

            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=(284, 284))
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)

            # Random horizontal flipping
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Random vertical flipping
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        return image, mask

    def __len__(self):
        return len(self.fns)

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

        # read dcm file with image
        dataset = pydicom.read_file(self.fns[idx])
        np_image = np.expand_dims(dataset.pixel_array, axis=2)

        # load mask
        try:
            # no pneumothorax
            if '-1' in self.labels_frame.loc[self.fns[idx].split('/')[-1][:-4],' EncodedPixels']:
                np_mask = np.zeros((im_height, im_width, 1), dtype=np.bool)
            else:
                # there is pneumothorax
                if type(self.labels_frame.loc[self.fns[idx].split('/')[-1][:-4],' EncodedPixels']) == str:
                    np_mask = np.expand_dims(rle2mask(self.labels_frame.loc[self.fns[idx].split('/')[-1][:-4],' EncodedPixels'], im_height, im_width), axis=2)
                else:
                    np_mask = np.zeros((1024, 1024, 1))
                    for x in self.labels_frame.loc[self.fns[idx].split('/')[-1][:-4],' EncodedPixels']:
                        np_mask =  np_mask + np.expand_dims(rle2mask(x, 1024, 1024), axis=2)
        except KeyError:
            # couldn't find mask in dataframe
            print(f"Key {self.fns[idx].split('/')[-1][:-4]} without mask, assuming healthy patient.")
            np_mask = np.zeros((im_height, im_width, 1), dtype=np.bool) # Assume missing masks are empty masks.

        # convert to PIL
        image = Image.fromarray(np_image.reshape(im_height, im_width) , 'L')
        mask = Image.fromarray(np_mask.reshape(im_height, im_width).astype(np.uint8) , 'L')

        # apply transformations
        self.apply_transform(image, mask)

        return [self.transforms(image), self.transforms(mask)]

def train_step(model, inputs, labels, optimizer, criterion):
    optimizer.zero_grad()
    # forward + backward + optimize
    outputs = model(inputs)
    # outputs.shape =(batch_size, n_classes, img_cols, img_rows)
    outputs = outputs.permute(0, 2, 3, 1)
    # outputs.shape =(batch_size, img_cols, img_rows, n_classes)
    outputs = outputs.resize(batch_size*width_out*height_out, 2)
    labels = labels.resize(batch_size*width_out*height_out)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss

def train(model, device, trainloader, testloader, optimizer, criterion, epochs):
    model.to(device)
    steps = 0
    running_loss = 0
    print_every = 1

    for epoch in range(epochs):
        for inputs, labels in trainloader:

            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model.forward(inputs)
                        batch_loss = criterion(outputs, labels)
                        test_loss += batch_loss.item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(testloader):.3f}.. ")
                running_loss = 0
                model.train()

def main():
    '''
    Script entry point.
    '''
    # Load data
    print('Loading data: \n')
    train_fns, test_fns, df_masks = load_data()

    # Training presets
    batch_size = 32
    epochs = 10
    learning_rate = 0.01

    width = 284
    height = 284
    width_out = 196
    height_out = 196

    # Create dataset and data loader
    print('Preparing the dataset: \n')
    train_ds = PneumothoraxDataset(train_fns, df_masks, transform=True)
    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    test_ds = PneumothoraxDataset(test_fns, df_masks, transform=False)
    testloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Prepare for training: initialize model, loss function, optimizer
    model = SimpleUNet(in_channel=1,out_channel=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.99)

    # Setup device for training
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    # Train the model
    print('Train the model: \n')
    train(model, device, trainloader, testloader, optimizer, criterion, epochs)

    # Save the models
    print('Save the model: \n')
    filepath = 'simple_unet.pth'
    checkpoint = {'state_dict': model.state_dict()}
    torch.save(checkpoint, filepath)

if __name__ == '__main__':
    main()
