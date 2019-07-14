'''
Script builds the model for Pneumothorax Segmentation competition.
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

# Import models
from simple_unet import UNet
from vgg_unet import UNet16

# Data loading utility
def load_data(datafilepath = '../siim-train-test/', healthy_num = 2000):
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
    # create a list of filenames with images to use

    counter = 0
    files_list = []
    for fname in train_fns:
        try:
            if '-1' in df_masks.loc[fname.split('/')[-1][:-4],' EncodedPixels']:
                if counter <= 2000:
                    files_list.append(fname)
                    counter += 1
            else:
                files_list.append(fname)
        except:
            pass

    return train_fns, test_fns, df_masks, files_list

# Define the dataset
class PneumothoraxDataset(Dataset):
    '''
    The dataset for pneumothorax segmentation.
    '''

    def __init__(self, fns, df_masks, files_list, transform=True, size = (224, 224), mode = 'train'):
        '''
        INPUT:
            fns - glob containing the images
            df_masks - dataframe containing image masks
            transform (optional) - enable transforms for the images
        '''
        self.labels_frame = df_masks
        self.fns = fns
        self.transform = transform
        self.size = size
        self.transforms_image = transforms.Compose([transforms.Resize(self.size), transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.transforms_mask = transforms.Compose([transforms.Resize(self.size), transforms.ToTensor()])
        self.mode = mode
        self.files_list = files_list

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
        im_chan = 3

        # get train image and mask
        np_image = np.zeros((im_height, im_width, im_chan), dtype=np.uint8)
        np_mask = np.zeros((im_height, im_width, im_chan), dtype=np.bool)

        # if validation then return filename instead of mask
        if self.mode == 'validation':
            # read dcm file with image
            dataset = pydicom.read_file(self.fns[idx])
            np_image = np.expand_dims(dataset.pixel_array, axis=2)

            image = Image.fromarray(np_image.reshape(im_height, im_width) , 'L')
            image = image.convert('RGB')
            image = self.transforms_image(image)
            return [image, self.fns[idx].split('/')[-1][:-4]]

        # read dcm file with image
        dataset = pydicom.read_file(self.files_list[idx])
        np_image = np.expand_dims(dataset.pixel_array, axis=2)

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
                    np_mask = np.zeros((1024, 1024, 1))
                    for x in self.labels_frame.loc[self.files_list[idx].split('/')[-1][:-4],' EncodedPixels']:
                        np_mask =  np_mask + np.expand_dims(rle2mask(x, 1024, 1024), axis=2)
        except KeyError:
            # couldn't find mask in dataframe
            np_mask = np.zeros((im_height, im_width, 1), dtype=np.bool) # Assume missing masks are empty masks.

        # convert to PIL
        image = Image.fromarray(np_image.reshape(im_height, im_width) , 'L')
        image = image.convert('RGB')
        np_mask = np.transpose(np_mask)
        mask = Image.fromarray(np_mask.reshape(im_height, im_width).astype(np.uint8) , 'L')

        # apply optional transforms (random rotation for the training set)
        if self.transform:
            # apply random rotation
            angle = random.uniform(-5, 5)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)

            # apply random horizontal flip
            #if (angle > 0):
            #    image = TF.hflip(image)
            #    mask = TF.hflip(mask)

        # apply required transforms normalization, reshape and convert to tensor
        image = self.transforms_image(image)
        mask = self.transforms_mask(mask)

        # convert to tensor and clip mask
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        mask = np.clip(mask, 0, 1)

        return image, mask

# Training the model
def train(model, device, trainloader, testloader, optimizer, criterion, epochs = 10):
    '''
    Function to train the model:
    INPUT:
        model - the model to train
        device - cuda or cpu
        trainloader - loader for the training data
        testloader - loader for testing data
        optimizer - optimizer (SGD or Adam for example)
        criterion - initialized loss function
        epochs - the number of epochs
    '''
    model.to(device)
    steps = 0
    running_loss = 0
    print_every = 100

    for epoch in range(epochs):
        for inputs, labels in trainloader:

            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels.reshape(-1, 224, 224).long())
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
                        batch_loss = criterion(outputs, labels.reshape(-1, 224, 224).long())
                        test_loss += batch_loss.item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(testloader):.3f}.. ")

                running_loss = 0

                model.train()

def main():
    '''
    Script entry point.
    TODO: add script comand line parameters.
    '''
    # Load data
    print('Loading data: \n')
    train_fns, test_fns, df_masks, files_list = load_data()

    # Training presets
    batch_size = 16
    epochs = 10
    learning_rate = 0.01
    test_split = .2

    # Set the size of images
    original_size = 1024
    width = 224
    height = 224

    # Create the dataset and data loaders
    print('Preparing the dataset: \n')
    train_ds = PneumothoraxDataset(train_fns, df_masks, files_list, transform=True, size = (height, width), mode = 'train')

    # Creating data indices for train and test splits with SubsetRandomSampler
    dataset_size = len(train_ds)
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    trainloader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, num_workers=4)
    testloader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, sampler=test_sampler, num_workers=4)

    valid_ds = PneumothoraxDataset(test_fns, None, None, transform=False, size = (height, width), mode = 'validation')
    validloader = DataLoader(valid_ds, batch_size=8, shuffle=False, num_workers=1)

    torch.cuda.empty_cache()

    # Prepare for training: initialize model, loss function, optimizer
    class param:
        unet_depth = 5
        unet_start_filters = 8
    #model = UNet(2, depth=param.unet_depth, start_filts=param.unet_start_filters, merge_mode='concat')
    model = UNet16(num_classes = 2, pretrained = False)

    # Initialize loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Setup device for training
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    # Train the model
    print('Train the model: \n')
    train(model, device, trainloader, testloader, optimizer, criterion, epochs)

    # Save the model
    print('Save the model: \n')
    filepath = 'simple_unet.pth'
    checkpoint = {'state_dict': model.state_dict()}
    torch.save(checkpoint, filepath)

    # Create submission file
    submission = {'ImageId': [], 'EncodedPixels': []}

    model.eval()
    torch.cuda.empty_cache()

    for X, fns in validloader:
        X = Variable(X).cuda()
        output = model(X)
        for i, fname in enumerate(fns):
            mask = torch.sigmoid(output[i, 0]).data.cpu().numpy()
            mask = binary_opening(mask > 0.001, disk(2))

            im = Image.fromarray((mask*255).astype(np.uint8)).resize((original_size,original_size))
            im = np.asarray(im)

            submission['EncodedPixels'].append(mask2rle(im, original_size, original_size))
            submission['ImageId'].append(fname)

if __name__ == '__main__':
    main()
