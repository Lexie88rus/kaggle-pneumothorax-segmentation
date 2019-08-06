'''
Script to build and train model for Kaggle pneumothorax segmentation competition.
'''
# Imports
import numpy as np
import pandas as pd
import argparse

# Import PyTorch
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

# Import data loading and dataset utilities
from dataset import PneumothoraxDataset
from load_data import build_dataloaders

# Import models
from models.simple_unet import UNet
from models.vgg_unet import UNet11, AlbuNet, UNet16
from models.unet_plusplus import NestedUNet
from models.unet_deepsupervision import Unet_2D
from models.phalanx import Res34Unetv4, Res34Unetv3, Res34Unetv5
from models.unet_brain import brain_unet
from models.att_unet import R2U_Net, AttU_Net, R2AttU_Net

# Import losses
from losses import BCEDiceLoss, WeightedBCEDiceLoss
from lovasz_loss import LovaszSoftmaxLoss
from jaccard_loss import JaccardLoss
from soft_IoU_loss import mIoULoss

# Import metrics
from metrics import iou_score

# Import training utulities
from train import train

# Import submission utilities
from submission import determine_threshold, make_submission

# Import model saving
from save_model import save_model

# Import model loading
from load_model import load_model

def build_from_checkpoint(filename, device, img_size, channels, test_split, batch_size, workers, epochs, learning_rate, swa, enable_scheduler, loss = 'BCEDiceLoss', all_data = False, tta = False):

    # create data loaders
    trainloader, testloader, validloader = build_dataloaders(image_size = (img_size, img_size), channels = channels,
    test_split = test_split,
    batch_size = batch_size,
    num_workers = workers,
    all_data = all_data)

    # setup the device
    if device == None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    # restore model
    model, model_arch, train_losses_0, test_losses_0, train_metrics_0, test_metrics_0 = load_model(filename, device, channels = channels)

    # setup criterion, optimizer and metrics
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if loss == 'BCEDiceLoss':
        criterion = BCEDiceLoss()

    if loss == 'LovaszSoftmaxLoss':
        criterion = LovaszSoftmaxLoss()

    if loss == 'JaccardLoss':
        criterion = JaccardLoss(device = device)

    if loss == 'mIoULoss':
        criterion = mIoULoss(n_classes = 1)

    if loss == 'WeightedBCEDiceLoss':
        criterion = WeightedBCEDiceLoss()

    metric = iou_score

    #train model
    model, train_losses, test_losses, train_metrics, test_metrics = train(model, device, trainloader, testloader, optimizer, criterion, metric, epochs, learning_rate, swa = swa, enable_scheduler = enable_scheduler, model_arch = model_arch)

    train_losses = train_losses + train_losses_0
    test_losses = test_losses + test_losses_0
    train_metrics = train_metrics + train_metrics_0
    test_metrics = test_metrics + test_metrics_0

    # create submission
    filename = 'submission_' + model_arch + '_lr' + str(learning_rate) + '_' + str(epochs) + '.csv'
    print('Generating submission to ' + filename + '\n')
    thresholds, ious, index_max, threshold_max = determine_threshold(model, device, testloader, image_size = (img_size, img_size))
    make_submission(filename, device, model, validloader, image_size = (img_size, img_size), channels = channels, threshold = threshold_max, original_size = 1024, tta = tta)

    # save the model
    save_model(model, model_arch, learning_rate, epochs, train_losses, test_losses, train_metrics, test_metrics, filepath = 'models_checkpoints')

def build_model(device, img_size, channels, test_split, batch_size, workers, model_arch, epochs, learning_rate, swa, enable_scheduler, loss = 'BCEDiceLoss', all_data = False, tta = False):
    # create data loaders
    trainloader, testloader, validloader = build_dataloaders(image_size = (img_size, img_size), channels = channels,
    test_split = test_split,
    batch_size = batch_size,
    num_workers = workers,
    all_data = all_data,
    data_filepath = '../siim-train-test/')

    # setup the device
    if device == None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    # initialize model
    if model_arch == 'UNet':
        model = UNet(num_classes = 1, depth=6, start_filts=8, merge_mode='concat')

    if model_arch == 'UNet11':
        model = UNet11(pretrained=True)

    if model_arch == 'UNet16':
        model = UNet16(num_classes = 1, pretrained=True)

    if model_arch == 'AlbuNet':
        model = AlbuNet(num_classes = 1, pretrained=True)

    if model_arch == 'NestedUNet':
        model = NestedUNet()

    if model_arch == 'Unet_2D':
        model = Unet_2D(n_channels = channels, n_classes = 1)

    if model_arch == 'Res34Unetv4':
        model = Res34Unetv4()

    if model_arch == 'Res34Unetv3':
        model = Res34Unetv3()

    if model_arch == 'Res34Unetv5':
        model = Res34Unetv5()

    if model_arch == 'BrainUNet':
        model = brain_unet(pretrained = True)

    if model_arch =='R2U_Net':
        model = R2U_Net()

    if model_arch == 'AttU_Net':
        model = AttU_Net()

    if model_arch == 'R2AttU_Net':
        model = R2AttU_Net()

    # setup criterion, optimizer and metrics
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if loss == 'BCEDiceLoss':
        criterion = BCEDiceLoss()

    if loss == 'LovaszSoftmaxLoss':
        criterion = LovaszSoftmaxLoss()

    if loss == 'JaccardLoss':
        criterion = JaccardLoss(device = device)

    if loss == 'mIoULoss':
        criterion = mIoULoss(n_classes = 1)

    if loss == 'WeightedBCEDiceLoss':
        criterion = WeightedBCEDiceLoss()

    metric = iou_score

    #train model
    model, train_losses, test_losses, train_metrics, test_metrics = train(model, device, trainloader, testloader, optimizer, criterion, metric, epochs, learning_rate, swa = swa, enable_scheduler = enable_scheduler, model_arch = model_arch)

    # create submission
    filename = 'submission_' + model_arch + '_lr' + str(learning_rate) + '_' + str(epochs) + '.csv'
    print('Generating submission to ' + filename + '\n')
    thresholds, ious, index_max, threshold_max = determine_threshold(model, device, testloader, image_size = (img_size, img_size), channels = channels)
    make_submission(filename, device, model, validloader, image_size = (img_size, img_size), channels = channels, threshold = threshold_max, original_size = 1024, tta = tta)

    # save the model
    save_model(model, model_arch, learning_rate, epochs, train_losses, test_losses, train_metrics, test_metrics, filepath = 'models_checkpoints')

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Argument parser')

    parser.add_argument('--learning_rate', type = float, action='store', default = 0.00001,
                    help='Model hyperparameters: learning rate.')

    parser.add_argument('--epochs', type = int, action='store', default = 1,
                    help='Number of training epochs.')

    parser.add_argument('--test_split', type = float, action='store', default = 0.1,
                    help='Train and test split ration for training.')

    parser.add_argument('--batch_size', type = int, action='store', default = 16,
                    help='Training batch size for training.')

    parser.add_argument('--workers', type = int, action='store', default = 4,
                    help='Number of workers for training.')

    parser.add_argument('--model', action='store', default = 'UNet',
    choices=['UNet', 'UNet11', 'AlbuNet', 'UNet16', 'NestedUNet', 'Unet_2D', 'Res34Unetv4', 'Res34Unetv3', 'Res34Unetv5',
    'BrainUNet', 'R2U_Net', 'AttU_Net', 'R2AttU_Net'],
    help='Model architecture.')

    parser.add_argument('--loss', action='store', default = 'BCEDiceLoss',
    choices=['BCEDiceLoss', 'LovaszSoftmaxLoss', 'Jaccard', 'mIoULoss'],
    help='Model architecture.')

    parser.add_argument('--optimizer', action='store', default = 'SGD',
    choices=['SGD', 'Adam'],
    help='Optimizer for fitting the model.')

    parser.add_argument('--swa', action='store_true',
                        help='Enable stochastic weight averaging')

    parser.add_argument('--lr_scheduler', action='store_true',
                        help='Enable learning rate scheduling (cosine annealing).')

    parser.add_argument('--checkpoint', action='store', default = None,
    help='Model checkpoint to load.')

    parser.add_argument('--create_submission', action='store_true',
                        help='Create submission from checkpoint.')

    parser.add_argument('--tta', action='store_true',
                        help='Enable Test Time Augmentation.')

    parser.add_argument('--threshold', action='store_true',
                        help='Find best threshold for mask.')

    results = parser.parse_args()

    learning_rate = results.learning_rate
    epochs = results.epochs
    test_split = results.test_split
    batch_size = results.batch_size
    workers = results.workers
    model_arch = results.model
    swa = results.swa
    lr_scheduler = results.lr_scheduler
    loss = results.loss
    checkpoint_filepath = results.checkpoint
    create_submission = results.create_submission
    tta = results.tta
    find_threshold = results.threshold

    # set the number of channels for images to train
    if model_arch == 'UNet' or model_arch == 'UNet_2D':
        channels = 1
    else:
        channels = 3

    # set the size of the images to train
    if model_arch == 'UNet' or model_arch =='R2U_Net' or model_arch == 'AttU_Net' or model_arch == 'R2AttU_Net':
        img_size = 224

    if model_arch == 'UNet11' or model_arch == 'UNet16' or model_arch == 'UNet_2D' or model_arch == 'NestedUNet':
        img_size = 224

    if model_arch == 'AlbuNet' or model_arch == 'Res34Unetv4' or model_arch == 'Res34Unetv3' or model_arch == 'BrainUNet':
        img_size = 256

    if model_arch == 'Res34Unetv5':
        img_size = 128

    # setup the device
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    if create_submission:

        if checkpoint_filepath is None:
            print('Filepath to saved model checkpoint should be specified.')
            return

        # restore model
        model, model_arch, train_losses_0, test_losses_0, train_metrics_0, test_metrics_0 = load_model(checkpoint_filepath, device, channels = channels)

        # create data loaders
        trainloader, testloader, validloader = build_dataloaders(image_size = (img_size, img_size), channels = channels,
        test_split = test_split,
        batch_size = batch_size,
        num_workers = workers,
        all_data = False,
        data_filepath = '../siim-train-test/')

        filename = 'submission.csv'

        if find_threshold:
            thresholds, ious, index_max, threshold = determine_threshold(model, device, testloader, image_size = (img_size, img_size), channels = channels)
            print('Threshold: {threshold} with {iou}\n'.format(threshold = threshold, iou = ious[index_max]))
        else:
            threshold = 0.9 # use default threshold for the submission

        make_submission(filename, device, model, validloader, image_size = (img_size, img_size), channels = channels, threshold = threshold, original_size = 1024, tta = tta)
    else:
        if checkpoint_filepath is not None:
            build_from_checkpoint(filename, device, img_size, channels, test_split, batch_size, workers, epochs, learning_rate, swa = swa, enable_scheduler = lr_scheduler, loss = loss, all_data = False, tta = tta)
        else:
            build_model(device, img_size, channels, test_split, batch_size, workers, model_arch, epochs, learning_rate, swa = swa, enable_scheduler = lr_scheduler ,loss = loss, all_data = False, tta = tta)

if __name__ == '__main__':
    main()
