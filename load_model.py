'''
Script to load the model from checkpoint.
'''

# Import PyTorch
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

# Import models
from models.simple_unet import UNet
from models.vgg_unet import UNet11, AlbuNet, UNet16
from models.unet_plusplus import NestedUNet
from models.unet_deepsupervision import Unet_2D
from models.phalanx import Res34Unetv4, Res34Unetv3, Res34Unetv5

def load_model(filename, channels = 3):
    '''
    Function loads the model from checkpoint
    INPUT:
        filename - filename of the checkpoint containing saved model
        channels - number of image channels
    '''
    checkpoint = torch.load(filename)
    model_arch = checkpoint['model_arch']
    train_losses = checkpoint['train_losses']
    test_losses = checkpoint['test_losses']
    train_metrics = checkpoint['train_metrics']
    test_metrics = checkpoint['test_metrics']

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

    model.load_state_dict(checkpoint['state_dict'])

    return model, model_arch, train_losses, test_losses, train_metrics, test_metrics
