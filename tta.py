'''
Script to perform Test Time Augmentation (TTA)
'''
# Imports
import numpy as np
import pandas as pd
from PIL import Image

# Import PyTorch
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF

def get_prediction_with_tta(model, img, device, img_size = (224, 224), channels = 3):
    '''
    Function to create submission using TTA:
    INPUT:
        model - trained model
        img - image to create prediction with TTA
        device - device the model is on
        img_size - size of images used for training
        channels - number of channels of images used for training
    OUTPUT:
        pred_mask - numpy array, mean prediction
    '''
    angles = [-3, 3, -2, 2]
    masks = []

    im_height = img_size[0]
    im_width = img_size[1]

    # get a PIL image from tensor
    if (channels == 3):
        image = TF.to_pil_image(img.detach().cpu(), mode='RGB')
    else:
        image = TF.to_pil_image(img.detach().cpu(), mode='L')

    for angle in angles:
        # rotate the image
        rotated = image.rotate(angle)

        # get rotated prediction
        pred = model(TF.to_tensor(rotated).reshape(-1, 3, im_height, im_width).to(device))
        pred_img = TF.to_pil_image(torch.sigmoid(pred.detach()).reshape(im_height, im_width).cpu() * 255 , mode = 'L')

        # rotate the prediction backward
        pred_img = pred_img.rotate(- angle)
        masks.append(np.array(pred_img) / 255)

    # get mean prediction
    pred_mask = np.zeros((im_height, im_width))

    for i in range(0, len(angles)):
        pred_mask = np.array(pred_mask) + np.array(masks[i])

    pred_mask = pred_mask / len(angles)

    return pred_mask
