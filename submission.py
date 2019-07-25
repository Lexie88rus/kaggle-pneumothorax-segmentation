'''
Script to create submission.
'''
# Imports
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
import tqdm

# import image utilities
from skimage.morphology import binary_opening, disk, label
from PIL import Image

# import mask utilities
from mask_functions import mask2rle

def make_submission(filename, device, model, validloader, image_size, threshold = 0.9, original_size = 1024):
    '''
    Function to create submission.csv file.
    INPUT:
        filename - submission filename
        model - model to create submission
        validloader - loader for validation dataset
        image_size - size of images for training
        threshold - threshold for submission
        original_size - original image size (1024)
    '''
    submission = {'ImageId': [], 'EncodedPixels': []}

    model.eval()
    torch.cuda.empty_cache()

    im_width = image_size[0]
    im_height = image_size[1]

    for X, fns in validloader:
        X = Variable(X).to(device)
        output = model(X)

        X_flipped = torch.flip(X, dims = (3,))
        output_flipped = torch.flip(model(X_flipped), dims = (3,))

        for i, fname in enumerate(fns):
            mask = torch.sigmoid(output[i].reshape(im_width,im_height)).data.cpu().numpy()
            mask = binary_opening(mask > threshold, disk(2))

            mask_flipped = torch.sigmoid(output_flipped[i].reshape(im_width,im_height)).data.cpu().numpy()
            mask_flipped = binary_opening(mask_flipped > threshold, disk(2))

            im = Image.fromarray(((mask + mask_flipped) / 2 * 255).astype(np.uint8)).resize((original_size,original_size))
            im = np.transpose(np.asarray(im))

            submission['EncodedPixels'].append(mask2rle(im, original_size, original_size))
            submission['ImageId'].append(fname)

    submission_df = pd.DataFrame(submission, columns=['ImageId', 'EncodedPixels'])
    submission_df.loc[submission_df.EncodedPixels=='', 'EncodedPixels'] = '-1'
    submission_df.to_csv(filename, index=False)

# function to calculate IOU to determine threshold
def get_iou(p, t):
    '''
    Function to calculate IOU to determine the best threshold for submission.
    INPUT:
        p - prediction vector
        t - true labels vector
    OUTPUT:
        metric - IOU metric
    '''
    metric = 0.0

    true = np.sum(t)
    pred = np.sum(p)

    # deal with empty mask first
    if true == 0:
        metric = (pred == 0)
        return metric

    # non empty mask case.  Union is never empty
    # hence it is safe to divide by its number of pixels

    intersection = np.sum(t * p)
    union = true + pred - intersection
    iou = intersection / union

    # iou metric is a stepwise approximation of the real iou over 0.5
    iou = np.floor(max(0, (iou - 0.45)*20)) / 10

    metric += iou

    return metric

def determine_threshold(model, device, testloader, image_size):
    '''
    Function to determine the best threshold for submission.
    INPUT:
        device - device used for computations
        testloader - loader for test dataset
        image_size - size of images for training
    OUTPUT:
        thresholds - array of thresholds for submission
        ious - array of IOU for different threshold values
        index_max - index of the best threshold
        threshold_max - the best threshold for submission
    '''
    im_width = image_size[0]
    im_height = image_size[1]
    thresholds = np.linspace(0.2,0.9,20)
    ious = []
    for threshold in thresholds:
        iou = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in testloader:
                for i in range(0, len(inputs)):
                    input = inputs[i]
                    label = labels[i]
                    input, label = input.to(device), label.to(device)
                    output = model.forward(input.reshape(-1, 1, im_width,im_height))
                    mask = binary_opening(torch.sigmoid(output.reshape(im_width,im_height)).data.cpu().numpy() > threshold, disk(2))
                    iou += get_iou(mask.reshape(im_width,im_height), label.reshape(im_width,im_height).float().data.cpu().numpy())
            ious.append(iou / len(testloader))

    index_max = np.argmax(ious)
    threshold_max = thresholds[index_max]

    return thresholds, ious, index_max, threshold_max
