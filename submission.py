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

# import tta utilities
from tta import get_prediction_with_tta

def make_submission(filename, device, model, validloader, image_size, channels, threshold = 0.9, original_size = 1024, tta = False):
    '''
    Function to create submission.csv file.
    INPUT:
        filename - submission filename
        model - model to create submission
        validloader - loader for validation dataset
        image_size - size of images for training
        channels - number of channels in training images
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

        if tta:
            for i, fname in enumerate(fns):
                mask = torch.sigmoid(output[i].reshape(im_width,im_height)).data.cpu().numpy()
                mask = binary_opening(mask > threshold, disk(2))

                mask_flipped = torch.sigmoid(output_flipped[i].reshape(im_width,im_height)).data.cpu().numpy()
                mask_flipped = binary_opening(mask_flipped > threshold, disk(2))

                mask_tta = get_prediction_with_tta(model, X[i], device, img_size = (im_width,im_height), channels = channels)
                im = Image.fromarray(((mask + mask_flipped + mask_tta) / 3 * 255).astype(np.uint8)).resize((original_size,original_size))

                im = np.transpose(np.asarray(im))

                labels = label(im)
                encodings = [mask2rle(labels == k, original_size, original_size) for k in np.unique(labels[labels > 0])]
                if len(encodings) > 0:
                    for encoding in encodings:
                        submission['ImageId'].append(fname)
                        submission['EncodedPixels'].append(encoding)
                else:
                    submission['ImageId'].append(fname)
                    submission['EncodedPixels'].append('-1')
        else:
            for i, fname in enumerate(fns):
                mask = torch.sigmoid(output[i].reshape(im_width,im_height)).data.cpu().numpy()
                mask = binary_opening(mask > threshold, disk(2))

                im = Image.fromarray((mask * 255).astype(np.uint8)).resize((original_size,original_size))

                im = np.transpose(np.asarray(im))

                labels = label(im)
                encodings = [mask2rle(labels == k, original_size, original_size) for k in np.unique(labels[labels > 0])]
                if len(encodings) > 0:
                    for encoding in encodings:
                        submission['ImageId'].append(fname)
                        submission['EncodedPixels'].append(encoding)
                else:
                    submission['ImageId'].append(fname)
                    submission['EncodedPixels'].append('-1')

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

# src: https://www.kaggle.com/aglotero/another-iou-metric
def iou_metric(y_pred_in, y_true_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in

    true_objects = 2
    pred_objects = 2

    intersection = np.histogram2d(labels, y_pred, bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)

    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)

def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)

def determine_threshold(model, device, testloader, image_size, channels):
    '''
    Function to determine the best threshold for submission.
    INPUT:
        device - device used for computations
        testloader - loader for test dataset
        image_size - size of images for training
        channels - the number of channels
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
                    output = model.forward(input.reshape(-1, channels, im_width,im_height))
                    mask = binary_opening(torch.sigmoid(output.reshape(im_width,im_height)).data.cpu().numpy() > threshold, disk(2))
                    # add to metric only if pneumothorax if found so empty images don't add to metric
                    if np.sum(mask) > 0:
                        iou += get_iou(mask.reshape(im_width,im_height), label.reshape(im_width,im_height).float().data.cpu().numpy())
            ious.append(iou / len(testloader))

    index_max = np.argmax(ious)
    threshold_max = thresholds[index_max]

    return thresholds, ious, index_max, threshold_max
