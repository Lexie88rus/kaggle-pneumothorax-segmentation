'''
Script containing custom losses for segmentation.
Source:
https://github.com/4uiiurz1/pytorch-nested-unet/blob/master/metrics.py
'''
import numpy as np
import torch
from torch.nn import functional as F
from torch import nn

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice

def weighted_BCE(logit_pixel, truth_pixel):
    logit = logit_pixel.view(-1)
    truth = truth_pixel.view(-1)
    assert(logit.shape==truth.shape)

    loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')
    if 0:
        loss = loss.mean()
    if 1:
        pos = (truth>0.5).float()
        neg = (truth<0.5).float()
        pos_weight = pos.sum().item() + 1e-12
        neg_weight = neg.sum().item() + 1e-12
        loss = (0.25*pos*loss/pos_weight + 0.75*neg*loss/neg_weight).sum()

    return loss

def weighted_soft_dice_criterion(logit_pixel, truth_pixel):
    batch_size = len(logit_pixel)
    logit = logit_pixel.view(batch_size,-1)
    truth = truth_pixel.view(batch_size,-1)
    assert(logit.shape==truth.shape)

    loss = soft_dice_criterion(logit, truth)

    loss = loss.mean()
    return loss

def soft_dice_criterion(logit, truth, weight=[0.2,0.8]):
    batch_size = len(logit)
    probability = torch.sigmoid(logit)

    p = probability.view(batch_size,-1)
    t = truth.view(batch_size,-1)
    w = truth.view(batch_size,-1).detach()
    w = w*(weight[1]-weight[0])+weight[0]

    p = w*(p*2-1)  #convert to [0,1] --> [-1, 1]
    t = w*(t*2-1)

    intersection = (p * t).sum(-1)
    union =  (p * p).sum(-1) + (t * t).sum(-1)
    dice  = 1 - 2*intersection/union

    loss = dice
    return loss

class WeightedBCEDiceLoss(nn.Module):
    def __init__(self):
        super(WeightedBCEDiceLoss, self).__init__()

    def forward(self, input, target):
        bce = weighted_BCE(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice
