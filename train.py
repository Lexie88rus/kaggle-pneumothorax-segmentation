'''
Script to perform training.
'''
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torchcontrib.optim import SWA

from save_model import save_model

def train(model, device, trainloader, testloader, optimizer, criterion, metric, epochs, learning_rate, swa = True, enable_scheduler = True, model_arch = ''):
    '''
    Function to perform model training.
    '''
    model.to(device)
    steps = 0
    running_loss = 0
    running_metric = 0
    print_every = 100

    train_losses = []
    test_losses = []
    train_metrics = []
    test_metrics = []

    if swa:
        # initialize stochastic weight averaging
        opt = SWA(optimizer)
    else:
        opt = optimizer

    # learning rate cosine annealing
    if enable_scheduler:
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, len(trainloader), eta_min=0.0000001)

    for epoch in range(epochs):

        if enable_scheduler:
            scheduler.step()

        for inputs, labels in trainloader:

            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            opt.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            opt.step()

            running_loss += loss
            running_metric += metric(outputs, labels.float())

            if steps % print_every == 0:
                test_loss = 0
                test_metric = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model.forward(inputs)

                        test_loss += criterion(outputs, labels.float())

                        test_metric += metric(outputs, labels.float())

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(testloader):.3f}.. "
                      f"Train metric: {running_metric/print_every:.3f}.. "
                      f"Test metric: {test_metric/len(testloader):.3f}.. ")

                train_losses.append(running_loss/print_every)
                test_losses.append(test_loss/len(testloader))
                train_metrics.append(running_metric/print_every)
                test_metrics.append(test_metric/len(testloader))

                running_loss = 0
                running_metric = 0

                model.train()
                if swa:
                    opt.update_swa()

        save_model(model, model_arch, learning_rate, epochs, train_losses, test_losses, train_metrics, test_metrics, filepath = 'models_checkpoints')

    if swa:
        opt.swap_swa_sgd()

    return model, train_losses, test_losses, train_metrics, test_metrics
