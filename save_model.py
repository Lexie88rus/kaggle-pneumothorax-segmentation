import torch

def save_model(model, model_arch, learning_rate, epochs, train_losses, test_losses, train_metrics, test_metrics, filepath = './models_checkpoints/'):
    """
    Functions to save the model checkpoint.
    """
    filename = filepath + 'model_' + model_arch + '_lr' + str(learning_rate) + '_' + str(epochs) + '.pth'

    print("Saving model to {}\n".format(filename))

    checkpoint = {'model_arch': model_arch,
    'train_losses' : train_losses,
    'test_losses' : test_losses,
    'train_metrics' : train_metrics,
    'test_metrics' : test_metrics,
    'state_dict': model.state_dict()}

    torch.save(checkpoint, filename)
