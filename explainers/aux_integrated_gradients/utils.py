import torch
import torch.nn.functional as F
import cv2
import numpy as np


def calculate_outputs_and_gradients(inputs, model, target_labels_idx, device='cpu'):
    """

    Parameters
    ----------
    inputs :
        
    model :
        
    target_labels_idx :
        
    device :
         (Default value = 'cpu')

    Returns
    -------

    """
    old_shape = inputs.shape
    inputs = inputs.view(-1, old_shape[2], old_shape[3], old_shape[4])
    inputs.requires_grad = True
    outputs = model(inputs)
    outputs = F.softmax(outputs, dim=1)
    outputs = outputs.view(old_shape[0], old_shape[1], -1)
    if target_labels_idx is None:
        target_labels_idx = outputs[-1].argmax(dim=1).detach()
    target_labels_idx = target_labels_idx[None, :].repeat(old_shape[0], 1)[:, :, None]

    outputs = outputs.gather(2, target_labels_idx)
    outputs = outputs.reshape(-1, outputs.shape[-1])
    outputs = outputs.split(1, dim=0)
    gradients = torch.autograd.grad(outputs, inputs)
    gradients = gradients[0].detach().reshape(old_shape)
    

    return gradients, target_labels_idx
