import numpy as np
import torch
from utils.model_utils import en_cuda


# integrated gradients
def integrated_gradients(inputs, model, target_labels_idx, predict_and_gradients, baseline, steps=50, device='cpu'):
    """
    Compute the integrated gradients for a batch of images
    Parameters
    ----------
    inputs :
        Images to process
        
    model :
        Model object
        
    target_labels_idx :
        Images target labels
    predict_and_gradients :
        Function to generate gradients
    baseline :
        Baseline for gradient images computations. If None, the latter is set to 0
    steps :
         Number of steps used to compute the integral of gradients(Default value = 50)
    device :
         Device on which operations will be done(Default value = 'cpu')

    Returns
    -------
        Batch of integrated gradient images with the same shape as the input images
    """
    if baseline is None:
        baseline = 0 * inputs
        # scale inputs and compute gradients
    baseline_expanded = baseline[None, :, :, :, :]
    scale = torch.linspace(0, 1, steps=steps + 1).to(device)
    # generate multiple images with different intensity multipliers for computing the integral of gradient

    scaled_inputs = torch.unsqueeze(inputs, 0).repeat(steps + 1, 1, 1, 1, 1)
    scaled_inputs = scale[:, None, None, None, None] * (scaled_inputs - baseline_expanded) + baseline_expanded
    del scale
    grads, target_labels_idx = predict_and_gradients(scaled_inputs, model, target_labels_idx, device)
    del scaled_inputs
    avg_grads = grads.mean(0) - (grads[0] + grads[-1])/(2*grads.shape[0])
    del grads
    integrated_grads = (inputs - baseline) * avg_grads

    return integrated_grads


def random_baseline_integrated_gradients(inputs, model, target_labels_idx,
                                         predict_and_gradients, steps, num_random_trials,
                                         device='cpu'):
    """
    Repeats the integrated gradient n times with random baselines and averages the result
    Parameters
    ----------
    inputs :
         Images to process

    model :
        Model object

    target_labels_idx :
        Images target labels
    predict_and_gradients :
        Function to generate gradients
    steps :
         Number of steps used to compute the integral of gradients(Default value = 50)
    device :
         Device on which operations will be done(Default value = 'cpu')
     num_random_trials :
        Number of integrated gradient trials

    Returns
    -------
        Batch of integrated gradient images with the same shape as the input image
    """
    all_intgrads = []
    for i in range(num_random_trials):
        integrated_grad = integrated_gradients(inputs, model, target_labels_idx, predict_and_gradients,
                                               baseline=255.0 * np.random.random(inputs.shape), steps=steps,
                                               device=device)
        all_intgrads.append(integrated_grad)
    avg_intgrads = torch.stack(all_intgrads).mean(dim=0)
    return avg_intgrads
