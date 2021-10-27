import numpy as np
import cv2
import torch


def convert_to_gray_scale(attributions):
    """

    Parameters
    ----------
    attributions :
        integrated gradients
        

    Returns
    -------
        Gray scale image of intergrated gradients
    """
    return attributions.mean(dim=1)


def linear_transform(attributions, clip_above_percentile=99.9, clip_below_percentile=70.0, low=0.2):
    """
    Transform the attributions by a linear function.
  Transform the attributions so that the specified percentage of top attribution
  values are mapped to a linear space between `low` and 1.0.

    Parameters
    ----------
    attributions :
        Resulting integrated gradients
    clip_above_percentile :
         (Default value = 99.9)
    clip_below_percentile :
         (Default value = 70.0)
    low :
        The low end of the linear space.
         (Default value = 0.2)

    Returns
    -------

    """
    sorted_attributions, percentiles = get_percentiles(attributions)
    m = compute_threshold_by_top_percentage(sorted_attributions, percentiles, percentage=100 - clip_above_percentile)[:,
        None, None]
    e = compute_threshold_by_top_percentage(sorted_attributions, percentiles, percentage=100 - clip_below_percentile)[:,
        None, None]
    print(m.shape, e.shape)
    transformed = (1 - low) * (torch.abs(attributions) - e) / (m - e) + low
    transformed *= torch.sign(attributions)
    transformed *= (transformed >= low).float()
    transformed = torch.clamp(transformed, 0.0, 1.0)
    return transformed


def get_percentiles(attributions):
    """
    Computes percentiles of gradients in a batch

    Parameters
    ----------
    attributions :
        Resulting integrated gradients

    Returns
    -------
    sorted_attributions:
        gradient values sorted in descending order for every image
    percentiles:
        percentile of every value in sorted_attributions
    """
    reshaped_attr = attributions.view(attributions.shape[0], -1)
    attribution_sum = reshaped_attr.sum(1)
    sorted_attributions = torch.sort(torch.abs(reshaped_attr), descending=True, dim=1)[0]
    percentiles = 100.0 * torch.cumsum(sorted_attributions, dim=1) / attribution_sum[:, None]
    return sorted_attributions, percentiles


def compute_threshold_by_top_percentage(sorted_attributions, percentiles, percentage=60):
    """
    Compute the threshold value that maps to the top percentage of values.
      This function takes the cumulative sum of attributions and computes the set
      of top attributions that contribute to the given percentage of the total sum.
      The lowest value of this given set is returned.
    Parameters
    ----------
    sorted_attributions :
        gradient values sorted in descending order for every image
    percentiles :
        percentile of every value in sorted_attributions
    percentage :
         Specified percentage by which to threshold. (Default value = 60)

    Returns
    -------

    """
    if percentage < 0 or percentage > 100:
        raise ValueError('percentage must be in [0, 100]')
    if percentage == 100:
        return sorted_attributions[:, -1]
    threshold_idxs = [torch.nonzero(percentiles[sample] >= percentage)[0][0].item() for sample in
                      range(percentiles.shape[0])]
    threshold = sorted_attributions[range(sorted_attributions.shape[0]), threshold_idxs]
    print(threshold_idxs, sorted_attributions.shape, threshold.shape)

    return threshold


def polarity_function(attributions, polarity):
    """

    Parameters
    ----------
    attributions :
        
    polarity :
        

    Returns
    -------

    """
    if polarity == 'positive':
        return torch.clamp(attributions, 0, 1)
    elif polarity == 'negative':
        return torch.clamp(attributions, -1, 0)
    else:
        raise NotImplementedError


def overlay_function(attributions, image):
    """

    Parameters
    ----------
    attributions :
        
    image :
        

    Returns
    -------

    """
    return torch.clamp(0.7 * image + 0.5 * attributions, 0, 1)


def visualize(attributions, image, polarity='positive',
              clip_above_percentile=99.9, clip_below_percentile=0, overlay=True):
    """
    Postprocess the images obtained by the integrated gradient method
    Parameters
    ----------
    attributions :
        Batch of integrated gradients
    image :
        Batch of generated images for the integrated gradients
    polarity :
         (Default value = 'positive')
    clip_above_percentile :
         (Default value = 99.9)
    clip_below_percentile :
         (Default value = 0)
    overlay :
         (Default value = True)
         If true overlay IG and original images

    Returns
    -------
        Post-processed IG results
    """
    if polarity == 'both':
        raise NotImplementedError

    elif polarity == 'positive':
        attributions = polarity_function(attributions, polarity=polarity)
    elif polarity == 'negative':
        attributions = polarity(attributions, polarity=polarity)
        attributions = torch.abs(attributions)
    else:
        raise RuntimeError

    # convert the attributions to the gray scale
    attributions = convert_to_gray_scale(attributions)
    # Map to 0-1 interval
    attributions = linear_transform(attributions, clip_above_percentile, clip_below_percentile, 0.0)
    if overlay:
        attributions = overlay_function(attributions, image)

    return attributions
