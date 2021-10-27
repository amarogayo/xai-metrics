import explainers
from .constants import SUPPORTED_MODELS
from .constants import SUPPORTED_EXPLAINERS

import torch
import numpy as np

def get_explainer(exp_name, model_name, model_dir):
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            "unknown model: '{}', should be one of {}".format(
                model_name, SUPPORTED_MODELS
            )
        )

    if exp_name not in SUPPORTED_EXPLAINERS:
        raise ValueError(
            "unknown explainer: '{}', should be one of {}".format(
                exp_name, SUPPORTED_EXPLAINERS
            )
        )

    if exp_name == "lime":
        exp = explainers.LIMEExplainer(model_name, model_dir)
    if exp_name == "gradcam":
        exp = explainers.GRADCAMExplainer(model_name, model_dir)
    if exp_name == "smoothgrad":
        exp = explainers.SmoothGradExplainer(model_name, model_dir)
    if exp_name == "integrated_gradient":
        exp = explainers.IntegratedGradientsExplainer(model_name, model_dir)
    return exp


def convert_to_grayscale(grads):
    # assumes input is in NHWC format
    grayscale_im = np.sum(np.abs(grads), axis=3)
    im_max = np.percentile(grayscale_im, 99, axis=(1, 2), keepdims=True)
    im_min = np.min(grayscale_im, axis=(1, 2), keepdims=True)
    grayscale_im = np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1)
    return grayscale_im
