import torch
import numpy as np

from utils import get_model
from utils.constants import DEFAULT_MODEL_DIR


class VanillaBP:
    def __init__(self, model_name, model_dir=None):
        if not model_dir:
            model_dir = DEFAULT_MODEL_DIR

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = get_model(model_name, model_dir)
        self.model = self.model.eval().to(self.device)

    def convert_to_grayscale(self, grads):
        # assumes input is in NHWC format
        grayscale_im = np.sum(np.abs(grads), axis=3)
        im_max = np.percentile(grayscale_im, 99, axis=(1, 2), keepdims=True)
        im_min = np.min(grayscale_im, axis=(1, 2), keepdims=True)
        grayscale_im = np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1)
        return grayscale_im

    def normalize_gradient_maps(self, grad):
        assert len(grad.shape) == 4
        min_vals = np.min(grad, axis=(1, 2, 3), keepdims=True)
        max_vals = np.percentile(grad, 99, axis=(1, 2, 3), keepdims=True)
        grad = (grad - min_vals) / (max_vals - min_vals)
        grad = np.clip(grad, 0, 1)
        return grad

    def make_one_hot(self, preds):
        _, labels = torch.max(preds, 1)
        labels.unsqueeze_(1)
        one_hot = torch.zeros_like(preds)
        one_hot.scatter_(1, labels, 1)
        return one_hot

    def explain_instances(self, images, one_hot_preds=None, permute=True):
        self.model.zero_grad()

        # make it a batch if just one image is given
        if len(images.shape) == 3:
            images = images.unsqueeze(0)

        images = images.to(self.device)

        # set to track gradients and zero them out
        images.requires_grad_(True)
        if images.grad is not None:
            images.grad.zero_()

        preds = self.model(images)
        if one_hot_preds is None:
            one_hot_preds = self.make_one_hot(preds)

        preds.backward(gradient=one_hot_preds)
        grad = images.grad

        if permute:
            grad = grad.permute(0, 2, 3, 1)

        return grad
