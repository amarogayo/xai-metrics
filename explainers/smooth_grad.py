import torch
from tqdm import tqdm
from .backprop import VanillaBP


class SmoothGradExplainer(VanillaBP):
    def __init__(self, model_name, model_dir=None, n_samples=50, sigma_multiplier=4):
        super(SmoothGradExplainer, self).__init__(
            model_name=model_name, model_dir=model_dir
        )

        self.n_samples = n_samples
        self.sigma_multiplier = sigma_multiplier

    def explain_instances(
        self, images, n_samples=None, sigma_multiplier=None, permute=True
    ):
        if not n_samples:
            n_samples = self.n_samples
        n_samples = int(n_samples)

        if not sigma_multiplier:
            sigma_multiplier = self.sigma_multiplier

        images = images.to(self.device)
        # make it a batch if just one image is given
        if len(images.shape) == 3:
            images = images.unsqueeze(0)

        clean_preds = self.model(images)
        one_hot_preds = self.make_one_hot(clean_preds)

        mins = (
            images.view(images.shape[0], -1)
            .min(dim=1)[0]
            .view(images.shape[0], 1, 1, 1)
        )
        maxes = (
            images.view(images.shape[0], -1)
            .max(dim=1)[0]
            .view(images.shape[0], 1, 1, 1)
        )

        sigma = sigma_multiplier / (maxes - mins)

        grads = torch.zeros_like(images)

        for _ in tqdm(range(n_samples), disable=None):
            images_with_noise = images + (torch.rand_like(images) * sigma)
            images_with_noise = images_with_noise.to(self.device)
            cur_grad = super(SmoothGradExplainer, self).explain_instances(
                images_with_noise, one_hot_preds, permute=False  # do it here later
            )
            grads.add_(cur_grad)

        grads.div_(n_samples)
        grads = grads.permute(0, 2, 3, 1)
        return grads
