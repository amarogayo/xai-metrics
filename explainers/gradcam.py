import torch
import numpy as np
import matplotlib.cm as cm

from .aux_gradcam.classes import GradCAM, BackPropagation

from utils import get_imagenet_image_preproc_pipeline
from utils import get_imagenet_tensor_preproc_pipeline
from utils import get_model
from utils.constants import TARGET_LAYER


class GRADCAMExplainer:
    def __init__(self, model_name, model_dir=None):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = get_model(model_name, model_dir)
        self.model = self.model.eval()
        self.model = self.model.to(self.device)

        self.target_layer = TARGET_LAYER[model_name]

        self.bp = BackPropagation(model=self.model)
        self.grad_cam = GradCAM(model=self.model, candidate_layers=self.target_layer)

        self.tensor_preproc = get_imagenet_tensor_preproc_pipeline()
        self.image_preproc = get_imagenet_image_preproc_pipeline(model_name)

    def overlap_heatmap(self, image, heatmap, paper_cmap=False):
        image = np.asarray(self.image_preproc(image))
        overlapped_heatmap = None
        cmap = cm.jet_r(heatmap)[..., :3] * 255.0
        if paper_cmap:
            alpha = heatmap[..., None]
            overlapped_heatmap = alpha * cmap + (1 - alpha) * image
        else:
            overlapped_heatmap = (cmap.astype(np.float) + image.astype(np.float)) / 2
        return overlapped_heatmap

    def batch_images(self, images):
        images = [self.image_preproc(i) for i in images]
        batch = torch.stack(tuple(self.tensor_preproc(i) for i in images), dim=0)
        batch = batch.to(self.device)
        return batch

    def batch_predict(self, batch):
        with torch.no_grad():
            logits = self.model(batch)
            probs = torch.nn.functional.softmax(logits, dim=1)
            probs, labels = probs.sort(dim=1, descending=True)
            probs = probs.detach().cpu().numpy()
            labels = labels[:, [0]].detach().cpu().numpy()
            return probs, labels

    def explain_instances(self, images, labels=None, **kwargs):
        self.model.zero_grad()

        if type(images) == list:
            images = self.batch_images(images)
        images = images.to(self.device)

        if labels is None:
            probs, labels = self.bp.forward(images)
            labels = labels[:, [0]]

        _ = self.grad_cam.forward(images)

        # print(labels)
        self.grad_cam.backward(ids=labels)
        regions = self.grad_cam.generate(target_layer=self.target_layer)

        # return regions[:, 0].detach().cpu().numpy()
        return regions[:, 0].detach().cpu()
