import torch
import numpy as np

from lime import lime_image

from utils import get_imagenet_tensor_preproc_pipeline
from utils import get_model


class LIMEExplainer:
    def __init__(self, model_name, model_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = get_model(model_name, model_dir)
        self.model = self.model.eval()
        self.model = self.model.to(self.device)

        self.tensor_preproc = get_imagenet_tensor_preproc_pipeline()

        self.explainer = lime_image.LimeImageExplainer()

    def batch_predict(self, images):
        with torch.no_grad():
            batch = torch.stack(tuple(self.tensor_preproc(i) for i in images), dim=0)
            batch = batch.to(self.device)
            logits = self.model(batch)
            probs = torch.nn.functional.softmax(logits, dim=1)
            return probs.detach().cpu().numpy()

    def explain_instance(self, image, **kwargs):
        image = np.array(image)
        explanation = self.explainer.explain_instance(image, self.batch_predict, kwargs)
        return explanation
