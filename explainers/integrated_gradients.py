import torch
import numpy as np
import matplotlib.cm as cm

import torch.nn.functional as F
from .aux_integrated_gradients.utils import calculate_outputs_and_gradients
from .aux_integrated_gradients.integrated_gradients import random_baseline_integrated_gradients
from .aux_integrated_gradients.integrated_gradients import integrated_gradients
from .aux_integrated_gradients.visualization import visualize
from utils import get_imagenet_image_preproc_pipeline
from utils import get_imagenet_tensor_preproc_pipeline
from utils import get_model
from utils.constants import TARGET_LAYER


class IntegratedGradientsExplainer:
    """
    """

    def __init__(self, model_name, model_dir=None):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = get_model(model_name, model_dir=model_dir)
        self.model = self.model.eval()
        self.model = self.model.to(self.device)

        self.target_layer = TARGET_LAYER[model_name]

        self.tensor_preproc = get_imagenet_tensor_preproc_pipeline()
        self.image_preproc = get_imagenet_image_preproc_pipeline(model_name)

    def explain_instances(self, image, label=None,
                          **kwargs):
        """

        Parameters
        ----------
        image :
            Batch of images to explain
        label :
             (Default value = None)
             Corresponding labels
        **kwargs :
            See docstring  for 'explain_batch' function

        Returns
        -------

        """
        # batch = self.batch_images(images)
        return self.explain_batch(image, label, self.model, **kwargs)

    def explain_batch(self, imgs, labels, model,
                      random_integrated_grad=False,
                      steps=20,
                      nb_random_trials=10,
                      baseline=None,
                      postprocess=False,
                      polarity='positive',
                      **kwargs):
        """

        Parameters
        ----------
        imgs :
            Batch of images to explain
        labels :
            Corresponding labels
        model :
            Model to explain
        random_integrated_grad :
             (Default value = False)
             If True random integrated gradient method is used to compute the gradients
        steps :
             (Default value = 50)
        nb_random_trials :
             (Default value = 10)
             Parameter only considered when random_integrated_grad=True
        baseline :
            (Default value = None)
            Parameter only considered when random_integrated_grad=False
            Baseline for integrated gradient computation. If None the latter is initialized to zeros
        postprocess :
             (Default value = False)
             Specifies if the resulting IGs should be postprocessed with the custom Visualisation function
        polarity :
             (Default value = 'positive')
             Only considered when postprocess=True
        **kwargs :


        Returns
        -------
            Integrated gradients for the input images
        """

        # image processing
        imgs = imgs.type(torch.float)

        if type(imgs) == list:
            imgs = self.batch_images(imgs)
        imgs = imgs.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)

        # calculate the gradient and the label index

        if random_integrated_grad:
            attributions = random_baseline_integrated_gradients(
                imgs,
                model,
                labels,
                calculate_outputs_and_gradients,
                steps=steps,
                num_random_trials=nb_random_trials,
                device=self.device
            )
        else:
            attributions = integrated_gradients(imgs,
                                                model,
                                                labels,
                                                calculate_outputs_and_gradients,
                                                baseline,
                                                steps=steps,
                                                device=self.device)

        if postprocess:
            img_integrated_gradient = visualize(
                attributions,
                imgs,
                clip_above_percentile=99,
                clip_below_percentile=0,
                overlay=False,
                polarity=polarity
            )
            ret = img_integrated_gradient
        else:
            ret = attributions.permute(0, 2, 3, 1)
        return ret.data.cpu()

    def batch_images(self, images):
        """
            Transform a list of images to a batch
        Parameters
        ----------
        images :
            list of images

        Returns
        -------
            torch batch of images
        """
        images = [self.image_preproc(i) for i in images]
        batch = torch.stack(tuple(self.tensor_preproc(i) for i in images), dim=0)
        batch = batch.to(self.device)
        return batch
