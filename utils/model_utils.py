import torch
from .constants import SUPPORTED_MODELS
from .constants import DEFAULT_MODEL_DIR

# TODO: have separate methods for cifar-10 and imagenet
# unify them under get model


def get_model(model_name, model_dir=None):
    # downloads model from torch-hub, sets to eval only and returns the model
    if not model_dir:
        model_dir = DEFAULT_MODEL_DIR
    torch.hub.set_dir(model_dir)
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            "unknonw model name: '{}', should be one of {}".format(
                model_name, SUPPORTED_MODELS
            )
        )

    model = torch.hub.load("pytorch/vision", model_name, pretrained=True)
    model.eval()

    return model

def en_cuda(tensor,cuda):
    if cuda:
        return tensor.cuda()
    else:
        return tensor.cpu()