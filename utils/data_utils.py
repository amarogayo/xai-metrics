import torch
import torchvision
from torchvision import transforms

from .constants import SUPPORTED_IMAGENET_MODELS
from .constants import SUPPORTED_CIFAR_10_MODELS

from .constants import DEFAULT_IMAGENET_DIR
from .constants import DEFAULT_CIFAR_10_DIR


def get_data_loader(
    dataset,
    model_name,
    data_dir=None,
    split="val",
    batch_size=1,
    shuffle=False,
    num_workers=1,
    transform=None,
    start_idx=0,
    end_idx=50000,
):

    if dataset not in ["cifar-10", "imagenet"]:
        raise ValueError("unknown dataset, should be cifar-10 or imagenet")

    if dataset == "imagenet":
        preprocess = get_imagenet_preprocessing_pipeline(model_name, transform)
        if not data_dir:
            data_dir = DEFAULT_IMAGENET_DIR

        data = torchvision.datasets.ImageNet(
            data_dir, split=split, transform=preprocess
        )

    elif dataset == "cifar-10":  # TODO
        if not data_dir:
            data_dir = DEFAULT_CIFAR_10_DIR
        raise NotImplementedError

    data = torch.utils.data.Subset(data, range(start_idx, end_idx))
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    return data_loader


def get_imagenet_preprocessing_pipeline(model_name, transform=None):
    if model_name not in SUPPORTED_IMAGENET_MODELS:
        raise ValueError(
            "unknown model, should be one of {}".format(
                model_name, SUPPORTED_IMAGENET_MODELS
            )
        )
    image_ops = get_imagenet_image_preproc_pipeline(model_name, transform)
    tensor_ops = get_imagenet_tensor_preproc_pipeline()

    ops = transforms.Compose([image_ops, tensor_ops])
    return ops


def get_imagenet_image_preproc_pipeline(model_name, transform=None):
    if model_name not in SUPPORTED_IMAGENET_MODELS:
        raise ValueError(
            "unknown model: '{}', should be one of {}".format(
                model_name, SUPPORTED_IMAGENET_MODELS
            )
        )
    if model_name == "inception_v3":
        ops = [transforms.Resize(299), transforms.CenterCrop(299)]
    else:  # same for vgg19, alexnet and resnet
        ops = [transforms.Resize(256), transforms.CenterCrop(224)]

    if transform:
        ops.append(transform)

    return transforms.Compose(ops)


def get_imagenet_tensor_preproc_pipeline():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_cifar_10_preprocessing_pipeline(model_name):  # TODO
    raise NotImplementedError
