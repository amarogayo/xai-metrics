SUPPORTED_IMAGENET_MODELS = ["inception_v3", "alexnet", "vgg19", "resnet101"]
SUPPORTED_CIFAR_10_MODELS = []

SUPPORTED_MODELS = SUPPORTED_IMAGENET_MODELS + SUPPORTED_CIFAR_10_MODELS

DEFAULT_IMAGENET_DIR = "/dataP/gst/imagenet"
DEFAULT_CIFAR_10_DIR = "/dataP/data/deeplearning/image_classification/cifar10"

DEFAULT_MODEL_DIR = "/dataP/gst/pytorch_pretrained_models"

SUPPORTED_TRANSFORMS = [
    "color_jitter",
    "horizontal_flip",
    "vertical_flip",
    "rotation",
    "translation",
    "resize",
]

SUPPORTED_EXPLAINERS = ["lime", "gradcam", "smoothgrad","integrated_gradient"]

TARGET_LAYER = {
    "inception_v3": "Mixed_7c",
    "resnet101": "layer4",
    "alexnet": "features.10",
    "vgg19": "features.36",
}
