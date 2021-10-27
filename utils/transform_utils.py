from torchvision import transforms
import torchvision.transforms.functional as TF

from .constants import SUPPORTED_TRANSFORMS


def get_args_from_string(string):
    # breaks down a comma-separated, key-value pairs into a dictionary
    # eg. arg1=val1,arg2=val2,arg3=val3 to {"arg1":"val1",...}

    if not string or len(string) == 0:
        return {}

    string = string.strip().split(",")
    string = [x.split("=") for x in string]

    # sanity check
    assert all([len(x) == 2 for x in string])

    return {x[0]: x[1] for x in string}


def get_transformation(transform_name, args_string):
    """
    takes a transform_name and args and returns back a torchvision.transforms
    object.

    Supported Transforms and behaviour:
    1. color_jitter: same args as transforms.ColorJitter
    2. horizontal_flip: no args needed, does transforms.RandomHorizontalFlip(p=1)
    3. vertical_flip: no args needed, does transforms.RandomVerticalFlip(p=1)
    4. rotation: needs a degree argument,
        does transforms.RandomRotation(0,(degree,degree))
    5. translation: see translate arg of transforms.functional.affine,
    6. resize:  TODO: have to figure this out, if we need this
    """

    if transform_name not in SUPPORTED_TRANSFORMS:
        raise ValueError(
            "Unknown transformation, must be one of {}".format(SUPPORTED_TRANSFORMS)
        )

    args = get_args_from_string(args_string)

    if transform_name == "color_jitter":  # TODO
        raise NotImplementedError
    if transform_name == "horizontal_flip":
        ret_transform = transforms.RandomHorizontalFlip(p=1)
    if transform_name == "vertical_flip":
        ret_transform = transforms.RandomVerticalFlip(p=1)
    if transform_name == "rotation":
        degree = float(args["degree"])
        ret_transform = transforms.RandomRotation((degree, degree))
    if transform_name == "translation":

        def trans_method(img):
            x = int(float(args.get("x", 0)) * img.size[0])
            y = int(float(args.get("y", 0)) * img.size[1])
            return TF.affine(img, angle=0, scale=1, shear=0, translate=[x, y])

        ret_transform = transforms.Lambda(trans_method)
    if transform_name == "resize":
        raise NotImplementedError

    return ret_transform
