# code to run explainers on ImageNet
import numpy as np
import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm

import argparse
import pickle as pkl

import os
from utils.constants import SUPPORTED_MODELS

from utils.constants import SUPPORTED_TRANSFORMS
from utils.constants import SUPPORTED_EXPLAINERS
from utils.constants import DEFAULT_IMAGENET_DIR
from utils.constants import DEFAULT_MODEL_DIR

from utils import get_data_loader

from utils import get_transformation
from utils import get_explainer
from utils import get_args_from_string
from utils import get_imagenet_image_preproc_pipeline

from utils import maybe_create_dir


# flags
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, help="batch size", default=100)
parser.add_argument(
    "--num_workers", type=int, help="how many data loader workers to use", default=50
)  # unused
parser.add_argument(
    "--start_idx", type=int, default=0, help="which subset of images to use"
)
parser.add_argument(
    "--end_idx", type=int, default=50000, help="which subset of images to use"
)

parser.add_argument(
    "--model",
    type=str,
    help="which model to use",
    default="inception_v3",
    choices=SUPPORTED_MODELS,
)

parser.add_argument(
    "--data_dir",
    type=str,
    help="""where the input data is stored,
            uses the constants in utils.constant if none given""",
    default=DEFAULT_IMAGENET_DIR,
)
parser.add_argument(
    "--output_dir",
    type=str,
    help="where to store the output heatmaps",
    default="/dataT/gst/exp_results",
)
parser.add_argument(
    "--model_dir",
    type=str,
    help="where to load models from ",
    default=DEFAULT_MODEL_DIR,
)
parser.add_argument(
    "--explainer",
    type=str,
    help="which explainer to use",
    default="lime",
    choices=SUPPORTED_EXPLAINERS,
)
parser.add_argument(  # unused RN
    "--exp_args",
    type=str,
    help="arguments to pass to the explainer, comma-separted arg1=val1,arg2=val2...",
    default="",
)
parser.add_argument(
    "--transform",
    type=str,
    help="what transforms to perform, default None",
    default=None,
    choices=SUPPORTED_TRANSFORMS,
)
parser.add_argument(
    "--transform_args",
    type=str,
    help="arguments to pass to transforms, see utils.get_transformation",
    default=None,
)


if __name__ == "__main__":
    args = parser.parse_args()
    if not args.transform:
        op_dir = os.path.join(args.output_dir, args.explainer, args.model, "normal")
    else:
        op_dir = os.path.join(
            args.output_dir,
            args.explainer,
            args.model,
            args.transform,
            args.transform_args,
        )
    maybe_create_dir(op_dir)

    batch_size = args.batch_size  # unused
    exp_args = get_args_from_string(args.exp_args)

    # get the explainer object
    explainer = get_explainer(args.explainer, args.model, args.model_dir)

    if args.explainer == "lime":  # LIME only takes one by one :(
        batch_size = 1

        # get the image preproc
        img_preproc = get_imagenet_image_preproc_pipeline(args.model)
        if args.transform:
            transform = get_transformation(args.transform, args.transform_args)
            img_preproc = transforms.Compose([img_preproc, transform])

        # get the image dataset object
        data = torchvision.datasets.ImageNet(
            root=args.data_dir, split="val", transform=img_preproc
        )
        data = torch.utils.data.Subset(data, range(args.start_idx, args.end_idx))

        # run the for loop over the images, run exp and store the result
        # NOTE: It is possible that job gets killed dumping to disk, leaving a
        # corrupted pickle, run a post processing step to check for these
        for i, (image, label) in tqdm(
            enumerate(data, start=args.start_idx), disable=None
        ):

            op_file = os.path.join(op_dir, "{}.pkl".format(i))
            if not os.path.exists(op_file):
                exp = explainer.explain_instance(image, **exp_args)
                pkl.dump(exp, open(op_file, "wb"))

    else:
        # hopefully we can feed as batches to the other methods
        if args.transform:
            transform = get_transformation(args.transform, args.transform_args)
        else:
            transform = None

        data_loader = get_data_loader(
            dataset="imagenet",
            model_name=args.model,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            transform=transform,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
        )

        all_explanations = []

        for images, labels in tqdm(data_loader, disable=None):
            explanations = explainer.explain_instances(images)
            all_explanations.append(explanations.detach().cpu().numpy())

        all_explanations = np.vstack(all_explanations)
        # TODO: ensure you store * only * the raw results, can do postproc later
        for i in range(0, len(all_explanations), 200):
            op_file = os.path.join(op_dir, "{}.pkl".format(i + args.start_idx))
            cur_explanation = all_explanations[i : i + 200]
            pkl.dump(cur_explanation, open(op_file, "wb"))

    print("all done!")
