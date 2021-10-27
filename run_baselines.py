# code to run baseline tests on ImageNet and CIFAR-10
# here baselines are run on both normal images as well as
# transformed images (rotation, flip, scale, translation)

import torch

from utils.constants import SUPPORTED_MODELS
from utils.constants import SUPPORTED_TRANSFORMS

from utils import get_data_loader
from utils import get_model
from utils import get_transformation

from utils import maybe_create_dir
from utils import accuracy

import argparse
import pickle as pkl

import os

# flags
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, help="batch size", default=100)
parser.add_argument(
    "--num_workers", type=int, help="how many data loader workers to use", default=30
)

parser.add_argument(
    "--dataset",
    type=str,
    help="which dataset to use",
    default="imagenet",
    choices=["imagenet", "cifar-10"],
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
    default=None,
)
parser.add_argument(
    "--output_dir",
    type=str,
    help="where to store the output heatmaps",
    default="results",
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
    maybe_create_dir(args.output_dir)

    res_file_name = "{}_{}_{}_{}_res.pkl".format(
        args.dataset, args.model, args.transform, args.transform_args
    )
    res_file_name = os.path.join(args.output_dir, res_file_name)

    # get data loader
    if args.transform:
        transform = get_transformation(args.transform, args.transform_args)
    else:
        transform = None

    data_loader = get_data_loader(
        dataset=args.dataset,
        model_name=args.model,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transform=transform,
    )

    # Get the model and put on GPU
    model = get_model(args.model)
    model = model.to("cuda")
    model = model.eval()

    all_logits = []
    all_labels = []

    # iterate through the data loader and collect results
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images = images.to("cuda")
            logits = model(images)

            all_logits.append(logits)
            all_labels.append(labels)

    # concatenate them and compute the metrics
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    all_logits = all_logits.to("cuda")
    all_labels = all_labels.to("cuda")

    all_probs = torch.nn.functional.softmax(all_logits, dim=1)
    all_preds = torch.max(all_probs, 1)[1]

    # acc@1
    # matches = all_preds == all_labels
    # acc_at_1 = torch.mean(matches.type(torch.FloatTensor))

    acc_at_1, acc_at_5 = accuracy(all_probs, all_labels, (1, 5))
    acc_at_1 = acc_at_1.item()
    acc_at_5 = acc_at_5.item()

    # put in pickle and dump
    all_preds = all_preds.to("cpu")
    all_probs = all_probs.to("cpu")

    results = {}
    results["probs"] = all_probs
    results["acc@1"] = acc_at_1
    results["acc@5"] = acc_at_5

    # dump to disk
    pkl.dump(results, open(res_file_name, "wb"))

    # print out, for sanity-check
    print(
        "exp: {}, acc at 1: {} , acc at 5: {}".format(
            res_file_name[:-4], acc_at_1, acc_at_5
        )
    )
