# code to create bsub scripts to submit to ZHC2
import os
from utils import maybe_create_dir
import numpy as np
from utils.constants import SUPPORTED_IMAGENET_MODELS

op_dir = "baseline_bsub_scripts"
log_dir = "baseline_bsub_logs"


def create_bsub_file(model, trans_name, trans_args="", dataset="imagenet"):
    config_name = "{}_{}_{}_{}".format(dataset, model, trans_name, trans_args)
    filename = "{}_bsub.sh".format(config_name)
    filename = os.path.join(op_dir, filename)

    with open(filename, "w") as f:
        f.write("#BSUB -J '{}'\n".format(config_name))
        f.write("#BSUB -n 4\n")
        # f.write("#BSUB -q prod.med\n")
        f.write("#BSUB -q prod.short\n")
        # f.write("#BSUB -q hpc.short\n")
        f.write("#BSUB -o '{}/{}.OUT'\n".format(log_dir, config_name))
        f.write("#BSUB -e '{}/{}.ERR'\n".format(log_dir, config_name))
        f.write("#BSUB -R 'select[ngpus>0] rusage [ngpus_excl_p=1]'\n")
        f.write("\n")
        f.write("source activate torch36\n")
        f.write(
            "python run_baselines.py --dataset {} --model {} --transform {} --transform_args {}\n".format(
                dataset, model, trans_name, trans_args
            )
        )

    return filename


if __name__ == "__main__":
    maybe_create_dir(op_dir)
    maybe_create_dir(log_dir)

    # lets do the flips first
    """
    for model in SUPPORTED_IMAGENET_MODELS:
        name = create_bsub_file(model, "horizontal_flip", "no=arg")
        print(name)
        os.system("bsub < {}".format(name))
        name = create_bsub_file(model, "vertical_flip", "no=arg")
        print(name)
        os.system("bsub < {}".format(name))

    # now rotation
    for model in SUPPORTED_IMAGENET_MODELS:
        for degree in range(-90, 91, 15):
            name = create_bsub_file(model, "rotation", "degree={}".format(degree))
            print(name)
            os.system("bsub < {}".format(name))
    """

    # finally, translation
    for model in SUPPORTED_IMAGENET_MODELS:
        # for x in np.linspace(0.1, 0.9, 9):
        for x in [0.2]:
            name = create_bsub_file(model, "translation", "x={},y=0".format(x))
            print(name)
            os.system("bsub < {}".format(name))
            name = create_bsub_file(model, "translation", "x={},y=0".format(-1 * x))
            print(name)
            os.system("bsub < {}".format(name))
            name = create_bsub_file(model, "translation", "y={},x=0".format(x))
            print(name)
            os.system("bsub < {}".format(name))
            name = create_bsub_file(model, "translation", "y={},x=0".format(-1 * x))
            print(name)
            os.system("bsub < {}".format(name))
