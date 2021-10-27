# code to create bsub scripts to submit for explainers
import os
import itertools
# from utils import maybe_create_dir

op_dir = "exp_transform_bsub_scripts"
log_dir = "exp_transform_bsub_logs"


def create_bsub_file(
        model,
        start_idx,
        end_idx,
        explainer,
        exp_args=None,
        transform=None,
        batch_size=7,
        transform_args="no=args",
):
    if not transform:
        config_name = "{}_{}_{}".format(model, explainer, start_idx)
    else:
        config_name = "{}_{}_{}_{}_{}".format(
            model, explainer, start_idx, transform, transform_args
        )

    filename = "{}_bsub.sh".format(config_name)
    filename = os.path.join(op_dir, filename)

    with open(filename, "w") as f:
        f.write("#BSUB -J '{}'\n".format(config_name))
        f.write("#BSUB -n 8\n")
        # f.write("#BSUB -q prod.med\n")
        f.write("#BSUB -q prod.short\n")
        f.write("#BSUB -o '{}/{}.OUT'\n".format(log_dir, config_name))
        f.write("#BSUB -e '{}/{}.ERR'\n".format(log_dir, config_name))
        f.write("#BSUB -R 'select[ngpus>0] rusage [ngpus_excl_p=1]'\n")
        f.write("\n")
        f.write("source activate torch36\n")

        args = [
            "--model {}".format(model),
            "--explainer {}".format(explainer),
            "--start_idx {}".format(start_idx),
            "--end_idx {}".format(end_idx),
        ]
        if batch_size:
            args.append("--batch_size {}".format(batch_size))
        if exp_args:
            args.append("--exp_args {}".format(exp_args))
        if transform:
            args.append("--transform {}".format(transform))
        if transform_args:
            args.append("--transform_args {}".format(transform_args))

        cmd = "python run_explainers.py " + " ".join(args) + "\n"
        f.write(cmd)

    return filename


if __name__ == "__main__":
    print("creating dirs")
    maybe_create_dir(op_dir)
    maybe_create_dir(log_dir)

    models = ["alexnet", "vgg19", "resnet101"]
    explainers = ["gradcam", "smoothgrad", "integrated_gradient"]
    chunk_size = 5000

    for model in models:
        for explainer in explainers:
            for start in range(0, 50000, chunk_size):

                print("create rot stuff")
                transform = "rotation"

                # let's do rotation
                for degree in range(-15, 16, 5):
                    name = create_bsub_file(
                        model=model,
                        start_idx=start,
                        end_idx=start + chunk_size,
                        explainer=explainer,
                        transform=transform,
                        transform_args="degree={}".format(degree),
                    )
                    os.system("bsub< {}".format(name))

                # now hflip and vflip
                print("create flip stuff")
                for transform in ["horizontal_flip", "vertical_flip"]:
                    name = create_bsub_file(
                        model=model,
                        start_idx=start,
                        end_idx=start + chunk_size,
                        explainer=explainer,
                        transform=transform,
                    )
                    os.system("bsub< {}".format(name))

                # now translation
                print("create translation stuff")
                for coord, val in itertools.product(["x", "y"], [-0.2, 0.2]):
                    name = create_bsub_file(
                        model=model,
                        start_idx=start,
                        end_idx=start + chunk_size,
                        explainer=explainer,
                        transform="translation",
                        transform_args="{}={}".format(coord, val),
                    )
                    os.system("bsub < {}".format(name))

    """
    explainer = "lime"
    for transform in ["rotation"]:  # ["horizontal_flip", "vertical_flip"]
        for degree in [5, 10, 15]:
            for i in range(0, 50000, chunk_size):
                name = create_bsub_file(
                    model=model,
                    start_idx=i,
                    end_idx=i + chunk_size,
                    explainer=explainer,
                    transform=transform,
                    transform_args="degree={}".format(degree),
                )
                os.system("bsub < {}".format(name))
    transform ="translation"  # ["horizontal_flip", "vertical_flip", "rotation"]
    trans = 0.2
    for i in range(0, 50000, chunk_size):
        name = create_bsub_file(
            model=model,
            start_idx=i,
            end_idx=i + chunk_size,
            explainer=explainer,
            transform=transform,
            transform_args="x={}".format(trans),
        )
        os.system("bsub < {}".format(name))
        name = create_bsub_file(
            model=model,
            start_idx=i,
            end_idx=i + chunk_size,
            explainer=explainer,
            transform=transform,
            transform_args="x={}".format(-1*trans),
        )
        os.system("bsub < {}".format(name))
        name = create_bsub_file(
            model=model,
            start_idx=i,
            end_idx=i + chunk_size,
            explainer=explainer,
            transform=transform,
            transform_args="y={}".format(trans),
        )
        os.system("bsub < {}".format(name))
        name = create_bsub_file(
            model=model,
            start_idx=i,
            end_idx=i + chunk_size,
            explainer=explainer,
            transform=transform,
            transform_args="y={}".format(-1*trans),
        )
        os.system("bsub < {}".format(name))

    """
