import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", type=str, help="which dataset to use", default="imagenet"
)
parser.add_argument(
    "--model", type=str, help="which model to use", default="inception_v3"
)
parser.add_argument(
    "--exp", type=str, help="which explainer to test on", default="lime"
)
parser.add_argument(
    "--data_dir", type=str, help="where the input data is stored", required=True
)
parser.add_argument(
    "--output_dir",
    type=str,
    help="where to store the output heatmaps",
    default="output",
)
