import os


def maybe_create_dir(dirname):
    os.makedirs(dirname, exist_ok=True)
