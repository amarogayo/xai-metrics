# code to check if pickles are downloaded correctly
from tqdm import tqdm
from glob import glob
import pickle as pkl
import os
import sys


if __name__ == "__main__":

    path = sys.argv[1]
    files = glob("{}/*.pkl".format(path))
    corrupted_files = []
    for f in tqdm(files):
        try:
            pkl.load(open(f, "rb"))
        except EOFError:
            corrupted_files.append(f)

    stuff = "\n".join(corrupted_files)

    with open("{}_corrupted_files.txt".format(os.path.basename(path)), "w") as f:
        f.write(stuff)

    print(stuff)
