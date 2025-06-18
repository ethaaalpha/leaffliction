import os
import sys

from preprocessing.loader import Loader

if __name__ == "__main__":
    if (len(sys.argv) == 2 and os.path.exists(sys.argv[1])):
        print(Loader(sys.argv[1]).count())
    else:
        print("./Distribution.py <dataset_folder>")