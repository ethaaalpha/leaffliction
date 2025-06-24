import os
import sys
import matplotlib
import matplotlib.pyplot as plt
from preprocessing.loader import Loader


def generate_histogram(data: dict[str, int]):
    classes = list(data.keys())
    counts = list(data.values())
    colors = matplotlib.colors.TABLEAU_COLORS

    plt.figure(figsize=(10, 6))
    plt.bar(classes, counts, color=colors, edgecolor='black')
    plt.xlabel("Classes")
    plt.ylabel("Number of Images")
    plt.title("Image Count per Class")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if (len(sys.argv) == 2 and os.path.exists(sys.argv[1])):
        count = Loader(sys.argv[1]).count()
        print(count)
        generate_histogram(count)
    else:
        print("./Distribution.py <dataset_folder>")
