from describe import load_csv
import matplotlib.pyplot as plt
import argparse
import sys


def print_histo(dataset):
    """Print a histogram grill to compare each courses."""
    courses = [
        "Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts",
        "Divination", "Muggle Studies", "Ancient Runes", "History of Magic",
        "Transfiguration", "Potions", "Care of Magical Creatures", "Charms", "Flying"
    ]
    houses = ["Gryffindor", "Slytherin", "Hufflepuff", "Ravenclaw"]
    colors = ['red', 'green', 'gold', 'blue']

    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(16, 10))
    axes = axes.flatten()

    for i, course in enumerate(courses):
        for house, color in zip(houses, colors):
            subset = dataset[dataset['Hogwarts House'] == house][course].dropna()
            axes[i].hist(subset, bins=25, alpha=0.5, label=house, color=color)
        axes[i].set_title(course, fontsize=10)
        axes[i].tick_params(axis='both', labelsize=8)

    for j in range(len(courses), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Score distribution per course across all houses", fontsize=16, fontweight='bold')
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', fontsize=12, bbox_to_anchor=(0.9, 0.1))

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.show()


def main():
    """Print infos on the dataset"""
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", dest="path_dataset", default="datasets/dataset_train.csv", help="Path to the dataset.")
    args = parser.parse_args()

    try:
        dataset = load_csv(args.path_dataset)
    except ValueError as e:
        print(e)
        sys.exit(1)
    print_histo(dataset)


if __name__ == "__main__":
    main()