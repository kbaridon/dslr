from describe import load_csv
import matplotlib.pyplot as plt
import argparse
import sys
import numpy as np


def print_pairplot(dataset):
    """Print a pair plot of all numerical features manually using matplotlib."""
    
    data = dataset.select_dtypes(include=[np.number])
    features = data.columns.tolist()
    num_features = len(features)
    houses_colors = {
        'Gryffindor': 'red',
        'Slytherin': 'green',
        'Hufflepuff': 'gold',
        'Ravenclaw': 'blue'
    }

    print(f"Generating pair plot of {num_features}x{num_features} ({num_features ** 2} graphiques)...")

    fig, axes = plt.subplots(num_features, num_features, figsize=(20, 20))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    grouped_data = dataset.groupby('Hogwarts House')

    for i in range(num_features):
        for j in range(num_features):
            ax = axes[i, j]
            feat_y = features[i]
            feat_x = features[j]

            ax.set_xticks([])
            ax.set_yticks([])

            for house, group in grouped_data:
                color = houses_colors.get(house, 'gray')
                
                if i == j:
                    ax.hist(group[feat_x].dropna(), bins=15, color=color, alpha=0.5)
                else:
                    ax.scatter(group[feat_x], group[feat_y], color=color, alpha=0.5, s=2)

            if j == 0:
                ax.set_ylabel(feat_y.replace(' ', '\n'), fontsize=8, rotation=0, ha='right', va='center', labelpad=15)
            if i == num_features - 1:
                ax.set_xlabel(feat_x.replace(' ', '\n'), fontsize=8, rotation=45, ha='right', va='top', labelpad=10)

    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=house)
               for house, color in houses_colors.items()]
    fig.legend(handles=handles, title='Hogwarts House', loc='upper right', fontsize=14)

    fig.suptitle("Pair Plot Matrix - Hogwarts Subjects", fontsize=20, fontweight='bold', y=0.98)
    
    plt.savefig('pair.png')
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
    print_pairplot(dataset)


if __name__ == "__main__":
    main()
