from describe import load_csv
import matplotlib.pyplot as plt
import argparse
import sys
import numpy as np
import random


def print_scatter(dataset):
    """Print two scatter plots comparing highly similar and random features."""
    data = dataset.select_dtypes(include=[np.number])
    houses_colors = {
        'Gryffindor': 'red',
        'Hufflepuff': 'gold',
        'Ravenclaw': 'blue', 
        'Slytherin': 'green'
    }

    corr_matrix = data.corr().abs()
    mask = np.ones(corr_matrix.shape, dtype=bool)
    np.fill_diagonal(mask, False)
    corr_matrix = corr_matrix.where(mask)

    sim_feat1, sim_feat2 = corr_matrix.unstack().idxmax()
    print(f"✅ Features les plus similaires : {sim_feat1} & {sim_feat2}")

    all_subjects = list(data.columns)
    pool_for_random = [s for s in all_subjects if s not in (sim_feat1, sim_feat2)]
    diff_feat1, diff_feat2 = random.sample(pool_for_random, 2)
    print(f"✅ Paire aléatoire pour comparaison : {diff_feat1} & {diff_feat2}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    ax1.set_title(f"Random features:\n{diff_feat1} vs {diff_feat2}")
    ax1.set_xlabel(diff_feat1)
    ax1.set_ylabel(diff_feat2)
    
    ax2.set_title(f"Most similar features:\n{sim_feat1} vs {sim_feat2}")
    ax2.set_xlabel(sim_feat1)
    ax2.set_ylabel(sim_feat2)
    
    for house, group in dataset.groupby('Hogwarts House'):
        color = houses_colors[house]
        ax1.scatter(group[diff_feat1], group[diff_feat2], color=color, label=house, alpha=0.7)
        ax2.scatter(group[sim_feat1], group[sim_feat2], color=color, label=house, alpha=0.7)

    for ax in (ax1, ax2):
        ax.legend(title='Hogwarts House', frameon=True, fontsize='small')
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('scatter.png')
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
    print_scatter(dataset)


if __name__ == "__main__":
    main()
