import argparse
import sys
import pandas as pd
from utils import *


def load_csv(path: str) -> pd.DataFrame:
    """Take a path to a csv and return the DataFrame of it."""
    try:
        dataset = pd.read_csv(path, header=0 ,index_col=0)
        print("Loading dataset of dimensions", dataset.shape)
        return dataset
    except (FileNotFoundError, PermissionError) as e:
        raise ValueError("bad parameters") from e


def print_infos(dataset: pd.DataFrame):
    num_dataset = dataset.select_dtypes(include='number')
    stats_df = pd.DataFrame(columns=["Count", "unique", "top", "freq", "Mean", "Std", "Min", "25%", "50%", "75%", "Max", "dtype"])
    for name_col in num_dataset:
        column = num_dataset[name_col]
        stats_df.loc[name_col] = {
            "Count": my_count(column),
            "unique": my_unique(column),
            "top": my_top(column),
            "freq": my_freq(column),
            "Mean": my_mean(column),
            "Std": my_std(column),
            "Min": my_min(column),
            "25%": my_quantile(column, 0.25),
            "50%": my_quantile(column, 0.50),
            "75%": my_quantile(column, 0.75),
            "Max": my_max(column),
            "dtype": type(column.iloc[0]).__name__
        }
    print(stats_df.T)


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
    print_infos(dataset)

if __name__ == "__main__":
    main()
