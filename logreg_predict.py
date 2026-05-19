import sys
import argparse
import numpy as np
import pandas as pd
from describe import load_csv
from logreg import LogisticRegressionOvR


def get_probas(model, dataset: pd.DataFrame):
    X_raw = dataset[model.columns].values
    X_std = (X_raw - model.mean) / model.std
    X_std = np.nan_to_num(X_std, nan=0.0)

    m = X_std.shape[0]
    X_b = np.c_[np.ones(m), X_std]

    probas = {
        house: model._sigmoid(X_b.dot(theta))
        for house, theta in model.weights.items()
    }

    return pd.DataFrame(probas).idxmax(axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", dest="dataset", type=str, default="datasets/dataset_test.csv")
    parser.add_argument("--weights", dest="weights", type=str, default="weights.json")
    args = parser.parse_args()

    try:
        model = LogisticRegressionOvR.load_model(args.weights)
        dataset = load_csv(args.dataset)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print("Error :", e)
        sys.exit(1)

    predictions = get_probas(model, dataset)

    output = pd.DataFrame({
        "Index":          range(len(predictions)),
        "Hogwarts House": predictions.values,
    })
    output.to_csv("houses.csv", index=False)
    print("houses.csv generated!")


if __name__ == "__main__":
    main()
