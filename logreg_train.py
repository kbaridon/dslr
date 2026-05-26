import sys
import argparse
from describe import load_csv
from logreg import LogisticRegressionOvR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  dest="dataset",  type=str,   default="datasets/dataset_train.csv")
    parser.add_argument("--lr",       dest="lr",       type=float, default=0.1)
    parser.add_argument("--epochs",   dest="epochs",   type=int,   default=10000)
    parser.add_argument("--epsilon",  dest="epsilon",  type=float, default=0.001)
    parser.add_argument("--patience", dest="patience", type=int,   default=10)
    parser.add_argument("--optimizer", dest="optimizer", type=str, default="classic", help="sgd, mini-batch or classic")
    parser.add_argument("--output",   dest="output",   type=str,   default="weights.json")
    args = parser.parse_args()

    try:
        df = load_csv(args.dataset)
    except ValueError as e:
        print("Wrong path:", e)
        sys.exit(1)

    model = LogisticRegressionOvR(args.lr, args.epochs, args.epsilon, args.patience, args.optimizer)
    model.fit(df)
    model.save_model(args.output)


if __name__ == "__main__":
    main()
