"""
split.py — Divise dataset_train.csv en :
  - datasets/dataset_train_split.csv    (70%) → entraînement
  - datasets/dataset_train_validate.csv (30%) → validation (avec la vraie maison)
Usage:
    python split.py
    python split.py --dataset datasets/dataset_train.csv --ratio 0.3
"""
import argparse
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/dataset_train.csv")
    parser.add_argument("--ratio",   type=float, default=0.3,
                        help="Proportion réservée à la validation (défaut: 0.3)")
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    print(f"Lignes totales : {len(df)}")

    # Filtre les lignes sans label
    labeled = df[df["Hogwarts House"].notna() & (df["Hogwarts House"].str.strip() != "")]
    print(f"Lignes avec label : {len(labeled)} / {len(df)}")

    # Filtre les lignes avec des NaN dans les colonnes de features (col 6+)
    feature_cols = df.columns[6:]
    complete = labeled.dropna(subset=feature_cols)
    print(f"Lignes complètes (sans NaN) : {len(complete)} / {len(labeled)}")

    # Shuffle + split
    shuffled = complete.sample(frac=1, random_state=42).reset_index(drop=True)
    split    = int(len(shuffled) * (1 - args.ratio))
    train    = shuffled.iloc[:split]
    validate = shuffled.iloc[split:]

    print(f"Train    : {len(train)} lignes  ({len(train)/len(shuffled)*100:.0f}%)")
    print(f"Validate : {len(validate)} lignes  ({len(validate)/len(shuffled)*100:.0f}%)")

    os.makedirs("datasets", exist_ok=True)
    train.to_csv("datasets/dataset_train_split.csv",    index=False)
    validate.to_csv("datasets/dataset_train_validate.csv", index=False)

    print("\nFichiers générés :")
    print("  datasets/dataset_train_split.csv    → logreg_train.py")
    print("  datasets/dataset_train_validate.csv → logreg_predict.py + score.py")
    print("\nWorkflow :")
    print("  python logreg_train.py   datasets/dataset_train_split.csv")
    print("  python logreg_predict.py datasets/dataset_train_validate.csv weights.csv")
    print("  python score.py")

if __name__ == "__main__":
    main()