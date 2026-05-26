#!/bin/bash

set -e

echo "=== [1/4] Split du dataset (70/30) ==="
python test/split.py

echo ""
echo "=== [2/4] Entraînement sur les 70% ==="
python logreg_train.py --dataset datasets/dataset_train_split.csv --optimizer sgd

echo ""
echo "=== [3/4] Prédiction sur les 30% ==="
python logreg_predict.py --dataset datasets/dataset_train_validate.csv

echo ""
echo "=== [4/4] Score ==="
python test/score.py