import json
import numpy as np
import pandas as pd


HOUSES = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]


class LogisticRegressionOvR:


    def __init__(self, lr=0.1, n_epochs=10000, epsilon=0.001, patience=10, optimizer="classic"):
        self.lr       = lr
        self.n_epochs = n_epochs
        self.epsilon  = epsilon
        self.patience = patience
        self.weights  = {}
        self.optimizer = optimizer
        self.mean     = None
        self.std      = None
        self.columns  = None


    @staticmethod
    def _standardize(X: np.ndarray):
        """Standardize grades (to make the gradient descent even)"""
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        std[std == 0] = 1
        return (X - mean) / std, mean, std


    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Sigmoid function"""
        return 1 / (1 + np.exp(-z))


    @staticmethod
    def _loss(y: np.ndarray, y_hat: np.ndarray) -> float:
        """Get the loss of our gradient descent (binary cross entropy error function)"""
        return -np.mean(y * np.log(y_hat + 1e-15) + (1 - y) * np.log(1 - y_hat + 1e-15))


    def _early_stopping(self, losses: list) -> bool:
        """Return True if the loss hasn't improved by epsilon over the last patience epochs."""
        if len(losses) < self.patience + 1:
            return False
        recent = losses[-self.patience:]
        best_before = min(losses[:-self.patience])
        return min(recent) > best_before - self.epsilon


    def _train_one(self, X: np.ndarray, y: np.ndarray, label: str, batch_size: int) -> np.ndarray:
        """Gradient descent with early stopping for one binary classifier."""
        m, n = X.shape
        theta = np.zeros(n + 1)
        X_bias = np.c_[np.ones(m), X]
        losses = []

        for epoch in range(1, self.n_epochs + 1):
            indices = np.random.permutation(m)
            X_s, y_s = X_bias[indices], y[indices]

            for start in range(0, m, batch_size):
                Xb = X_s[start:start + batch_size]
                yb = y_s[start:start + batch_size]
                y_hat = self._sigmoid(Xb.dot(theta))
                theta -= self.lr * (1 / len(yb)) * Xb.T.dot(y_hat - yb)

            loss = self._loss(y, self._sigmoid(X_bias.dot(theta)))
            losses.append(loss)

            if epoch % 200 == 0:
                print(f"  [{label}] epoch {epoch}/{self.n_epochs} - loss: {loss:.4f}")

            if self._early_stopping(losses):
                print(f"  [{label}] Early stopping at epoch {epoch} - loss: {loss:.4f}")
                break

        return theta


    def fit(self, df: pd.DataFrame):
        """Train one classifier per house on the dataset."""
        self.columns = df.select_dtypes(include=[np.number]).columns.tolist()
        X_raw = df[self.columns].values
        X_std, mean, std = self._standardize(X_raw)
        X_std = np.nan_to_num(X_std, nan=0.0)
        self.mean, self.std = mean, std

        for house in HOUSES:
            print(f"\n-- Training {house} vs Rest --")
            y = (df["Hogwarts House"] == house).astype(int).values
            if (self.optimizer == "sgd"):
                self.weights[house] = self._train_one(X_std, y, house, 1)
            elif (self.optimizer == "mini-batch"):
                self.weights[house] = self._train_one(X_std, y, house, 32)
            else:
                self.weights[house] = self._train_one(X_std, y, house, len(X_std))


    def save_model(self, path: str):
        model = {
            "lr":       self.lr,
            "n_epochs": self.n_epochs,
            "epsilon":  self.epsilon,
            "patience": self.patience,
            "mean":     self.mean.tolist(),
            "std":      self.std.tolist(),
            "columns":  self.columns,
            "weights":  {house: theta.tolist() for house, theta in self.weights.items()},
        }
        with open(path, "w") as f:
            json.dump(model, f, indent=2)
        print(f"\nModel saved: {path}")


    @classmethod
    def load_model(cls, path: str):
        with open(path, "r") as f:
            model = json.load(f)
        instance          = cls(lr=model["lr"], n_epochs=model["n_epochs"],
                                epsilon=model["epsilon"], patience=model["patience"])
        instance.mean     = np.array(model["mean"])
        instance.std      = np.array(model["std"])
        instance.columns  = model["columns"]
        instance.weights  = {house: np.array(theta) for house, theta in model["weights"].items()}
        return instance