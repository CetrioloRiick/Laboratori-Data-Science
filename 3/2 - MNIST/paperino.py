import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA


def sparsity(weights):
    return np.mean(weights == 0)


def sparsity_tol(weights, tol=1e-4):
    return np.mean(np.isclose(weights, 0, atol=tol))


X, y = fetch_openml("mnist_784", version=1, return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=10000, train_size=50000, random_state=42
)

pca = PCA(n_components=50, random_state=42)
pca.fit(X_train)

pippo = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                solver="sag", max_iter=100000, tol=0.001, random_state=42
            ),
        ),
    ]
)

pippo.fit(pca.transform(X_train), y_train)
y_test_pred = pippo.predict(pca.transform(X_test))

weights = pippo.named_steps["model"].coef_.copy()
weights.ravel()
print("Lenght of weights:", len(weights))
print("Sparsity:", sparsity(weights))
print("Sparsity with tollerance:", sparsity_tol(weights))
print("Accuracy:", accuracy_score(y_test, y_test_pred))
