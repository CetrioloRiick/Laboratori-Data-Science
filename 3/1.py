import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import pickle


def stats(data, pred):
    data = np.asarray(data).ravel()
    pred = np.asarray(pred).ravel()

    if len(data) != len(pred):
        raise ValueError(
            f"data e pred devono avere stessa lunghezza: {len(data)} vs {len(pred)}"
        )

    print("Numero di dati:", len(data))
    print("MSE totale:", mean_squared_error(data, pred))

    thresholds = np.linspace(0, 8.7, 25)

    prev = -np.inf
    for t in thresholds:
        mask = (data > prev) & (data <= t)
        if np.any(mask):
            e = mean_squared_error(data[mask], pred[mask])
            print(f"errore su ({prev:.3f}, {t:.3f}]   {e}")
        else:
            print(f"errore su ({prev:.3f}, {t:.3f}]   (nessun dato)")
        prev = t


def printStats(model):
    Y_train_pred = model.predict(X_train)
    Y_test_pred = model.predict(X_test)

    train_error = mean_squared_error(Y_train, Y_train_pred)
    test_error = mean_squared_error(Y_test, Y_test_pred)

    print("Errore sul train:", train_error)
    print("Errore sul test:", test_error)

FEATURES = [
    "formation_energy_per_atom",
    "energy_above_hull",
    "density",
    "volume",
    "nsites",
    "symmetry.number",
    "efermi",
    "total_magnetization",
    "total_magnetization_normalized_vol",
    "num_magnetic_sites",
]

X = pickle.load(open("X.dat", "rb"))
Y = pickle.load(open("Y.dat", "rb"))

# Fit e test
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.5, random_state=42
)

linear = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])

linear_ridge = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])

linear_lasso = Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=0.05))])


pipelines = [
    ("Linear regression", linear),
    ("Ridge linear regression", linear_ridge),
    ("Lasso linear regression", linear_lasso),
    # ("Polinomial regression", polinomial),
]

for name, pipe in pipelines:
    print("\nEseguo:", name)
    pipe.fit(X_train, Y_train)
    printStats(pipe)

