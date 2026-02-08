import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# --- FUNZIONI DI ANALISI ---


def analizza_errore_per_range(y_true, y_pred, y_min=0.0, y_max=9.0, n_bins=10):
    """
    Calcola e stampa l'MSE per fasce di valore del band-gap (Y).
    """
    data = np.asarray(y_true).ravel()
    pred = np.asarray(y_pred).ravel()

    if len(data) != len(pred):
        raise ValueError(
            f"Dati e predizioni devono avere stessa lunghezza: {len(data)} vs {len(pred)}"
        )

    print("\n--- Analisi Errore per fasce di Band Gap ---")
    thresholds = np.linspace(y_min, y_max, n_bins)

    prev = -np.inf
    for t in thresholds:
        mask = (data > prev) & (data <= t)
        if np.any(mask):
            e = mean_squared_error(data[mask], pred[mask])
            count = np.sum(mask)
            print(f"Range ({prev:6.2f}, {t:6.2f}]: MSE = {e:.4f} (su {count} campioni)")
        prev = t
    print("-" * 50)


def coefficients_table_linear(pipe: Pipeline, feature_names):
    """
    Tabella coefficienti per modelli lineari (senza PolynomialFeatures).
    """
    model = pipe.named_steps["model"]
    coefs = np.asarray(model.coef_).ravel()

    df = pd.DataFrame(
        {"feature": feature_names, "coef": coefs, "abs_coef": np.abs(coefs)}
    ).sort_values("abs_coef", ascending=False)

    return df


def coefficients_table_poly(pipe: Pipeline, base_feature_names):
    """
    Tabella coefficienti per pipeline con PolynomialFeatures.
    Usa i nomi espansi corretti (x0, x0^2, x0 x1, ... ma con i nomi reali delle feature).
    """
    poly = pipe.named_steps["poli"]
    model = pipe.named_steps["model"]

    expanded_names = poly.get_feature_names_out(base_feature_names)
    coefs = np.asarray(model.coef_).ravel()

    if len(expanded_names) != len(coefs):
        raise ValueError(
            f"Mismatch: {len(expanded_names)} nomi vs {len(coefs)} coefficienti"
        )

    df = pd.DataFrame(
        {"poly_feature": expanded_names, "coef": coefs, "abs_coef": np.abs(coefs)}
    ).sort_values("abs_coef", ascending=False)

    return df


def printStats(pipe: Pipeline, X_train, X_test, Y_train, Y_test, feature_names):
    Y_train_pred = pipe.predict(X_train)
    Y_test_pred = pipe.predict(X_test)

    train_error = mean_squared_error(Y_train, Y_train_pred)
    test_error = mean_squared_error(Y_test, Y_test_pred)
    r2 = r2_score(Y_test, Y_test_pred)

    print(f"MSE Train: {train_error:.4f}")
    print(f"MSE Test:  {test_error:.4f}")
    print(f"R2 Score:  {r2:.4f}")

    analizza_errore_per_range(Y_test, Y_test_pred)

    # Coefficienti in tabella (solo se il modello è lineare e non polinomiale)
    df_coef = coefficients_table_linear(pipe, feature_names)
    print("\nCoefficienti (ordinati per |coef|):")
    print(df_coef.to_string(index=False, justify="left", col_space=2))


def printStatsPoli(
    pipe: Pipeline, X_train, X_test, Y_train, Y_test, base_feature_names, top_k=None
):
    Y_train_pred = pipe.predict(X_train)
    Y_test_pred = pipe.predict(X_test)

    train_error = mean_squared_error(Y_train, Y_train_pred)
    test_error = mean_squared_error(Y_test, Y_test_pred)
    r2 = r2_score(Y_test, Y_test_pred)

    print(f"MSE Train: {train_error:.4f}")
    print(f"MSE Test:  {test_error:.4f}")
    print(f"R2 Score:  {r2:.4f}")

    analizza_errore_per_range(Y_test, Y_test_pred)

    # Coefficienti polynomial in tabella
    df_poly = coefficients_table_poly(pipe, base_feature_names)

    if top_k is not None:
        df_poly = df_poly.head(top_k)

    print("\nCoefficienti Polynomial (ordinati per |coef|):")
    print(df_poly.to_string(index=False, justify="left", col_space=2))


# --- CARICAMENTO DATI ---

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

try:
    X = pickle.load(open("X.dat", "rb"))
    Y = pickle.load(open("Y.dat", "rb"))
except FileNotFoundError:
    print("Errore: File .dat non trovati. Assicurati di averli generati.")
    raise SystemExit(1)

X = np.asarray(X)
Y = np.asarray(Y).ravel()  # IMPORTANTISSIMO: 1D


# --- ANALISI CORRELAZIONI ---

print("### ANALISI CORRELAZIONE FEATURE-TARGET ###")
df_analysis = pd.DataFrame(X, columns=FEATURES)
df_analysis["BAND_GAP_TARGET"] = Y

corr_matrix = df_analysis.corr(numeric_only=True)

target_corr = corr_matrix["BAND_GAP_TARGET"].drop("BAND_GAP_TARGET")
target_corr_sorted = target_corr.abs().sort_values(ascending=False)

print("Feature più correlate con il Band Gap (in valore assoluto):")
for feature in target_corr_sorted.index:
    corr_value = target_corr[feature]
    print(f"{feature:35s}: {corr_value:+.4f}")
print("-" * 50)

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix, annot=True, square=True, cmap="icefire", fmt=".2f", linewidths=0.5
)
plt.title("Matrice di correlazione", fontsize=16)
plt.tight_layout()
plt.savefig("CorrelationMatrix.png", dpi=300, bbox_inches="tight")
print("Figura Salvata: CorrelationMatrix.png")


# --- TRAIN / TEST ---

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42
)

pipelines = [
    (
        "Linear regression",
        Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    ),
    (
        "Ridge linear regression",
        Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=475.0))]),
    ),
    (
        "Lasso linear regression",
        Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=0.006))]),
    ),
]

for name, pipe in pipelines:
    print(f"\n{'='*20}\nModello: {name}\n{'='*20}")
    pipe.fit(X_train, Y_train)
    printStats(pipe, X_train, X_test, Y_train, Y_test, FEATURES)


# --- POLYNOMIAL REGRESSION ---

print(f"\n{'='*20}\nModello: Regressione Polinomiale\n{'='*20}")

last_pipe = Pipeline(
    [
        ("poli", PolynomialFeatures(degree=3, include_bias=False)),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=110.0)),
    ]
)

last_pipe.fit(X_train, Y_train)

# top_k=None stampa tutti; metti tipo top_k=30 per i 30 più importanti
printStatsPoli(last_pipe, X_train, X_test, Y_train, Y_test, FEATURES, top_k=50)
