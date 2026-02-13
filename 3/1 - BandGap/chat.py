import numpy as np
import pandas as pd  # Aggiunto per gestire meglio le correlazioni
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle

# --- FUNZIONI DI ANALISI ---


def analizza_errore_per_range(y_true, y_pred):
    """
    Calcola e stampa l'MSE (errore quadratico medio) per diverse fasce
    di valore del band-gap (Y).
    """
    data = np.asarray(y_true).ravel()
    pred = np.asarray(y_pred).ravel()

    if len(data) != len(pred):
        raise ValueError(
            f"Dati e predizioni devono avere stessa lunghezza: {len(data)} vs {len(pred)}"
        )

    print("\n--- Analisi Errore per fasce di Band Gap ---")
    # Definiamo dei range (bin) per il band gap.
    # Nota: Ho esteso leggermente il range massimo per sicurezza.
    thresholds = np.linspace(0, 9.0, 10)

    prev = -np.inf
    for t in thresholds:
        # Maschera per selezionare i dati nel range corrente
        mask = (data > prev) & (data <= t)

        if np.any(mask):
            # Calcolo MSE solo su questo sottoinsieme
            e = mean_squared_error(data[mask], pred[mask])
            count = np.sum(mask)
            print(f"Range ({prev:5.2f}, {t:5.2f}]: MSE = {e:.4f} (su {count} campioni)")
        else:
            pass  # Non stampiamo nulla se il bin è vuoto per pulizia
        prev = t
    print("-" * 40)


def printStats(model, X_train, X_test, Y_train, Y_test):
    Y_train_pred = model.predict(X_train)
    Y_test_pred = model.predict(X_test)

    train_error = mean_squared_error(Y_train, Y_train_pred)
    test_error = mean_squared_error(Y_test, Y_test_pred)
    r2 = r2_score(Y_test, Y_test_pred)

    print(f"MSE Train: {train_error:.4f}")
    print(f"MSE Test:  {test_error:.4f}")
    print(f"R2 Score:  {r2:.4f}")

    # Richiamo la funzione per vedere l'errore al variare del band-gap
    analizza_errore_per_range(Y_test, Y_test_pred)

    # Print dei coefficenti
    model = pipe.named_steps["model"]
    coefs = model.coef_
    print("\nCoefficienti:")
    for f, c in zip(FEATURES, coefs):
        if abs(c) > 1e-4:  # Stampa solo quelli rilevanti
            print(f"{f}: {c:.4f}")
        else:
            print(f"{f}: --- (Scartata)")

def printStatsPoli(model, X_train, X_test, Y_train, Y_test):
    Y_train_pred = model.predict(X_train)
    Y_test_pred = model.predict(X_test)

    train_error = mean_squared_error(Y_train, Y_train_pred)
    test_error = mean_squared_error(Y_test, Y_test_pred)
    r2 = r2_score(Y_test, Y_test_pred)

    print(f"MSE Train: {train_error:.4f}")
    print(f"MSE Test:  {test_error:.4f}")
    print(f"R2 Score:  {r2:.4f}")

    # Richiamo la funzione per vedere l'errore al variare del band-gap
    analizza_errore_per_range(Y_test, Y_test_pred)

    # Print dei coefficenti
    model = last_pipe.named_steps["model"]
    coefs = model.coef_
    print("\nCoefficienti:")
    for f, c in zip(FEATURES, coefs):
        if abs(c) > 1e-4:  # Stampa solo quelli rilevanti
            print(f"{f}: {c:.4f}")
        else:
            print(f"{f}: --- (Scartata)")


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

# Assicurati che i file X.dat e Y.dat siano nella stessa cartella
try:
    X = pickle.load(open("X.dat", "rb"))
    Y = pickle.load(open("Y.dat", "rb"))
except FileNotFoundError:
    print("Errore: File .dat non trovati. Assicurati di averli generati.")
    exit()

# --- ANALISI CORRELAZIONI (Nuova Sezione) ---

print("### ANALISI CORRELAZIONE FEATURE-TARGET ###")
# Creiamo un DataFrame temporaneo per calcolare facilmente le correlazioni
df_analysis = pd.DataFrame(X, columns=FEATURES)
df_analysis["BAND_GAP_TARGET"] = Y

# Calcoliamo la matrice di correlazione
corr_matrix = df_analysis.corr()

# Estraiamo le correlazioni con il target e ordiniamo per valore assoluto
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

# Salvataggio
plt.savefig("CorrelationMatrix.png", dpi=300, bbox_inches="tight")
print(f"Figura Salvata.")

# --- TRAINING E TEST ---

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

    # Passiamo anche i dati a printStats per poter calcolare le predizioni dentro
    printStats(pipe, X_train, X_test, Y_train, Y_test)

    # Se è Lasso, stampiamo quali feature ha azzerato (Feature Selection)



print(f"\n{'='*20}\nModello: Regressione Polinomiale\n{'='*20}")
last_pipe = Pipeline(
    [
        ("poli", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=5000.0)),
    ]
)
last_pipe.fit(X_train, Y_train)
printStatsPoli(pipe, X_train, X_test, Y_train, Y_test)
