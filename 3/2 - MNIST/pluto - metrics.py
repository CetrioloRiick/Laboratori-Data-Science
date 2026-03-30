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

vuoiStampareLeFoto = False
# RECUPERO DATI E MODELLO
X, y = fetch_openml("mnist_784", version=1, return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=10000, train_size=50000, random_state=42
)

# STAMPO LE METRICHE DEL DATASET
try:
    pippo = pickle.load(
        open(
            "/home/diego/Documents/Uni/Data_science/Laboratori/3/2 - MNIST/lastrun.pkl",
            "rb",
        )
    )
except FileNotFoundError:
    print("Errore: File non trovati. Assicurati di averli generati.")
    raise SystemExit(1)

""" while input("continuo?") == "s":
    n = int(input("numerino fino a 10.000"))
    plt.imshow(np.reshape(X_test.iloc[[n]], shape=(28, 28)), cmap="gray")
    print("questo era un:", y_test.iloc[[n]])
    print("secondo il modello era un:", pippo.predict(X_test.iloc[[n]]))
    plt.show() """

# TEST
y_test_pred = pippo.predict(X_test)

# STAMPO LE METRICHE (e)
weights = pippo.named_steps["model"].coef_.copy()
weights.ravel()
print("Lenght of weights:", len(weights))
print("Sparsity:", sparsity(weights))
print("Sparsity with tollerance:", sparsity_tol(weights))
print("Accuracy:", accuracy_score(y_test, y_test_pred))


# IMMAGINI DEI PIXEL (f)
scale = np.max(np.abs(weights))
coef_pictures = weights / scale
coef_pictures = coef_pictures.reshape(10, 28, 28)
fig, axes = plt.subplots(2, 5, figsize=(10, 5))

print("Scala", scale)
for i, (ax, pic) in enumerate(zip(axes.ravel(), coef_pictures)):
    im = ax.imshow(pic, cmap="RdYlGn", vmin=-1, vmax=1)
    # Correzione: usa una f-string per unire testo e numero
    ax.set_title(f"Coefficienti di {i}")
    
cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cax)
plt.show()
# PCA SUL DATASET (g, h)

n_components = np.unique(np.geomspace(1, 783, num=60, dtype=int))
variances = []
for n in n_components:
    pca_instance = PCA(n_components=n, random_state=42)
    X_train_transformed = pca_instance.fit_transform(X_train)

    cumulative_variance = np.sum(pca_instance.explained_variance_ratio_)

    print("\nNumero di compontenti:", n)
    print("Varianza:", cumulative_variance)
    variances.append(cumulative_variance)

    if vuoiStampareLeFoto:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
        fig.suptitle(f"Originale vs Ricostruzione", fontsize=12)
        image_num = 25
        X_sample = X_test.iloc[[image_num]]
        axes[0].imshow(np.reshape(X_sample, shape=(28, 28)), cmap="gray")
        # print("questo era un:", y_test.iloc[[n]])
        # print("secondo il modello era un:", pippo.predict(X_test.iloc[[n]]))

        X_pca_sample = pca_instance.transform(X_sample)
        X_reconstructed = pca_instance.inverse_transform(X_pca_sample)
        axes[1].imshow(np.reshape(X_reconstructed, shape=(28, 28)), cmap="gray")
        plt.savefig(str(n) + ".svg")
        """ while input("continue?") == "s":
            fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
            fig.suptitle(f"Originale vs Ricostruzione", fontsize=12)
            n = int(input("numerino fino a 10.000"))
            X_sample = X_test.iloc[[n]]
            axes[0].imshow(np.reshape(X_sample, shape=(28, 28)), cmap="gray")
            # print("questo era un:", y_test.iloc[[n]])
            # print("secondo il modello era un:", pippo.predict(X_test.iloc[[n]]))

            X_pca_sample = pca_instance.transform(X_sample)
            X_reconstructed = pca_instance.inverse_transform(X_pca_sample)
            axes[1].imshow(np.reshape(X_reconstructed, shape=(28, 28)), cmap="gray")
            plt.show() """
plt.close("all")

## ROBA DI GEMINI PER FARE IL GRAFICO CARINO
plt.figure(figsize=(10, 6))
plt.style.use('seaborn-v0_8-whitegrid') 

# Grafico principale con marker per evidenziare i punti
plt.plot(n_components, variances, marker='o', linestyle='-', color='b', linewidth=2, label='Varianza singola')

# Grafico della varianza cumulata (opzionale ma molto utile nella PCA)
cumulative_variance = np.cumsum(variances)
plt.step(n_components, cumulative_variance, where='mid', alpha=0.5, color='red', label='Varianza cumulata')

# Titoli e etichette
plt.title('Analisi della Varianza Spiegata (PCA)', fontsize=15, pad=20)
plt.xlabel('Numero di Componenti Principali (n_components)', fontsize=12)
plt.ylabel('Rapporto di Varianza Spiegata', fontsize=12)

# Personalizzazione degli assi
plt.xticks(n_components) # Forza la visualizzazione di tutti i numeri di componenti
plt.ylim(0, 1.05)        # La varianza va da 0 a 1

# Aggiunta di una linea di soglia (es. 90% varianza)
plt.axhline(y=0.90, color='green', linestyle='--', alpha=0.7)
plt.text(0.5, 0.92, 'Soglia 90%', color='green', fontweight='bold')

plt.legend(loc='best')
plt.tight_layout() # Ottimizza lo spazio tra i vari elementi

plt.show()

# PCA IN 2D (i)
pca_instance = PCA(n_components=2, random_state=42)
X_train_transformed = pca_instance.fit_transform(X_train)
plt.scatter(X_train_transformed[:, 0], X_train_transformed[:, 1], s=1)
plt.show()
