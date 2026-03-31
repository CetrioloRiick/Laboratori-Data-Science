import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from matplotlib.colors import ListedColormap
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

n_components = []
variances = []
for n in n_components:
    pca_instance = PCA(n_components=n, random_state=42)
    X_train_transformed = pca_instance.fit_transform(X_train)

    cumulative_variance = np.sum(pca_instance.explained_variance_ratio_)

    print("\nNumero di compontenti:", n)
    print("Varianza:", cumulative_variance)
    variances.append(cumulative_variance)

    if vuoiStampareLeFoto:
        plt.plot(figsize=(10, 8), constrained_layout=True)
        image_num = 26
        X_sample = X_test.iloc[[image_num]]

        X_pca_sample = pca_instance.transform(X_sample)
        X_reconstructed = pca_instance.inverse_transform(X_pca_sample)
        plt.imshow(np.reshape(X_reconstructed, shape=(28, 28)), cmap="gray")
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
plt.plot(figsize=(10, 8), constrained_layout=True)
image_num = 26
X_sample = X_test.iloc[[image_num]]

plt.imshow(np.reshape(X_sample, shape=(28, 28)), cmap="gray")
plt.savefig("original.svg")
plt.close("all")


pca_instance = PCA(n_components=2, random_state=42)
X_train_transformed = pca_instance.fit_transform(X_train)

# PCA IN 2D (i)
# 1. Prepariamo i colori (usiamo la tua lista)
colors = ['red', 'blue', 'pink', 'yellow', 'black', 'orange', 'purple', 'green', 'brown', 'gray']
cmap_custom = ListedColormap(colors)

# 2. Convertiamo y_train in interi (se sono stringhe) per usarli come indici dei colori
y_numeric = y_train.astype(int)

# 3. Scatter plot unico
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_train_transformed[:, 0], X_train_transformed[:, 1], 
                      c=y_numeric, cmap=cmap_custom, s=1)

# 4. Legenda automatica
# legend_elements() crea i "quadratini" colorati basandosi sulla colormap
plt.legend(handles=scatter.legend_elements()[0], 
           labels=[str(i) for i in range(10)],
           title="Classi",
           markerscale=2) # Ingrandisce i pallini nella legenda (sennò con s=1 non li vedi)

plt.title("Visualizzazione PCA dei dati")
plt.xlabel("Componente Principale 1")
plt.ylabel("Componente Principale 2")
plt.show()