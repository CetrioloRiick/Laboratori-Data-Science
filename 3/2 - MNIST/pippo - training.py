import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.datasets import fetch_openml

X, y = fetch_openml("mnist_784", version=1, return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=10000, train_size=50000, random_state=42
)

pippo = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(solver="sag", max_iter=100000, tol=0.001, random_state=42),
        ),
    ]
)

pippo.fit(X_train, y_train)

pickle.dump(pippo, open("pippo.pkl", "wb"))


""" # TEST

y_test_pred = pippo.predict(X_test)
weight = pippo.named_steps["model"].coef_.copy()

# printo le immmaginini qui
prova = np.reshape(weight[0], shape=(28, 28))
immagini = np.reshape(weight, shape=(28, 28, 10))
plt.imshow(immagini[0])

plt.show()

weight.ravel()
print("\n lunghezza daudsifona:", len(weight))

number_of_non_zero_weight = np.count_nonzero(weight)
sparsita = number_of_non_zero_weight / len(weight)
print(
    "Sparsità:",
    sparsita,
    "\n non zero weirjfdifajfe:",
    number_of_non_zero_weight,
    "\n lunghezza daudsifona:",
    len(weight),
)
print("Accuracy:", accuracy_score(y_test, y_test_pred))

tol = 1e-4
sparsity = np.mean(weight == 0)
sparsity_tol = np.mean(np.isclose(weight, 0, atol=tol))
print(f"Sparsity:   {sparsity:.2%}")
print(f"Sparsity (with tolerance):   {sparsity_tol:.2%}")
 """