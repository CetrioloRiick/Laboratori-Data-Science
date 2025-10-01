import numpy as np
import matplotlib.pyplot as plt

A = np.array([[-1, 1], [-1, -1]], dtype=float)

eigvals, P = np.linalg.eig(A)
inv_P = np.linalg.inv(P)

t = np.linspace(0, 5, 200)

E_list = []
for ti in t:
    Dti = np.diag(np.exp(eigvals * ti))
    Eti = P @ Dti @ inv_P
    Eti = np.real_if_close(Eti)
    E_list.append(Eti)

E = np.stack(E_list, axis=0)

fig, ax = plt.subplots(4, 1, figsize=(7, 10), sharex=True)

labels = ["(1,1)", "(1,2)", "(2,1)", "(2,2)"]
indices = [(0, 0), (0, 1), (1, 0), (1, 1)]

for i, (row, col) in enumerate(indices):
    ax[i].plot(t, E[:, row, col], label=f"Elemento {labels[i]}")
    ax[i].legend(loc="best")
    ax[i].grid(True)
    ax[i].set_ylabel(f"{labels[i]}")

ax[-1].set_xlabel("t")
fig.suptitle("Elementi della matrice $e^{At}$", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

print("Autovalori:", eigvals)
print("Autovettori (colonne di P):\n", P)
