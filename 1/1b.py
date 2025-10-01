import numpy as np
import matplotlib.pyplot as plt

N = np.linspace(3, 200, 4, dtype=int)
s = 100
sigma = 1.0
matrices_list = [np.random.normal(0.0, sigma, size=(s, n, n)) for n in N]

eigvals_list = [np.linalg.eigvals(mat) for mat in matrices_list]

eigvals_scaled = [
    (eig.real / np.sqrt(n), eig.imag / np.sqrt(n)) for n, eig in zip(N, eigvals_list)
]

for n, (x, y) in zip(N, eigvals_scaled):
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, color="blue", marker="x", s=10, alpha=0.6)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Parte reale")
    plt.ylabel("Parte immaginaria")
    plt.title(f"Autovalori normalizzati - Matrice {n}×{n}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

plt.show()
