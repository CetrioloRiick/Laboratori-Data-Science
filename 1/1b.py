import numpy as np
import matplotlib.pyplot as plt

def pippo(num):
    return num/np.sqrt(n)

n = 300
s = 1000
sigma = 1.0
matrices = np.random.normal(0.0, sigma, size=(s, n, n))  # array (s, n, n)

eigvals = np.linalg.eigvals(matrices)
 #[np.linalg.eigvals(i) for i in matrices]

div = np.sqrt(n)
x = [z.real / div for z in eigvals]
y = [z.imag / div for z in eigvals]


# Creo il grafico
plt.figure(figsize=(6,6))
plt.scatter(x, y, color="blue", marker="o")

# Aggiungo etichette
plt.axhline(0, color="black", linewidth=0.8)
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Parte reale")
plt.ylabel("Parte immaginaria")
plt.title("Numeri complessi sul piano")

plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

print(eigvals)
