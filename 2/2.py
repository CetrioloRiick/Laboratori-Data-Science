import numpy as np
import matplotlib.pyplot as plt


def polynomial_model(coeffs, x):
    powers = x ** np.arange(len(coeffs))
    return np.dot(coeffs, powers)


def cost_function(coeffs, y_true, y):
    N = len(coeffs)
    res = np.sum((y_true - y) ** 2)
    return 1 / (2 * N)


epsilon = 0.2
n = 200

x = np.random.uniform(0, 1, n)
noise = np.random.uniform(-epsilon, epsilon, n)
y = np.sin(2 * np.pi * x) + noise

x_true = np.linspace(0, 1, 500)
y_true = np.sin(2 * np.pi * x_true)

grado = 10

coeffs = np.random.uniform(-0.5, 0.5, grado)
pippo = [polynomial_model(coeffs, x) for x in x_true]


plt.figure(figsize=(8, 5))
plt.scatter(x, y, s=25, alpha=0.6, label="Punti con rumore")
plt.plot(x_true, y_true, color="black", lw=2, label="sin(2πx)")
plt.plot(x_true, pippo, color="blue", lw=2, label="model")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Punti con rumore intorno a sin(2πx)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)


plt.show()
