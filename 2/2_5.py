from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random


def funzioneBella():
    return np.sin(2 * np.pi * x_true)


def polynomial_model(coeffs, x):
    x = np.asarray(x)
    powers = x[:, None] ** np.arange(len(coeffs))
    return powers @ coeffs


def cost_function(coeffs: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    predictions = polynomial_model(coeffs, x)
    errors = predictions - y
    return np.mean(errors**2) / 2


def gradient_cost_function(
    coeffs: np.ndarray, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    predictions = polynomial_model(coeffs, x)
    errors = predictions - y
    N = len(x)

    gradient = np.array([np.sum(errors * x**i) / N for i in range(len(coeffs))])
    return gradient


epsilon = 0.02
n = 2000
len_batches = 100

x = np.random.uniform(0, 1, n)
noise = np.random.uniform(-epsilon, epsilon, n)
y = np.sin(2 * np.pi * x) + noise

dati = [[x_i, y_i] for x_i, y_i in zip(x, y)]

x_true = np.linspace(0, 1, 500)
y_true = np.sin(2 * np.pi * x_true)

grado = 30

coeffs = np.random.uniform(-0.5, 0.5, grado)

with tqdm(range(7000), desc="Ottimizzazione", unit="iter") as t:
    for i in t:
        dati = np.random.permutation(dati)
        batch = [dati[i : i + len_batches] for i in range(0, n, len_batches)]
        batch = np.array(batch)
        for element in batch:
            grad = gradient_cost_function(coeffs, element[:, 0], element[:, 1])
            costo = cost_function(coeffs, dati[:, 0], dati[:, 1])
            t.set_postfix({"Costo": f"{costo:.5f}"})
            for i, g in enumerate(grad):
                coeffs[i] -= 0.1 * g
pippo = polynomial_model(coeffs, x_true)


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
