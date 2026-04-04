from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import joblib
import time


def polynomial_model(coeffs: np.ndarray, x: np.ndarray):
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

    X_poly = x[:, None] ** np.arange(len(coeffs))
    gradient = errors @ X_poly / len(x)
    return gradient




x_true = np.linspace(0, 1, 500)
y_true = np.sin(2 * np.pi * x_true)



coeffs = joblib.load("1775252507.1026.coeffs300g")

y_plot = polynomial_model(coeffs, x_true)
plt.figure(figsize=(12, 8))
plt.plot(x_true, y_true, color="black", lw=2, label="sin(2πx)")
plt.plot(x_true, y_plot, color="red", lw=2, label="model")

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.show()
