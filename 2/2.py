from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


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
    deg = len(coeffs)

    gradient = np.array([np.sum(errors * x**i) for i in range(deg)]) / N
    return gradient


noise_amp = 0.2
n_samples = 200

x = np.random.uniform(0, 1, n_samples)
noise = np.random.uniform(-noise_amp, noise_amp, n_samples)
y = np.sin(2 * np.pi * x) + noise

x_true = np.linspace(0, 1, 500)
y_true = np.sin(2 * np.pi * x_true)

degree = 16

coeffs = np.random.uniform(-0.5, 0.5, degree)

for _ in range(3):
    y_plot = polynomial_model(coeffs, x_true)
    print(coeffs)
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, s=25, alpha=0.6, label="Punti con rumore")
    plt.plot(x_true, y_true, color="black", lw=2, label="sin(2πx)")
    plt.plot(x_true, y_plot, color="red", lw=2, label="model")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Punti con rumore intorno a sin(2πx)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.show()

    with tqdm(range(40000), desc="Ottimizzazione", unit="iter") as t:
        for i in t:
            grad = gradient_cost_function(coeffs, x, y)
            cost = cost_function(coeffs, x, y)
            t.set_postfix({"Costo": f"{cost:.6e}"})
            coeffs -= 0.1 * grad


plt.figure(figsize=(8, 5))
plt.scatter(x, y, s=25, alpha=0.6, label="Punti con rumore")
plt.plot(x_true, y_true, color="black", lw=2, label="sin(2πx)")
plt.plot(x_true, y_plot, color="red", lw=2, label="model")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Punti con rumore intorno a sin(2πx)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)


plt.show()
