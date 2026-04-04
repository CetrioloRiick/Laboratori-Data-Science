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


noise_amp = 0.2
n_samples = 200
n_iter = 1000
eta = 0.5
idRun = str(time.time())


x = np.random.uniform(0, 1, n_samples)
noise = np.random.uniform(-noise_amp, noise_amp, n_samples)
y = np.sin(2 * np.pi * x) + noise

x_true = np.linspace(0, 1, 500)
y_true = np.sin(2 * np.pi * x_true)

degree = 9


"""
costs = []
colors = ["red", "green", "purple", "yellow", "brown", "black"]
etas = [1.21, 1, 0.1, 0.05, 0.01, 0.001]
coeffs_start = np.random.uniform(-0.5, 0.5, degree)

for eta, c in zip(etas, colors):
    costs = []
    coeffs = coeffs_start.copy()
    print(f"Inizio con {eta}...")
    with tqdm(range(n_iter), desc="Ottimizzazione", unit="iter") as t:
        for i in t:
            grad = gradient_cost_function(coeffs, x, y)
            cost = cost_function(coeffs, x, y)
            costs.append(cost)
            t.set_postfix({"Costo": f"{cost:.6e}"})
            coeffs -= eta * grad
    print(coeffs)
    print("Costo finale: " + str(cost))
    plt.plot(range(n_iter), costs, color=c)
    joblib.dump(coeffs, idRun + ".coeffs.eta=" + str(eta))
    joblib.dump(costs, idRun + ".costs.eta=" + str(eta))


plt.grid(True, linestyle="--", alpha=0.7)
plt.show()
 """


def get_batches(x, y, batch_size):
    indices = np.arange(len(x))
    np.random.shuffle(indices) 

    for i in range(0, len(x), batch_size):
        batch_idx = indices[i : i + batch_size]
        yield x[batch_idx], y[batch_idx]


# Esempio di utilizzo nel tuo loop:
n_epchos = []
times = []
conv_cost = 6.7e-3
coeffs_init = np.random.uniform(-0.5, 0.5, degree)

for batch_size in range(1, 101):
    epoch = 0
    start_time = time.perf_counter()
    coeffs = coeffs_init.copy()
    cost = cost_function(coeffs, x, y)

    while cost > conv_cost:
        epoch += 1

        for x_batch, y_batch in get_batches(x, y, batch_size):
            grad = gradient_cost_function(coeffs, x_batch, y_batch)
            coeffs -= eta * grad

        cost = cost_function(coeffs, x, y)

    joblib.dump(coeffs, idRun + ".minibatch.batchsize=" + str(batch_size))
    n_epchos.append(epoch)
    end_time = time.perf_counter()

    times.append(end_time - start_time)

joblib.dump(times, idRun + ".minibatch.times")
joblib.dump(n_epchos, idRun + ".minibatch.epochs")

plt.bar(range(1, 101), times)
plt.show()
plt.bar(range(1, 101), n_epchos)
plt.show()


""" y_plot = polynomial_model(coeffs, x_true)
plt.figure(figsize=(8, 5))
plt.scatter(x, y, s=25, alpha=0.6, label="Punti con rumore")
plt.plot(x_true, y_true, color="black", lw=2, label="sin(2πx)")
plt.plot(x_true, y_plot, color="red", lw=2, label="model")

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.show() """
