from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


# ---- utility ----
def build_vandermonde(x, n_coeffs):
    # x assumed scaled to [-1, 1] for better conditioning
    return np.vander(x, N=n_coeffs, increasing=True)


def loss_and_grad(theta, V, y):
    # predictions
    y_pred = V @ theta
    # errors
    err = y_pred - y
    N = y.shape[0]
    # MSE/2
    loss = 0.5 * np.mean(err**2)
    # gradient for (MSE/2): (V^T err)/N
    grad = (V.T @ err) / N
    return loss, grad, y_pred


# ---- data ----
rng = np.random.default_rng(42)
noise_amp = 0.02
n_samples = 5

x = rng.uniform(0.0, 1.0, n_samples)
noise = rng.uniform(-noise_amp, noise_amp, n_samples)
y = np.sin(2 * np.pi * x) + noise

# scaling to [-1, 1] for conditioning
x_scaled = 2.0 * x - 1.0

x_plot = np.linspace(0, 1, 500)
x_plot_scaled = 2.0 * x_plot - 1.0
y_true = np.sin(2 * np.pi * x_plot)

# ---- model ----
degree = 15  # sii onesto: >15 su base monomiale è un bagno di sangue
n_coeffs = degree + 1
theta = rng.uniform(-0.5, 0.5, n_coeffs)

V = build_vandermonde(x_scaled, n_coeffs)
V_plot = build_vandermonde(x_plot_scaled, n_coeffs)

# ---- GD params ----
lr = 0.1
max_iter = 200000
patience = 1000
tol = 1e-10
best_loss = np.inf
bad_count = 0

with tqdm(range(max_iter), desc="Ottimizzazione", unit="iter") as t:
    for _ in t:
        loss, grad, _ = loss_and_grad(theta, V, y)
        t.set_postfix({"Loss": f"{loss:.6e}", "lr": f"{lr:.3g}"})

        # early stopping bookkeeping
        if loss + tol < best_loss:
            best_loss = loss
            bad_count = 0
        else:
            bad_count += 1
            if bad_count % 200 == 0:
                lr = lr * 0.5  # semplice schedule: dimezza se non migliora per un po'

        # gradient step (with simple clipping for safety)
        gnorm = np.linalg.norm(grad)
        if gnorm > 1e6:  # paranoia: evita esplosioni
            grad = grad * (1e6 / gnorm)

        theta -= lr * grad

        if bad_count > patience:
            break

# final predictions
y_pred_plot = V_plot @ theta
print("theta:", theta)

# ---- plot ----
plt.figure(figsize=(8, 5))
plt.scatter(x, y, s=15, alpha=0.5, label="Punti con rumore")
plt.plot(x_plot, y_true, lw=2, label="sin(2πx)")
plt.plot(x_plot, y_pred_plot, lw=2, label=f"Polinomio grado {degree}")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Fit polinomiale su sin(2πx) con rumore")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()
