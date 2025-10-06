import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


img = mpimg.imread("1/fig.png")
X = rgb2gray(img)
plt.imshow(X, cmap="gray", vmin=0, vmax=1)

# SINGULAR VALUES
U, S, Vh = np.linalg.svd(X, full_matrices=True)
fig, ax = plt.subplots()
ax.set_xlabel("index")
ax.set_ylabel("singular values")
plt.semilogy()
ax.plot(S, lw=2)

# CUMULATIVE ENERGY
CumEn = np.cumsum(S) / np.sum(S)
fig, ax = plt.subplots()
ax.set_xlabel("index")
ax.set_ylabel("Cumulative Energy")
ax.set_ylim([0, 1.1])
ax.plot(CumEn, lw=2)

# COMPRESSED IMAGE
R = np.linspace(5, 150, 6, dtype=int)
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.ravel()

for i, r in enumerate(R):
    U_tilde = U[:, :r]
    S_tilde = S[:r]
    Vh_tilde = Vh[:r, :]

    X_tilde = U_tilde @ np.diag(S_tilde) @ Vh_tilde

    axes[i].imshow(X_tilde, cmap="gray", vmin=0, vmax=1)
    axes[i].set_title(f"r = {r}")
    axes[i].axis("off")

plt.tight_layout()
plt.show()
