import numpy as np
import matplotlib.pyplot as plt

A = np.array([[-1, 1], [-1,-1]])
eigvals, eigvecs = np.linalg.eig(A)

diagonal_A = np.diag(eigvals)
P = eigvecs
inv_P = np.linalg.inv(P)

t = np.linspace(0, 5, 20)

results = []
fig, ax = plt.subplots()

for t_i in t:
    results.append(P @ np.exp(diagonal_A*t_i) @ inv_P)

for j in range(len(results)):
    for i in range(2):
        for k in range(2):
            ax.plot(t[j], results[j][i][k])



plt.show()
