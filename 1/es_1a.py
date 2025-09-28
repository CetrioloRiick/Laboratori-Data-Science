import numpy as np
import matplotlib.pyplot as plt


A = np.array([[-1,1], [-1,-1]])
eigvals, eigvecs = np.linalg.eig(A)

D = np.diag(eigvals)

P = eigvecs

P_inv = np.linalg.inv(P)

ricostruzione = P @ D @ P_inv

# print("Autovalori:", eigvals)
# print("Autovettori (colonne di P):\n", P)
# print("Matrice diagonale D:\n", D)
# print("Ricostruzione A:\n", ricostruzione)

# Calcolo di Exp(A*t) per t in [0,5]
t = np.linspace(0, 5, 100)
ExpAt = np.zeros((len(t), 2, 2), dtype=complex)
print(ExpAt)

for i in range(len(t)):
    expD_t = np.diag(np.exp(D* t[i]))
    expA_t = P @ expD_t @ P_inv

    ExpAt[i,:,:] = expA_t

fig, ax = plt.subplots(2, 2, figsize=(10, 8))

for (i, j), axis in np.ndenumerate(ax):
    axis.plot(t, ExpAt[:, i, j].real, label=f'Re(Exp(A*t)[{i},{j}])')
    axis.plot(t, ExpAt[:, i, j].imag, label=f'Im(Exp(A*t)[{i},{j}])')
    axis.grid(True, ls='dotted')
   # axis.set_aspect('equal')
    axis.set_xlim(0, 5)
    axis.set_ylim(-1, 1)
    axis.set_xticks(np.linspace(0, 5, 6))
    axis.set_yticks(np.linspace(-1, 1, 6))

# ax[0, 0].plot(t, ExpAt[:, 0, 0].real, label='Re(Exp(A*t)[0,0])')
# ax[0, 0].plot(t, ExpAt[:, 0, 0].imag, label='Im(Exp(A*t)[0,0])')
# ax[0, 1].plot(t, ExpAt[:, 0, 1].real, label='Re(Exp(A*t)[0,1])')
# ax[0, 1].plot(t, ExpAt[:, 0, 1].imag, label='Im(Exp(A*t)[0,1])')
# ax[1, 0].plot(t, ExpAt[:, 1, 0].real, label='Re(Exp(A*t)[1,0])')
# ax[1, 0].plot(t, ExpAt[:, 1, 0].imag, label='Im(Exp(A*t)[1,0])')
# ax[1, 1].plot(t, ExpAt[:, 1, 1].real, label='Re(Exp(A*t)[1,1])')
# ax[1, 1].plot(t, ExpAt[:, 1, 1].imag, label='Im(Exp(A*t)[1,1])')

# ax[0, 0].grid(True, ls='dotted')
# ax[0, 0].set_aspect('equal')
# ax[0, 1].grid(True, ls='dotted')
# ax[0, 1].set_aspect('equal')
# ax[1, 0].grid(True, ls='dotted')
# ax[1, 0].set_aspect('equal')
# ax[1, 1].grid(True, ls='dotted')
# ax[1, 1].set_aspect('equal')

# ax[0, 0].set_xlim(-5, 5)
# ax[0, 0].set_ylim(-5, 5)
# ax[0, 0].set_xticks(np.linspace(-5,5,6))
# ax[0, 0].set_yticks(np.linspace(-5,5,6))
# ax[0, 1].set_xlim(-5, 5)
# ax[0, 1].set_ylim(-5, 5)
# ax[0, 1].set_xticks(np.linspace(-5,5,6))
# ax[0, 1].set_yticks(np.linspace(-5,5,6))
# ax[1, 0].set_xlim(-5, 5)
# ax[1, 0].set_ylim(-5, 5)
# ax[1, 0].set_xticks(np.linspace(-5,5,6))
# ax[1, 0].set_yticks(np.linspace(-5,5,6))
# ax[1, 1].set_xlim(-5, 5)
# ax[1, 1].set_ylim(-5, 5)
# ax[1, 1].set_xticks(np.linspace(-5,5,6))
# ax[1, 1].set_yticks(np.linspace(-5,5,6))

plt.show()





