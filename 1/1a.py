import numpy as np
import matplotlib.pyplot as plt

# Matrice
A = np.array([[-1, 1],
              [-1,-1]], dtype=float)

# Autovalori e autovettori (destri)
eigvals, P = np.linalg.eig(A)
inv_P = np.linalg.inv(P)

# Intervallo temporale
t = np.linspace(0, 5, 200)

# Calcolo e^{A t} con diagonalizzazione
E_list = []
for ti in t:
    # diag(exp(lambda_i * t))
    Dti = np.diag(np.exp(eigvals * ti))
    Eti = P @ Dti @ inv_P
    Eti = np.real_if_close(Eti)     # rimuove residui immaginari numerici
    E_list.append(Eti)

E = np.stack(E_list, axis=0)  # shape: (len(t), 2, 2)

# Grafici degli elementi di matrice in funzione di t
fig, ax = plt.subplots(4,1)
ax[0].plot(t, E[:,0,0], label='(1,1)')
ax[1].plot(t, E[:,0,1], label='(1,2)')
ax[2].plot(t, E[:,1,0], label='(2,1)')
ax[3].plot(t, E[:,1,1], label='(2,2)')
ax[3].set_xlabel('t')
ax[3].set_ylabel('elementi di $e^{At}$')
ax[3].legend()
ax[3].grid(True)
plt.show()

print("Autovalori:", eigvals)
print("Autovettori (colonne di P):\n", P)


#Note: finisci la grafica e considera gli immaginari