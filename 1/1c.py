from ase import Atoms
from ase.io import write, read
import numpy as np


def calcDiagVal(x, y, m):
    res = m * (np.power(x, 2) + np.power(y, 2))
    return np.sum(res)


def calcIlResto(x, y, m):
    res = m * x * y
    return -np.sum(res)


DataSet = read("1/dataset.xyz", index=":")
masses = [atom.get_masses() for atom in DataSet]
positions = [atom.get_positions() for atom in DataSet]
n = len(positions)

centri_di_massa = [
    np.average(pos, axis=0, weights=mass) for pos, mass in zip(positions, masses)
]

I = np.zeros((n, 3, 3))
for pos, mass, i in zip(positions, masses, I):
    xx = calcDiagVal(pos[:, 1], pos[:, 2], mass)
    yy = calcDiagVal(pos[:, 0], pos[:, 2], mass)
    zz = calcDiagVal(pos[:, 0], pos[:, 1], mass)
    xy = calcIlResto(pos[:, 0], pos[:, 1], mass)
    yz = calcIlResto(pos[:, 1], pos[:, 2], mass)
    xz = calcIlResto(pos[:, 0], pos[:, 2], mass)
    i = [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]


pluto = 0


print()
