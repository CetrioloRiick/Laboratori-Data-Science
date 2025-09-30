from ase import Atoms
from ase.io import write, read
import numpy as np


def calc_diag_val(x: np.ndarray, y: np.ndarray, m: np.ndarray) -> float:
    return np.sum(m * (np.square(x) + np.square(y)))


def calc_off_val(x: np.ndarray, y: np.ndarray, m: np.ndarray) -> float:
    return -np.sum(m * x * y)


def classify_eigs(vals: np.ndarray, atol: float = 1.0) -> str:
    a, b, c = vals
    if np.isclose(a, b, atol=atol) and np.isclose(b, c, atol=atol):
        return "Sferica"
    if np.isclose(a, b, atol=atol) and b < c:
        return "Oblata"
    if a < b and np.isclose(b, c, atol=atol):
        return "Prolata"
    if a < b < c:
        return "Asimmetrica"
    # Fallback teoricamente non raggiunto con i casi sopra
    return "Asimmetrica"


DataSet = read("1/dataset.xyz", index=":")
masses = [atom.get_masses() for atom in DataSet]
positions = [atom.get_positions() for atom in DataSet]
n = len(positions)

centers_of_mass = [
    np.average(pos, axis=0, weights=mass) for pos, mass in zip(positions, masses)
]

positions_cm = [pos - com for pos, com in zip(positions, centers_of_mass)]

I = []
for pos, mass in zip(positions_cm, masses):
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    xx = calc_diag_val(y, z, mass)
    yy = calc_diag_val(x, z, mass)
    zz = calc_diag_val(x, y, mass)
    xy = calc_off_val(x, y, mass)
    yz = calc_off_val(y, z, mass)
    xz = calc_off_val(x, z, mass)

    i = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], dtype=float)
    I.append(i)


eigvals = [np.sort(np.linalg.eigvals(i)) for i in I]

counts = {"Sferica": 0, "Oblata": 0, "Prolata": 0, "Asimmetrica": 0}
for vals in eigvals:
    counts[classify_eigs(vals, atol=1.0)] += 1

total = len(eigvals)
print("Molecole analizzate: ", total, "\n")
for k in ("Sferica", "Oblata", "Prolata", "Asimmetrica"):
    n = counts[k]
    perc = n / total * 100.0
    print(f"{k:>12}: {n:4d}  ({perc:6.2f}%)")
