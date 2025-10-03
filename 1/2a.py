import numpy as np
import matplotlib.pyplot as plt

epsilon_range = np.linspace(-1e-5, 1e-5, 21)

for epsilon in epsilon_range:
    matrix = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 3.0],
            [epsilon, 0.0, 0.0, 0.0],
        ]
    )

    eigenvalues = np.linalg.eigvals(matrix)
    singular_values = np.linalg.svdvals(matrix)

    print("=" * 60)
    print(f"ε = {epsilon:.2e}")
    print(f"Eigenvalues      : {np.round(eigenvalues, 6)}")
    print(f"Singular values  : {np.round(singular_values, 6)}")
