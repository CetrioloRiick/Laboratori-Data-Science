import numpy as np
from sklearn.decomposition import PCA


pippo = np.random.rand(70000, 750)

n_components = np.linspace(5, 750, num=15, dtype=int)
for n in n_components:
    pca_instance = PCA(n_components=n, random_state=42)
    X_train_transformed = pca_instance.fit_transform(pippo)

    cumulative_variance = np.sum(pca_instance.explained_variance_ratio_)

    print("\nNumero di compontenti:", n)
    print("Varianza:", cumulative_variance)
