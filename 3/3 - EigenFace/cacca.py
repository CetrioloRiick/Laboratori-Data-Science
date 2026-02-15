import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


mat_contents = scipy.io.loadmat("3/3 - EigenFace/allFaces.mat")
faces = mat_contents["faces"].T

st_instance = StandardScaler()
faces_scaled = st_instance.fit_transform(faces)

ca_instance = PCA(n_components=50, random_state=42)
faces_transformed = pca_instance.fit_transform(faces_scaled)

while True:
    i = int(input("primo ciclo quante componenti vuoi fare la pca: "))
    pca_instance = PCA(n_components=i, random_state=42)
    faces_transformed = pca_instance.fit_transform(faces_scaled)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    fig.suptitle(f"Originale vs Ricostruzione", fontsize=12)
    n = int(input("numerino fino a 10.000"))
    faces_inverse = pca_instance.inverse_transform(faces_transformed)
    X_sample = faces_inverse[n]
    axes[0].imshow(np.reshape(X_sample, shape=(168, 192)), cmap="gray")
    # print("questo era un:", y_test.iloc[[n]])
    # print("secondo il modello era un:", pippo.predict(X_test.iloc[[n]]))
    axes[1].imshow(np.reshape(faces_scaled[n], (168, 192)), cmap="gray")
    plt.show()


print("ho finito cazzoni")
