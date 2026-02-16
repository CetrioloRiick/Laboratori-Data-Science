import numpy as np
import scipy
import matplotlib.pyplot as plt
import joblib

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def facesProcessing(data):
    faces = mat_contents["faces"].T
    faces_2d = faces.reshape(-1, 168, 192)   # (N,168,192)
    faces_rotate = faces_2d.transpose(0, 2, 1)  # (N,192,168)
    faces_final = faces_rotate.reshape(len(faces), -1)
    return faces_final

fc_shape = (192, 168)
mat_contents = scipy.io.loadmat("3/3 - EigenFace/allFaces.mat")
faces = facesProcessing(mat_contents)

plt.imshow(faces[0].reshape(fc_shape))
plt.show()

""" 

 
st_instance = StandardScaler()
faces_scaled = st_instance.fit_transform(faces)


# CARICAAMENTO DEL MODELLO
path_pca = "3/3 - EigenFace/pca.pkl"
if os.path.exists(path_pca):
    print(f"Caricamento modello PCA...\n")
    pca = joblib.load(path_pca)

else:
    pca = PCA(random_state=42)

    print("Addestramento PCA...")
    pca.fit(X_train)
    print("Addestramento completato.\n")

    # Salvataggio modello
    joblib.dump(pca, path_pca)
    print(f"Modello salvato in '{path_pca}'.")

mean_face = pca.mean_.reshape(m, n).T  # sklearn salva cosi la faccia media
eigenfaces = pca.components_  # lista con tutte le autofacce
# Visualizzo le auto facce una alla volta, iniziamo con le prime 10
fig, axes = plt.subplots(2, 5, figsize=(10, 8), constrained_layout=True)
axes = axes.ravel()
for i, ax in enumerate(axes):

    eigenFace = pca.components_[i]
    image = np.reshape(eigenFace, shape=(168, 192))
    ax.imshow(image, cmap="gray")
plt.show()


img = plt.imread("3/3 - EigenFace/foto.png").T.ravel()
print(img)
plt.imshow(np.reshape(img, shape=(168, 192)), cmap="gray")
plt.show()

k = np.linspace(10, 50, dtype=int, num=5)
for i in k:
    img_transformed = pca.transform([img])
    img_transformed[:, i:] = 0
    img_reverse = pca.inverse_transform(img_transformed)
    plt.imshow(np.reshape(img_reverse[0], shape=(168, 192)), cmap="gray")
    plt.show()

print("ho finito cazzoni")
 """