import numpy as np
import scipy
import matplotlib.pyplot as plt
import joblib
import os
import cv2

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from PIL import Image


def facesProcessing(data):
    faces = mat_contents["faces"].T
    faces_2d = faces.reshape(-1, 168, 192)  # (N,168,192)
    faces_rotate = faces_2d.transpose(0, 2, 1)  # (N,192,168)
    faces_final = faces_rotate.reshape(len(faces), -1)
    return faces_final


def process_image(image_path = "3/3 - EigenFace/foto.png"):
    # 1. Carica l'immagine dal percorso fornito
    img = cv2.imread(image_path)
    
    if img is None:
        raise FileNotFoundError(f"Impossibile trovare o caricare l'immagine al percorso: {image_path}")

    # 2. Conversione in scala di grigi
    # Passiamo da 3 canali (BGR) a 1 singolo canale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Riscalaggio a 192x168
    # Nota: OpenCV usa il formato (larghezza, altezza)
    resized_img = cv2.resize(gray_img, (192, 168))

    # 4. Flattening: trasforma la matrice in un vettore riga singolo
    flat_vector = resized_img.flatten()

    return flat_vector

fc_shape = (192, 168)
mat_contents = scipy.io.loadmat("3/3 - EigenFace/allFaces.mat")
faces = facesProcessing(mat_contents)

""" plt.imshow(faces[0].reshape(fc_shape))
plt.show() """
X_train = faces

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

print(len(pca.components_))

# Le eigenfaces sono contenute in pca.components_
# Ogni riga è una 'faccia' nel linguaggio delle componenti principali
for i in range(100):
    # 1. Prendi la i-esima componente
    eigen_vector = pca.components_[i]
    
    # 2. Reshape (attenzione: verifica se serve il .T in base a come hai caricato i dati)
    eigen_face = eigen_vector.reshape(fc_shape)
    
    # 3. NORMALIZZAZIONE (Fondamentale)
    # Le eigenfaces hanno valori positivi e negativi. 
    # Per vederle come PNG dobbiamo portarle nel range [0, 255]
    f_min, f_max = eigen_face.min(), eigen_face.max()
    eigen_face_normalized = (eigen_face - f_min) / (f_max - f_min) * 255
    
    # 4. Conversione e salvataggio
    img_array = eigen_face_normalized.astype(np.uint8)
    img = Image.fromarray(img_array)
    img.save(f"{i}.png")


""" 
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
 """
print("ho finito cazzoni")
