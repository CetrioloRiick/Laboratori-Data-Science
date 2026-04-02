import numpy as np
import scipy
import matplotlib.pyplot as plt
import joblib
import os
import time

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from PIL import Image


def facesProcessing(data):
    faces = mat_contents["faces"].T
    faces_2d = faces.reshape(-1, 168, 192)  # (N,168,192)
    faces_rotate = faces_2d.transpose(0, 2, 1)  # (N,192,168)
    faces_final = faces_rotate.reshape(len(faces), -1)
    return faces_final


def process_image(image_path="3/3 - EigenFace/foto.png"):
    # Carica, converte in scala di grigi ('L') e riscala
    img = Image.open(image_path).convert("L")
    visualizzamiStaPIL(img)
    # Converte in array numpy e appiattisce
    return np.array(img).flatten()


def visualizzamiStaFaccia(vettore, k=0):
    plt.imshow(np.reshape(vettore, fc_shape), cmap="gray")
    plt.savefig("/tmp/" + idRun + "k" + str(k) + ".png")

    plt.show()

def visualizzamiStaPIL(vettore, k=0):
    img_array = np.reshape(vettore, fc_shape)
    img_array = img_array.astype(np.uint8)
    img = Image.fromarray(img_array)
    img.save(f"/tmp/{idRun}k={k}.png")



def pss(model, vector):
    """Esegue il transform su un singolo vettore flat."""
    return model.transform(vector.reshape(1, -1))[0]


def pss_inverse(model, vector):
    """Esegue il transform su un singolo vettore flat."""
    return model.inverse_transform(vector.reshape(1, -1))[0]


idRun = str(time.time())
vuoiSalvareLe100AutoFacce = False
fc_shape = (192, 168)
mat_contents = scipy.io.loadmat("3/3 - EigenFace/allFaces.mat")
faces = facesProcessing(mat_contents)

""" plt.imshow(faces[0].reshape(fc_shape))
plt.show() """
# X_train = faces

st_instance = StandardScaler()
faces_scaled = st_instance.fit_transform(faces)
X_train = faces_scaled


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
if vuoiSalvareLe100AutoFacce:
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

myFace = pss(st_instance, process_image())
tigre = pss(st_instance, process_image("3/3 - EigenFace/tigre.png"))
# myFace = process_image()
# myFace = faces_scaled[np.random.randint(0, 2410)]
visualizzamiStaFaccia(myFace)
visualizzamiStaFaccia(tigre)

myFacePCA = pss(pca, myFace)
tigrePCA = pss(pca, tigre)



if False:
    componentiPerVisualizzareLaMiaFaccia = np.linspace(5, len(myFacePCA), 120, dtype=int)

    for element in componentiPerVisualizzareLaMiaFaccia:
        myFacePCA_filtered = myFacePCA.copy()

        # Imposta a zero tutti gli elementi dalla posizione 'element' in poi
        myFacePCA_filtered[element:] = 0
        facciaDaVedere = pss_inverse(pca, myFacePCA_filtered)
        facciaDaVedere2 = pss_inverse(st_instance, facciaDaVedere)
        visualizzamiStaPIL(facciaDaVedere2, element)
        plt.close("all")

if False:
    componentiPerVisualizzareTigre = np.linspace(5, len(tigrePCA), 20, dtype=int)

    for element in componentiPerVisualizzareTigre:
        tigrePCA_filtered = tigrePCA.copy()

        # Imposta a zero tutti gli elementi dalla posizione 'element' in poi
        tigrePCA_filtered[element:] = 0
        facciaDaVedere = pss_inverse(pca, tigrePCA_filtered)
        visualizzamiStaFaccia(facciaDaVedere)
        facciaDaVedere2 = pss_inverse(st_instance, facciaDaVedere)

        visualizzamiStaPIL(facciaDaVedere2, element)
        plt.close("all")


if True:
    plt.close("all")
    shanelX = range(len(myFacePCA))
    shanelY = []
    for i in shanelX:
        myFacePCA_filtered = myFacePCA.copy()
        myFacePCA_filtered[i:] = 0
        myFaceRestored = pss_inverse(pca, myFacePCA_filtered)
        err = mean_squared_error(myFace, myFaceRestored)
        shanelY.append(err)
    print("errore faccia: " + str(err))

    petrelX = range(len(tigrePCA))
    petrelY = []
    for i in petrelX:
        tigrePCA_filtered = tigrePCA.copy()
        tigrePCA_filtered[i:] = 0
        tigreRestored = pss_inverse(pca, tigrePCA_filtered)
        err = mean_squared_error(tigre, tigreRestored)
        petrelY.append(err)
    print("errore tigre: " + str(err))

    plt.plot(shanelX, shanelY, color='red')
    plt.plot(petrelX, petrelY, color='blue')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.show()

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
