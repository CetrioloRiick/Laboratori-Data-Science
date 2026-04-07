import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score



# Fissiamo il seed per la riproducibilità
SEED = 1004
np.random.seed(SEED)
torch.manual_seed(SEED)

# ==========================================
# (a) Preparazione dei dati
# ==========================================
# Creiamo 10000 coppie di numeri casuali tra 0 e 999
X_numpy = np.random.randint(0, 1000, size=(10000, 2))
# Calcoliamo la somma per ogni coppia
y_numpy = np.sum(X_numpy, axis=1, keepdims=True)

# Convertiamo i dati in Tensori di PyTorch (usiamo float32 perché i pesi della rete sono float)
X = torch.tensor(X_numpy, dtype=torch.float32)
y = torch.tensor(y_numpy, dtype=torch.float32)

# ==========================================
# (b) Definizione della rete neurale
# ==========================================
# Modello sequenziale con 2 layer nascosti (64 neuroni, ReLU) e 1 layer di output
model = nn.Sequential(
    nn.Linear(in_features=2, out_features=64),  # Input layer -> Hidden 1
    nn.ReLU(),  # Attivazione ReLU
    nn.Linear(in_features=64, out_features=64),  # Hidden 1 -> Hidden 2
    nn.ReLU(),  # Attivazione ReLU
    nn.Linear(in_features=64, out_features=1),  # Hidden 2 -> Output (1 neurone)
)

# ==========================================
# (c) Configurazione dell'addestramento
# ==========================================
# Ottimizzatore Adam e funzione di perdita MSE (Mean Squared Error)
optimizer = optim.Adam(model.parameters(), lr=0.01)  # lr è il learning rate
criterion = nn.MSELoss()

# ==========================================
# (d) Addestramento del modello
# ==========================================
# Dividiamo i dati: 80% training, 20% validation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)
print(X_train[1])

epochs = 100
costsTrainToPlot = []
costsTestToPlot = []

for epoch in range(epochs):
    model.train()  # Impostiamo il modello in modalità addestramento

    # 1. Forward pass (Calcolo delle predizioni)
    predictions = model(X_train)

    # 2. Calcolo della perdita (errore)
    loss = criterion(predictions, y_train)

    # 3. Backward pass e ottimizzazione
    optimizer.zero_grad()  # Azzeriamo i gradienti precedenti
    loss.backward()  # Calcoliamo i nuovi gradienti (backpropagation)
    optimizer.step()  # Aggiorniamo i pesi

    model.eval()  # Modalità valutazione
    with torch.no_grad():
        val_predictions = model(X_train)
        val_loss_train = criterion(val_predictions, y_train)

        val_predictions = model(X_test)
        val_loss_test = criterion(val_predictions, y_test)


        y_test_numpy = y_test.numpy()
        val_pred_numpy = val_predictions.numpy()

        # Calcoliamo la metrica
        r2 = r2_score(y_test_numpy, val_pred_numpy)


    # print(f'Epoca [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
    costsTrainToPlot.append(val_loss_train)
    costsTestToPlot.append(val_loss_test)

plt.plot(range(epochs), costsTrainToPlot)
print(f"Costo train: {val_loss_train}")
print(f"Costo test: {val_loss_test}")
print(f"R^2: {r2}")

plt.plot(range(epochs), costsTestToPlot)
plt.show()



# ==========================================
# (e) Predizioni su nuovi dati
# ==========================================
print("\n--- (e) Predizione su una nuova coppia ---")
coppietest = [
    [123.0, 456.0],
    [0.1, 0.1],
    [-20.0, 10],
    [-20, -15],
    [999, 998],
    [2000, 4000],
    [-2000, -4000],
]

for i in coppietest:
    model.eval()  # Modalità valutazione prima di fare inferenza
    nuova_coppia = torch.tensor([i], dtype=torch.float32)

    with torch.no_grad():
        predizione_e = model(nuova_coppia)

    somma_reale_e = i[1] + i[0]
    print(f"Input: {i}")
    print(f"Somma Reale: {somma_reale_e}")
    print(f"Somma Predetta: {predizione_e.item():.4f}")

# Commento: La rete neurale non "impara l'algoritmo" dell'addizione esatta,
# ma approssima una funzione continua. Per questo motivo il risultato è
# un numero con la virgola (float) vicinissimo a 579, ma non esattamente 579.

# ==========================================
# (f) Lista di test finale
# ==========================================
print("\n--- (f) Test su lista specifica ---")
lista_test = [[5, 7], [100, 200], [400, 200], [999, 999]]
test_tensor = torch.tensor(lista_test, dtype=torch.float32)

with torch.no_grad():
    predizioni_f = model(test_tensor)

for i, coppia in enumerate(lista_test):
    somma_reale = sum(coppia)
    somma_predetta = predizioni_f[i].item()
    print(
        f"Coppia: {coppia} | Somma Reale: {somma_reale} | Predizione: {somma_predetta:.4f}"
    )
