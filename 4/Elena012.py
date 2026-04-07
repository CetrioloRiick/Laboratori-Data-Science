import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# 1. Preparazione Dati (con iniezione di zeri)
SEED = 1004
np.random.seed(SEED)
torch.manual_seed(SEED)

N_tot = 10000
X_random = np.random.randint(-999, 1000, size=(9000, 2))
X_zeros_base = np.random.randint(-999, 1000, size=(1000, 1))
X_zeros = np.hstack((X_zeros_base, -X_zeros_base))  # Crea coppie come [15, -15]
X_numpy = np.vstack((X_random, X_zeros))
np.random.shuffle(X_numpy)

somme = np.sum(X_numpy, axis=1)
# (I dati X_numpy e somme rimangono gli stessi della V1)

# Codifica: -1, 0, 1
y_v2_numpy = np.sign(somme).reshape(-1, 1)  # np.sign fa esattamente questo!

X = torch.tensor(X_numpy, dtype=torch.float32)
y_v2 = torch.tensor(y_v2_numpy, dtype=torch.float32)


X_train, X_val, y_train_v2, y_val_v2 = train_test_split(
    X, y_v2, test_size=0.2, random_state=SEED
)
print(X_train[1])

# Modello identico all'Es 1 (1 neurone in output)
model_v2 = nn.Sequential(
    nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1)
)

optimizer_v2 = optim.Adam(model_v2.parameters(), lr=0.01)
criterion_v2 = nn.MSELoss()  # Torniamo all'MSE

costsTestToPlot = []
costsTrainToPlot = []
epochs = 100

for epoch in range(epochs):
    model_v2.train()
    optimizer_v2.zero_grad()
    predictions = model_v2(X_train)
    loss = criterion_v2(predictions, y_train_v2)
    loss.backward()
    optimizer_v2.step()


    
    model_v2.eval()
    with torch.no_grad():
        val_preds = model_v2(X_val)
        # Arrotondiamo il valore predetto all'intero più vicino (es. 0.8 diventa 1.0)
        pred_rounded = torch.round(val_preds)
        corrette = (pred_rounded == y_val_v2).sum().item()
        accuratezza = corrette / y_val_v2.size(0) * 100
        
        val_predictions = model_v2(X_train)
        val_loss_train = criterion_v2(val_predictions, y_train_v2)

        val_predictions = model_v2(X_val)
        val_loss_test = criterion_v2(val_predictions, y_val_v2)


        y_test_numpy = y_val_v2.numpy()
        val_pred_numpy = val_predictions.numpy()

        # Calcoliamo la metrica
        r2 = r2_score(y_test_numpy, val_pred_numpy)
    costsTrainToPlot.append(val_loss_test)
    costsTestToPlot.append(val_loss_test)
    print(
        f"V2 - Epoca [{epoch+1}/{epochs}], Accuratezza Validazione: {accuratezza:.2f}%"
    )

plt.plot(range(epochs), costsTrainToPlot)
print(f"Costo train: {val_loss_train}")
print(f"Costo test: {val_loss_test}")
print(f"R^2: {r2}")

plt.plot(range(epochs), costsTestToPlot)
plt.show()
