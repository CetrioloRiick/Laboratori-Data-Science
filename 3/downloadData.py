import pickle
from mp_api.client import MPRester
import numpy as np

# Querying dei dati nulla di sostanziale
API_KEY = "U3nYOxQMoZafxTSlipfQfKGq7lgAJLoB"

FIELDS = [
    "material_id",
    "formula_pretty",
    "band_gap",
    "formation_energy_per_atom",
    "energy_above_hull",
    "density",
    "volume",
    "nsites",
    "symmetry.number",
    "efermi",
    "total_magnetization",
    "total_magnetization_normalized_vol",
    "num_magnetic_sites",
]

FEATURES = [
    "formation_energy_per_atom",
    "energy_above_hull",
    "density",
    "volume",
    "nsites",
    "symmetry.number",
    "efermi",
    "total_magnetization",
    "total_magnetization_normalized_vol",
    "num_magnetic_sites",
]

with MPRester(API_KEY) as mpr:
    docs = mpr.materials.summary.search(
        num_elements=2,
        # band_gap=(0.0001, 10000000000),
        fields=FIELDS,
        num_chunks=1,
        chunk_size=1000,
    )


X = []
Y = []
for doc in docs:
    row = []
    skip = False

    for feat in FEATURES:
        # Questo perchè mp-api è davvero stupido e non capisce se provo a gettagli cose annidate
        if feat == "symmetry.number":
            value = doc.symmetry.number if doc.symmetry else None
        else:
            value = doc.get(feat, None)
        if value is None:
            skip = True
            break
        row.append(value)

    if skip or doc.band_gap is None:
        continue

    X.append(row)
    Y.append(doc.band_gap)

X = np.array(X)
Y = np.array(Y)
pickle.dump(X, open("X.dat", "wb"))
pickle.dump(Y, open("Y.dat", "wb"))
