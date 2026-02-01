import numpy as np
from mp_api.client import MPRester
from sklearn.linear_model import LinearRegression

# Legge la API key dall'ambiente
API_KEY = "U3nYOxQMoZafxTSlipfQfKGq7lgAJLoB"

FIELDS = [
    "material_id",
    "formula_pretty",
    "band_gap",  # <-- aggiunto target
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

# ordine delle 10 feature (come richiesto)
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
        fields=FIELDS,
        num_chunks=1,
        chunk_size=10,
    )

X=np.array(docs)

print(X[0])
