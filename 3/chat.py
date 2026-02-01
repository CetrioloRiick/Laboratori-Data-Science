#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esercizio 1 (Foglio 3) — Regressione lineare per predire il band gap dei composti binari
Database: Materials Project (Summary endpoint)
Librerie consentite: numpy, sklearn (+ standard library)

Cosa fa questo script:
1) Scarica da Materials Project i composti binari (nelements=2) con:
   - material_id, formula_pretty (solo identificazione, NON usate come feature)
   - band_gap (target)
   - 10 feature richieste nel testo dell'esercizio
2) Preprocessa (imputazione mediana + standardizzazione)
3) Calcola correlazioni (Pearson) tra ciascuna feature e band_gap
4) Addestra e valuta:
   - LinearRegression (baseline)
   - RidgeCV
   - LassoCV
   - Regressione polinomiale (PolynomialFeatures + RidgeCV)
5) Risponde alle domande stampando:
   - feature più correlate
   - dove sbaglia di più (MAE per bin di band_gap)
   - se regolarizzazione migliora la generalizzazione + confronto coefficienti
   - effetto della regressione polinomiale

Uso rapido:
  export MP_API_KEY="LA_TUA_API_KEY"
  python esercizio1_bandgap_mp.py --max-items 5000

Nota:
- La chiamata API richiede una API key di Materials Project.
- Se la rete è lenta, usa --cache per salvare/riprendere i dati.
"""

import argparse
import json
import time
import urllib.parse
import urllib.request
from urllib.error import HTTPError, URLError

import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_URL = "https://api.materialsproject.org"

# Mappatura "nome colonna" -> (come trovarla nel JSON MP)
# NB: symmetry.number può arrivare come symmetry={"number":...} oppure come "symmetry.number"
FEATURES = [
    ("formation_energy_per_atom", "formation_energy_per_atom"),
    ("energy_above_hull", "energy_above_hull"),
    ("density", "density"),
    ("volume", "volume"),
    ("nsites", "nsites"),
    ("symmetry_number", "symmetry.number"),
    ("efermi", "efermi"),
    ("total_magnetization", "total_magnetization"),
    ("total_magnetization_normalized_vol", "total_magnetization_normalized_vol"),
    ("num_magnetic_sites", "num_magnetic_sites"),
]

ID_FIELDS = ["material_id", "formula_pretty"]
TARGET_FIELD = "band_gap"


def _get_api_key(cmdline_key: str | None) -> str:
    key = cmdline_key or ("" if "MP_API_KEY" not in os.environ else os.environ["MP_API_KEY"])
    key = key.strip()
    if not key:
        raise SystemExit(
            "Errore: manca la API key.\n"
            "Imposta la variabile d'ambiente MP_API_KEY oppure passa --api-key."
        )
    return key


def _mp_get(url: str, api_key: str, retries: int = 5, backoff_s: float = 1.5) -> dict:
    """GET con retry basilare (gestisce 429 e errori temporanei)."""
    headers = {"X-API-KEY": api_key}
    req = urllib.request.Request(url, headers=headers, method="GET")
    last_err = None
    for k in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw)
        except HTTPError as e:
            last_err = e
            # 429 = rate limit: aspetta e riprova
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(backoff_s * (k + 1))
                continue
            raise
        except URLError as e:
            last_err = e
            time.sleep(backoff_s * (k + 1))
            continue
    raise RuntimeError(f"Richiesta fallita dopo {retries} tentativi: {last_err}")


def fetch_binary_summary(
    api_key: str,
    max_items: int | None,
    chunk_size: int = 500,
    sleep_s: float = 0.0,
) -> list[dict]:
    """
    Scarica composti binari (nelements=2) dal Summary endpoint.

    Ritorna: lista di dict (uno per materiale).
    """
    # Campi da richiedere: id + formula + target + feature
    # Per sicurezza richiediamo sia "symmetry.number" sia "symmetry"
    # (alcune versioni dell'API possono differire nel formato di output).
    fields = ID_FIELDS + [TARGET_FIELD] + [src for _, src in FEATURES]
    if "symmetry" not in fields:
        fields.append("symmetry")

    all_rows: list[dict] = []
    skip = 0

    while True:
        limit = chunk_size
        params = {
            "nelements": 2,
            "fields": ",".join(fields),
            "skip": skip,
            "limit": limit,
        }
        url = f"{BASE_URL}/materials/summary/?" + urllib.parse.urlencode(params)
        payload = _mp_get(url, api_key=api_key)
        data = payload.get("data", [])
        if not data:
            break
        all_rows.extend(data)
        skip += len(data)

        if max_items is not None and len(all_rows) >= max_items:
            all_rows = all_rows[:max_items]
            break

        if len(data) < limit:
            break

        if sleep_s > 0:
            time.sleep(sleep_s)

    return all_rows


def _extract_symmetry_number(row: dict) -> float | None:
    # Possibili casi:
    # 1) row["symmetry.number"] già presente
    # 2) row["symmetry"] è un dict con chiave "number"
    if "symmetry.number" in row and row["symmetry.number"] is not None:
        return row["symmetry.number"]
    sym = row.get("symmetry", None)
    if isinstance(sym, dict):
        val = sym.get("number", None)
        return val
    return None


def rows_to_arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Converte lista di record in:
    X: (n_samples, 10) float
    y: (n_samples,) float
    ids: (n_samples,) str
    formulas: (n_samples,) str
    feature_names: lista di 10 nomi
    """
    feat_names = [name for name, _ in FEATURES]
    X_list = []
    y_list = []
    id_list = []
    formula_list = []

    for r in rows:
        # target
        y = r.get(TARGET_FIELD, None)
        if y is None:
            continue

        # id/formula (possono mancare)
        mid = r.get("material_id", "")
        fpr = r.get("formula_pretty", "")

        # feature vector
        vec = []
        ok = True
        for name, src in FEATURES:
            if src == "symmetry.number":
                val = _extract_symmetry_number(r)
            else:
                val = r.get(src, None)
            # teniamo anche i None (verranno imputati), ma scartiamo se non numerico
            if val is None:
                vec.append(np.nan)
            else:
                try:
                    vec.append(float(val))
                except (TypeError, ValueError):
                    ok = False
                    break
        if not ok:
            continue

        X_list.append(vec)
        y_list.append(float(y))
        id_list.append(str(mid))
        formula_list.append(str(fpr))

    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=float)
    ids = np.array(id_list, dtype=str)
    formulas = np.array(formula_list, dtype=str)
    return X, y, ids, formulas, feat_names


def pearson_corr(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Pearson r tra ogni colonna di X e y.
    X e y possono contenere NaN: usiamo imputazione mediana prima.
    """
    X_imp = np.array(X, copy=True)
    # imputazione mediana per colonna (numpy-only)
    for j in range(X_imp.shape[1]):
        col = X_imp[:, j]
        mask = ~np.isnan(col)
        if not np.any(mask):
            # colonna completamente NaN -> correlazione non definita
            X_imp[:, j] = 0.0
            continue
        med = np.median(col[mask])
        col[~mask] = med
        X_imp[:, j] = col

    y_imp = np.array(y, copy=True)
    # se per qualche motivo y avesse NaN
    if np.isnan(y_imp).any():
        medy = np.median(y_imp[~np.isnan(y_imp)])
        y_imp[np.isnan(y_imp)] = medy

    # Pearson per colonne: corr(X_j, y)
    y0 = y_imp - y_imp.mean()
    y_std = y0.std()
    if y_std == 0:
        return np.full(X_imp.shape[1], np.nan, dtype=float)

    r = np.empty(X_imp.shape[1], dtype=float)
    for j in range(X_imp.shape[1]):
        x0 = X_imp[:, j] - X_imp[:, j].mean()
        x_std = x0.std()
        if x_std == 0:
            r[j] = np.nan
        else:
            r[j] = float((x0 @ y0) / (len(y0) * x_std * y_std))
    return r


def evaluate_regressor(name: str, model, X_train, X_test, y_train, y_test, feature_names: list[str]) -> dict:
    """Fit + metriche + coefficienti (se disponibili)."""
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = math.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    coefs = None
    intercept = None
    # gestiamo Pipeline: l'ultimo step ha coef_
    est = model
    if hasattr(model, "named_steps"):
        est = model.named_steps[list(model.named_steps.keys())[-1]]
    if hasattr(est, "coef_"):
        coefs = np.array(est.coef_, dtype=float).ravel()
        intercept = float(est.intercept_) if hasattr(est, "intercept_") else None

    return {
        "name": name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pred": pred,
        "coef": coefs,
        "intercept": intercept,
        "feature_names": feature_names,
        "model": model,
    }


def mae_by_gap_bins(y_true: np.ndarray, y_pred: np.ndarray, bins: list[float]) -> list[tuple[str, float, int]]:
    """MAE per intervalli di band_gap."""
    abs_err = np.abs(y_pred - y_true)
    out = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_true >= lo) & (y_true < hi)
        n = int(mask.sum())
        if n == 0:
            out.append((f"[{lo}, {hi})", float("nan"), 0))
        else:
            out.append((f"[{lo}, {hi})", float(abs_err[mask].mean()), n))
    # ultimo bin [bins[-1], +inf)
    lo = bins[-1]
    mask = (y_true >= lo)
    n = int(mask.sum())
    if n == 0:
        out.append((f"[{lo}, +inf)", float("nan"), 0))
    else:
        out.append((f"[{lo}, +inf)", float(abs_err[mask].mean()), n))
    return out


def print_coef_table(feature_names: list[str], coef_dict: dict[str, np.ndarray]) -> None:
    """Stampa tabella coefficienti affiancati (feature x modelli)."""
    models = list(coef_dict.keys())
    widths = [max(len("feature"), max(len(f) for f in feature_names))]
    widths += [max(len(m), 12) for m in models]

    header = "  ".join(
        ["feature".ljust(widths[0])]
        + [m.rjust(widths[i + 1]) for i, m in enumerate(models)]
    )
    print(header)
    print("-" * len(header))

    for j, f in enumerate(feature_names):
        row = [f.ljust(widths[0])]
        for i, m in enumerate(models):
            c = coef_dict[m]
            if c is None:
                row.append(" " * widths[i + 1])
            else:
                row.append(f"{c[j]: .6f}".rjust(widths[i + 1]))
        print("  ".join(row))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", type=str, default=None, help="API key MP (in alternativa a MP_API_KEY)")
    parser.add_argument("--cache", type=str, default="mp_binary_bandgap_cache.npz", help="File cache .npz")
    parser.add_argument("--no-cache", action="store_true", help="Ignora cache e riscarica dati")
    parser.add_argument("--max-items", type=int, default=5000, help="Numero massimo di materiali da scaricare (None=all)")
    parser.add_argument("--chunk-size", type=int, default=500, help="Dimensione pagine API")
    parser.add_argument("--sleep", type=float, default=0.0, help="Pausa (s) tra chiamate API (utile per rate limit)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Frazione test set")
    parser.add_argument("--seed", type=int, default=0, help="Seed random")
    parser.add_argument("--poly-degree", type=int, default=2, help="Grado polinomiale (>=2)")
    args = parser.parse_args()

    api_key = _get_api_key(args.api_key)

    # 1) Carica cache o scarica
    cache_path = args.cache
    if (not args.no_cache) and cache_path and os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        X = z["X"]
        y = z["y"]
        ids = z["ids"]
        formulas = z["formulas"]
        feature_names = list(z["feature_names"])
        print(f"[cache] Caricati {len(y)} materiali da {cache_path}")
    else:
        print("[download] Scarico dati da Materials Project (summary, nelements=2)...")
        rows = fetch_binary_summary(
            api_key=api_key,
            max_items=args.max_items if args.max_items and args.max_items > 0 else None,
            chunk_size=args.chunk_size,
            sleep_s=args.sleep,
        )
        print(f"[download] Ricevuti {len(rows)} record grezzi.")
        X, y, ids, formulas, feature_names = rows_to_arrays(rows)
        print(f"[prep] Campioni dopo parsing (target presente): {len(y)}")

        if cache_path and not args.no_cache:
            np.savez_compressed(
                cache_path,
                X=X,
                y=y,
                ids=ids,
                formulas=formulas,
                feature_names=np.array(feature_names, dtype=str),
            )
            print(f"[cache] Salvato cache in {cache_path}")

    if len(y) < 50:
        raise SystemExit("Troppi pochi campioni: aumenta --max-items o verifica la API key/connessione.")

    # 2) Correlazioni (Pearson) feature-target
    r = pearson_corr(X, y)
    order = np.argsort(-np.abs(r))  # decrescente per |r|
    print("\n=== Correlazioni (Pearson) con band_gap (ordinate per |r|) ===")
    for idx in order:
        print(f"{feature_names[idx]:35s}  r = {r[idx]: .4f}")

    # 3) Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )

    # Pipeline comune: imputazione mediana + standardizzazione
    pre = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]

    # 4) Modelli
    lr = Pipeline(pre + [("reg", LinearRegression())])

    alphas = np.logspace(-3, 3, 13)
    ridge = Pipeline(pre + [("reg", RidgeCV(alphas=alphas))])

    # LassoCV seleziona alpha via CV interna (qui 5-fold di default)
    lasso = Pipeline(pre + [("reg", LassoCV(alphas=None, cv=5, random_state=args.seed, max_iter=50000))])

    results = []
    results.append(evaluate_regressor("LinearRegression", lr, X_train, X_test, y_train, y_test, feature_names))
    results.append(evaluate_regressor("RidgeCV", ridge, X_train, X_test, y_train, y_test, feature_names))
    results.append(evaluate_regressor("LassoCV", lasso, X_train, X_test, y_train, y_test, feature_names))

    print("\n=== Performance su test set ===")
    for res in results:
        print(f"{res['name']:15s}  R2={res['r2']:.4f}  MAE={res['mae']:.4f} eV  RMSE={res['rmse']:.4f} eV")

    # 5) Dove sbaglia di più? (bin di band gap)
    # bins scelti "ragionevoli" per gap: 0-1-2-4 eV e oltre
    bins = [0.0, 1.0, 2.0, 4.0]
    base = results[0]
    bybin = mae_by_gap_bins(y_test, base["pred"], bins=bins)
    print("\n=== Dove sbaglia di più? (MAE per bin di band_gap) — baseline LinearRegression ===")
    for label, mae, n in bybin:
        print(f"{label:12s}  n={n:5d}  MAE={mae:.4f} eV")

    # 6) Regolarizzazione: confronto coefficienti
    coef_dict = {
        "Linear": results[0]["coef"],
        "Ridge": results[1]["coef"],
        "Lasso": results[2]["coef"],
    }
    print("\n=== Confronto coefficienti (sulle feature standardizzate) ===")
    print_coef_table(feature_names, coef_dict)

    # Coeff. zero di Lasso?
    if results[2]["coef"] is not None:
        nz = int(np.sum(np.abs(results[2]["coef"]) > 1e-12))
        print(f"\n[Lasso] coefficienti non nulli: {nz}/{len(feature_names)}")

    # 7) La regolarizzazione migliora la generalizzazione? (CV su training)
    # Nota: CV su training per confronto più robusto
    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    print("\n=== 5-fold CV sul training (MAE medio ± std; R2 medio ± std) ===")
    for name, model in [("LinearRegression", lr), ("RidgeCV", ridge), ("LassoCV", lasso)]:
        mae_scores = -cross_val_score(model, X_train, y_train, cv=kf, scoring="neg_mean_absolute_error")
        r2_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2")
        print(f"{name:15s}  MAE={mae_scores.mean():.4f}±{mae_scores.std():.4f}  R2={r2_scores.mean():.4f}±{r2_scores.std():.4f}")

    # 8) Regressione polinomiale
    if args.poly_degree >= 2:
        poly = Pipeline(pre + [
            ("poly", PolynomialFeatures(degree=args.poly_degree, include_bias=False)),
            ("reg", RidgeCV(alphas=alphas)),
        ])
        poly_res = evaluate_regressor(
            f"Poly(deg={args.poly_degree})+RidgeCV",
            poly, X_train, X_test, y_train, y_test,
            feature_names=feature_names,  # base names (per coeff serve feature expanded)
        )
        print("\n=== Regressione polinomiale ===")
        print(f"{poly_res['name']:22s}  R2={poly_res['r2']:.4f}  MAE={poly_res['mae']:.4f} eV  RMSE={poly_res['rmse']:.4f} eV")
        print("Nota: il numero di feature aumenta molto (termini quadratici/cross-term), quindi Ridge è consigliato.")

    # 9) (Opzionale) mostra i peggiori errori (baseline)
    abs_err = np.abs(base["pred"] - y_test)
    worst_idx = np.argsort(-abs_err)[:10]
    print("\n=== Top 10 peggiori predizioni (baseline) ===")
    # ids/formulas sono allineati con X originale, ma qui abbiamo fatto train_test_split senza tenere indici.
    # Ricaviamo gli indici originali ricostruendo con una maschera: più semplice è rifare lo split sugli indici.
    # Qui stampiamo solo y_true, y_pred e errore.
    for k, i in enumerate(worst_idx, 1):
        print(f"{k:2d}) y_true={y_test[i]:.3f}  y_pred={base['pred'][i]:.3f}  |err|={abs_err[i]:.3f}")

    print("\nFatto.")


if __name__ == "__main__":
    import os
    import math
    main()
